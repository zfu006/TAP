import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn import init
import numpy as np

"""
# --------------------------------------------
# GPU parallel setup
# --------------------------------------------
# """

def model_to_device(network, gpu_ids=[0]):
    device = torch.device('cuda:{}'.format(gpu_ids[0]) if gpu_ids and torch.cuda.is_available() else 'cpu')  
    message = ''
    message += '--------------------------------------------\n'
    
    if torch.cuda.device_count()==0:
        message += 'use cpu.\n'
    elif len(gpu_ids)==1 or torch.cuda.device_count()==1:
        message += 'use single gpu.\n'
    else:
        network = DataParallel(network, device_ids=gpu_ids)
        message += 'use multiple-gpus (DataParallel).\n'

    message += '--------------------------------------------\n'
    network.to(device)
    print(message)

    return network, message, device

"""
# -------------------------------------------
# init networks
# -------------------------------------------
"""
def init_weights(network, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        network (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s\n' % init_type)
    network.apply(init_func)  # apply the initialization function <init_func>

    return network

"""
# -----------------------------------------------
# get bare model, unwrap models from Dataparallel
# -----------------------------------------------
"""
def get_bare_model(network):
    if isinstance(network, (DataParallel, DistributedDataParallel)):
        network = network.module
    return network

"""
# --------------------------------------------
# print network params 
# --------------------------------------------
"""
def print_networks(network, network_name, verbose=False):
    """Print the total number of parameters in the network and (if verbose) network architecture

    Parameters:
    verbose (bool) -- if verbose: print the network architecture
    """
    network = get_bare_model(network=network)
    message = '------------ Network initialized -----------\n'
    # calculate the number of network params
    message += '[Network {}] params number: {:.3f} M'.format(network_name, sum(map(lambda x: x.numel(), network.parameters()))/1e6) + '\n'
    
    # print network structure for debug
    if verbose:
        message += 'Net structure:\n{}'.format(str(network)) + '\n'
    message += '--------------------------------------------\n'
    print(message)

    return message

"""
# --------------------------------------------
# pixel level loss
# --------------------------------------------
"""
def pixel_loss(type='L1'):
    if type == 'L1':
        return nn.L1Loss()
    
    elif type == 'L2':
        return nn.MSELoss()
    
    elif type == 'psnr':
        return PSNRLoss()

    else:
        raise NotImplementedError('loss type [{}] is not found, default loss types are L1, L2'.format(type))

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        
"""
# --------------------------------------------
# Blind-spot conv V1
# --------------------------------------------
"""
class Blind_spotConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Blind_spotConv2d, self).__init__(*args, **kwargs)
        self.weight_mask = self.get_weight_mask()

    def get_weight_mask(self):
        weight = np.ones((1, 1, self.kernel_size[0], self.kernel_size[1]))
        weight[:, :, self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
        return torch.tensor(weight.copy(), dtype=torch.float32)
        
    def forward(self, x):
        if self.weight_mask.type() != self.weight.type():
            with torch.no_grad():
                self.weight_mask = self.weight_mask.type(self.weight.type())
        device = self.weight.device
        self.weight_mask = self.weight_mask.to(device)
        w=torch.mul(self.weight,self.weight_mask)
        output = F.conv2d(x, w, self.bias, self.stride,
                           self.padding)
        
        return output
    
"""
# --------------------------------------------
# Blind-spot conv V2
# --------------------------------------------
"""
class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(CentralMaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask = torch.ones_like(self.weight.data)
        self.mask[:, :, self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
        self.mask.requires_grad = False

    def forward(self, x):
        self.mask = self.mask.to(self.weight.device)
        masked_weight = self.weight * self.mask
    
        output = F.conv2d(x, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output
        