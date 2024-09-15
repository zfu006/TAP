import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import logging
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

from options.option_pretrain import BaseOptions
from utils.utils_logger import logger_info
from data.pretrain_im_dataset import pretrain_raw_im_dataset, test_raw_im_dataset
from models.network_nafnet import Baseline
import utils.utils_network as net_util
import utils.utils_image as utils

def test_pretrain_crvd(net, device, dataloader):
    psnr = 0
    ssim = 0
    num = 0

    net.eval()
    for i, data in enumerate(tqdm(dataloader, desc="Testing", leave=False)):
        noisy_img = data['noisy'].to(device)
        clean_img = data['clean'].to(device)
        num = num + noisy_img.shape[0]
        with torch.no_grad():
            pred_img = net(noisy_img)

        # discrete the output image
        pred_img = pred_img.cpu().numpy()
        pred_img = np.clip(pred_img, 0, 1)
        pred_img = (np.uint16(pred_img*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240)
        pred_img = torch.from_numpy(pred_img).to(device)
        psnr_ = utils.calculate_psnr_pt(pred_img, clean_img, crop_border=0)
        ssim_ = utils.calculate_ssim_pt(pred_img, clean_img, crop_border=0)
        psnr = psnr + psnr_.sum()   
        ssim = ssim + ssim_.sum()

    return psnr/num, ssim/num

def main():
    # -------------------------------------------------
    # Initiate logs and args
    # -------------------------------------------------
    opt, opt_message = BaseOptions().parse()

    # create save folders
    if not os.path.exists(opt.save_logs_dir):
        os.makedirs(opt.save_logs_dir)
    if not os.path.exists(opt.save_models_dir):
        os.makedirs(opt.save_models_dir)
    if not os.path.exists(opt.save_imgs_dir):
        os.makedirs(opt.save_imgs_dir)

    logger_name = 'train'
    logger_info(logger_name, log_path=opt.save_logs_dir + logger_name + '.log')
    logger = logging.getLogger(logger_name)
    logger.info(opt_message)

    # -------------------------------------------------
    # Prepare data
    # -------------------------------------------------
    train_dataset = pretrain_raw_im_dataset(source_path=opt.train_clean_img_dir, 
                                            patch_size=opt.patch_size, 
                                            datasets=opt.datasets,
                                            noise_aug=opt.noise_aug,
                                            bayer_aug=opt.bayer_aug)
    
    test_dataset = test_raw_im_dataset(gt_dir=opt.test_clean_img_dir,
                                       noisy_dir=opt.test_noisy_img_dir)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.train_batch_size,
        shuffle=not opt.not_train_shuffle,
        drop_last=not opt.not_train_drop_last,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=opt.test_batch_size, 
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True
    )

    # -------------------------------------------------
    # Networks, optimizers, loss
    # -------------------------------------------------
    
    # define network
    net_G = Baseline(img_channel=opt.in_nc, width=opt.nc, middle_blk_num=6, 
                     enc_blk_nums=[2, 2, 4], dec_blk_nums=[2, 2, 2], dw_expand=2, ffn_expand=2)
    
    # print network
    net_message = net_util.print_networks(network=net_G, network_name='G', verbose=opt.verbose)
    logger.info(net_message)
    # initiate networks
    net_G = net_util.init_weights(network=net_G, init_type=opt.init_type)
    # model to device -- return net, gpu message and device info 
    net_G, gpu_message, device = net_util.model_to_device(network=net_G, gpu_ids=opt.gpu_ids)
    logger.info(gpu_message)

    # optimizer setting
    optim_G = optim.AdamW(net_G.parameters(), opt.G_lr, betas=(0.9, 0.9))
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer=optim_G, T_max=opt.epochs, eta_min=1e-6, verbose=True)

    # loss setting
    Cri_rec = net_util.pixel_loss(type=opt.pixel_loss_type)

    # -------------------------------------------------
    # Training, testing
    # -------------------------------------------------
    current_step = 0
    for epoch in range(opt.epochs):
        epoch += 1
        for i, data in enumerate(train_loader):
            current_step += 1
            net_G.train()

            noisy_img = data['noisy'].to(device)
            clean_img = data['clean'].to(device)

            log_dict = OrderedDict()

            # 1) feed data
            rec_img = net_G(noisy_img) # reconstruction image

            # 2) optimizer networks
            optim_G.zero_grad()
            rec_loss = Cri_rec(rec_img, clean_img) # reconstruction loss
            rec_loss.backward()
            optim_G.step()

            # 3) logs
            log_dict['rec_loss'] = rec_loss.item()

            loss_message = '<epoch:{:d}, step: {:08d}> '.format(epoch, current_step)
            for k, v in log_dict.items():  # merge log information into message
                loss_message += '{:s}: {:.3e} '.format(k, v)
            print(loss_message)
            logger.info(loss_message)

            if current_step % opt.test_step == 0:
                net_G.eval()
                psnr, ssim = test_pretrain_crvd(net=net_G, device=device, dataloader=test_loader)
                message = '<epoch:{:d}, step: {:08d}> psnr: {:.3f} ssim: {:.3f}'.format(epoch, current_step, psnr.item(), ssim.item())
                print(message)
                logger.info(message)
                net_G.train()

                if opt.save_models:
                    torch.save(net_util.get_bare_model(net_G).state_dict(), opt.save_models_dir+'net_{:05d}_{:05d}.pth'.format(epoch, current_step))


        scheduler_G.step()


if __name__ == '__main__':
    main()