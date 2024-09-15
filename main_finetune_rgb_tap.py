import argparse
import os
import logging
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import cv2
from tqdm import tqdm
from collections import OrderedDict


from options.option_finetune_rgb import BaseOptions
from utils.utils_logger import logger_info
from data.finetune_vid_dataset import test_rgb_vid_dataset, finetune_rgb_vid_dataset
from models.network_nafnet import Baseline_Video
import utils.utils_image as util
import utils.utils_network as net_util

def save_pseudo_datasets(opt, save_datasets_dir):
    prev_stage = opt.stage - 1
    checkpoints_dir = opt.pretrained_checkpoints_dir
    test_dataset = test_rgb_vid_dataset(gt_dir=opt.ft_clean_vid_dir, n_frames=opt.n_frames, noise_level=opt.noise_level, dataset=opt.ft_dataset) 
    test_dataloader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.num_workers)
    
    net_G = Baseline_Video(n_frames=opt.n_frames, img_channel=opt.in_nc, width=opt.nc, middle_blk_num=6, 
                     enc_blk_nums=[2, 2, 4], dec_blk_nums=[2, 2, 2], dw_expand=2, ffn_expand=2, current_stage=prev_stage)
    net_G.load_state_dict(torch.load(checkpoints_dir, map_location='cpu'), strict=False)
    net_G, _, device = net_util.model_to_device(network=net_G, gpu_ids=opt.gpu_ids)
    net_G.eval()

    for data in tqdm(test_dataloader, desc="Processing images"):
        noisy = data['noisy'].to(device) # B, T, C, H, W
        key_frame_paths = data['key_frame_path']

        with torch.no_grad():
            recs = net_G(noisy) # B, C, H, W

        # save pseudo frames in a batch
        for rec, key_frame_path in zip(recs, key_frame_paths):
            rec = torch.clamp(rec, 0, 1)
            rec = (rec.cpu().numpy() * 255).astype(np.uint8)
            rec = rec.transpose(1, 2, 0)

            folder, key_frame_name = key_frame_path.split('/')[-2:]
            save_path = os.path.join(save_datasets_dir, folder, key_frame_name.replace('jpg', 'png'))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, rec)

    print("Finished")

def test_finetune_rgb(net, device, dataloader1, dataloader2):

    def test_pretrain(dataloader):
        total_psnr, total_ssim, num = 0, 0, 0
        
        net.eval()
        for data in tqdm(dataloader, desc="Testing", leave=False):
            noisy = data['noisy'].to(device) # B, T, C, H, W
            clean = data['clean'].to(device) # B, C, H, W
            num = num + noisy.shape[0]

            with torch.no_grad():
                rec = net(noisy) # B, C, H, W

            rec = torch.clamp(rec, 0, 1)
            rec = torch.round(rec * 255)/255.
            psnr_ = util.calculate_psnr_pt(rec, clean, crop_border=0)
            ssim_ = util.calculate_ssim_pt(rec, clean, crop_border=0)
            total_psnr = total_psnr + psnr_.sum()
            total_ssim = total_ssim + ssim_.sum()

        results = {
            "mean_psnr": total_psnr / num,
            "mean_ssim": total_ssim / num,
        }

        return results
    
    results = {
        "davis": test_pretrain(dataloader1),
        "set8": test_pretrain(dataloader2)
    }

    return results
        

def main():
    # ---------------------------------------------------------
    # Initiate logs and args
    # ---------------------------------------------------------
    opt, opt_message = BaseOptions().parse()

    # create save folders
    save_logs_dir = os.path.join(opt.save_dir, f'sigma{opt.noise_level}', f's{opt.stage}', 'logs/')
    save_models_dir = os.path.join(opt.save_dir, f'sigma{opt.noise_level}', f's{opt.stage}', 'models/')
    save_datasets_dir = os.path.join(opt.save_dir, f'sigma{opt.noise_level}', f's{opt.stage}', 'datasets/') 

    os.makedirs(save_logs_dir, exist_ok=True)
    os.makedirs(save_models_dir, exist_ok=True)
    os.makedirs(save_datasets_dir, exist_ok=True)

    logger_name = 'train'
    logger_info(logger_name, log_path=save_logs_dir + logger_name + '.log')
    logger = logging.getLogger(logger_name)
    logger.info(opt_message)

    # --------------------------------------------------------------------------------------------
    # Generate pseudo datasets
    # Generating pseudo datasets requires the fine-tuned model from the previous stage.
    # If you don't have the corresponding checkpoints, you need to train the previous model first.
    # --------------------------------------------------------------------------------------------
    if opt.prepare_datasets:
        save_pseudo_datasets(opt, save_datasets_dir)

    # ---------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------
    train_dataset = finetune_rgb_vid_dataset(gt_dir=save_datasets_dir, n_frames=opt.n_frames, noise_level=opt.noise_level,
                                             patch_size=opt.patch_size, img_aug=opt.img_aug) # requires pseudo datasets
    
    test_dataset_davis = test_rgb_vid_dataset(gt_dir=opt.test_clean_vid_dir[0], n_frames=opt.n_frames, noise_level=opt.noise_level, dataset='davis')
    test_dataset_set8 = test_rgb_vid_dataset(gt_dir=opt.test_clean_vid_dir[1], n_frames=opt.n_frames, noise_level=opt.noise_level, dataset='set8')
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.train_batch_size,
        shuffle=not opt.not_train_shuffle,
        drop_last=not opt.not_train_drop_last,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    test_loader_davis = DataLoader(
        dataset=test_dataset_davis,
        batch_size=opt.test_batch_size, 
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True
    )
    test_loader_set8 = DataLoader(
        dataset=test_dataset_set8,
        batch_size=opt.test_batch_size, 
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True
    )

    # ---------------------------------------------------------
    # Networks, optimizers, loss
    # ---------------------------------------------------------

    # define network
    net_G = Baseline_Video(n_frames=opt.n_frames, img_channel=opt.in_nc, width=opt.nc, middle_blk_num=6,
                            enc_blk_nums=[2, 2, 4], dec_blk_nums=[2, 2, 2], dw_expand=2, ffn_expand=2, current_stage=opt.stage)
    
    # print network
    net_message = net_util.print_networks(network=net_G, network_name='G', verbose=opt.verbose)
    logger.info(net_message)

    # networks, optimizer
    net_G = net_util.init_weights(network=net_G, init_type=opt.init_type)
    net_G.load_state_dict(torch.load(opt.pretrained_checkpoints_dir, map_location='cpu'), strict=False)
    optim_G = optim.AdamW(net_G.pcd_align[3-opt.stage].parameters(), opt.G_lr, betas=(0.9, 0.9))

    # model to device -- return net, gpu message and device info
    net_G, gpu_message, device = net_util.model_to_device(network=net_G, gpu_ids=opt.gpu_ids)
    logger.info(gpu_message)

    # scheduler
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer=optim_G, T_max=opt.epochs, eta_min=1e-6, verbose=True)

    # loss
    Cri_rec = net_util.pixel_loss(type=opt.pixel_loss_type)

    # ---------------------------------------------------------
    # Training, testing
    # ---------------------------------------------------------
    current_step = 0
    for epoch in range(opt.epochs):
        epoch += 1
        for i, data in enumerate(train_loader):
            current_step += 1
            net_G.train()

            noisy = data['noisy'].to(device) # B, T, C, H, W
            clean = data['clean'].to(device) # B, C, H, W

            log_dict = OrderedDict()

            # 1) feed data
            rec = net_G(noisy) # B, C, H, W

            # 2) optimizer networks
            optim_G.zero_grad()
            rec_loss = Cri_rec(rec, clean) # reconstruction loss
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
                if opt.save_models:
                    torch.save(net_util.get_bare_model(net_G).state_dict(), save_models_dir+'net_{:05d}_{:05d}.pth'.format(epoch, current_step))
                
                net_G.eval()
                results = test_finetune_rgb(net=net_G, device=device, dataloader1=test_loader_davis, dataloader2=test_loader_set8)
                message = '<epoch: {:d}, step: {:08d}>'.format(epoch, current_step)
                message += '\nDavis Results: '
                message += 'Mean PSNR: {:.4f}, Mean SSIM: {:.4f}\n'.format(results['davis']['mean_psnr'], results['davis']['mean_ssim'])
                message += 'Set8 Results: '
                message += 'Mean PSNR: {:.4f}, Mean SSIM: {:.4f}\n'.format(results['set8']['mean_psnr'], results['set8']['mean_ssim'])
                print(message)
                logger.info(message)
                net_G.train()

        if epoch == opt.epochs:
            if opt.save_models:
                    torch.save(net_util.get_bare_model(net_G).state_dict(), save_models_dir+'net_{:05d}_{:05d}.pth'.format(epoch, current_step))
                
            net_G.eval()
            results = test_finetune_rgb(net=net_G, device=device, dataloader1=test_loader_davis, dataloader2=test_loader_set8)
            message = '<epoch: {:d}, step: {:08d}>'.format(epoch, current_step)
            message += '\nDavis Results: '
            message += 'Mean PSNR: {:.4f}, Mean SSIM: {:.4f}\n'.format(results['davis']['mean_psnr'], results['davis']['mean_ssim'])
            message += 'Set8 Results: '
            message += 'Mean PSNR: {:.4f}, Mean SSIM: {:.4f}\n'.format(results['set8']['mean_psnr'], results['set8']['mean_ssim'])
            print(message)
            logger.info(message)
            net_G.train()




        scheduler_G.step()


if __name__ == '__main__':
    main()


