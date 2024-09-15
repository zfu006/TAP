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


from options.option_finetune_crvd import BaseOptions
from utils.utils_logger import logger_info
from data.finetune_vid_dataset import test_crvd_vid_indoor_dataset, test_crvd_vid_outdoor_dataset, finetune_crvd_vid_dataset
from models.network_nafnet import Baseline_Video
import utils.utils_image as util
import utils.utils_network as net_util

def save_pseudo_datasets(opt, save_datasets_dir):
    scene_type = opt.scene_type
    prev_stage = opt.stage - 1
    checkpoints_dir = opt.pretrained_checkpoints_dir
    test_dataset = test_crvd_vid_indoor_dataset(gt_dir=opt.test_clean_vid_dir, noisy_dir=opt.test_noisy_vid_dir, 
                                                n_frames=opt.n_frames) if scene_type == 'indoor' else \
                   test_crvd_vid_outdoor_dataset(gt_dir=opt.test_clean_vid_dir, noisy_dir=opt.test_noisy_vid_dir,
                                                n_frames=opt.n_frames)
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
            rec = (rec.squeeze(0).cpu().numpy() * (2**12-1-240) + 240).astype(np.uint16)
            rec = util.depack_gbrg_raw(rec)

            scene, iso, key_frame_name = key_frame_path.split('/')[-3:]
            if scene_type == 'indoor':
                key_frame_name = key_frame_name.replace('_clean_and_slightly_denoised.tiff', '_denoised.tiff')
            else:
                iso = iso.replace('iso', 'ISO')
                key_frame_name = key_frame_name[:-5] + '_denoised.tiff'
            
            save_path = os.path.join(save_datasets_dir, scene, iso, key_frame_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, rec)

    print("Finished")

def test_finetune_crvd(net, device, dataloader):
    psnr = 0
    ssim = 0
    num = 0

    net.eval()
    for i, data in enumerate(tqdm(dataloader, desc="Testing", leave=False)):
        noisy = data['noisy'].to(device) # B, T, C, H, W
        clean = data['clean'].to(device) # B, C, H, W
        num = num + noisy.shape[0]

        with torch.no_grad():
            rec = net(noisy) # B, C, H, W
        
        rec = torch.clamp(rec, 0, 1)
        rec = (torch.round(rec * (2**12-1-240) + 240) - 240)/(2**12-1-240)
        psnr_ = util.calculate_psnr_pt(rec, clean, crop_border=0)
        ssim_ = util.calculate_ssim_pt(rec, clean, crop_border=0)
        psnr = psnr + psnr_.sum()
        ssim = ssim + ssim_.sum()

    return psnr/num, ssim/num
        

def main():
    # ---------------------------------------------------------
    # Initiate logs and args
    # ---------------------------------------------------------
    opt, opt_message = BaseOptions().parse()

    # create save folders
    save_logs_dir = os.path.join(opt.save_dir, f's{opt.stage}', 'logs/')
    save_models_dir = os.path.join(opt.save_dir, f's{opt.stage}', 'models/')
    save_datasets_dir = os.path.join(opt.save_dir, f's{opt.stage}', 'datasets/') 

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
    train_dataset = finetune_crvd_vid_dataset(gt_dir=save_datasets_dir, n_frames=opt.n_frames, 
                                             patch_size=opt.patch_size, scene_type=opt.scene_type) # requires pseudo datasets
    
    test_dataset = test_crvd_vid_indoor_dataset(gt_dir=opt.test_clean_vid_dir, noisy_dir=opt.test_noisy_vid_dir, 
                                                n_frames=opt.n_frames) if opt.scene_type == 'indoor' else \
                   test_crvd_vid_outdoor_dataset(gt_dir=opt.test_clean_vid_dir, noisy_dir=opt.test_noisy_vid_dir,
                                                n_frames=opt.n_frames)
    
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
                
                if opt.scene_type == 'indoor':
                    net_G.eval()
                    psnr, ssim = test_finetune_crvd(net=net_G, device=device, dataloader=test_loader)
                    message = '<epoch:{:d}, step: {:08d}> psnr: {:.3f} ssim: {:.3f}'.format(epoch, current_step, psnr.item(), ssim.item())
                    print(message)
                    logger.info(message)
                    net_G.train()

                if opt.save_models:
                    torch.save(net_util.get_bare_model(net_G).state_dict(), save_models_dir+'net_{:05d}_{:05d}.pth'.format(epoch, current_step))


        if epoch == opt.epochs:

            if opt.scene_type == 'indoor':
                net_G.eval()
                psnr, ssim = test_finetune_crvd(net=net_G, device=device, dataloader=test_loader)
                message = '<epoch:{:d}, step: {:08d}> psnr: {:.3f} ssim: {:.3f}'.format(epoch, current_step, psnr.item(), ssim.item())
                print(message)
                logger.info(message)
                net_G.train()

            if opt.save_models:
                torch.save(net_util.get_bare_model(net_G).state_dict(), save_models_dir+'net_{:05d}_{:05d}.pth'.format(epoch, current_step))


        scheduler_G.step()


if __name__ == '__main__':
    main()


