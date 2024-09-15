import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import logging
from collections import OrderedDict
import numpy as np

from options.option_pretrain import BaseOptions
from utils.utils_logger import logger_info
from data.pretrain_im_dataset import pretrain_rgb_im_dataset, test_rgb_im_dataset
from models.network_nafnet import Baseline
import utils.utils_network as net_util
import utils.utils_image as util

def test_pretrain_davis_set8(net, device, dataloader1, dataloader2):

    def test_pretrain(dataloader):
        noise_levels = [10, 20, 30, 40, 50]
        psnr_dict = {nl: 0 for nl in noise_levels}
        ssim_dict = {nl: 0 for nl in noise_levels}
        total_psnr, total_ssim, num = 0, 0, 0
        
        net.eval()
        for data in dataloader:
            clean_img = data['clean'].to(device)
            num += clean_img.shape[0]
            for noise_level in noise_levels:
                noisy_img = clean_img + torch.randn_like(clean_img) * (noise_level / 255.0)
                with torch.no_grad():
                    pred_img = torch.clamp(net(noisy_img).detach(), 0, 1)
                    pred_img = torch.round(pred_img * 255) / 255.0

                psnr = util.calculate_psnr_pt(pred_img, clean_img, crop_border=0).sum()
                ssim = util.calculate_ssim_pt(pred_img, clean_img, crop_border=0).sum()

                psnr_dict[noise_level] += psnr
                ssim_dict[noise_level] += ssim
                total_psnr += psnr
                total_ssim += ssim

        results = {
            "mean_psnr": total_psnr / (num * len(noise_levels)),
            "mean_ssim": total_ssim / (num * len(noise_levels)),
            "psnr_noise_levels": {nl: psnr_dict[nl] / num for nl in noise_levels},
            "ssim_noise_levels": {nl: ssim_dict[nl] / num for nl in noise_levels},
        }

        return results

    results = {
        "davis": test_pretrain(dataloader1),
        "set8": test_pretrain(dataloader2)
    }

    return results

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
    train_dataset = pretrain_rgb_im_dataset(source_path=opt.train_clean_img_dir, 
                                            patch_size=opt.patch_size)
    
    test_davis_dataset = test_rgb_im_dataset(gt_dir=os.path.join(opt.test_clean_img_dir, 'DAVIS-2017-test-dev-480p/DAVIS/JPEGImages/480p/'))
    test_set8_dataset = test_rgb_im_dataset(gt_dir=os.path.join(opt.test_clean_img_dir, 'Set8'))
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.train_batch_size,
        shuffle=not opt.not_train_shuffle,
        drop_last=not opt.not_train_drop_last,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    test_davis_loader = DataLoader(
        dataset=test_davis_dataset,
        batch_size=opt.test_batch_size, 
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True
    )

    test_set8_loader = DataLoader(
        dataset=test_set8_dataset,
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
                if opt.save_models:
                    torch.save(net_util.get_bare_model(net_G).state_dict(), opt.save_models_dir+'net_{:05d}_{:05d}.pth'.format(epoch, current_step))

                net_G.eval()
                results = test_pretrain_davis_set8(net_G, device, test_davis_loader, test_set8_loader)
                message = '<epoch: {:d}, step: {:08d}>'.format(epoch, current_step)
                message += '\nDavis Results: '
                message += 'Mean PSNR: {:.4f}, Mean SSIM: {:.4f}\n'.format(results['davis']['mean_psnr'], results['davis']['mean_ssim'])
                for noise_level in [10, 20, 30, 40, 50]:
                    message += 'Noise Level {}: PSNR: {:.4f}, SSIM: {:.4f}\n'.format(
                        noise_level,
                        results['davis']['psnr_noise_levels'][noise_level],
                        results['davis']['ssim_noise_levels'][noise_level]
                    )

                message += 'Set8 Results: '
                message += 'Mean PSNR: {:.4f}, Mean SSIM: {:.4f}\n'.format(results['set8']['mean_psnr'], results['set8']['mean_ssim'])
                for noise_level in [10, 20, 30, 40, 50]:
                    message += 'Noise Level {}: PSNR: {:.4f}, SSIM: {:.4f}\n'.format(
                        noise_level,
                        results['set8']['psnr_noise_levels'][noise_level],
                        results['set8']['ssim_noise_levels'][noise_level]
                    )

                print(message)
                logger.info(message)
                net_G.train()


        scheduler_G.step()


if __name__ == '__main__':
    main()