from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm

from data.finetune_vid_dataset import test_crvd_vid_indoor_dataset
from models.network_nafnet import Baseline_Video
import utils.utils_image as util
import utils.utils_network as net_util

def test_crvd(net, device, dataloader):
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

    # configurations
    test_clean_vid_dir = './Datasets/video_denoising/CRVD_dataset/indoor_raw_gt/'    # specify the path 
    test_noisy_vid_dir = './Datasets/video_denoising/CRVD_dataset/indoor_raw_noisy/'
    n_frames = 5 # specify the number of frames (slide window size)
    batch_size = 2
    model_path = './checkpoints/tap_crvd_indoor/tap_crvd_indoor_s3.pth' # specify the checkpoint path
    gpu_ids = [1] # specify the gpu ids, can use multiple gpus



    test_dataset = test_crvd_vid_indoor_dataset(gt_dir=test_clean_vid_dir, noisy_dir=test_noisy_vid_dir, 
                                                n_frames=n_frames)
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True)

    net_G = Baseline_Video(n_frames=n_frames, img_channel=4, width=64, middle_blk_num=6,
                            enc_blk_nums=[2, 2, 4], dec_blk_nums=[2, 2, 2], dw_expand=2, ffn_expand=2, current_stage=3)
    net_G.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    net_G, gpu_message, device = net_util.model_to_device(network=net_G, gpu_ids=gpu_ids)

    psnr, ssim = test_crvd(net_G, device, test_loader)

    print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")

if __name__ == '__main__':
    main()