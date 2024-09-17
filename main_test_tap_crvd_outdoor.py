import os
import torch
import numpy as np
import cv2
import torchvision.transforms as T
import natsort
from tqdm import tqdm

from models.network_nafnet import Baseline_Video
from models.network_isp import ISP
import utils.utils_image as util
import utils.utils_network as net_util

def video_tensor_to_gif(tensor, path, duration = 100, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 0))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

def main():

    # configurations
    test_noisy_vid_root =  '/media/user/HDD/zixuan.fu/Files/research/projects/unsupervised_video_restoration/Datasets/video_denoising/CRVD_dataset/outdoor_raw_noisy/' # specify the path
    scene = 'scene4'
    iso = 'iso25600'
    test_noisy_vid_dir = os.path.join(test_noisy_vid_root, scene, iso)
    save_dir = './saves/tap_crvd_outdoor/'
    n_frames = 5 # specify the number of frames (slide window size)
    denoiser_path = './checkpoints/tap_crvd_outdoor/tap_crvd_outdoor_s3.pth' # specify the checkpoint path
    isp_path = 'checkpoints/isp/ISP_CNN.pth'
    gpu_ids = [1] 
    save_dir_imgs = os.path.join(save_dir, 'image')
    save_dir_gif = os.path.join(save_dir, 'gif')
    os.makedirs(save_dir_imgs, exist_ok=True)
    os.makedirs(save_dir_gif, exist_ok=True)

    net_G = Baseline_Video(n_frames=n_frames, img_channel=4, width=64, middle_blk_num=6,
                            enc_blk_nums=[2, 2, 4], dec_blk_nums=[2, 2, 2], dw_expand=2, ffn_expand=2, current_stage=3)
    net_G.load_state_dict(torch.load(denoiser_path, map_location='cpu'), strict=True)
    net_G, _, device = net_util.model_to_device(network=net_G, gpu_ids=gpu_ids)
    net_G.eval()

    net_isp = ISP()
    net_isp.load_state_dict(torch.load(isp_path, map_location='cpu'), strict=True)
    net_isp, _, _ = net_util.model_to_device(network=net_isp, gpu_ids=gpu_ids)
    net_isp.eval()

    img_lists = os.listdir(test_noisy_vid_dir)
    img_lists = natsort.natsorted(img_lists)
    img_lists = [os.path.join(test_noisy_vid_dir, img) for img in img_lists]

    num_frames = len(img_lists)
    m = n_frames//2
    video_start_idx = 0
    video_end_idx = num_frames - 1
    rec_rgbs = []
    
    for idx in tqdm(range(len(img_lists)), desc="Processing", leave=False):
        indices = [idx + i for i in range(-m, m+1)]
        noisy_frames = []

        for i, frame_idx in enumerate(indices):
            if frame_idx < video_start_idx:
                frame_idx = video_start_idx + (video_start_idx - frame_idx)
            elif frame_idx > video_end_idx:
                frame_idx = video_end_idx - (frame_idx - video_end_idx)
            
            noisy_frame = cv2.imread(img_lists[frame_idx], -1).astype(np.float32)
            noisy_frame = util.pack_gbrg_raw(noisy_frame)
            noisy_frames.append(noisy_frame)

        noisy_frames_stacked = np.stack(noisy_frames, axis=0) # T, C, H, W
        noisy_frames_stacked = np.maximum(noisy_frames_stacked - 240, 0) / (2**12 - 1 - 240)
        noisy_frames_stacked = torch.from_numpy(noisy_frames_stacked)

        with torch.no_grad():
            rec = net_G(noisy_frames_stacked.unsqueeze(0).to(device))
            rec = torch.clamp(rec, 0, 1)
            rec_rgb = torch.clamp(net_isp(rec), 0, 1)
            rec_rgbs.append(rec_rgb.squeeze(0).cpu())
            rec_rgb = rec_rgb.squeeze(0).cpu().numpy()
            rec_rgb = (rec_rgb * 255).astype(np.uint8).transpose(1, 2, 0)
            cv2.imwrite(os.path.join(save_dir_imgs, 'frame_{:04d}.png'.format(idx)), rec_rgb)

    rec_rgbs = torch.stack(rec_rgbs, dim=0)
    video_tensor_to_gif(rec_rgbs[:, [2, 1, 0], :, :], os.path.join(save_dir_gif, 'rec.gif'))

if __name__ == '__main__':
    main()


