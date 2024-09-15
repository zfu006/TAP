import torch
import torch.utils.data as data
import os
import numpy as np
from torch.utils.data import Dataset
import random
import cv2
import re
import utils.utils_image as util

# ---------------------------------------------------
# Define dataset for raw video denoiser finetuning
# For CRVD
# ---------------------------------------------------
class finetune_crvd_vid_dataset(Dataset):
    """
    During finetuning, only pseudo clean videos are used.
    """
    def __init__(self, gt_dir, n_frames=3, patch_size=256, scene_type='indoor', bayer_aug=True):
        self.gt_dir = gt_dir
        self.n_frames = n_frames
        self.patch_size = patch_size
        self.bayer_aug = bayer_aug
        self.frame_paths = []
        self.video_start_indices = []

        iso_list = [1600,3200,6400,12800,25600]
        self.a_list = [3.513262,6.955588,13.486051,26.585953,52.032536]
        self.g_noise_var_list = [11.917691,38.117816,130.818508,484.539790,1819.818657]
        self.iso_list = iso_list
        scene_ids = range(7, 12) if scene_type == 'indoor' else range(1, 11)

        for scene_id in scene_ids:

            scene = 'scene{}'.format(scene_id) 

            for iso_id in iso_list:

                iso = 'ISO{}'.format(iso_id)

                folder = os.path.join(gt_dir, scene, iso)
                frames = sorted(os.listdir(folder))
                self.video_start_indices.append(len(self.frame_paths))
                self.frame_paths.extend([os.path.join(folder, f) for f in frames])
    
    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        m = self.n_frames // 2
        key_frame_path = self.frame_paths[idx]
        iso_value = key_frame_path.split('/')[-2]
        iso_value = int(iso_value[3:])
        video_start_idx = max([i for i in self.video_start_indices if i <= idx])
        video_end_idx = min([i for i in self.video_start_indices if i > idx], default=len(self.frame_paths)) - 1
        
        # Calculate the indices for the surrounding frames
        indices = [idx + i for i in range(-m, m + 1)]
        
        # Handle boundary conditions
        frames = []
        noisy_frames = []
        noise_dict = {}

        aug_mode = np.random.randint(0, 4)
        for i, frame_idx in enumerate(indices):
            if frame_idx < video_start_idx:
                frame_idx = video_start_idx + (video_start_idx - frame_idx)
            elif frame_idx > video_end_idx:
                frame_idx = video_end_idx - (frame_idx - video_end_idx)

            frame = cv2.imread(self.frame_paths[frame_idx], -1)
            frame = frame.astype(np.float32)
            frame = np.clip(frame, 240, 2**12-1)
            if self.bayer_aug:
                frame = self.bayer_preserving_augmentation(frame, aug_mode)
            
            frame = util.pack_gbrg_raw(frame) # 4, H, W
            frames.append(frame)
        
            if frame_idx not in noise_dict:
                a = self.a_list[self.iso_list.index(iso_value)]
                b = self.g_noise_var_list[self.iso_list.index(iso_value)]
                noise_dict[frame_idx] = self.generate_noise(frame, a, b) - frame

            noisy_frame = frame + noise_dict[frame_idx]
            noisy_frames.append(noisy_frame)
        
        # Stack frames into T, C, H, W format
        frames_stacked = np.stack(frames, axis=0) # T, C, H, W
        noisy_frames_stacked = np.stack(noisy_frames, axis=0) # T, C, H, W
        # random crop:
        h, w = frames_stacked.shape[2], frames_stacked.shape[3]
        rnd_h = random.randint(0, max(0, h - self.patch_size))
        rnd_w = random.randint(0, max(0, w - self.patch_size))
        frames_stacked = frames_stacked[self.n_frames//2, :, rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size] # C, H, W
        noisy_frames_stacked = noisy_frames_stacked[:, :, rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size] # T, C, H, W
        
        frames_stacked = np.maximum(frames_stacked - 240, 0) / (2**12 - 1 - 240)
        noisy_frames_stacked = np.maximum(noisy_frames_stacked - 240, 0) / (2**12 - 1 - 240)
        frames_stacked = torch.from_numpy(frames_stacked)
        noisy_frames_stacked = torch.from_numpy(noisy_frames_stacked)
        
        return {'clean': frames_stacked, 'noisy': noisy_frames_stacked}
    
    def generate_noise(self, gt_raw, a, b):
        # generate poisson gaussian noise
        """
        gt_raw: range 240-4095
        a: sigma_s^2
        b: sigma_r^2
        """
        gaussian_noise_var = b
        z = np.random.poisson((gt_raw-240)/a).astype(np.float32) * a
        z = z + np.random.randn(*z.shape).astype(np.float32) * np.sqrt(gaussian_noise_var) + 240 # add black level
        z = np.clip(z, 0, 2**12-1) # CRVD 12-bit
        return z
    
    def bayer_preserving_augmentation(self, raw, aug_mode):  
        if aug_mode == 0:  
            aug_raw = raw      
        elif aug_mode == 1:  # horizontal flip
            aug_raw = np.flip(raw, axis=1)[:,1:-1]
        elif aug_mode == 2: # vertical flip
            aug_raw = np.flip(raw, axis=0)[1:-1,:]
        else:  # random transpose
            aug_raw = np.transpose(raw, (1, 0))

        return aug_raw
    
class test_crvd_vid_indoor_dataset(Dataset):
    def __init__(self, gt_dir, noisy_dir, n_frames=3):
        self.n_frames = n_frames
        self.gt_frame_paths = []
        self.noisy_frame_paths = []
        self.video_start_indices = []

        iso_list = [1600,3200,6400,12800,25600]
        gt_frames_suffix = '_clean_and_slightly_denoised.tiff'
        noisy_frames_suffix = '_noisy0.tiff'

        for scene_id in range(7, 12):

            scene = 'scene{}'.format(scene_id) 

            for iso_id in iso_list:

                iso = 'ISO{}'.format(iso_id)

                gt_folder = os.path.join(gt_dir, scene, iso)
                noisy_folder = os.path.join(noisy_dir, scene, iso)
                gt_frames = sorted(os.listdir(gt_folder))
                self.video_start_indices.append(len(self.gt_frame_paths))
                self.gt_frame_paths.extend([os.path.join(gt_folder, f) for f in gt_frames])
                self.noisy_frame_paths.extend([os.path.join(noisy_folder, f.replace(gt_frames_suffix, noisy_frames_suffix)) for f in gt_frames])
    
    def __len__(self):
        return len(self.gt_frame_paths)

    def __getitem__(self, idx):
        m = self.n_frames // 2
        video_start_idx = max([i for i in self.video_start_indices if i <= idx])
        video_end_idx = min([i for i in self.video_start_indices if i > idx], default=len(self.gt_frame_paths)) - 1
        
        # Calculate the indices for the surrounding frames
        indices = [idx + i for i in range(-m, m + 1)]
        
        # Handle boundary conditions
        gt_frames = []
        noisy_frames = []

        for i, frame_idx in enumerate(indices):
            if frame_idx < video_start_idx:
                frame_idx = video_start_idx + (video_start_idx - frame_idx)
            elif frame_idx > video_end_idx:
                frame_idx = video_end_idx - (frame_idx - video_end_idx)

            gt_frame = cv2.imread(self.gt_frame_paths[frame_idx], -1)
            noisy_frame = cv2.imread(self.noisy_frame_paths[frame_idx], -1)
            gt_frame = gt_frame.astype(np.float32)
            noisy_frame = noisy_frame.astype(np.float32)
            gt_frame = util.pack_gbrg_raw(gt_frame)
            noisy_frame = util.pack_gbrg_raw(noisy_frame)
            gt_frames.append(gt_frame)
            noisy_frames.append(noisy_frame)
        
        # Stack frames into T, C, H, W format
        gt_frames_stacked = np.stack(gt_frames, axis=0)[m] # C, H, W
        noisy_frames_stacked = np.stack(noisy_frames, axis=0) # T, C, H, W
        gt_frames_stacked = np.maximum(gt_frames_stacked - 240, 0) / (2**12 - 1 - 240)
        noisy_frames_stacked = np.maximum(noisy_frames_stacked - 240, 0) / (2**12 - 1 - 240)

        gt_frames_stacked = torch.from_numpy(gt_frames_stacked)
        noisy_frames_stacked = torch.from_numpy(noisy_frames_stacked)
        key_frame_path = self.gt_frame_paths[idx]
        
        return {'clean': gt_frames_stacked, 'noisy': noisy_frames_stacked, 'key_frame_path': key_frame_path}

class test_crvd_vid_outdoor_dataset(Dataset):
    def __init__(self, gt_dir, noisy_dir, n_frames=3):
        self.n_frames = n_frames
        self.noisy_frame_paths = []
        self.video_start_indices = []

        iso_list = [1600,3200,6400,12800,25600]

        for scene_id in range(1, 11):

            scene = 'scene{}'.format(scene_id) 

            for iso_id in iso_list:

                iso = 'iso{}'.format(iso_id)

                noisy_folder = os.path.join(noisy_dir, scene, iso)
                noisy_frames = sorted(os.listdir(noisy_folder))
                self.video_start_indices.append(len(self.noisy_frame_paths))
                self.noisy_frame_paths.extend([os.path.join(noisy_folder, f) for f in noisy_frames])
    
    def __len__(self):
        return len(self.noisy_frame_paths)

    def __getitem__(self, idx):
        m = self.n_frames // 2
        video_start_idx = max([i for i in self.video_start_indices if i <= idx])
        video_end_idx = min([i for i in self.video_start_indices if i > idx], default=len(self.noisy_frame_paths)) - 1
        
        # Calculate the indices for the surrounding frames
        indices = [idx + i for i in range(-m, m + 1)]
        
        # Handle boundary conditions
        noisy_frames = []

        for i, frame_idx in enumerate(indices):
            if frame_idx < video_start_idx:
                frame_idx = video_start_idx + (video_start_idx - frame_idx)
            elif frame_idx > video_end_idx:
                frame_idx = video_end_idx - (frame_idx - video_end_idx)

            noisy_frame = cv2.imread(self.noisy_frame_paths[frame_idx], -1)
            noisy_frame = noisy_frame.astype(np.float32)
            noisy_frame = util.pack_gbrg_raw(noisy_frame)
            noisy_frames.append(noisy_frame)
        
        # Stack frames into T, C, H, W format
        noisy_frames_stacked = np.stack(noisy_frames, axis=0) # T, C, H, W
        noisy_frames_stacked = np.maximum(noisy_frames_stacked - 240, 0) / (2**12 - 1 - 240)
        noisy_frames_stacked = torch.from_numpy(noisy_frames_stacked)
        key_frame_path = self.noisy_frame_paths[idx]
        
        return {'noisy': noisy_frames_stacked, 'key_frame_path': key_frame_path}

# ---------------------------------------------------
# Define dataset for RGB video denoiser finetuning
# For DAVIS and Set8
# ---------------------------------------------------
class finetune_rgb_vid_dataset(Dataset):
    def __init__(self, gt_dir, n_frames=3, noise_level=30, patch_size=256, img_aug=True):
        self.n_frames = n_frames
        self.noise_level = noise_level
        self.patch_size = patch_size
        self.img_aug = img_aug
        self.gt_frame_paths = []
        self.video_start_indices = []

        for vid_folder in sorted(os.listdir(gt_dir)):
            vid_folder_path = os.path.join(gt_dir, vid_folder)
            print(vid_folder_path)
            if os.path.isdir(vid_folder_path):
                frames = sorted(os.listdir(vid_folder_path))
                self.video_start_indices.append(len(self.gt_frame_paths))
                self.gt_frame_paths.extend(os.path.join(vid_folder_path, f) for f in frames)    
    
    def __len__(self):
        return len(self.gt_frame_paths)

    def __getitem__(self, idx):
        m = self.n_frames // 2
        video_start_idx = max([i for i in self.video_start_indices if i <= idx])
        video_end_idx = min([i for i in self.video_start_indices if i > idx], default=len(self.gt_frame_paths)) - 1
        
        # Calculate the indices for the surrounding frames
        indices = [idx + i for i in range(-m, m + 1)]
        
        # Handle boundary conditions
        gt_frames = []
        noisy_frames = []
        noise_dict = {}
        aug_mode = np.random.randint(0, 8) if self.img_aug else 0

        for i, frame_idx in enumerate(indices):
            if frame_idx < video_start_idx:
                frame_idx = video_start_idx + (video_start_idx - frame_idx)
            elif frame_idx > video_end_idx:
                frame_idx = video_end_idx - (frame_idx - video_end_idx)

            gt_frame = cv2.imread(self.gt_frame_paths[frame_idx], cv2.IMREAD_COLOR) # H, W, C
            gt_frame = gt_frame.astype(np.float32)/255.
            gt_frame = util.augment_img(gt_frame, aug_mode)
            gt_frame = gt_frame.transpose(2, 0, 1) # C, H, W    
            gt_frames.append(gt_frame)

            if frame_idx not in noise_dict:
                noise = np.random.randn(*gt_frame.shape).astype(np.float32) * self.noise_level/255.
                noise_dict[frame_idx] = noise

            noisy_frame = gt_frame + noise_dict[frame_idx]
            noisy_frames.append(noisy_frame)
        
        # Stack frames into T, C, H, W format
        gt_frames_stacked = np.stack(gt_frames, axis=0) # T, C, H, W
        noisy_frames_stacked = np.stack(noisy_frames, axis=0) # T, C, H, W
        # random crop:
        h, w = gt_frames_stacked.shape[2], gt_frames_stacked.shape[3]
        rnd_h = random.randint(0, max(0, h - self.patch_size))
        rnd_w = random.randint(0, max(0, w - self.patch_size))
        gt_frames_stacked = gt_frames_stacked[m, :, rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size] # C, H, W
        noisy_frames_stacked = noisy_frames_stacked[:, :, rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size] # T, C, H, W

        gt_frames_stacked = torch.from_numpy(gt_frames_stacked)
        noisy_frames_stacked = torch.from_numpy(noisy_frames_stacked)
        
        return {'clean': gt_frames_stacked, 'noisy': noisy_frames_stacked}    

class test_rgb_vid_dataset(Dataset):
    def __init__(self, gt_dir, n_frames=3, noise_level=30, dataset='davis'):
        self.n_frames = n_frames
        self.noise_level = noise_level
        self.dataset = dataset
        self.resize_h, self.resize_w = (480, 854) if dataset == 'davis' else (540, 960)
        self.gt_frame_paths = []
        self.video_start_indices = []

        for vid_folder in sorted(os.listdir(gt_dir)):
            vid_folder_path = os.path.join(gt_dir, vid_folder)
            if os.path.isdir(vid_folder_path):
                frames = sorted(os.listdir(vid_folder_path))
                self.video_start_indices.append(len(self.gt_frame_paths))
                self.gt_frame_paths.extend(os.path.join(vid_folder_path, f) for f in frames)    
    
    def __len__(self):
        return len(self.gt_frame_paths)

    def __getitem__(self, idx):
        m = self.n_frames // 2
        video_start_idx = max([i for i in self.video_start_indices if i <= idx])
        video_end_idx = min([i for i in self.video_start_indices if i > idx], default=len(self.gt_frame_paths)) - 1
        
        # Calculate the indices for the surrounding frames
        indices = [idx + i for i in range(-m, m + 1)]
        
        # Handle boundary conditions
        gt_frames = []
        noisy_frames = []
        noise_dict = {}

        for i, frame_idx in enumerate(indices):
            if frame_idx < video_start_idx:
                frame_idx = video_start_idx + (video_start_idx - frame_idx)
            elif frame_idx > video_end_idx:
                frame_idx = video_end_idx - (frame_idx - video_end_idx)

            gt_frame = cv2.imread(self.gt_frame_paths[frame_idx], cv2.IMREAD_COLOR) # H, W, C
            gt_frame = gt_frame.astype(np.float32)/255.
            h, w, _ = gt_frame.shape
            if h != self.resize_h or w != self.resize_w:
                gt_frame = cv2.resize(gt_frame, (self.resize_w, self.resize_h))
            gt_frame = gt_frame.transpose(2, 0, 1) # C, H, W    
            gt_frames.append(gt_frame)

            if frame_idx not in noise_dict:
                noise = np.random.randn(*gt_frame.shape).astype(np.float32) * self.noise_level/255.
                noise_dict[frame_idx] = noise

            noisy_frame = gt_frame + noise_dict[frame_idx]
            noisy_frames.append(noisy_frame)
        
        # Stack frames into T, C, H, W format
        gt_frames_stacked = np.stack(gt_frames, axis=0)[m] # C, H, W
        noisy_frames_stacked = np.stack(noisy_frames, axis=0) # T, C, H, W

        gt_frames_stacked = torch.from_numpy(gt_frames_stacked)
        noisy_frames_stacked = torch.from_numpy(noisy_frames_stacked)
        key_frame_path = self.gt_frame_paths[idx]
        
        return {'clean': gt_frames_stacked, 'noisy': noisy_frames_stacked, 'key_frame_path': key_frame_path}