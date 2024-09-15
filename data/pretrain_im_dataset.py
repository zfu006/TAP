import torch
import torch.utils.data as data
import os
import numpy as np
from torch.utils.data import Dataset
import random
import cv2
import re
import utils.utils_image as util

# -----------------------------------------------------
# Define dataset for raw image denoiser pretraining
# -----------------------------------------------------
class pretrain_raw_im_dataset(Dataset):
    def __init__(self, source_path, patch_size=128, datasets=['SID'], noise_aug=True, bayer_aug=True):
        full_img_lists = os.listdir(source_path)
        selected_img_lists = []
        for dataset_name in datasets:
            pattern = re.compile(r'^' + re.escape(dataset_name) + r'[\W_]')
            selected_img_lists.extend([file for file in full_img_lists if pattern.match(file)])
        
        
        self.selected_img_lists = selected_img_lists
        self.source_path = source_path
        self.patch_size = patch_size
        self.iso_list = [1600,3200,6400,12800,25600]
        self.a_list = [3.513262,6.955588,13.486051,26.585953,52.032536]
        self.g_noise_var_list = [11.917691,38.117816,130.818508,484.539790,1819.818657]
        self.noise_aug = noise_aug
        self.bayer_aug = bayer_aug
        # print(len(self.selected_img_lists))

    def __len__(self):
        return len(self.selected_img_lists)
    
    def __getitem__(self, index):
        img = np.load(os.path.join(self.source_path, self.selected_img_lists[index]))
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        clean = img.copy()
        if self.bayer_aug:
            clean = self.bayer_preserving_augmentation(clean)
        
        if self.patch_size:
            h, w = clean.shape[1], clean.shape[2]
            rnd_h = random.randint(0, max(0, h - self.patch_size))
            rnd_w = random.randint(0, max(0, w - self.patch_size))
            clean = clean[:, rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size]

        if self.noise_aug:
            a, g = self.noise_model_augmentation()
        else:
            i = random.randint(0, 4)
            a = self.a_list[i]
            g = self.g_noise_var_list[i]

        noisy = clean.copy()
        noisy = noisy*(2**12-1-240.) + 240.
        noisy = self.generate_noise(noisy, a, g)
        noisy = np.clip((noisy-240.)/(2**12-1-240.), 0, 1)
        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)

        return {'clean': clean, 'noisy': noisy}

    def generate_noise(self, gt_raw, a, b):
        # generate poisson gaussian noise
        """
        a: sigma_s^2
        b: sigma_r^2
        """
        gaussian_noise_var = b
        # print(np.min(gt_raw-240))
        z = np.random.poisson((gt_raw-240)/a).astype(np.float32) * a
        z = z + np.random.randn(*z.shape).astype(np.float32) * np.sqrt(gaussian_noise_var) + 240 # add black level of CRVD
        z = np.clip(z, 0, 2**12-1) # CRVD 12-bit
        return z
    
    def noise_model_augmentation(self):
        # We calculate the linear relationship between log_g and log_a 
        # based on CRVD noise model: log_g = m * log_a + c
        m = 1.87
        c = 0.06
        a = np.random.uniform(3.0, 53.0)
        log_a = np.log(a)
        log_g = m * log_a + c
        g = np.exp(log_g)
        return a, g
    
    def bayer_preserving_augmentation(self, raw):
        aug_mode = np.random.randint(0, 4)
        raw = util.depack_gbrg_raw(raw)
        
        if aug_mode == 0:  # no augmentation
            aug_raw = raw
        elif aug_mode == 1:  # horizontal flip
            aug_raw = np.flip(raw, axis=1)[:,1:-1]
        elif aug_mode == 2: # vertical flip
            aug_raw = np.flip(raw, axis=0)[1:-1,:]
        else:  # random transpose
            aug_raw = np.transpose(raw, (1, 0))

        aug_raw = util.pack_gbrg_raw(aug_raw)
        return aug_raw

    
# -----------------------------------------------------
# Define dataset for raw image denoiser evaluation
# Denoiser is evaluated on CRVD test set
# -----------------------------------------------------
class test_raw_im_dataset(Dataset):
    def __init__(self, gt_dir, noisy_dir):
        self.gt_img_lists = []
        self.noisy_img_lists = []

        iso_list = [1600,3200,6400,12800,25600]
        for scene_id in range(7, 12):

            scene = 'scene{}'.format(scene_id) 

            for iso_id in iso_list:
                
                iso = 'ISO{}'.format(iso_id)

                # get two folder
                gt_img_path = os.path.join(gt_dir, scene, iso) 
                noisy_img_path = os.path.join(noisy_dir, scene, iso)

                gt_img_lists = os.listdir(gt_img_path)
                num_imgs = len(gt_img_lists)
                for i in range(1, num_imgs+1):
                    frame_id = 'frame{}'.format(i)
                    gt_img_name = frame_id + '_clean_and_slightly_denoised.tiff'
                    noisy_img_name = frame_id + '_noisy0.tiff'
                    self.gt_img_lists.append(os.path.join(gt_img_path, gt_img_name))
                    self.noisy_img_lists.append(os.path.join(noisy_img_path, noisy_img_name))

    def __getitem__(self, index):

        # uint16 image range: 240 - 4095
        black_level = 240
        white_level = 2**12 - 1
        gt_img = cv2.imread(self.gt_img_lists[index], -1)
        noisy_img = cv2.imread(self.noisy_img_lists[index], -1)
        gt_img = np.maximum(gt_img.astype(np.float32) - black_level, 0) / (white_level - black_level)
        noisy_img = np.maximum(noisy_img.astype(np.float32) - black_level, 0) / (white_level - black_level)

        # pack to RGBG image, range: 0 - 1, float32, shape (4, 1080, 1920)
        gt_img = util.pack_gbrg_raw(gt_img)
        noisy_img = util.pack_gbrg_raw(noisy_img)
        gt_img = torch.from_numpy(gt_img)
        noisy_img = torch.from_numpy(noisy_img)

        return {'clean': gt_img, 'noisy': noisy_img}
    
    def __len__(self):
        assert len(self.gt_img_lists) == len(self.noisy_img_lists)
        return len(self.gt_img_lists)

# -----------------------------------------------------
# Define dataset for rgb image denoiser pretraining
# -----------------------------------------------------
class pretrain_rgb_im_dataset(Dataset):
    def __init__(self, source_path, patch_size=128):
        full_img_lists = os.listdir(source_path)     
        self.full_img_lists = full_img_lists
        self.source_path = source_path
        self.patch_size = patch_size

    def __len__(self):
        return len(self.full_img_lists)
    
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.source_path, self.full_img_lists[index]), cv2.IMREAD_COLOR)
        img = img.astype(np.float32)
        clean = img.copy()
        
        if self.patch_size:
            h, w, _ = clean.shape
            rnd_h = random.randint(0, max(0, h - self.patch_size))
            rnd_w = random.randint(0, max(0, w - self.patch_size))
            clean = clean[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]

        
        mode = random.randint(0, 7)
        clean = util.augment_img(clean, mode)
        clean = util.uint2tensor3(clean) 
        noisy = clean.clone()
        noisy = self._generate_gaussian_noise(noisy, blind=True)

        return {'clean': clean, 'noisy': noisy}

    def _generate_gaussian_noise(self, clean_img, noise_level=25, blind=False):
        # noise basic settings
        if blind:
            noise_level = np.random.randint(10, 55)

        # synthesize noise
        # add AWGN to the clean image
        # noisy_img = clean_img + N(0, sigma)
        # sigma = (noise_level/255)^2
        noise = torch.randn(*clean_img.size())*(noise_level/255.0)
        noisy_img = noise + clean_img

        return noisy_img
    
class test_rgb_im_dataset(data.Dataset):
    def __init__(self, gt_dir):
        super().__init__()
        self.gt_dir = gt_dir
        self.gt_img_lists = []
        for root, dirs, files in os.walk(gt_dir):
            for file in files:
                self.gt_img_lists.append(os.path.join(root, file))

    def __len__(self):
        return len(self.gt_img_lists)
    
    def __getitem__(self, index):
        gt_img_path = self.gt_img_lists[index]
        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
        h, w, _ = gt_img.shape
        # if h and w is not 480 and 854, resize the image
        if h != 480 or w != 854:
            gt_img = cv2.resize(gt_img, (854, 480))
        gt_img = util.uint2tensor3(gt_img) # (c, h, w) range: [0, 1]

        return {'clean': gt_img}