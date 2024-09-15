import os
import numpy as np
import rawpy
import pickle
import mat73
import argparse
import cv2
import glob

# All packed images are normalized to [0, 1] 
# -------------------------------------------------
# SID 
# -------------------------------------------------
def pack_raw_bayer_sid(raw):
    img = raw.raw_image_visible.astype(np.float32)
    # Bayer pattern in SID: RGGB
    white_level = 16383
    black_level = 512
    # the bayer pattern of sony A7R3 is RGGB, to be consistent with CRVD, we need to change it to GBRG
    img = img[1:-1, :] # now the patterm is GBRG
    img_shape = img.shape
    H = img_shape[0] - img_shape[0] % 2
    W = img_shape[1] - img_shape[1] % 2

    out = np.stack(
        (
            img[1:H:2,0:W:2], #RGBG
            img[1:H:2,1:W:2],
            img[0:H:2,1:W:2],
            img[0:H:2,0:W:2]
        ), axis=2
    ) # H, W, 4

    out = (out - black_level) / (white_level - black_level)
    out = np.clip(out, 0, 1)

    return out

# -------------------------------------------------
# SIDD
# -------------------------------------------------
def raw_pattern_sidd(camera_name):
    if camera_name == 'GP':
        return 'bggr'
    
    elif camera_name == 'IP':
        return 'rggb'
    
    elif camera_name == 'S6':
        return 'grbg'
    
    elif camera_name == 'N6':
        return 'bggr'
    
    elif camera_name == 'G4':
        return 'bggr'
    
    else:
        raise NotImplementedError
    

def pack_raw_bayer_sidd(raw, raw_pattern):
    img = raw.astype(np.float32)

    if raw_pattern == 'bggr':
        img = img[:, 1:-1]
        
    elif raw_pattern == 'rggb':
        img = img[1:-1, :]
        
    elif raw_pattern == 'grbg':
        img = img[1:-1, 1:-1]
        
    else: 
        raise Exception('raw pattern not supported!')
    
    img_shape = img.shape
    H = img_shape[0] - img_shape[0] % 2
    W = img_shape[1] - img_shape[1] % 2
    
    out = np.stack(
        (
            img[1:H:2,0:W:2], #RGBG
            img[1:H:2,1:W:2],
            img[0:H:2,1:W:2],
            img[0:H:2,0:W:2]
        ), axis=2
    )
    out = np.clip(out, 0, 1)
    
    return out

# -------------------------------------------------
# CRVD
# -------------------------------------------------
def pack_raw_bayer_crvd(raw):
    # bayer pattern: GBRG
    black_level = 240
    white_level = 2**12 - 1
    img = raw.astype(np.float32)
    img_shape = img.shape
    H = img_shape[0] - img_shape[0] % 2
    W = img_shape[1] - img_shape[1] % 2

    out = np.stack((img[1:H:2,0:W:2], #RGBG
                    img[1:H:2,1:W:2],
                    img[0:H:2,1:W:2],
                    img[0:H:2,0:W:2]), axis=2)
    out = (out - black_level) / (white_level - black_level)
    out = np.clip(out, 0, 1)
    
    return out

# -------------------------------------------------
# Sensenoise
# -------------------------------------------------
def pack_raw_bayer_sensenoise(raw):
    # bayer pattern: RGGB
    black_level = 64
    white_level = 1023
    img = raw.astype(np.float32)
    img = img[1:-1, :] # GBRG

    img_shape = img.shape
    H = img_shape[0] - img_shape[0] % 2
    W = img_shape[1] - img_shape[1] % 2

    out = np.stack((img[1:H:2,0:W:2], #RGBG
                    img[1:H:2,1:W:2],
                    img[0:H:2,1:W:2],
                    img[0:H:2,0:W:2]), axis=2)
    out = (out - black_level) / (white_level - black_level)
    out = np.clip(out, 0, 1)

    return out

# -------------------------------------------------
# Image cropping
# -------------------------------------------------
def crop_imgs(img, target_path, p_size=512, p_overlap=64, p_max=800):
    '''
    img: original full size raw image, numpy array, shape (H, W, 4), bayer pattern RGBG
    p_size: patch size
    p_overlap: overlap size between patches
    p_max: maximum size of original image, if larger than this size, crop the image to patches
    '''
    num_patch = 0
    h, w, _ = img.shape

    if w>p_max and h>p_max:
        w1 = list(np.arange(0, w-p_size, p_size-p_overlap, dtype=np.int16))
        h1 = list(np.arange(0, h-p_size, p_size-p_overlap, dtype=np.int16))
        w1.append(w-p_size)
        h1.append(h-p_size)
        for i in h1:
            for j in w1:
                num_patch += 1
                img_patch = img[i:i+p_size, j:j+p_size]
                img_patch_name = target_path + '_{:03d}.npy'.format(num_patch)
                np.save(
                    img_patch_name,
                    img_patch
                )

    else:
        img_patch_name = target_path + '_{:03d}.npy'.format(num_patch)
        np.save(
            img_patch_name,
            img
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='*', help='crop selected datasets to patches, dataset names: [SID, SIDD, CRVD, Sensenoise]')
    parser.add_argument('--save_dir', type=str, default='./Datasets/image_denoising/pretrain_raw_im_patch/', help='save directory')
    args = parser.parse_args()

    sid_dir = './Datasets/image_denoising/raw_im_source/SID/Sony/long/'
    sidd_dir = './Datasets/image_denoising/raw_im_source/SIDD_Medium_Raw/Data/'
    crvd_dir = './Datasets/video_denoising/CRVD_dataset/indoor_raw_gt/'
    sensenoise_dir = './Datasets/image_denoising/raw_im_source/sensenoise/sensenoise_raw_v4_230730/gt/'
    
    save_dir = args.save_dir
    selected_datasets = args.datasets

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if 'SID' in selected_datasets:
        sid_img_lists = os.listdir(sid_dir)
        for file in sid_img_lists:
            if file.endswith('ARW'):
                raw = rawpy.imread(sid_dir + file)
                raw_name = file.split('.')[0]
                raw_name = 'SID_'+raw_name
                img = pack_raw_bayer_sid(raw)
                crop_imgs(img, save_dir+raw_name)

    if 'SIDD' in selected_datasets:
        # walk into the folder
        for root, dirs, files in os.walk(sidd_dir):
            for file in files:
                if 'GT_RAW_010' in file:
                    camera_id = root.split('/')[-1]
                    camera_name = camera_id[9:11]
                    bayer_pattern = raw_pattern_sidd(camera_name)   
                    raw = mat73.loadmat(os.path.join(root, file))['x']
                    raw_name = file.split('.')[0]
                    raw_name = 'SIDD_'+ camera_id + '_' + raw_name
                    img = pack_raw_bayer_sidd(raw, bayer_pattern)
                    crop_imgs(img, save_dir+raw_name)

    if 'CRVD' in selected_datasets:
        iso_list = [1600,3200,6400,12800,25600]
        for scene_id in range(1, 7):
            # first 7 scenes for training 
            scene = 'scene{}'.format(scene_id)
            for iso_id in iso_list:
                iso = 'ISO{}'.format(iso_id)

                folder = os.path.join(crvd_dir, scene, iso)
                files = sorted(glob.glob(os.path.join(folder, "*.tiff")))

                for file in files:
                    raw = cv2.imread(file, -1)
                    raw_name = file.split('/')[-1].split('.')[0]
                    raw_name = 'CRVD_'+ 'scene{:02d}'.format(scene_id) + '_' + 'ISO{:05d}'.format(iso_id) + '_' + raw_name[:6]
                    img = pack_raw_bayer_crvd(raw)
                    crop_imgs(img, save_dir+raw_name)

    if 'Sensenoise' in selected_datasets:
        sensenoise_img_lists = os.listdir(sensenoise_dir)
        for file in sensenoise_img_lists:
            if file.endswith('npy'):
                raw = np.load(sensenoise_dir + file)
                raw_name = file.split('.')[0]
                raw_name = 'Sensenoise_'+raw_name
                img = pack_raw_bayer_sensenoise(raw)
                crop_imgs(img, save_dir+raw_name)
        
if __name__ == '__main__':
    main()