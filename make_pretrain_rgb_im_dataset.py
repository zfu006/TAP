import os
import numpy as np
import rawpy
import pickle
import mat73
import argparse
import cv2
import glob

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
                img_patch_name = target_path + '_{:03d}.png'.format(num_patch)
                cv2.imwrite(
                    img_patch_name,
                    img_patch
                )

    else:
        img_patch_name = target_path + '_{:03d}.png'.format(num_patch)
        cv2.imwrite(
            img_patch_name,
            img
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./Datasets/image_denoising/pretrain_raw_im_patch/', help='save directory')
    args = parser.parse_args()

    div2k_dir = './Datasets/image_denoising/rgb_im_source/DIV2K-tr_1X/'
    wed_dir = './Datasets/image_denoising/rgb_im_source/exploration_database_and_code/pristine_images/'
    flickr2k_dir = './Datasets/image_denoising/rgb_im_source/Flickr2K/'
    bsd500_dir = './Datasets/image_denoising/rgb_im_source/BSD500/images/'
    
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    div2k_img_lists = os.listdir(div2k_dir)
    for file in div2k_img_lists:
        if file.endswith('png'):
            img = cv2.imread(os.path.join(div2k_dir, file), -1)
            img_name = file.split('.')[0]
            img_name = 'DIV2K_'+img_name
            crop_imgs(img, os.path.join(save_dir, img_name))

    wed_img_lists = os.listdir(wed_dir)
    for file in wed_img_lists:
        if file.endswith('bmp'):
            img = cv2.imread(os.path.join(wed_dir, file), cv2.IMREAD_COLOR)
            img_name = file.split('.')[0]
            img_name = 'WED_'+img_name
            crop_imgs(img, os.path.join(save_dir, img_name))

    # walk into the folder
    for root, dirs, files in os.walk(flickr2k_dir):
        for file in files:
            if file.endswith('png'):
                img = cv2.imread(os.path.join(root, file), -1)
                img_name = file.split('.')[0]
                img_name = 'Flickr2K_'+img_name
                crop_imgs(img, os.path.join(save_dir, img_name))
            
    for root, dirs, files in os.walk(bsd500_dir):
        for file in files:
            if file.endswith('jpg'):
                img = cv2.imread(os.path.join(root, file), -1)
                img_name = file.split('.')[0]
                img_name = 'BSD500_'+img_name
                crop_imgs(img, os.path.join(save_dir, img_name))

        
if __name__ == '__main__':
    main()

