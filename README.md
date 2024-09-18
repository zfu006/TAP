<p align="center">
  <h1 align="center">Temporal As a Plugin: Unsupervised Video Denoising with Pre-Trained Image Denoisers</h1>
   <h3 align="center">
    <a href="https://arxiv.org/abs/2409.11256" target='_blank'><img src="https://img.shields.io/badge/arXiv-2407.16125-b31b1b.svg"></a>
  </h3>
  
  <p align="center">
    <a href="https://github.com/zfu006" target="_blank">Zixuan Fu</a>,
    <a href="https://github.com/GuoLanqing" target="_blank">Lanqing Guo</a>,
    <a href="https://github.com/ChongWang1024" target="_blank">Chong Wang</a>,
    <a href="https://github.com/wyf0912" target="_blank">Yufei Wang</a>,
    <a href="https://github.com/lizhihao6" target="_blank">Zhihao Li</a> and
    <a href="https://personal.ntu.edu.sg/bihan.wen/" target="_blank">Bihan Wen*</a>.
  
  </p>

</p>

This is the official implementation of Temporal As a Plugin: Unsupervised Video Denoising with Pre-Trained Image Denoisers (ECCV 2024).

## Abstract 

>Recent advancements in deep learning have shown impressive results in image and video denoising, leveraging extensive pairs of noisy and noise-free data for supervision. However, the challenge of acquiring paired videos for dynamic scenes hampers the practical deployment of deep video denoising techniques. In contrast, this obstacle is less pronounced in image denoising, where paired data is more readily available. Thus, a well-trained image denoiser could serve as a reliable spatial prior for video denoising. In this paper, we propose a novel unsupervised video denoising framework, named ``**T**emporal **A**s a **P**lugin'' (TAP), which integrates tunable temporal modules into a pre-trained image denoiser. By incorporating temporal modules, our method can harness temporal information across noisy frames, complementing its power of spatial denoising. Furthermore, we introduce a progressive fine-tuning strategy that refines each temporal module using the generated *pseudo clean* video frames, progressively enhancing the network's denoising performance. Compared to other unsupervised video denoising methods, our framework demonstrates superior performance on both sRGB and raw video denoising datasets.


## Setup

### Requirements

- PyTorch 1.13.0
- torchvision 0.14.0
- CUDA 11.2
- Python 3.8
- opencv 4.8.1

### Pretrained models

Download [pretrained models](https://drive.google.com/drive/folders/1REk44iw0usXG9QTjQUH21Epu2MNLBk40?usp=sharing)  here and place them into `/checkpoints/` folder.

## Quick test

### Raw video denoising
Run ```main_test_tap_crvd_indoor.py/main_test_tap_crvd_outdoor.py``` for evaluation.

## Samples

<div style="text-align: center;">
  <img src="./sample/crvd_outdoor_scene3_denoised.gif" alt="denoised" width="400" style="display: inline-block;"/>
  <img src="./sample/crvd_outdoor_scene3_noisy.gif" alt="noisy" width="400" style="display: inline-block;"/>
</div>
<br/>
<div style="text-align: center;">
  <img src="./sample/crvd_outdoor_scene4_denoised.gif" alt="denoised" width="400" style="display: inline-block;"/>
  <img src="./sample/crvd_outdoor_scene4_noisy.gif" alt="noisy" width="400" style="display: inline-block;"/>
</div>

## Train

### Fine-tuning on raw videos
Run the following command to train video denoiser on CRVD
```
python main_finetune_crvd_tap.py --stage 1 --pretrained_checkpoints_dir './checkpoints/nafnet/nafnet_raw.pth' --save_dir ./saves/ft_crvd_indoor/ --test_clean_vid_dir ./Datasets/video_denoising/CRVD_dataset/indoor_raw_gt/ --test_noisy_vid_dir ./Datasets/video_denoising/CRVD_dataset/indoor_raw_noisy/ --n_frames 5 --scene_type indoor --patch_size 256 --train_batch_size 6 --test_batch_size 6 --num_workers 32 --gpu_ids 0 1 3 --in_nc 4 --nc 64 --verbose --G_lr 1e-3 --pixel_loss_type L1 --epochs 1000 --test_step 120 --save_models --bayer_aug --prepare_datasets
```
Note to change the stages and the corresponding checkpoints directory for progressive tuning.

## Reference

Our implementation is based on [KAIR](https://github.com/cszn/KAIR), [RViDeNet](https://github.com/cao-cong/RViDeNet), [NAFNet](https://github.com/megvii-research/NAFNet). We would like to thank them.