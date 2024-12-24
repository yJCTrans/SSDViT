# [ICASSP2025] SSDViT: Exploring Siamese and Self Distillation in ViTs for Generalizable Person Re-identification

## Pipeline
![image](https://github.com/yJCTrans/SSDViT/blob/main/figs/framework.jpg)

## Ablation Study
![image](https://github.com/yJCTrans/SSDViT/blob/main/figs/Ablation.png)

## Requirements
### Installation
we use /torch 1.13.1 /torchvision 0.14.1 /cuda 11.7 /four 24G RTX 4090 for training and evaluation
### Prepare Datasets
Training Data:
The model is trained and evoluated on Market-1501, MSMT17_V1, CUHK03, CUHK-SYSU.</br>
```
data
├── market1501
│    └── Market-1501-v15.09.15
│        └── images ..
├── cuhk03
│    └── cuhk03_release
│        └── images ..
├── msmt17v1
│    └── MSMT17_V1
│        └── images ..
└── cuhk_sysu
│    └── croppped_images
│        └── images ..
```
### Prepare ViT Pre-trained Models
You need to download the pretrained vision transformer model : [ViT-B/16](https://drive.google.com/file/d/1ZFMCBZ-lNFMeBD5K8PtJYJfYEk5D9isd/view)
## Training
We utilize 4 GPUs for training</br>
Single-source domain
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_vit.py -d msmt17v1 --dataset_src1 market1501 -b 128 --test-batch-size 256 --height 256 --width 128 --num-instances 8 -a vit_base --BNNeck --SD --sd_temp 3.0 --sd_weight 0.7 --epochs 40 --iters 400 --data-dir your dataset path --logs-dir your log path
```
Three source domains
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_vit.py -d msmt17v1 --dataset_src1 market1501 --dataset_src2 cuhk_sysu --dataset_src3 cuhk03 --multi_source -b 128 --test-batch-size 256 --height 256 --width 128 --num-instances 8 -a vit_base --BNNeck --SD --sd_temp 3.0 --sd_weight 0.7 --epochs 60 --iters 400 --data-dir your dataset path --logs-dir your log path
```
## Evaluation
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python test_vit.py -d msmt17v1 --test-batch-size 256 --height 256 --width 128 -a vit_base --data-dir your dataset path --logs-dir your model path
```
## Acknowledgement
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.</br>
[TransReID](https://github.com/damo-cv/TransReID), [TransReID-SSL](https://github.com/damo-cv/TransReID-SSL)
## Contact
If you have any question, please feel free to contact us. E-mail: [jierujia@sxu.edu.cn](mailto:jierujia@sxu.edu.cn), [202322405023@email.sxu.edu.cn](mailto:202322405023@email.sxu.edu.cn)
