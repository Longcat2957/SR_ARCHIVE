# MOBILE_SR_ARCHIVE

## Introduction  
개인 공부 목적으로 임베디드시스템(Jetson, 라즈베리파이) 등에 적합한 초해상화 모델을 정리하는 공간입니다.  


## To-Do list
##### 높은 우선순위 🔥🔥🔥 
`libs.network` 정리하기 🔥  
`test model`  test.py 코드 마무리하기  
`single image inference with OpenCV` 50%   
`video inference with OpenCV`  

##### 중간 우선순위 ⭐⭐⭐
`pretrained weights 추가하기` RLFN(2023-01-15 시작)  🚀  
`gan based learning`  

##### 낮은 우선순위 ⭐
`Quantization`  
`TensorRT Inference`  
`ONNX-Runtime Inference`  
`Mobile Inference Benchmark`  Jetson Xavier NX, Jetson Nano(2G), raspberry pi(4B+8g)  


## Requirements
```
pytorch
torchvision
torchmetrics
colorama
opencv-python
tqdm
matplotlib
numpy
```

## Models
|Model|내용|논문|Code|progress|
|-------------|--------------|------------------|-------------|----|
|RLFN|First place in Runtime Track(NTIRE 2022)|https://arxiv.org/abs/2205.07514|https://github.com/bytedance/RLFN|O|
|ABPN|Anchor-based Plain Net for Mobile Image Super-Resolution|https://arxiv.org/pdf/2105.09750.pdf|-|O|
|VapSR|Efficient Image Super-Resolution using Vast-Receptive-Field Attention|https://arxiv.org/abs/2210.05960|https://github.com/zhoumumu/VapSR|X|
|MTESR|First place in Model Complexity Track(NTIRE2022)|-|https://github.com/sunny2109/MobileSR-NTIRE2022|X|
|HPINet|Hierarchical Pixel Integration for Lightweight Image Super-Resolution|https://arxiv.org/abs/2211.16776|https://github.com/passerer/hpinet|X|



## Pre-trained weights & PSNR/SSIM Score
Ubuntu 22.04, Pytorch 1.13.1+cu11.7  
13900k + 64G(DDR5) + RTX4090  


|Model|upscale_ratio|train|link|
|-------------|--------------|------------------|------------------|
|RLFN|x4|DIV2K + Flickr2K|-|
|ABPN|x4|DIV2K + Flickr2K|-|



##### Supervised Learning (L1 Loss)
|Model|upscale_ratio|PSNR|SSIM|
|-----|-------------|----|----|
|RLFN|x4|-|-|
|ABPN|x4|-|-|

##### GAN Based Learning + Enhanced data-augmentation tactics  
|Model|upscale_ratio|PSNR|SSIM|
|-----|-------------|----|----|
|RLFN|x4|-|-|
|ABPN|x4|-|-|


## Train (FP32)
```
  in progress
```

## Train (QAT) 
Quantization Aware Training(in Pytorch)  
```
  in progress
```

## Inference
#### Single Image Super Resolution (fp32/fp16)
```
  python train.py --model RLFN
```
#### Single Video Super Resolution (fp32/fp16)
```
  in progress
```
#### Single Image Super Resolution with Quantization (int8)
```
  in progress
 ```
#### Single Video Super Resolution with Quantization (int8)
```
  in progress
```
## ONNX-Runtime Inference
```
  in progress
```
## TensorRT Inference
```
  in progress
```
## Contact
lucete030@konkuk.ac.kr  
