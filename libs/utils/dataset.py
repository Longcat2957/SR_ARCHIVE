import os
from .base import *

import torch
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def splitIndex(data_root:str, ratio:Tuple[float, float, float], seed:int=0):
    # ratio = (train, validation, test)
    
    length = len(os.listdir(data_root))
    assert length > 0
    
    r_sum = sum(ratio)
    rf1 = (length*(ratio[0]/r_sum), length*(ratio[1]/r_sum), length*(ratio[2]/r_sum))
    ri = list(map(int, rf1))
    remain = length - sum(ri)
    
    # 나머지는 훈련데이터에 포함시킨다.
    ri[0] += remain
    
    # Random Split
    np.random.seed(seed)
    id = np.arange(length)
    np.random.shuffle(id)
    
    io1, io2, io3 = id[:ri[0]], id[:ri[1]], id[:ri[2]]
    # Check data
    assert length == len(io1) + len(io2) + len(io3)
    assert len(io1) > 0 and len(io2) > 0 and len(io3) >= 0
    
    imgs_path_array = np.array([
        os.path.join(data_root, x) for x in os.listdir(data_root)
    ])

    # if include test sets..
    if len(io3) > 0:
        train_arr, val_arr = imgs_path_array[io1], imgs_path_array[io2]
        test_arr = imgs_path_array[io3]
        return train_arr, val_arr, test_arr

    train_arr, val_arr = imgs_path_array[io1], imgs_path_array[io2]
    return train_arr, val_arr

class baseDataset(Dataset):
    # DATASET CLASS for Train & VALIDATION
    def __init__(self,
                 imgs_list,
                 hr_transform,
                 lr_transform,
                 post_transform=None):
        super().__init__()
        self.imgs_list = imgs_list
        self.length = len(imgs_list)
        
        if hr_transform is None:
            raise NotImplementedError(hr_transform)
        if lr_transform is None:
            raise NotImplementedError(lr_transform)
        
        # data-augmentation
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        
        # albumentations base code
        self.to_tensor = ToTensorV2()
        
        # e.x) torchvision.transform.normalize
        if post_transform is None:
            self.post_transform = None
        else:
            self.post_transform = post_transform
        

    def __len__(self):
        return self.length

    def __getitem__(self, idx:int):

        img_path = self.imgs_list[idx]
        img = openImg(img_path)
    
        hr_obj = self.hr_transform(image=img)['image']
        lr_obj = self.lr_transform(image=hr_obj)['image']
        
        hr_tensor = TF.to_tensor(hr_obj)
        lr_tensor = TF.to_tensor(lr_obj)
        
        if self.post_transform is not None:
            hr_tensor = self.post_transform(hr_obj)
            lr_tensor = self.post_transform(lr_obj)
        
        
        return hr_tensor, lr_tensor

class testDataset(Dataset):
    def __init__(self, imgs_list, upscale:int=4):
        
        self.imgs_list = imgs_list
        self.length = len(imgs_list)
        self.upscale_ratio = upscale
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx:int):
        img_path = self.imgs_list[idx]
        img = openImg(img_path)

        H, W, _ = img.shape
        
        hr_h = (H//self.upscale_ratio)*self.upscale_ratio
        hr_w = (W//self.upscale_ratio)*self.upscale_ratio
        
        hr_obj = A.resize(img, height=hr_h, width=hr_w,
                          interpolation=cv2.INTER_CUBIC)
        
        hr_tensor = TF.to_tensor(hr_obj)
        
        lr_obj = A.resize(hr_obj, height=hr_h//self.upscale_ratio,\
            width=hr_w//self.upscale_ratio, interpolation=cv2.INTER_CUBIC)
        
        lr_tensor = TF.to_tensor(lr_obj)
        
        bicubic_obj = A.resize(lr_obj, height=hr_h, width=hr_w,
                          interpolation=cv2.INTER_CUBIC)
        
        bicubic_tensor = TF.to_tensor(bicubic_obj)
        
        return hr_tensor, lr_tensor, bicubic_tensor

# if __name__ == '__main__':
    
#     pass