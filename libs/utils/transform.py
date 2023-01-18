import cv2
import albumentations as A
from .base import to_2tuple

class SimpleDataAugmentation(object):
    def __init__(self, hr_size, lr_size, train:bool=True):
        # simple data augmentation for supervised train
        self.hr_size = to_2tuple(hr_size)
        self.lr_size = to_2tuple(lr_size)
        self.train = train

    def __call__(self, *args, **kwargs):
        return self.pull(*args, **kwargs)
    
    def pull(self):
        if self.train:
            # if train
            hr_transform = A.Compose([
                A.RandomCrop(height=self.hr_size[0], width=self.hr_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5)
            ])
            lr_transform = A.Compose([
                A.Resize(height=self.lr_size[0], width=self.lr_size[1])
            ])

        else:
            # if validation
            hr_transform = A.Compose([
                A.CenterCrop(height=self.hr_size[0], width=self.hr_size[1])
            ])
            lr_transform = A.Compose([
                A.Resize(height=self.lr_size[0], width=self.lr_size[1], interpolation=cv2.INTER_CUBIC)
            ])
        return hr_transform, lr_transform

class ComplexDataAugmentation(object):
    def __init__(self, hr_size, lr_size, train:bool=True):
        # simple data augmentation for supervised train
        self.hr_size = to_2tuple(hr_size)
        self.lr_size = to_2tuple(lr_size)
        self.train = train

    def __call__(self, *args, **kwargs):
        return self.pull(*args, **kwargs)
    
    def pull(self):
        if self.train:
            # if train
            hr_transform = A.Compose([
                A.RandomCrop(height=self.hr_size[0], width=self.hr_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5)
            ])
            lr_transform = A.Compose([
                A.AdvancedBlur(p=0.5),
                A.Downscale(scale_min=0.7, scale_max=0.95, p=0.5, interpolation=cv2.INTER_CUBIC),
                A.OneOf([A.GaussNoise(), A.ISONoise()], p=0.5),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                A.AdvancedBlur(p=0.5),
                A.Resize(height=self.lr_size[0], width=self.lr_size[1], interpolation=cv2.INTER_CUBIC),
                A.OneOf([A.GaussNoise(), A.ISONoise()], p=0.5),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5)
            ])

        else:
            # if validation
            hr_transform = A.Compose([
                A.CenterCrop(height=self.hr_size[0], width=self.hr_size[1])
            ])
            lr_transform = A.Compose([
                A.Resize(height=self.lr_size[0], width=self.lr_size[1], interpolation=cv2.INTER_CUBIC)
            ])
        return hr_transform, lr_transform