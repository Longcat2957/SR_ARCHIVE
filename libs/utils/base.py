import os
import torch
from typing import Union, Tuple
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2
import numpy as np

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else \
        torch.device('cpu')

def to_2tuple(x:Union[int, tuple, list])->tuple:
    if isinstance(x, int):
        height, width = x, x
    elif isinstance(x, Union[tuple, list]) and len(x)==2:
        height, width = x[0], x[1]
    return (height, width)

def openImg(p:str):
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    try:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        raise ValueError(p)
    return img

def sisr_preprocess(p:str):
    img = openImg(p)
    img = TF.to_tensor(img)
    img = img.unsqueeze(dim=0)
    return img

def sisr_postprocess(t:torch.Tensor):
    if len(t.size()) == 4:
        t = t.squeeze(dim=0)
    elif len(t.size()) == 3:
        t = t
    else:
        raise NotImplementedError(f"SIZE ERROR = {len(t.size())}")
    t = t.cpu().clamp_(0.0, 1.0)
    t *= 255.0
    o = t.round().to(dtype=torch.uint8).numpy()
    o = np.transpose(o, axes=[1,2,0])
    o = cv2.cvtColor(o, cv2.COLOR_BGR2RGB)
    return o