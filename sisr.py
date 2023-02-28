# Single Image Super Resolution
import os
import argparse
import torch
import cv2
from libs.model import get_network, rmbn
from libs.core import load_model
from libs.utils.base import sisr_preprocess, sisr_postprocess, get_device

parser = argparse.ArgumentParser()
parser.add_argument(
    '-w', '--weights', type=str
)
parser.add_argument(
    '--img', type=str, default='./test.png'
)

if __name__ == '__main__':
    opt = parser.parse_args()
    weight_abs_path = os.path.join('./weights', opt.weights)
    
    model_name = opt.weights.split('_')[0]
    bn = opt.weights.split('_')[-2]
    
    net = get_network(model_name)
    
    if bn == 'wobn':
        net.apply(rmbn)
    
    DEVICE = get_device()
    
    net = load_model(net, weight_abs_path)
    net = net.eval()
    net = net.to(DEVICE)
    
    lr_tensor = sisr_preprocess(opt.img).to(DEVICE)
    with torch.no_grad():
        sr = net(lr_tensor).detach()
        
    sr_np = sisr_postprocess(sr)
    cv2.imwrite('sr_output.png', sr_np)