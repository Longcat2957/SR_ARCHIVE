# Single Image Super Resolution
import os
import argparse
import torch
import cv2
from torch.utils.data import DataLoader
from libs.model import get_network, rmbn
from libs.core import load_model, test
from libs.utils.base import sisr_preprocess, sisr_postprocess, get_device
from libs.utils.dataset import testDataset


parser = argparse.ArgumentParser()
parser.add_argument(
    '-w', '--weights', type=str
)
parser.add_argument(
    '--test_datasets', type=str, default='../data/Classic Data'
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
    
    for td in os.listdir(opt.test_datasets):
        print(f"# TEST DATASET = {td}")
        td = os.path.join(opt.test_datasets, td)
        test_dataset = testDataset([os.path.join(td, x) for x in os.listdir(td)], )
        test_loader = DataLoader(
            test_dataset,
            1,
            False
        )
        test(net, test_loader)