import os
import argparse
import torch
from torch.utils.data import DataLoader
from libs.model import *
from libs.utils.transform import SimpleDataAugmentation, ComplexDataAugmentation
from libs.utils.dataset import baseDataset, splitIndex
from libs.core import train_loop, load_model



parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', type=str, choices=NETWORKS
)
parser.add_argument(
    '-b', '--batch_size', type=int, default=32
)
parser.add_argument(
    '-d', '--dataset', type=str, default='../data/DF2K'
)
parser.add_argument(
    "--hr_size", type=int, default=256
)
parser.add_argument(
    "--lr_size", type=int, default=64
)
parser.add_argument(
    "-e", "--epochs", type=int, default=1000
)
parser.add_argument(
    '--wobn', action='store_true'
)
parser.add_argument(
    '--complex_da', action='store_true'
)
parser.add_argument(
    '--weights', type=str
)

if __name__ == '__main__':
    opt = parser.parse_args()

    net = get_network(opt.model)
    
    TAG = ''
    if opt.wobn:
        print(f"# REMOVE Batch Normalization Layer ... ")
        net.apply(rmbn)
        TAG += 'wobn'
    else:
        TAG += 'baseline'
    
    if opt.weights is not None:
        net = load_model(net, opt.weight)
        
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5)
    criterion = torch.nn.L1Loss()
    
    if opt.complex_da:
        dat = ComplexDataAugmentation
        TAG += '_complex'
    else:
        dat = SimpleDataAugmentation
        TAG += '_simple'
    
    train_arr, val_arr = splitIndex(opt.dataset, [0.9, 0.1, 0.0])
    
    train_hr_t, train_lr_t = dat(hr_size=opt.hr_size, lr_size=opt.lr_size)()
    valid_hr_t, valid_lr_t = dat(hr_size=opt.hr_size, lr_size=opt.lr_size, train=False)()

    train_dataset = baseDataset(
        train_arr,
        train_hr_t,
        train_lr_t
    )
    
    val_dataset = baseDataset(
        val_arr,
        valid_hr_t,
        valid_lr_t
    )

    train_loader = DataLoader(
        train_dataset,
        opt.batch_size,
        True,
        num_workers=os.cpu_count()//2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        opt.batch_size,
        False,
        num_workers=os.cpu_count()//2,
        pin_memory=True
    )

    train_loop(
        net,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        opt.epochs,
        20,
        './weights',
        TAG,
        None,
        lr_scheduler
    )