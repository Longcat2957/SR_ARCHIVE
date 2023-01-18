import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

VALIDATION_INTERVAL = 5


# save weights function
def save_model(model:nn.Module, name:str):
    if os.path.exists(name):
        os.remove(name)
    torch.save(model.state_dict(), name)

# load weights function
def load_model(model:nn.Module, p:str):
    if not os.path.exists(p):
        raise FileNotFoundError(p)

    model.load_state_dict(torch.load(p))
    return model

# train function (1 epoch)
def train(model:nn.Module,
          train_loader:DataLoader,
          criterion:nn.Module,
          optimizer:torch.optim.Optimizer,
          e:list,
          ):
    # DEVICE
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else \
        torch.device('cpu')
    model = model.to(DEVICE)
    model = model.train()
    train_bar = tqdm(train_loader, ncols=100)
    
    total_loss = 0.0
    iteration = 0
    for hr, lr in train_bar:
        hr, lr = hr.to(DEVICE), lr.to(DEVICE)
        sr = model(lr)
        
        optimizer.zero_grad()
        
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.detach().mean()
        iteration += 1
        
        train_bar.set_description(
            desc=f"# TRAIN[{e[0]}/{e[1]}]| LOSS = {total_loss/iteration:.8f} | "
        )
    
# validation function (1 epoch)
def validation(model:nn.Module,
          val_loader:DataLoader,
          criterion:nn.Module,
          e:list,
          lr_scheduler:nn.Module=None
          ):
    # DEVICE
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else \
        torch.device('cpu')
        
    model = model.to(DEVICE)
    model = model.eval()
    val_bar = tqdm(val_loader, ncols=100)

    total_loss = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    iteration = 0
    
    for hr, lr in val_bar:
        hr, lr = hr.to(DEVICE), lr.to(DEVICE)
        
        with torch.no_grad():        
            sr = model(lr).detach()
            
        loss = criterion(hr, sr)
        total_loss += loss.mean()
        
        psnr_value = float(
            peak_signal_noise_ratio(
                sr, hr
            )
        )
        ssim_value = float(
            structural_similarity_index_measure(
                sr, hr
            )
        )
        psnr_sum += psnr_value
        ssim_sum += ssim_value
        iteration += 1
        
        val_bar.set_description(
            desc=f"# VALID[{e[0]}/{e[1]}]| LOSS = {total_loss/iteration:.8f} | PSNR = {psnr_sum/iteration:.3f}| SSIM = {ssim_sum/iteration:.5f}"
        )
    if lr_scheduler is not None:
        lr_scheduler.step()

# train-loop function (n epochs)
def train_loop(model:nn.Module,
               trainloader:DataLoader,
               valloader:DataLoader,
               optimizer:torch.optim.Optimizer,
               criterion:nn.Module,
               total_epochs:int,
               save_interval:int,
               save_path:str,
               tag:str=None,
               pre_trained=None,
               lr_scheduler=None):
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    if pre_trained is not None:
        model = load_model(model, pre_trained)

    for e in range(1, total_epochs+1):
        train(
            model=model,
            train_loader=trainloader,
            criterion=criterion,
            optimizer=optimizer,
            e=[e, total_epochs]
        )
        if e % VALIDATION_INTERVAL == 0:
            validation(
                model=model,
                val_loader=valloader,
                criterion=criterion,
                e=[e, total_epochs],
                lr_scheduler=lr_scheduler
            )
        
        if e % save_interval == 0:
            name = model.name + f"_e{e}_{total_epochs}_{tag}.pth"
            name = os.path.join(save_path, name)
            save_model(model, name)

# ToDo : TEST

def test(model:nn.Module,
         test_loader:DataLoader):
    
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else \
        torch.device('cpu')
        
    model = model.to(DEVICE)
    model = model.eval()
    test_bar = test_loader
    
    iteration = 0

    psnr_sr_hr = 0.0
    ssim_sr_hr = 0.0
    
    psnr_bc_hr = 0.0
    ssim_bc_hr = 0.0
    
    for hr, lr, bc in test_bar:
        iteration += 1
        hr = hr.to(DEVICE)
        lr = lr.to(DEVICE)
        bc = bc.to(DEVICE)
        
        with torch.no_grad():
            sr = model(lr).detach()

        psnr_value = float(
            peak_signal_noise_ratio(
                sr, hr
            )
        )
        ssim_value = float(
            structural_similarity_index_measure(
                sr, hr
            )
        )

        psnr_sr_hr += psnr_value
        ssim_sr_hr += ssim_value
        
        psnr_value = float(
            peak_signal_noise_ratio(
                bc, hr
            )
        )
        ssim_value = float(
            structural_similarity_index_measure(
                bc, hr
            )
        )
        
        psnr_bc_hr += psnr_value
        ssim_bc_hr += ssim_value
    
    print('')
    print("# TEST RESULT")
    print(f"# HT(ground_truth) <-> SUPERES | PSNR = {psnr_sr_hr/iteration:.3f} | SSIM = {ssim_sr_hr/iteration:.5f}")
    print(f"# HT(ground_truth) <-> BICUBIC | PSNR = {psnr_bc_hr/iteration:.3f} | SSIM = {ssim_bc_hr/iteration:.5f}")
    print('')


# ToDo : GAN based train

# ToDo : GAN based train_loop