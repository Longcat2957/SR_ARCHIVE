import torch
import torch.nn as nn
from .common import Concatenation, BasicConv
    
class ABPN(nn.Module):
    def __init__(self, upscale:int=4):
        super().__init__()
        self.name = 'ABPN'
        self.concat = Concatenation(upscale)
        
        hidden_dim = 3 * upscale ** 2 + 1
        fe = []
        fe.append(BasicConv(3, hidden_dim, kernel_size=3, stride=1, padding=1))
        for _ in range(4):
            fe.append(BasicConv(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1))
        fe.append(BasicConv(hidden_dim, hidden_dim-1, kernel_size=3, stride=1, padding=1))
        fe.append(BasicConv(hidden_dim-1, hidden_dim-1, act=nn.Identity(), kernel_size=3, stride=1, padding=1))
        
        self.fe = nn.Sequential(*fe)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

        
    def forward(self, x):
        c = self.concat(x)
        fe = self.fe(x)
        x = torch.add(c, fe)
        x = self.pixel_shuffle(x)

        return x
