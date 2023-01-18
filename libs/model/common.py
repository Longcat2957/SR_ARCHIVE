import torch
import torch.nn as nn
import torch.nn.functional as F

class Concatenation(nn.Module):
    def __init__(self, scale_ratio:int=4):
        super().__init__()
        self.rep = scale_ratio ** 2
        
    def forward(self, x):
        length = len(x.size())
        temp = [x for _ in range(self.rep)]
        if length == 3:
            return torch.cat(temp, dim=0)
        elif length == 4:
            return torch.cat(temp, dim=1)
        
class Clip(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.clip(x, min=0.0, max=255.0)

class BasicConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int,
                 bn:bool=True,
                 act:nn.Module=nn.ReLU(),
                 **kwargs):
        super().__init__()
        # convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        
        # batch-normalization layer
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()
        
        # activation function
        if isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()
            
    def forward(self, x:torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


def rmbn(m):
    if isinstance(m, nn.BatchNorm2d):
        print(f"find batch norm and change to nn.Identity")
        m = nn.Identity()

if __name__ == '__main__':
    # Test area
    conv_layer = BasicConv(3,6,False, kernel_size=3)
    a = torch.randn(1, 3, 224, 224)
    b = conv_layer(a)
    print(b.shape)
    print(conv_layer)