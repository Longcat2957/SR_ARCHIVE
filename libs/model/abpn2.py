import torch
import torch.nn as nn

class basicBlock(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int=None,
                 act:bool=True
                 ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                groups=in_channels, bias=False
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0,
                stride=1, bias=False
            )
        )
        if act:
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()
            
    def forward(self, x):
        x = self.net(x)
        x = self.act(x)
        return x

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x:torch.Tensor):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)
    


class v3Baiscblock(nn.Module):
    def __init__(self, in_channels, rep:int=2):
        super().__init__()
        body = []
        for _ in range(rep - 2):
            body.append(basicBlock(in_channels, in_channels))
        body.append(basicBlock(in_channels, in_channels, False))
        self.body = nn.Sequential(*body)
        self.channel_shuffle = ShuffleBlock(groups=3)
        self.head = nn.Sequential(
            basicBlock(in_channels+3, in_channels+3),
            basicBlock(in_channels+3, in_channels+3),
        )
    
    def forward(self, x:torch.Tensor, residual:torch.Tensor):
        x = self.body(x)
        x = torch.cat([residual, x], dim=1)
        x = self.channel_shuffle(x)
        x = self.head(x)
        return x

class ABPNv3(nn.Module):
    def __init__(self, scale_ratio:int=4, rep:int=4):
        super().__init__()
        self.number_of_layer = scale_ratio ** 2 - 1
        self.main = nn.ModuleList([v3Baiscblock(3*i, rep) \
            for i in range(1, self.number_of_layer+1)])
        self.depth_to_space = nn.PixelShuffle(scale_ratio)
    
    def forward(self, x):
        residual = x
        for i in range(self.number_of_layer):
            x = self.main[i](x, residual)
        x = self.depth_to_space(x)
        return x
