import torch
import torch.nn as nn

class GAM_Channel(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(GAM_Channel, self).__init__()
        reduced_channels = in_channels // ratio
        
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  # (B, H*W, C)
        x_att = self.channel_mlp(x_permute).view(b, h, w, c).permute(0, 3, 1, 2)  # (B, C, H, W)
        return self.sigmoid(x_att)
    
class GAM_Spatial(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(GAM_Spatial, self).__init__()
        reduced_channels = in_channels // ratio
        padding = (kernel_size - 1) // 2
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size, padding=padding, padding_mode='replicate'),
            nn.InstanceNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size, padding=padding, padding_mode='replicate'),
            nn.InstanceNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.spatial_attention(x)

class GAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(GAM, self).__init__()
        self.channel_attention = GAM_Channel(in_channels, ratio)
        self.spatial_attention = GAM_Spatial(in_channels, ratio, kernel_size)

    def forward(self, x):
        x_out = x * self.channel_attention(x)
        x_out = x_out * self.spatial_attention(x_out)
        return x_out