import torch
import torch.nn as nn

class CAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Output size: (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Output size: (B, C, 1, 1)

        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        padding = (kernel_size - 1) // 2  # Ensure the output size is the same as input
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        x = torch.cat([avg_out, max_out], dim=1)  # Shape: (B, 2, H, W)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = CAM(in_planes, ratio)
        self.spatial_attention = SAM(kernel_size)

    def forward(self, x):
        x_out = x * self.channel_attention(x)
        x_out = x_out * self.spatial_attention(x_out)
        return x_out
