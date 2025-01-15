import torch
import torch.nn as nn
from .attention.get_att_module import get_att_module

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False, attention_type=None):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            (nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") if down 
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(out_channels, affine=True),
            (nn.ReLU(inplace=True) if act == "relu" else nn.LeakyReLU(0.2, inplace=True)),
        )

        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
        self.attention = get_att_module(attention_type, out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
        return self.dropout(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64, attention_type=None):
        super(UNetGenerator, self).__init__()
        # Encoder (Downsampling)
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),  # 256 -> 128
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False, attention_type=attention_type)    # 128 -> 64
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False, attention_type=attention_type)  # 64 -> 32
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False, attention_type=attention_type)  # 32 -> 16
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False, attention_type=attention_type)  # 16 -> 8
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False, attention_type=attention_type)  # 8 -> 4
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False, attention_type=attention_type)  # 4 -> 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),  # 2 -> 1
            nn.ReLU(inplace=True)
        )

        # Decoder (Upsampling)
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True, attention_type=attention_type)     # 1 -> 2
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True, attention_type=attention_type)  # 2 -> 4
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True, attention_type=attention_type)  # 4 -> 8
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False, attention_type=attention_type) # 8 -> 16
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False, attention_type=attention_type) # 16 -> 32
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False, attention_type=attention_type) # 32 -> 64
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False, attention_type=attention_type)     # 64 -> 128
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)    # 128x128
        d2 = self.down1(d1)          # 64x64
        d3 = self.down2(d2)          # 32x32
        d4 = self.down3(d3)          # 16x16
        d5 = self.down4(d4)          # 8x8
        d6 = self.down5(d5)          # 4x4
        d7 = self.down6(d6)          # 2x2
        bottleneck = self.bottleneck(d7)  # 1x1
        up1 = self.up1(bottleneck)                         # 2x2
        up2 = self.up2(torch.cat([up1, d7], dim=1))        # 4x4
        up3 = self.up3(torch.cat([up2, d6], dim=1))        # 8x8
        up4 = self.up4(torch.cat([up3, d5], dim=1))        # 16x16
        up5 = self.up5(torch.cat([up4, d4], dim=1))        # 32x32
        up6 = self.up6(torch.cat([up5, d3], dim=1))        # 64x64
        up7 = self.up7(torch.cat([up6, d2], dim=1))        # 128x128
        return self.final_up(torch.cat([up7, d1], dim=1))  # 256x256


if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256))

    model = UNetGenerator()
    preds = model(x)
    print(preds.shape)  # torch.Size([2, 1, 256, 256])

    model = UNetGenerator(attention_type="cbam")
    preds = model(x)
    print(preds.shape)  # torch.Size([2, 1, 256, 256])

    model = UNetGenerator(attention_type="gam")
    preds = model(x)
    print(preds.shape)  # torch.Size([2, 1, 256, 256])

    model = UNetGenerator(attention_type="simam")
    preds = model(x)
    print(preds.shape)  # torch.Size([2, 1, 256, 256])

