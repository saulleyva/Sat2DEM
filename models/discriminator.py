import torch
import torch.nn as nn
from .attention.get_att_module import get_att_module

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=3, target_channels=1, attention_type=None):
        super(PatchGANDiscriminator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(input_channels + target_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.attention1 = get_att_module(attention_type, 128)

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.attention2 = get_att_module(attention_type, 256)

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.attention3 = get_att_module(attention_type, 512)

        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.initial(x)
        x = self.attention1(self.layer1(x))
        x = self.attention2(self.layer2(x))
        x = self.attention3(self.layer3(x))
        return self.final(x)


if __name__ == "__main__":
    import torch
    x = torch.rand((2, 3, 256, 256))
    y = torch.rand((2, 1, 256, 256))

    model = PatchGANDiscriminator()
    preds = model(x,y)
    print(preds.shape)  # torch.Size([2, 1, 30, 30])

    model = PatchGANDiscriminator(attention_type="ca")
    preds = model(x,y)
    print(preds.shape)  # torch.Size([2, 1, 30, 30])

    model = PatchGANDiscriminator(attention_type="cbam")
    preds = model(x,y)
    print(preds.shape)  # torch.Size([2, 1, 30, 30])

    model = PatchGANDiscriminator(attention_type="gam")
    preds = model(x,y)
    print(preds.shape)  # torch.Size([2, 1, 30, 30])