import torch.nn as nn

class PatchGANReviser(nn.Module):
    def __init__(self, input_channels=3, target_channels=1):
        super(PatchGANReviser, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(input_channels + target_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2)
        )

        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.final(x)


if __name__ == "__main__":
    import torch
    x = torch.rand((2, 4, 256, 256))

    model = PatchGANReviser()
    preds = model(x)
    print(preds.shape)  # torch.Size([2, 1, 30, 30])