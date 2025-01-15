import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

class DRPnet(nn.Module):
    def __init__(self, window_width=5, window_height=5, region_width=20, region_height=20,
                 receptive_field=70.0, mask_opt=True, spatial_scale=1.0, sampling_ratio=2):
        super(DRPnet, self).__init__()
        self.window_width = window_width
        self.window_height = window_height
        self.region_width = region_width
        self.region_height = region_height
        self.stride = 1
        self.receptive_field = receptive_field
        self.mask_opt = mask_opt
        self.roialign = RoIAlign((self.region_height, self.region_width),
                                 spatial_scale=spatial_scale,
                                 sampling_ratio=sampling_ratio)
        window_area = self.window_width * self.window_height
        kernel = torch.ones((1, 1, self.window_height, self.window_width)) / window_area
        self.register_buffer('mean_kernel', kernel)

    def _localize(self, score_map, input_image):
        batch_size, _, H, W = score_map.size()
        
        mean_scores = F.conv2d(score_map, self.mean_kernel, bias=None, stride=self.stride, padding=0)
        pro_height, pro_width = mean_scores.shape[2], mean_scores.shape[3]
        mean_scores_flat = mean_scores.view(batch_size, -1)

        _, min_indices = torch.min(mean_scores_flat, dim=1)
        min_indices_2d = torch.stack((min_indices // pro_width, min_indices % pro_width), dim=1) 

        y1 = min_indices_2d[:, 0] * self.stride
        x1 = min_indices_2d[:, 1] * self.stride

        scale_factor_x = input_image.size(3) / W
        scale_factor_y = input_image.size(2) / H

        x1_scaled = x1.float() * scale_factor_x 
        y1_scaled = y1.float() * scale_factor_y 

        x1_scaled = torch.clamp(x1_scaled, min=0, max=input_image.size(3) - self.receptive_field)
        y1_scaled = torch.clamp(y1_scaled, min=0, max=input_image.size(2) - self.receptive_field)

        return torch.stack((x1_scaled, y1_scaled), dim=1)  
    
    def _mask_operation(self, real_AB, fake_AB, ax):
        batch_size = real_AB.size(0)
        mask = torch.zeros_like(fake_AB)

        x = torch.clamp(ax[:, 0].round().long(), min=0)
        y = torch.clamp(ax[:, 1].round().long(), min=0)

        x_end = torch.clamp(x + int(self.receptive_field), max=fake_AB.size(3))
        y_end = torch.clamp(y + int(self.receptive_field), max=fake_AB.size(2))

        for i in range(batch_size):
            mask[i, :, y[i]:y_end[i], x[i]:x_end[i]] = 1.0 

        fake_ABm = fake_AB * mask + real_AB * (1 - mask) 
        return fake_ABm

    def forward(self, img, target, fake_target, score_map):
        real_AB = torch.cat((img, target), dim=1)      
        fake_AB = torch.cat((img, fake_target), dim=1)  

        ax = self._localize(score_map, real_AB)

        batch_size = real_AB.size(0)
        device = real_AB.device
        box_indices = torch.arange(batch_size, device=device).float()
        x1 = ax[:, 0]
        y1 = ax[:, 1]
        x2 = x1 + self.region_width
        y2 = y1 + self.region_height
        boxes = torch.stack([box_indices, x1, y1, x2, y2], dim=1) 

        fake_Br = self.roialign(fake_target, boxes)
        real_Br = self.roialign(target, boxes)

        if not self.mask_opt:
            fake_ABr = self.roialign(fake_AB, boxes)
            real_ABr = self.roialign(real_AB, boxes)
            return fake_ABr, real_ABr, real_Br, fake_Br
        else:
            fake_ABm = self._mask_operation(real_AB, fake_AB, ax)
            return fake_ABm, real_Br, fake_Br
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    x = torch.randn((3, 3, 256, 256)) 
    y = torch.randn((3, 1, 256, 256))
    y_fake = torch.zeros_like(y)
    score_map = torch.ones((3, 1, 30, 30))

    proposal = DRPnet()

    score_map[0,:,0:1, 0:1] = 0      # Fakest top left corner
    score_map[1,:,28:30, 27:30] = 0  # Fakest bottom right corner
    score_map[2,:,12:15, 17:18] = 0  # Fakest middle

    masked_img, real_Br, fake_Br = proposal(x, y, y_fake, score_map)

    for masked, scores in zip(masked_img, score_map):
        plt.imshow(np.squeeze(scores))
        plt.show()
        plt.imshow(masked[3,:,:])
        plt.show()

    print(masked_img.shape)  # Should print: torch.Size([3, 4, 256, 256])
    print(real_Br.shape)     # Should print: torch.Size([3, 1, 20, 20])
    print(fake_Br.shape)     # Should print: torch.Size([3, 1, 20, 20])