import os
import numpy as np
import rasterio
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset
from utils import extract_numeric_index

class SatelliteDEMDataset(Dataset):
    """
    PyTorch Dataset for loading paired DEM and satellite RGB images.

    - **DEMs**: Assumed to be pre-scaled within [-1, 1].
    - **Satellite Images**: Assumed to be in [0, 255] range.
    
    Applies different transformations based on training or testing mode.
    """
    def __init__(self, dem_dir, sat_dir, indices=None, training=True):
        """
        Args:
            dem_dir (str): Path to the DEM images.
            sat_dir (str): Path to the satellite images.
            indices (list of int, optional): List of image indices to include. If None, include all images.
            training (bool, optional): Flag to indicate training mode. If True, applies training transformations.
        """
        self.dem_dir = dem_dir
        self.sat_dir = sat_dir

        self.dem_files = sorted(os.listdir(dem_dir), key=lambda x: extract_numeric_index(x))    
        self.sat_files = sorted(os.listdir(sat_dir), key=lambda x: extract_numeric_index(x))
        if indices is not None:
            self.dem_files = [self.dem_files[i] for i in indices]
            self.sat_files = [self.sat_files[i] for i in indices]

        if training:
            self.common_trans = transforms.Compose([  
                transforms.ToImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        else:
            self.common_trans = transforms.Compose([   
                transforms.ToImage(),
            ])
        self.sat_trans = transforms.Compose([   
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dem_trans = transforms.ToDtype(torch.float32, scale=False)

    
    def __len__(self):
        return len(self.dem_files)
    
    def __getitem__(self, index):
        dem_path = os.path.join(self.dem_dir, self.dem_files[index])
        sat_path = os.path.join(self.sat_dir, self.sat_files[index])
        
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
        
        with rasterio.open(sat_path) as src:
            sat_red = src.read(1)
            sat_green = src.read(2)
            sat_blue = src.read(3)
            sat_rgb = np.dstack((sat_red, sat_green, sat_blue))
        
        dem, sat_rgb = self.common_trans(dem, sat_rgb)
        sat_rgb = self.sat_trans(sat_rgb)
        dem = self.dem_trans(dem)
        
        return dem, sat_rgb


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from utils import load_all_splits, get_split

    dem_folder = 'data/normalized_dem'
    sentinel_folder = 'data/sentinel_images'
    landsat_folder = 'data/landsat_images'

    all_splits = load_all_splits()

    dataset_sentinel = SatelliteDEMDataset(dem_folder, sentinel_folder, indices=get_split(all_splits, 1, 'train'), training=True)
    print(f"Number of samples in Sentinel dataset: {len(dataset_sentinel)}")

    dataset_landsat = SatelliteDEMDataset(dem_folder, landsat_folder, indices=get_split(all_splits, 1, 'train'), training=True)
    print(f"Number of samples in Landsat dataset: {len(dataset_landsat)}")

    def plt_images(dataset, title_prefix, num_examples=4):
        for i in range(num_examples):
            dem_image, sat_image = dataset[i]
            
            sat_image_np = sat_image.numpy().transpose(1, 2, 0)
            dem_image_np = dem_image.numpy().squeeze()

            fig = plt.figure(figsize=(8, 4))
            gs = GridSpec(1, 3, width_ratios=[47.5, 47.5, 5]) 

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(sat_image_np*0.5+0.5)
            ax1.set_title(f'{title_prefix} Satellite Image {i+1}')
            ax1.axis('off')

            ax2 = fig.add_subplot(gs[0, 1])
            im = ax2.imshow(dem_image_np, cmap='terrain')
            ax2.set_title(f'{title_prefix} DEM {i+1}')
            ax2.axis('off')

            cbar_ax = fig.add_subplot(gs[0, 2])
            fig.colorbar(im, cax=cbar_ax, orientation='vertical')

            plt.tight_layout()
            plt.show()

    plt_images(dataset_sentinel, 'Sentinel')
    plt_images(dataset_landsat, 'Landsat')