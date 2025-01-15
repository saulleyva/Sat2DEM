import os
import rasterio
import numpy as np

original_folder = '../original_dem'  # Folder containing DEM with null value
original_filename = f'original_DEM_cell_{8941}.tif'  # DEM file with null value

original_path = os.path.join(original_folder, original_filename)

with rasterio.open(original_path) as src:
    original_dem = src.read(1)  
    original_dem = np.nan_to_num(original_dem, nan=0)
    meta = src.meta.copy()

with rasterio.open(original_path, 'w', **meta) as dst:
    dst.write(original_dem.astype(rasterio.float32), 1)
