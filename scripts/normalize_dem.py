import os
import rasterio
import numpy as np

def normalize_dem(dem_array, min_val, max_val, with_shift):
    if max_val == min_val:
        raise ValueError("max_val and min_val cannot be the same for normalization.")

    normalized = (dem_array - min_val) / (max_val - min_val)
    normalized = (normalized * 2) - 1

    if with_shift:
        dem_min = np.min(normalized)
        normalized = normalized - (dem_min + 1)

    return normalized

def process_dems(input_dir, output_dir, min_val, max_val, with_shift):
    # List all .tif files in the input directory
    dem_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.tif')]

    if not dem_files:
        print(f"No DEM (.tif) files found in '{input_dir}'.")
        return

    print(f"Found {len(dem_files)} DEM file(s) in '{input_dir}'. Starting normalization...\n")

    for dem_file in dem_files:
        input_path = os.path.join(input_dir, dem_file)

        if dem_file.startswith('original_DEM_'):
            output_filename = dem_file.replace('original_DEM_', 'normalized_DEM_')
        else:
            exit(f"Error: Unexpected DEM filename format: {dem_file}")
        output_path = os.path.join(output_dir, output_filename)

        try:
            with rasterio.open(input_path) as src:
                dem_data = src.read(1)  
                meta = src.meta.copy() 

                dem_data = np.nan_to_num(dem_data, nan=0)

                normalized_dem = normalize_dem(dem_data, min_val, max_val, with_shift)

                meta.update(dtype=rasterio.float32)

                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(normalized_dem.astype(rasterio.float32), 1)

                print(f"Normalized DEM saved: {output_path}")

        except Exception as e:
            print(f"Error processing '{dem_file}': {e}")

    print("\nNormalization complete.")


if __name__ == "__main__":
    input_folder = '../data/original_dem'     # Folder containing cleaned DEMs
    output_folder = '../data/normalized_dem'  # Folder to save normalized DEMs
    with_shift = False  # If True, Global Normalization with Shift

    # Define overall min and max values provided
    overall_min = -168.6189727783203
    overall_max = 3472.300537109375

    process_dems(input_folder, output_folder, overall_min, overall_max, with_shift)
