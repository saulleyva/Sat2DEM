import os
import rasterio

def find_overall_min_max(dem_folder):

    overall_min = None
    overall_max = None

    min_files = []
    max_files = []

    if not os.path.isdir(dem_folder):
        print(f"Error: The folder '{dem_folder}' does not exist.")
        return

    dem_files = [f for f in os.listdir(dem_folder) if f.lower().endswith('.tif')]

    if not dem_files:
        print(f"No DEM (.tif) files found in '{dem_folder}'.")
        return

    print(f"Found {len(dem_files)} DEM files in '{dem_folder}'.\n")

    for dem_file in dem_files:
        dem_path = os.path.join(dem_folder, dem_file)
        try:
            with rasterio.open(dem_path) as src:
                dem_data = src.read(1)  

                # Compute min and max for this DEM
                dem_min = dem_data.min()
                dem_max = dem_data.max()

                print(f"{dem_file}: min={dem_min}, max={dem_max}")

                # Update overall minimum
                if (overall_min is None) or (dem_min < overall_min):
                    overall_min = dem_min
                    min_files = [dem_file]
                elif dem_min == overall_min:
                    min_files.append(dem_file)

                # Update overall maximum
                if (overall_max is None) or (dem_max > overall_max):
                    overall_max = dem_max
                    max_files = [dem_file]
                elif dem_max == overall_max:
                    max_files.append(dem_file)

        except Exception as e:
            print(f"Error processing '{dem_file}': {e}")

    print("\n--- Overall Statistics ---")
    if overall_min is not None and overall_max is not None:
        print(f"Overall Minimum Elevation: {overall_min}")
        print(f"Found in files: {min_files}\n")

        print(f"Overall Maximum Elevation: {overall_max}")
        print(f"Found in files: {max_files}")
    else:
        print("No valid DEM data found to compute statistics.")


if __name__ == "__main__":
    dem_folder = '../data/zero_normalized_dem' # Change this to the path of your DEM folder

    find_overall_min_max(dem_folder)
