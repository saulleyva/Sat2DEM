create_splits.py -> This script creates a json file that contains n number of different splits of the dataset images index splited with a 70/15/15 rate.

dataset_min_max.py -> This script analyzes all Digital Elevation Model (DEM) GeoTIFF files in a specified directory to determine the overall minimum and maximum elevation values.

generate_figures.py -> This script generates figures with the training and eval learning curves.

normalize_dem_0_1.py -> This script normalizes Digital Elevation Model (DEM) GeoTIFF files within a specified input directory to a [-1, 1] range based on provided minimum and maximum elevation values.

replace_null_val.py -> This script cleans a specific Digital Elevation Model (DEM) GeoTIFF file by replacing all NaN (null) values with zero. DEMs with index 5209, 8734, 8839 and 8941 where processed.