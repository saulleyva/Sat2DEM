# Sat2DEM: Digital Elevation Model Generation from Satellite Imagery

Sat2DEM is a deep learning framework for generating Digital Elevation Models (DEMs) from satellite imagery. This project integrates advanced attention mechanisms and novel normalization strategies to improve DEM generation performance. **The code in this repository** was initially developed as part of a master’s thesis (TFM), but it is designed to be easily extended or adapted for research and application in geospatial analysis.

---

## Features
- **State-of-the-art architectures**: Implements U-Net, Pix2Pix, and DRPAN for DEM generation.  
- **Advanced attention mechanisms**: Incorporates CBAM, GAM, and SimAM to enhance spatial and channel attention.  
- **Normalization strategies**: Explores Global Normalization and Global Normalization with Shift for focused terrain modeling.  
- **Publicly available dataset**: Pairs Sentinel-2 and Landsat 9 imagery with Copernicus DEM GLO-30 for the Iberian Peninsula.  
- **Scalability**: Offers a cost-effective and efficient alternative to traditional DEM generation methods, such as LiDAR and radar.


---

## Dataset
### Description
The dataset includes:
- **Satellite imagery**: Harmonized Sentinel-2 (10m resolution) and USGS Landsat 9 (30m resolution).  
- **DEM data**: Copernicus DEM GLO-30 with elevation values ranging from -168.62 m (Las Cruces mine, Seville) to 3472.30 m (Sierra Nevada, Granada).  
- **Image tiles**: 256×256 pixels, each covering ~7.68×7.68 km.  
- **Train/val/test splits**: Split in a 70/15/15 ratio. Configurations are provided in `data/dataset_splits.json`.

**Download**: The full dataset is publicly available on Zenodo:  
[https://doi.org/10.5281/zenodo.14647632](https://doi.org/10.5281/zenodo.14647632)

### Preprocessing

- **DEM normalization**  
  - *Global Normalization*: Maps elevation to `[-1, 1]` based on the dataset-wide min/max values.  
  - *Global Normalization with Shift*: Focuses on local terrain by setting each tile’s minimum to -1.  
- **Cloud artifact removal**: Uses a median operation to reduce interference caused by clouds.  

---

## Installation
### Requirements
- Python 3.11  
- PyTorch 2.4.0  
- Additional Python packages listed in `requirements.txt`.

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/saulleyva/Sat2DEM.git
   cd Sat2DEM
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Repository Structure
```plaintext
Sat2DEM/
├── data/
│   └── dataset_splits.json   # Train/val/test configurations
├── loss/
│   └── loss.py               # Loss function
├── models/
│   ├── discriminator.py      # PatchGAN discriminator
│   ├── drpnet.py             # Discriminative Region Proposal Network
│   ├── generator.py          # UNet generator
│   ├── reviser.py            # DRPAN reviser module
│   └── attention/            # Attention mechanisms (CBAM, GAM, SimAM)
├── scripts/
│   ├── create_splits.py      # Data split creation
│   ├── normalize_dem.py      # DEM normalization
│   ├── generate_figures.py   # Learning curve visualization
│   └── replace_null_val.py   # Handling missing DEM values
├── main_pix2pix.py           # Pix2Pix training and testing script
├── main_unet.py              # UNet training and testing script
├── main_drpan.py             # DRPAN training and testing script
├── test.py                   # Model evaluation
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Usage
### Training and testing
To train the Pix2Pix model:
```bash
python main_pix2pix.py
```

### Testing a given configuration
Evaluate a trained model:
```bash
python test.py
```

---

## Results
- **Best configuration**: Pix2Pix with GAM and Global Normalization with Shift achieved a Mean Absolute Error (MAE) of 65m and a Structural Similarity Index (SSIM) of 0.85.
- Detailed metrics are saved in the `metrics/` directory.
