# Sat2DEM: Satellite-Based Digital Elevation Model Generation Using Attention Mechanisms and Gradient-Based Loss Optimizations

## Introduction

Sat2DEM is a deep learning framework for generating Digital Elevation Models (DEMs) from satellite imagery. The code in this repository was developed as part of the research presented in the paper: *"Satellite-Based Digital Elevation Model Generation Using Attention Mechanisms and Gradient-Based Loss Optimizations"* by S. Leyva, M. Ortega, J. de Moura. It is designed to be easily extended or adapted for research and applications in geospatial analysis.

---

## Features
- **State-of-the-art architectures**: Implements U-Net, Pix2Pix, and DRPAN for DEM generation.  
- **Advanced attention mechanisms**: Incorporates CBAM, GAM, and SimAM to enhance spatial and channel attention.  
- **Normalization strategies**: Explores Global Normalization and Global Normalization with Shift for focused terrain modeling.  
- **Publicly available dataset**: Pairs Sentinel-2 and Landsat 9 imagery with Copernicus DEM GLO-30 for the Iberian Peninsula.  
- **Scalability**: Offers a cost-effective and efficient alternative to traditional DEM generation methods, such as LiDAR and radar.

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

## Dataset
### Description
The dataset includes:
- **Satellite imagery**: Harmonized Sentinel-2 (10m resolution) and USGS Landsat 9 (30m resolution).  
- **DEM data**: Copernicus DEM GLO-30 with elevation values ranging from -168.62 m (Las Cruces mine, Seville) to 3472.30 m (Sierra Nevada, Granada).  
- **Image tiles**: 10,237 tiles of 256×256 pixels each, covering approximately 7.68×7.68 km per tile.
- **Train/val/test splits**: Split in a 70/15/15 ratio. Configurations are provided in `data/dataset_splits.json`.

**Download**: The full dataset is publicly available [here](https://doi.org/10.5281/zenodo.14647632).


### Preprocessing

- **DEM normalization**  
  - *Global Normalization*: Maps elevation to `[-1, 1]` based on the dataset-wide min/max values.  
  - *Global Normalization with Shift*: Focuses on local terrain by setting each tile’s minimum to -1.  
- **Cloud artifact removal**: Uses a median operation to reduce interference caused by clouds.  

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
└── requirements.txt          # Python dependencies
```

---

## Configuration Parameters

The scripts allow customization of parameters to adapt the model to different datasets or experimental setups. These can be found at the beginning of each main script (e.g., `main_pix2pix.py`). Below are some commonly modified parameters:

- **EXPERIMENT_NAME**: Define the name of the experiment for organizing results.
- **NUM_EPOCHS**: Number of training epochs (default: 500).
- **BATCH_SIZE**: Batch size for training (default: 32).
- **LEARNING_RATE**: Initial learning rate for optimization (default: 2e-4).
- **ATT**: Attention mechanism used (`cbam`, `gam`, `simam`, or `None`).

---

## Usage
### Training and testing
To train and test the Pix2Pix model:
```bash
python main_pix2pix.py
```

### Testing a given configuration
Evaluate a trained configuration:
```bash
python test.py
```

---

## Citation

S. Leyva, M. Ortega, J. de Moura, "Satellite-Based Digital Elevation Model Generation Using Attention Mechanisms and Gradient-Based Loss Optimizations", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2025. (pending of acceptation).
