# Sat2DEM: Digital Elevation Model Generation from Satellite Imagery

Sat2DEM is a deep learning framework for generating high-resolution Digital Elevation Models (DEMs) from satellite imagery. This repository supports research in geospatial analysis by integrating advanced attention mechanisms and novel normalization strategies. The project is based on the findings of the study "Digital Elevation Model Generation from Satellite Imagery Using Deep Learning," which focuses on leveraging Sentinel-2 and Landsat 9 imagery paired with DEMs.

---

## Features
- **State-of-the-art architectures**: Includes implementations of U-Net, Pix2Pix, and DRPAN for DEM generation.
- **Advanced attention mechanisms**: Incorporates CBAM, GAM, and SimAM to enhance spatial and channel attention.
- **Normalization strategies**: Introduces Global Normalization and Global Normalization with Shift for effective terrain modeling.
- **Custom dataset**: Utilizes a curated dataset featuring Sentinel-2 and Landsat 9 imagery paired with high-resolution DEMs from the Iberian Peninsula.
- **Scalability**: Offers a cost-effective and efficient alternative to traditional DEM generation methods such as LiDAR and radar.

---

## Dataset
### Description
The dataset includes:
- **Satellite imagery**: Harmonized Sentinel-2 (10m resolution) and USGS Landsat 9 (30m resolution).
- **DEM data**: Copernicus DEM GLO-30 with elevation values ranging from -168.62m (Las Cruces mine, Seville) to 3472.30m (Sierra Nevada, Granada).
- **Image tiles**: 256x256 pixels corresponding to 7.68x7.68 km physical areas.
- **Train/val/test splits**: 70/15/15 ratio, with configurations provided in `data/dataset_splits.json`.

### Preprocessing
- **DEM normalization**:
  - *Global Normalization*: Maps elevation to [-1, 1] based on dataset-wide min/max values.
  - *Global Normalization with Shift*: Focuses on local terrain by setting each tile’s minimum to -1.
- **Cloud artifact removal**: Median operation applied to reduce cloud interference.

---

## Installation
### Requirements
- Python 3.11
- PyTorch 2.4.0
- Required packages listed in `requirements.txt`.

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
│   └── dataset_splits.json    # Train/val/test configurations
├── loss/
│   └── loss.py               # Loss functions
├── models/
│   ├── discriminator.py      # PatchGAN discriminator
│   ├── generator.py          # UNet generator
│   ├── reviser.py            # DRPAN reviser module
│   └── attention/            # Attention mechanisms (CBAM, GAM, SimAM)
├── scripts/
│   ├── create_splits.py      # Data split creation
│   ├── normalize_dem.py      # DEM normalization
│   ├── generate_figures.py   # Visualization tools
│   └── replace_null_val.py   # Handling missing DEM values
├── main_pix2pix.py           # Pix2Pix training script
├── main_unet.py              # UNet training script
├── main_drpan.py             # DRPAN training script
├── test.py                   # Model evaluation
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Usage
### Training
To train the Pix2Pix model:
```bash
python main_pix2pix.py
```

### Testing
Evaluate a trained model:
```bash
python test.py
```

### Visualization
Generate sample figures:
```bash
python scripts/generate_figures.py
```

---

## Results
- **Best configuration**: Pix2Pix with GAM and Global Normalization with Shift achieved a Mean Absolute Error (MAE) of 65m and a Structural Similarity Index (SSIM) of 0.85.
- Detailed metrics are saved in the `metrics/` directory.

---

## Citation
If you use this repository, please cite the associated paper:
```bibtex
@article{Sat2DEM,
  title={Digital Elevation Model Generation from Satellite Imagery Using Deep Learning},
  author={Saúl Leyva, Marcos Ortega, Joaquim de Moura},
  journal={University of A Coruña, Spain},
  year={2025}
}
```

---

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

---

## Acknowledgments
- **Dataset creation**: Leveraged Google Earth Engine and Copernicus DEM resources.
- **Hardware**: Experiments conducted on NVIDIA RTX 4070 Ti and RTX 4080 GPUs.
