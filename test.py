import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from torchmetrics.image import StructuralSimilarityIndexMeasure

from dataset import SatelliteDEMDataset
from metrics import compute_metrics
from models.generator import UNetGenerator
from utils import load_all_splits, get_split, initialize_weights, save_some_examples

# ==========================
# Configuration Parameters
# ==========================
# Define your experiment configuration name
EXPERIMENT_NAME = "pix2pix_sentinel111"

# Define base directories for saving outputs
BASE_WEIGHT_DIR = "weights"
BASE_METRIC_DIR = "metrics"
BASE_EXAMPLE_DIR = "examples"

# Dataset directories
DEM_DIR = "data/normalized_dem"
SAT_DIR = "data/sentinel_images"

# Model
ATT = 'simam'         # None, 'cbam', 'gam', 'simam'

# Training hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 2
SSIM_RANGE = 1.0  # 1 or 0.6439 


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Connected to {device}")

    metric_dir = os.path.join(BASE_METRIC_DIR, EXPERIMENT_NAME)
    os.makedirs(metric_dir, exist_ok=True)

    consolidated_test_csv_path = os.path.join(BASE_METRIC_DIR, EXPERIMENT_NAME, "aggregated_test_metrics.csv")
    if not os.path.exists(consolidated_test_csv_path):
        df_header = pd.DataFrame(columns=['Split', 'MAE', 'RMSE', 'SSIM', 'R2'])
        df_header.to_csv(consolidated_test_csv_path, index=False)
        print(f"Created consolidated test metrics file at {consolidated_test_csv_path}")
    else:
        print(f"Consolidated test metrics file already exists at {consolidated_test_csv_path}. Metrics will be appended.")

    all_splits = load_all_splits()
    aggregated_test_metrics_list  = []     

    for split in range(1, 5):
        print("\n===============================")
        print(f"           Split {split}/4\n")
        print("===============================\n")

        model_dir = os.path.join(BASE_WEIGHT_DIR, EXPERIMENT_NAME, f"split_{split}")
        example_dir = os.path.join(BASE_EXAMPLE_DIR, EXPERIMENT_NAME, f"split_{split}")

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(example_dir, exist_ok=True)

        test_dataset = SatelliteDEMDataset(DEM_DIR, SAT_DIR, indices=get_split(all_splits, split, 'test'), training=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        # Generator
        generator = UNetGenerator(attention_type=ATT).to(device)
        initialize_weights(generator)
        
        # ==========================
        #       Testing Phase
        # ==========================
        print(f"Starting Testing for Split {split}...\n")

        test_metrics_dict = {
            'MAE': MeanAbsoluteError().to(device),
            'RMSE': MeanSquaredError(squared=False).to(device),
            'SSIM': StructuralSimilarityIndexMeasure(data_range=SSIM_RANGE).to(device),
            'R2': R2Score().to(device)
        }
        for metric in test_metrics_dict.values():
            metric.reset()

        best_gen_checkpoint_path = os.path.join(model_dir, "best_generator.pth")
        checkpoint_G = torch.load(best_gen_checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint_G['state_dict'])
        generator.eval()

        with torch.no_grad():
            for dem, sat in test_loader:
                dem = dem.to(device)
                sat = sat.to(device)

                fake_dem = generator(sat)

                compute_metrics(fake_dem, dem, test_metrics_dict)
            
            save_some_examples(generator, test_loader, 0, folder=example_dir)
            print(f"Saved example images at test")

        avg_test_metrics = {
            'MAE': test_metrics_dict['MAE'].compute().item(),
            'RMSE': test_metrics_dict['RMSE'].compute().item(),
            'SSIM': test_metrics_dict['SSIM'].compute().item(),
            'R2': test_metrics_dict['R2'].compute().item()
        }
        aggregated_test_metrics_list.append({
            'Split': split,
            'MAE': avg_test_metrics['MAE'],
            'RMSE': avg_test_metrics['RMSE'],
            'SSIM': avg_test_metrics['SSIM'],
            'R2': avg_test_metrics['R2']
        })
        df_split_test = pd.DataFrame([{
            'Split': split,
            'MAE': avg_test_metrics['MAE'],
            'RMSE': avg_test_metrics['RMSE'],
            'SSIM': avg_test_metrics['SSIM'],
            'R2': avg_test_metrics['R2']
        }])
        df_split_test.to_csv(consolidated_test_csv_path, mode='a', header=False, index=False)
        print(f"Appended test metrics for Split {split} to {consolidated_test_csv_path}")

        print(f"Test Results for Split {split}:")
        print(f"MAE: {avg_test_metrics['MAE']:.4f} | RMSE: {avg_test_metrics['RMSE']:.4f} | "
              f"SSIM: {avg_test_metrics['SSIM']:.4f} | R2: {avg_test_metrics['R2']:.4f}\n")

    # ==========================
    #  Aggregating Test Results
    # ==========================
    if aggregated_test_metrics_list:
        df_all_test_metrics = pd.DataFrame(aggregated_test_metrics_list)
        
        metrics_to_aggregate = ['MAE', 'RMSE', 'SSIM', 'R2']
        mean_metrics = df_all_test_metrics[metrics_to_aggregate].mean()
        std_metrics = df_all_test_metrics[metrics_to_aggregate].std()

        summary_df = pd.DataFrame({
            'Split': ['Mean', 'Std'],
            'MAE': [mean_metrics['MAE'], std_metrics['MAE']],
            'RMSE': [mean_metrics['RMSE'], std_metrics['RMSE']],
            'SSIM': [mean_metrics['SSIM'], std_metrics['SSIM']],
            'R2': [mean_metrics['R2'], std_metrics['R2']]
        })

        summary_df.to_csv(consolidated_test_csv_path, mode='a', header=False, index=False)
        print(f"Appended mean and std metrics to {consolidated_test_csv_path}")

        print("Aggregated Test Metrics Across All Splits:")
        print(df_all_test_metrics)
        print("\nSummary Metrics:")
        print(summary_df)
    else:
        print("No test metrics were collected. Please ensure that the best model checkpoints exist for each split.")
