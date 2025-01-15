import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from torchmetrics.image import StructuralSimilarityIndexMeasure

from dataset import SatelliteDEMDataset
from metrics import compute_metrics
from models.generator import UNetGenerator
from loss.loss import GradientLoss
from utils import load_all_splits, get_split, initialize_weights, save_checkpoint, save_some_examples

# ==========================
# Configuration Parameters
# ==========================

# Define your experiment configuration name
EXPERIMENT_NAME = "unet_sentinel"

# Define base directories for saving outputs
BASE_MODEL_DIR = "weights"
BASE_METRIC_DIR = "metrics"
BASE_EXAMPLE_DIR = "examples"

# Dataset directories
DEM_DIR = "data/normalized_dem"
SAT_DIR = "data/sentinel_images"

# Attention
ATT = None      # None, 'cbam', 'gam', 'simam'

# Training parameters
NUM_EPOCHS = 500
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = 2e-4
BETA1 = 0.5
BETA2 = 0.999
FACTOR = 0.5
PATIENCE = 20
LAMBDA_L1 = 100
MU = 20.0 
USE_GRADIENT_LOSS = False
SSIM_RANGE = 1.0  # 1 with Global Norm or 0.6439 Global Norm with Shift

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Connected to {device}")

    criterion_L1 = nn.L1Loss()
    if USE_GRADIENT_LOSS:
        criterion_gradient = GradientLoss().to(device)

    metric_dir = os.path.join(BASE_METRIC_DIR, EXPERIMENT_NAME)
    os.makedirs(metric_dir, exist_ok=True)

    all_splits = load_all_splits()
    aggregated_test_metrics_list  = [] 

    consolidated_test_csv_path = os.path.join(BASE_METRIC_DIR, EXPERIMENT_NAME, "aggregated_test_metrics.csv")
    if not os.path.exists(consolidated_test_csv_path):
        df_header = pd.DataFrame(columns=['Split', 'MAE', 'RMSE', 'SSIM', 'R2'])
        df_header.to_csv(consolidated_test_csv_path, index=False)
        print(f"Created consolidated test metrics file at {consolidated_test_csv_path}")
    else:
        print(f"Consolidated test metrics file already exists at {consolidated_test_csv_path}. Metrics will be appended.")

    for split in range(1, 5):
        print("\n===============================")
        print(f"           Split {split}/4\n")
        print("===============================\n")

        model_dir = os.path.join(BASE_MODEL_DIR, EXPERIMENT_NAME, f"split_{split}")
        example_dir = os.path.join(BASE_EXAMPLE_DIR, EXPERIMENT_NAME, f"split_{split}")

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(example_dir, exist_ok=True)

        train_dataset = SatelliteDEMDataset(DEM_DIR, SAT_DIR, indices=get_split(all_splits, split, 'train'), training=True)
        val_dataset = SatelliteDEMDataset(DEM_DIR, SAT_DIR, indices=get_split(all_splits, split, 'validation'), training=False)
        test_dataset = SatelliteDEMDataset(DEM_DIR, SAT_DIR, indices=get_split(all_splits, split, 'test'), training=False)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        generator = UNetGenerator(attention_type=ATT).to(device)
        initialize_weights(generator)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
        scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=FACTOR, patience=PATIENCE)

        metrics = {
            'epoch': [],
            'train_MAE': [], 'train_RMSE': [], 'train_SSIM': [], 'train_R2': [], 'train_G_loss': [],
            'val_MAE': [],   'val_RMSE': [],   'val_SSIM': [],   'val_R2': [],   'val_G_loss': []
        }

        train_metrics_dict = {
            'MAE': MeanAbsoluteError().to(device),
            'RMSE': MeanSquaredError(squared=False).to(device),
            'SSIM': StructuralSimilarityIndexMeasure(data_range=SSIM_RANGE).to(device),
            'R2': R2Score().to(device)
        }

        val_metrics_dict = {
            'MAE': MeanAbsoluteError().to(device),
            'RMSE': MeanSquaredError(squared=False).to(device),
            'SSIM': StructuralSimilarityIndexMeasure(data_range=SSIM_RANGE).to(device),
            'R2': R2Score().to(device)
        }

        best_val_mae = np.inf

        for epoch in range(1, NUM_EPOCHS + 1):
            generator.train()

            train_G_losses = []

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)

            for metric in train_metrics_dict.values():
                metric.reset()

            for dem, sat in train_loader:
                dem = dem.to(device)
                sat = sat.to(device)

                optimizer_G.zero_grad()

                fake_dem = generator(sat)

                loss_G_L1 = criterion_L1(fake_dem, dem) * LAMBDA_L1
                if USE_GRADIENT_LOSS:
                    gradient_loss = criterion_gradient(fake_dem, dem) * MU
                    loss_G = loss_G_L1 + gradient_loss
                else:
                    loss_G = loss_G_L1

                loss_G.backward()
                optimizer_G.step()

                train_G_losses.append(loss_G.item())

                compute_metrics(fake_dem, dem, train_metrics_dict)

                progress_bar.update(1)
                progress_bar.set_postfix({
                    'G Loss': f"{loss_G.item():.4f}"
                })

            progress_bar.close()
            
            avg_train_metrics = {
                'MAE': train_metrics_dict['MAE'].compute().item(),
                'RMSE': train_metrics_dict['RMSE'].compute().item(),
                'SSIM': train_metrics_dict['SSIM'].compute().item(),
                'R2': train_metrics_dict['R2'].compute().item()
            }
            avg_G_loss = np.mean(train_G_losses)


            # ==========================
            #      Validation Phase
            # ==========================
            generator.eval()

            for metric in val_metrics_dict.values():
                metric.reset()

            val_G_losses = [] 

            with torch.no_grad():
                for dem, sat in val_loader:
                    dem = dem.to(device)
                    sat = sat.to(device)

                    fake_dem = generator(sat)

                    loss_G_L1_val = criterion_L1(fake_dem, dem) * LAMBDA_L1
                    if USE_GRADIENT_LOSS:
                        gradient_loss_val = criterion_gradient(fake_dem, dem) * MU
                        loss_G_val = loss_G_L1_val + gradient_loss_val
                    else:
                        loss_G_val = loss_G_L1_val
                    val_G_losses.append(loss_G_val.item())

                    compute_metrics(fake_dem, dem, val_metrics_dict)

            avg_val_metrics = {
                'MAE': val_metrics_dict['MAE'].compute().item(),
                'RMSE': val_metrics_dict['RMSE'].compute().item(),
                'SSIM': val_metrics_dict['SSIM'].compute().item(),
                'R2': val_metrics_dict['R2'].compute().item()
            }
            avg_val_G_loss = np.mean(val_G_losses)

            scheduler_G.step(avg_val_metrics['MAE'])

            metrics['epoch'].append(epoch)
            metrics['train_G_loss'].append(avg_G_loss)
            metrics['train_MAE'].append(avg_train_metrics['MAE'])
            metrics['train_RMSE'].append(avg_train_metrics['RMSE'])
            metrics['train_SSIM'].append(avg_train_metrics['SSIM'])
            metrics['train_R2'].append(avg_train_metrics['R2'])
            metrics['val_G_loss'].append(avg_val_G_loss)
            metrics['val_MAE'].append(avg_val_metrics['MAE'])
            metrics['val_RMSE'].append(avg_val_metrics['RMSE'])
            metrics['val_SSIM'].append(avg_val_metrics['SSIM'])
            metrics['val_R2'].append(avg_val_metrics['R2'])

            print(f"Epoch {epoch}/{NUM_EPOCHS} Summary:")
            print(f"Train MAE: {avg_train_metrics['MAE']:.4f} | Train RMSE: {avg_train_metrics['RMSE']:.4f} | "
                  f"Train SSIM: {avg_train_metrics['SSIM']:.4f} | Train R2: {avg_train_metrics['R2']:.4f} | "
                  f"Train G Loss: {avg_G_loss:.4f}")
            print(f"Val MAE: {avg_val_metrics['MAE']:.4f} | Val RMSE: {avg_val_metrics['RMSE']:.4f} | "
                  f"Val SSIM: {avg_val_metrics['SSIM']:.4f} | Val R2: {avg_val_metrics['R2']:.4f} | "
                  f"Val G Loss: {avg_val_G_loss:.4f}")

            df_metrics = pd.DataFrame(metrics)
            csv_path = os.path.join(metric_dir, f"metrics_split_{split}.csv")
            df_metrics.to_csv(csv_path, index=False)

            if epoch % 50 == 0:
                gen_checkpoint_path = os.path.join(model_dir, f"generator_epoch_{epoch}.pth")
                save_checkpoint(generator, optimizer_G, filename=gen_checkpoint_path)
                print(f"Saved model checkpoints at epoch {epoch}")

                save_some_examples(generator, test_loader, epoch, save_gt=(epoch==50), folder=example_dir)
                print(f"Saved example images at epoch {epoch}")

            current_val_mae = avg_val_metrics['MAE']
            if current_val_mae < best_val_mae:
                print(f"New best model found at epoch {epoch} with Val MAE: {current_val_mae:.4f}")
                best_val_mae = current_val_mae
                best_gen_checkpoint_path = os.path.join(model_dir, "best_generator.pth")
                save_checkpoint(generator, optimizer_G, filename=best_gen_checkpoint_path)
                
            print()
            

        print(f"Completed training for split {split}/4\n")

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

        best_gen_checkpoint_path = os.path.join(model_dir, "best_generator.pth")
        checkpoint_G = torch.load(best_gen_checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint_G['state_dict'])
        generator.eval()

        for metric in test_metrics_dict.values():
            metric.reset()

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

    # ======================
    # Aggregating Test Results
    # ======================

    if aggregated_test_metrics_list:
        df_all_test_metrics = pd.DataFrame(aggregated_test_metrics_list)
        
        metrics_to_aggregate = ['MAE', 'RMSE', 'SSIM', 'R2']
        mean_metrics = df_all_test_metrics[metrics_to_aggregate].mean()
        std_metrics = df_all_test_metrics[metrics_to_aggregate].std()

        # Create a summary DataFrame
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
