from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from torchmetrics.image import StructuralSimilarityIndexMeasure

def denormalize(tensor, min_val = -168.6189727783203, max_val = 3472.300537109375):
    return ((tensor + 1) * (max_val - min_val) / 2) + min_val

def compute_metrics(fake_dem, dem, metrics_dict):
    fake_dem = fake_dem.float()
    dem = dem.float()

    fake_dem_denorm = denormalize(fake_dem)
    dem_denorm = denormalize(dem)
    
    metrics_dict['MAE'].update(fake_dem_denorm, dem_denorm)
    metrics_dict['RMSE'].update(fake_dem_denorm, dem_denorm)
    metrics_dict['R2'].update(fake_dem_denorm.flatten(), dem_denorm.flatten())

    fake_dem_ssim = (fake_dem + 1) / 2  # Now in [0, 1]
    dem_ssim = (dem + 1) / 2  # Now in [0, 1]

    metrics_dict['SSIM'].update(fake_dem_ssim, dem_ssim)
    
    return metrics_dict