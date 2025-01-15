import re
import os
import json
import torch
import torch.nn as nn
from torchvision.utils import save_image


def load_all_splits(file_path="data/dataset_splits.json"):
    """
    Load all dataset splits from a single JSON file.

    Args:
        file_path (str): Path to the JSON file containing all splits.

    Returns:
        dict: Dictionary containing all splits and their configurations.
    """
    with open(file_path, 'r') as f:
        splits = json.load(f)
    return splits


def get_split(splits, split_number, split_type):
    """
    Retrieve a specific split type from a specific split configuration.

    Args:
        splits (dict): The dictionary containing all splits.
        split_number (int): The split number (1 to num_splits).
        split_type (str): One of 'train', 'validation', 'test'.

    Returns:
        list: List of image indices for the specified split type.

    Raises:
        ValueError: If the split number is not found or the split type is invalid.
    """
    split_key = f"split_{split_number}"
    if split_key not in splits:
        raise ValueError(f"Split {split_number} not found in splits.")
    if split_type not in splits[split_key]:
        raise ValueError(f"Split type '{split_type}' not found in split {split_number}.")
    return splits[split_key][split_type]


def extract_numeric_index(filename):
    """
    Extracts the numeric index from a filename. It assumes that the numeric index is the part 
    of the filename that appears after the last underscore '_' and before the '.tif' extension.

    Args:
        filename (str): The filename to extract the index from.

    Returns:
        int: The extracted numeric index.

    Raises:
        ValueError: If the filename does not contain an underscore followed by digits before '.tif'.
    """
    pattern = r'_(\d+)\.tif$'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Filename '{filename}' does not match the expected pattern.")


def initialize_weights(model, init_type='normal', init_gain=0.02):
    """
    Initialize the weights of the network.

    Parameters:
        model (nn.Module): The neural network model to initialize.
        init_type (str): The type of initialization ('normal', 'xavier', 'kaiming', 'orthogonal').
        init_gain (float): Scaling factor for normal, xavier, and orthogonal.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"Initialization method '{init_type}' is not implemented.")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.affine:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)


def save_checkpoint(model, optimizer, filename):
    """
    Save the model and optimizer state dictionaries to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer associated with the model.
        filename (str): The path where the checkpoint will be saved.
    """
    print(f"=> Saving checkpoint to {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Load the model and optimizer state dictionaries from a checkpoint file.

    Args:
        checkpoint_file (str): The path to the checkpoint file.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        lr (float): The learning rate to set for the optimizer.
    """
    print(f"=> Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Update optimizer's learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_some_examples(gen, val_loader, epoch, folder, save_gt = False, device="cuda"):
    """
    Generate and save example images from the validation set.

    Args:
        gen (torch.nn.Module): The generator model.
        val_loader (torch.utils.data.DataLoader): The validation DataLoader.
        epoch (int): Current epoch number.
        folder (str): Directory where the images will be saved.
        save_gt (bool): Whether to save the ground truth images.
        device (torch.device): Device to perform computations on.
    """
    os.makedirs(folder, exist_ok=True)
    dem, sat = next(iter(val_loader))  # Corrected to dem, sat
    dem, sat = dem.to(device), sat.to(device)  # Send both to device
    gen.eval()
    with torch.no_grad():
        dem_fake = gen(sat)  # Pass sat to the generator to generate the fake DEM
        # Assuming the generator's output is normalized to [-1, 1]
        dem_fake = (dem_fake + 1) / 2  # Scale to [0, 1]
        sat_denorm = (sat + 1) / 2  # Scale input satellite to [0, 1]
        dem_denorm = (dem + 1) / 2  # Scale ground truth DEM to [0, 1]

        if epoch == 0:
            save_image(dem_fake, os.path.join(folder, f"dem_gen_test.png"))
        else:
            save_image(dem_fake, os.path.join(folder, f"dem_gen_epoch_{epoch}.png"))
        if save_gt:
            save_image(sat_denorm, os.path.join(folder, f"input_sat_epoch_{epoch}.png"))
            save_image(dem_denorm, os.path.join(folder, f"gt_dem_epoch_{epoch}.png"))


def gradient_penalty(reviser, real_data, fake_data, device):
    """
    Compute the gradient penalty for the reviser.

    Args:
        reviser (torch.nn.Module): The reviser model.
        real_data (torch.Tensor): Real data samples.
        fake_data (torch.Tensor): Fake data samples.
        device (torch.device): Device to perform computations on.
    """
    alpha = torch.rand(real_data.size(0), 1, 1, 1, device=device).expand_as(real_data)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_(True)
    
    reviser_output = reviser(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=reviser_output,
        inputs=interpolates,
        grad_outputs=torch.ones_like(reviser_output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty