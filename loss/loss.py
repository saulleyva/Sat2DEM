import torch
from torch import nn

class GradientLoss(nn.Module):
    """
    Computes the first-order gradient loss between the predicted and real DEM images.
    """
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted DEM images of shape [B, C, H, W]
            target (torch.Tensor): Real DEM images of shape [B, C, H, W]
        
        Returns:
            torch.Tensor: Scalar tensor representing the gradient loss
        """
        # Compute gradients in the x direction
        grad_pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_loss_x = torch.abs(grad_pred_x - grad_target_x).mean()

        # Compute gradients in the y direction
        grad_pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        grad_target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad_loss_y = torch.abs(grad_pred_y - grad_target_y).mean()

        # Total gradient loss
        return grad_loss_x + grad_loss_y