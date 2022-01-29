import torch
from torch import nn


class RMSELoss(nn.Module):
    """
    Implement RMSE loss using PyTorch MSELoss existing class.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class NON_ZERO_RMSELoss(nn.Module):
    """
    For the non-MF models where we rebuild the user/item vectors, we want to measure only the actual ratings and not the zero ones.
    Therefore, we implement another RMSE loss that clear the zero indices.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y):
        non_zero_mask = torch.nonzero(y, as_tuple=True)
        y_non_zeros = y[non_zero_mask]
        yhat_non_zeros = yhat[non_zero_mask]

        loss = torch.sqrt(self.mse(yhat_non_zeros, y_non_zeros) + self.eps)
        return loss