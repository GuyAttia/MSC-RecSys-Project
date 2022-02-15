import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ignite.metrics import Metric
from torch import nn


# --------------------------------------------------- Losses ---------------------------------------------------------------------
class RMSELoss(nn.Module):
    """
    Implement RMSE loss using PyTorch MSELoss existing class.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y, **kwargs):
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
        self.eps = eps  # Add eps to avoid devision by 0

    def forward(self, yhat, y, **kwargs):
        # Create mask for all non zero items in the tensor
        non_zero_mask = torch.nonzero(y, as_tuple=True)
        y_non_zeros = y[non_zero_mask]  # Keep only non zero in y
        yhat_non_zeros = yhat[non_zero_mask]    # Keep only non zero in y_hat

        loss = torch.sqrt(self.mse(yhat_non_zeros, y_non_zeros) + self.eps)
        return loss


class VAE_Loss(nn.Module):
    """
    Build an nn.Module type class for the VAE loss which is build from both, KLD and BCE
    """

    def __init__(self, anneal=1.0):
        super().__init__()
        self.anneal = anneal

    def forward(self, recon_x, x, mu, logvar):
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar -
                                mu.pow(2) - logvar.exp(), dim=1))
        loss = BCE + self.anneal * KLD
        return loss

# --------------------------------------------------- MRR ---------------------------------------------------------------------


def mrr(pred, actual, cutoff=5, mrr_threshold=4):
    """
    Function to calculate the MRR score
    cutoff : mrr cutoff
    mrr_threshold :  relevancy rating threshold
    """
    mrr_sum, zero_cnt, i = 0, 0, 0
    for row in pred:
        cnt, j = 1, len(row) - 1
        index = np.argsort(row)
        while j >= 0 and actual[i][index[j]] < mrr_threshold and cnt <= cutoff:
            if actual[i][index[j]] != 0:
                cnt += 1
            j -= 1
        if j != -1:
            mrr_sum += 1 / cnt
        if np.all(actual[i] == 0):
            zero_cnt += 1
        i += 1
    mrr = mrr_sum / (len(pred) - zero_cnt)
    return mrr


class MetricMRR_Vec(Metric):
    """
    Create custom metric class for calculating the MRR over the entire batch output for Vectorized models (AutoRec, VAE)
    """

    def __init__(self, cutoff=5, mrr_threshold=4, output_transform=lambda x: x, device='cpu'):
        self.cutoff = cutoff
        self.mrr_threshold = mrr_threshold
        self.df_test = None
        self.df_preds = None
        super().__init__(output_transform, device)

    def reset(self):
        self.df_test = torch.Tensor()
        self.df_preds = torch.Tensor()
        super().reset()

    def update(self, output):
        batch_y_preds, batch_y_test = output[0].detach(), output[1].detach()
        self.df_test = torch.cat((self.df_test, batch_y_test), 0)
        self.df_preds = torch.cat((self.df_preds, batch_y_preds), 0)

    def compute(self):
        mrr_ = mrr(pred=self.df_preds.numpy(), actual=self.df_test.numpy(
        ), cutoff=self.cutoff, mrr_threshold=self.mrr_threshold)

        return mrr_


class MetricMRR_MF(Metric):
    """
    Create custom metric class for calculating the MRR over the entire batch output for the MF model
    """

    def __init__(self, cutoff=5, mrr_threshold=4, output_transform=lambda x: x, device='cpu'):
        self.cutoff = cutoff
        self.mrr_threshold = mrr_threshold
        self.test_users = None
        self.test_items = None
        self.y_test = None
        self.y_preds = None
        super().__init__(output_transform, device)

    def reset(self):
        self.test_users = torch.Tensor(device=self._device)
        self.test_items = torch.Tensor(device=self._device)
        self.y_test = torch.Tensor(device=self._device)
        self.y_preds = torch.Tensor(device=self._device)
        super().reset()

    def update(self, output):
        batch_y_preds, batch_y_targets_stack = output[0].detach(
        ), output[1].detach()

        batch_test_users = batch_y_targets_stack[0]
        batch_test_items = batch_y_targets_stack[1]
        batch_y_test = batch_y_targets_stack[2]

        self.test_users = torch.cat((self.test_users, batch_test_users))
        self.test_items = torch.cat((self.test_items, batch_test_items))
        self.y_test = torch.cat((self.y_test, batch_y_test))
        self.y_preds = torch.cat((self.y_preds, batch_y_preds))

    def compute(self):
        df_true = pd.DataFrame(
            {'user_id': self.test_users, 'item_id': self.test_items, 'rating': self.y_test})
        ratings_true = df_true.pivot(
            index='user_id', columns='item_id', values='rating').fillna(0)

        df_preds = pd.DataFrame(
            {'user_id': self.test_users, 'item_id': self.test_items, 'rating': self.y_preds})
        ratings_preds = df_preds.pivot(
            index='user_id', columns='item_id', values='rating').fillna(0)

        mrr_ = mrr(pred=ratings_preds.values, actual=ratings_true.values,
                   cutoff=self.cutoff, mrr_threshold=self.mrr_threshold)

        return mrr_
