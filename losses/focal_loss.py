import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = 1
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert targets to float for smooth handling
        targets = targets.float()

        # Apply sigmoid to get probabilities
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)

        # Compute probabilities and focal term
        pt = torch.exp(-BCE_loss)

        # If alpha is a tensor, it should have the same shape as targets
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(inputs.device)  # Ensure alpha is on the same device
            alpha = alpha.unsqueeze(0).expand_as(targets)  # Expand alpha to match the shape of targets
        else:
            alpha = self.alpha

        F_loss = alpha * (1 - pt) ** self.gamma * BCE_loss

        # Apply reduction method
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss