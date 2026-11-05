"""
Loss Functions for Forest Segmentation
Combined Dice Loss and Binary Cross-Entropy (BCE) Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred_logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Computes soft Dice loss for binary segmentation.

    Args:
        pred_logits: Raw model predictions before sigmoid (B, 1, H, W).
        targets: Ground truth binary masks (B, 1, H, W).
        smooth: Smoothing epsilon to prevent division by zero.
    """
    probs = torch.sigmoid(pred_logits)

    probs = probs.view(-1)
    targets = targets.view(-1)

    intersection = (probs * targets).sum()
    dice_coeff = (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return 1.0 - dice_coeff


class DiceBCELoss(nn.Module):
    """Combines Binary Cross-Entropy with Logits and Soft Dice Loss.

    Loss = bce_weight * BCE + dice_weight * Dice
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_fn(pred_logits, targets)
        dice = dice_loss(pred_logits, targets, smooth=self.smooth)
        return self.bce_weight * bce + self.dice_weight * dice
