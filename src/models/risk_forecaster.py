"""
Deforestation Risk Forecaster Module
Predicts future deforestation vulnerability (T2 -> T3) by learning spatial hazard
relationships between landscape covariates (distance to frontiers, roads, fragmentation,
and Prithvi Foundation Model embeddings).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score


class DeforestationHazardNet(nn.Module):
    """Convolutional Hazard Network for Spatial Risk Modeling.

    Maps multi-channel spatial frontier covariates to a continuous hazard probability map.
    """

    def __init__(self, in_channels: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeforestationRiskForecaster:
    """Predictive Risk Forecaster estimating prospective deforestation probability for remaining forest stands."""

    def __init__(self, in_channels: int = 5, learning_rate: float = 1e-3):
        self.in_channels = in_channels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DeforestationHazardNet(in_channels=in_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.BCEWithLogitsLoss()

    def fit_from_temporal_transition(
        self,
        features_t1: np.ndarray,
        actual_cleared_t2: np.ndarray,
        forest_mask_t1: np.ndarray,
        epochs: int = 20,
    ) -> float:
        """Trains the hazard network using observed historical transitions (T1 features -> T2 clearings).

        Only evaluates loss on forest pixels at T1 (since non-forest cannot be deforested).
        """
        self.model.train()
        x_tensor = torch.from_numpy(features_t1).float().unsqueeze(0).to(self.device)
        y_tensor = torch.from_numpy(actual_cleared_t2.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(forest_mask_t1.astype(bool)).unsqueeze(0).unsqueeze(0).to(self.device)

        final_loss = 0.0
        for _ in range(epochs):
            self.optimizer.zero_grad()
            logits = self.model(x_tensor)

            # Masked loss strictly on forest pixels
            masked_logits = logits[mask_tensor]
            masked_targets = y_tensor[mask_tensor]

            if len(masked_targets) == 0:
                continue

            # Handle class imbalance: newly cleared pixels are rarer than background forest
            pos_weight = (len(masked_targets) - masked_targets.sum()) / (masked_targets.sum() + 1e-6)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fn(masked_logits, masked_targets)

            loss.backward()
            self.optimizer.step()
            final_loss = float(loss.item())

        print(f"[Risk Forecaster] Hazard Net trained. Final Loss: {final_loss:.4f}")
        return final_loss

    def predict_risk_map(
        self,
        features_t2: np.ndarray,
        forest_mask_t2: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generates continuous probability risk map for all remaining forest at T2.

        Returns:
            Tuple of (risk_map_2d, risk_statistics_dict).
        """
        self.model.eval()
        x_tensor = torch.from_numpy(features_t2).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Non-forest pixels have 0 future deforestation risk
        risk_map = np.where(forest_mask_t2.astype(bool), probs, 0.0).astype(np.float32)

        # Quantitative tier breakdown
        forest_pixels = np.sum(forest_mask_t2)
        ha_per_px = 0.01  # 10m x 10m = 0.01 ha

        imminent_px = np.sum((risk_map >= 0.80) & forest_mask_t2)
        high_px = np.sum((risk_map >= 0.50) & (risk_map < 0.80) & forest_mask_t2)
        mod_px = np.sum((risk_map >= 0.20) & (risk_map < 0.50) & forest_mask_t2)
        low_px = np.sum((risk_map < 0.20) & forest_mask_t2)

        stats = {
            "total_forest_ha": round(float(forest_pixels * ha_per_px), 2),
            "imminent_risk_ha": round(float(imminent_px * ha_per_px), 2),
            "high_risk_ha": round(float(high_px * ha_per_px), 2),
            "moderate_risk_ha": round(float(mod_px * ha_per_px), 2),
            "low_risk_ha": round(float(low_px * ha_per_px), 2),
            "mean_risk_score": round(float(np.mean(risk_map[forest_mask_t2.astype(bool)])), 4),
        }

        return risk_map, stats
