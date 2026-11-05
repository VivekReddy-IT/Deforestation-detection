"""
NDVI Baseline Classifier
Computes Normalized Difference Vegetation Index (NDVI) from Sentinel-2
Red (B4) and Near-Infrared (B8) bands and evaluates thresholded binary forest masks.
"""

from typing import Dict, List, Tuple
import numpy as np


def compute_ndvi(red_b4: np.ndarray, nir_b8: np.ndarray) -> np.ndarray:
    """Computes Normalized Difference Vegetation Index:

    NDVI = (NIR - Red) / (NIR + Red + 1e-6)
    Values range from -1.0 to +1.0. Dense green forest typically has NDVI > 0.6.
    """
    numerator = nir_b8 - red_b4
    denominator = nir_b8 + red_b4 + 1e-6
    return numerator / denominator


class NDVIBaseline:
    """Rule-based domain baseline classifier using NDVI thresholding."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predicts binary forest mask from 6-channel Sentinel-2 imagery.

        Channels:
            [0: B2, 1: B3, 2: B4 (Red), 3: B8 (NIR), 4: B11, 5: B12]

        Args:
            image: Array of shape (6, H, W) or (B, 6, H, W).

        Returns:
            Binary boolean mask of predicted forest.
        """
        if image.ndim == 3:
            red = image[2]
            nir = image[3]
        elif image.ndim == 4:
            red = image[:, 2]
            nir = image[:, 3]
        else:
            raise ValueError(f"Expected 3D or 4D array, got ndim={image.ndim}")

        ndvi = compute_ndvi(red, nir)
        return ndvi > self.threshold

    def fit_threshold(
        self,
        train_tiles: List[Dict],
        thresholds: np.ndarray = np.linspace(0.3, 0.8, 26),
    ) -> float:
        """Finds the optimal NDVI threshold maximizing IoU on the training split."""
        best_threshold = self.threshold
        best_iou = -1.0

        for thresh in thresholds:
            total_intersection = 0
            total_union = 0

            for tile in train_tiles:
                data = np.load(tile["file_path"])
                img = data["image"]
                mask = data["mask"] > 0

                pred = compute_ndvi(img[2], img[3]) > thresh
                intersection = np.logical_and(pred, mask).sum()
                union = np.logical_or(pred, mask).sum()

                total_intersection += intersection
                total_union += union

            iou = total_intersection / (total_union + 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_threshold = float(thresh)

        print(f"[NDVI Baseline] Best training threshold: {best_threshold:.2f} (Train IoU: {best_iou:.4f})")
        self.threshold = best_threshold
        return best_threshold

    def evaluate(self, tiles: List[Dict]) -> Dict[str, float]:
        """Evaluates NDVI baseline against ground truth on given tiles."""
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        for tile in tiles:
            data = np.load(tile["file_path"])
            img = data["image"]
            mask = data["mask"] > 0

            pred = self.predict(img)

            total_tp += np.logical_and(pred, mask).sum()
            total_fp += np.logical_and(pred, ~mask).sum()
            total_fn += np.logical_and(~pred, mask).sum()
            total_tn += np.logical_and(~pred, ~mask).sum()

        iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return {
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "threshold": self.threshold,
        }
