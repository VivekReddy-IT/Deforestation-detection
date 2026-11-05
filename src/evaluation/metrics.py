"""
Evaluation Metrics & Error Analysis Module
Computes IoU (Jaccard), F1 (Dice), Precision, and Recall on spatial holdouts,
formats markdown summary tables for the README, and identifies failure mode tiles.
"""

from typing import Dict, List, Tuple, Any
import numpy as np


def compute_segmentation_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes rigorous binary segmentation metrics.

    Args:
        y_pred: Predicted probabilities or binary mask.
        y_true: Ground truth binary mask (same shape).
        threshold: Decision threshold if y_pred contains continuous probabilities.

    Returns:
        Dict containing IoU, F1, Precision, Recall, Specificity, TP, FP, FN, TN.
    """
    if y_pred.dtype == bool or np.issubdtype(y_pred.dtype, np.integer):
        pred_binary = y_pred > 0
    else:
        pred_binary = y_pred >= threshold

    true_binary = y_true > 0

    tp = int(np.logical_and(pred_binary, true_binary).sum())
    fp = int(np.logical_and(pred_binary, ~true_binary).sum())
    fn = int(np.logical_and(~pred_binary, true_binary).sum())
    tn = int(np.logical_and(~pred_binary, ~true_binary).sum())

    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def find_worst_performing_tiles(
    tile_predictions: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Sorts evaluated tiles by lowest IoU to enable diagnostic failure mode inspection

    (e.g., cloud shadows, meandering rivers, plantation monocultures).
    """
    sorted_tiles = sorted(tile_predictions, key=lambda x: x["metrics"]["iou"])
    return sorted_tiles[:top_k]


def format_results_table(
    ndvi_metrics: Dict[str, float],
    unet_metrics: Dict[str, float],
) -> str:
    """Generates a GitHub-flavored Markdown benchmark table comparing

    the NDVI domain baseline against the Deep Learning U-Net on held-out spatial blocks.
    """
    iou_delta = (unet_metrics["iou"] - ndvi_metrics["iou"]) * 100
    f1_delta = (unet_metrics["f1"] - ndvi_metrics["f1"]) * 100

    table = f"""| Model / Approach | Spatial Split | Forest IoU | Forest F1-Score | Precision | Recall | Delta vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **NDVI Threshold (Baseline)** | Eastern Block (Held-Out) | {ndvi_metrics['iou']:.4f} | {ndvi_metrics['f1']:.4f} | {ndvi_metrics['precision']:.4f} | {ndvi_metrics['recall']:.4f} | Baseline |
| **U-Net (ResNet-34, 6-Bands)** | Eastern Block (Held-Out) | **{unet_metrics['iou']:.4f}** | **{unet_metrics['f1']:.4f}** | **{unet_metrics['precision']:.4f}** | **{unet_metrics['recall']:.4f}** | **+{iou_delta:+.2f}% IoU** |
"""
    return table
