"""
Publication-Quality Figure Generator
Renders high-resolution visual comparisons, multispectral false-color composites,
NDVI maps, change detection alert overlays, and predictive deforestation risk heatmaps.
"""

from typing import List, Dict, Any, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _rescale_rgb(band_r: np.ndarray, band_g: np.ndarray, band_b: np.ndarray) -> np.ndarray:
    """Enhances multispectral bands into an 8-bit RGB image with percentile stretching."""
    stack = np.stack([band_r, band_g, band_b], axis=-1)
    p2 = np.percentile(stack, 2)
    p98 = np.percentile(stack, 98)
    stretched = np.clip((stack - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
    return stretched


def save_visual_comparison_figure(
    img_t1: np.ndarray,
    img_t2: np.ndarray,
    ground_truth_t1: np.ndarray,
    ground_truth_t2: np.ndarray,
    unet_t1: np.ndarray,
    unet_t2: np.ndarray,
    cleared_mask: np.ndarray,
    stats: Dict[str, Any],
    output_path: str = "results/figures/deforestation_summary_figure.png",
) -> str:
    """Creates a comprehensive 6-panel summary figure showcasing the retrospective change detection."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=200)

    # 1. True Color T1 (B4, B3, B2)
    rgb_t1 = _rescale_rgb(img_t1[2], img_t1[1], img_t1[0])
    axes[0, 0].imshow(rgb_t1)
    axes[0, 0].set_title("Sentinel-2 T1 (Baseline Year)\nTrue Color (B4-B3-B2)", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. True Color T2 (B4, B3, B2)
    rgb_t2 = _rescale_rgb(img_t2[2], img_t2[1], img_t2[0])
    axes[0, 1].imshow(rgb_t2)
    axes[0, 1].set_title("Sentinel-2 T2 (+2 Years Later)\nTrue Color (B4-B3-B2)", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")

    # 3. False Color Infrared T1 (B8 NIR, B4 Red, B3 Green)
    cir_t1 = _rescale_rgb(img_t1[3], img_t1[2], img_t1[1])
    axes[0, 2].imshow(cir_t1)
    axes[0, 2].set_title("Sentinel-2 T1 False-Color Infrared\n(B8-B4-B3, Foliage in Red)", fontsize=13, fontweight="bold")
    axes[0, 2].axis("off")

    # 4. NDVI Vegetation Index Map T1
    ndvi_t1 = (img_t1[3] - img_t1[2]) / (img_t1[3] + img_t1[2] + 1e-6)
    im_ndvi = axes[1, 0].imshow(ndvi_t1, cmap="YlGn", vmin=-0.2, vmax=0.9)
    axes[1, 0].set_title("Normalized Difference Vegetation Index (NDVI)\nBaseline Vegetation Density", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")
    cbar = fig.colorbar(im_ndvi, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    # 5. U-Net / Prithvi vs Ground Truth (T2)
    diff = np.zeros((*unet_t2.shape, 3), dtype=np.float32)
    gt_bool = ground_truth_t2.astype(bool)
    pred_bool = unet_t2.astype(bool)

    diff[gt_bool & pred_bool] = [0.15, 0.70, 0.15]   # TP: Green
    diff[(~gt_bool) & pred_bool] = [0.95, 0.80, 0.1]  # FP: Yellow
    diff[gt_bool & (~pred_bool)] = [0.85, 0.20, 0.20] # FN: Red
    diff[(~gt_bool) & (~pred_bool)] = [0.95, 0.95, 0.95] # TN: Light Gray

    axes[1, 1].imshow(diff)
    axes[1, 1].set_title("Segmentation vs Dynamic World (T2)\nGreen: Match | Yellow: FP | Red: FN", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")

    # 6. Detected Deforestation Overlay on T2
    overlay = rgb_t2.copy()
    overlay[cleared_mask] = [1.0, 0.05, 0.15]
    axes[1, 2].imshow(overlay)
    ha_val = stats.get("total_cleared_hectares", 0.0)
    events_val = stats.get("num_clearing_events", 0)
    axes[1, 2].set_title(
        f"Detected Deforestation Alert (>0.5 ha)\nCleared: {ha_val:.1f} ha across {events_val} patches",
        fontsize=13,
        fontweight="bold",
        color="#B71C1C",
    )
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"[Plot] Saved summary comparison figure to {output_path}")
    return output_path


def save_predictive_risk_figure(
    img_t2: np.ndarray,
    covariates: Dict[str, np.ndarray],
    risk_map: np.ndarray,
    forest_mask_t2: np.ndarray,
    risk_stats: Dict[str, Any],
    output_path: str = "results/figures/predictive_risk_forecast.png",
) -> str:
    """Renders a 4-panel diagnostic figure showcasing the predictive risk forecasting engine.

    Panels:
        1. True Color T2 landscape with active logging roads
        2. Euclidean distance to active logging frontier (meters)
        3. Forest edge fragmentation & degradation density
        4. Calibrated Deforestation Vulnerability Heatmap (Next 12 Months)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=200)

    # 1. Landscape RGB
    rgb = _rescale_rgb(img_t2[2], img_t2[1], img_t2[0])
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Sentinel-2 T2 Monitoring Landscape\nActive Logging Roads & Clear-Cuts Visible", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. Distance to Deforestation Frontier
    dist_clearing = covariates.get("dist_clearing_meters", np.zeros_like(risk_map))
    im_dist = axes[0, 1].imshow(dist_clearing, cmap="viridis_r")
    axes[0, 1].set_title("Proximity to Active Logging Frontier\nEuclidean Distance to Nearest Clear-Cut (Meters)", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")
    cbar1 = fig.colorbar(im_dist, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label("Distance (m)", fontsize=10)

    # 3. Forest Edge Degradation & Fragmentation Index
    edge_density = covariates.get("edge_density", np.zeros_like(risk_map))
    im_edge = axes[1, 0].imshow(edge_density, cmap="magma")
    axes[1, 0].set_title("Canopy Fragmentation & Edge Density\nExposed Boundaries Vulnerable to Timber Poaching", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")
    cbar2 = fig.colorbar(im_edge, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar2.set_label("Edge Density Index", fontsize=10)

    # 4. Calibrated Future Vulnerability Risk Heatmap
    # Mask non-forest to light gray
    risk_display = np.copy(risk_map)
    risk_display[~forest_mask_t2.astype(bool)] = np.nan

    cmap_risk = plt.cm.YlOrRd.copy()
    cmap_risk.set_bad(color="#ECEFF1")

    im_risk = axes[1, 1].imshow(risk_display, cmap=cmap_risk, vmin=0.0, vmax=1.0)
    imminent_ha = risk_stats.get("imminent_risk_ha", 0.0)
    high_ha = risk_stats.get("high_risk_ha", 0.0)
    axes[1, 1].set_title(
        f"Forecasted Deforestation Risk (Next 12 Mos)\nImminent (>80%): {imminent_ha} ha | High (50-80%): {high_ha} ha",
        fontsize=13,
        fontweight="bold",
        color="#B71C1C",
    )
    axes[1, 1].axis("off")
    cbar3 = fig.colorbar(im_risk, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar3.set_label("Probability of Clearing P(Loss)", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"[Plot] Saved predictive risk forecast figure to {output_path}")
    return output_path


def save_worst_tiles_figure(
    worst_tiles: List[Dict[str, Any]],
    output_path: str = "results/figures/worst_tiles_diagnostics.png",
) -> str:
    """Plots diagnostic inspection of the lowest-performing held-out test tiles."""
    if not worst_tiles:
        return ""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    k = min(len(worst_tiles), 4)
    fig, axes = plt.subplots(k, 3, figsize=(12, 3.5 * k), dpi=180)
    if k == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(k):
        tile_data = worst_tiles[i]
        img = tile_data["image"]
        gt = tile_data["mask"]
        pred = tile_data["pred"]
        metrics = tile_data["metrics"]

        rgb = _rescale_rgb(img[2], img[1], img[0])
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Tile #{tile_data['tile_id']} (RGB)", fontsize=11)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt, cmap="Greens", vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth Forest", fontsize=11)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="Blues", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Pred (IoU: {metrics['iou']:.2f}, F1: {metrics['f1']:.2f})", fontsize=11)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close()
    print(f"[Plot] Saved diagnostic worst-tiles figure to {output_path}")
    return output_path
