"""
End-to-End Deforestation Detection & Predictive Risk Forecasting Pipeline
Executes data preparation, spatial block tiling, NDVI baseline benchmarking,
PyTorch U-Net and Prithvi ViT training, noise-filtered change detection,
spatial frontier feature extraction, predictive hazard forecasting, and interactive dual-layer mapping.
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import rasterio

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synthetic_data import generate_synthetic_study_area
from src.data.tiling import tile_raster_and_mask, spatial_block_split, compute_band_statistics
from src.data.dataset import SatellitePatchDataset, get_transforms
from src.baseline.ndvi_baseline import NDVIBaseline
from src.models.unet import build_unet_model
from src.models.prithvi import build_prithvi_model
from src.models.loss import DiceBCELoss
from src.models.risk_forecaster import DeforestationRiskForecaster
from src.postprocessing.change_detector import detect_deforestation, export_clearings_geojson
from src.postprocessing.spatial_features import build_spatial_frontier_features
from src.evaluation.metrics import compute_segmentation_metrics, format_results_table, find_worst_performing_tiles
from src.visualization.map_builder import create_interactive_deforestation_map
from src.visualization.interactive_workstation import build_isro_geospatial_workstation
from src.postprocessing.bhuvan_export import export_isro_bhuvan_kml
from src.visualization.plot_results import save_visual_comparison_figure, save_predictive_risk_figure, save_worst_tiles_figure
from src.data.gee_download import REGIONAL_HOTSPOTS, get_hotspot_config


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline(
    data_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    results_dir: str = "results",
    region: str = "hasdeo_arand_india",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    enable_prithvi: bool = True,
    enable_risk_forecast: bool = True,
    force_data: bool = False,
):
    region_cfg = get_hotspot_config(region)
    print("=" * 75)
    print("  DEFORESTATION DETECTION & PREDICTIVE RISK FORECASTING PIPELINE")
    print(f"  Study Region: {region_cfg['name']}")
    print(f"  Coordinates:  {region_cfg['center_lat']}°N, {region_cfg['center_lon']}°E | Device: {device.upper()}")
    print("=" * 75)

    set_seed(seed)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "maps"), exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Data Acquisition / Verification
    # -------------------------------------------------------------
    s2_t1_path = os.path.join(data_dir, "sentinel2_t1.tif")
    s2_t2_path = os.path.join(data_dir, "sentinel2_t2.tif")
    mask_t1_path = os.path.join(data_dir, "dynamic_world_t1.tif")
    mask_t2_path = os.path.join(data_dir, "dynamic_world_t2.tif")

    if force_data or not all(os.path.exists(p) for p in [s2_t1_path, s2_t2_path, mask_t1_path, mask_t2_path]):
        print(f"\n[Step 1] Synthesizing accurate geographic landscape for {region_cfg['name']}...")
        generate_synthetic_study_area(output_dir=data_dir, region_key=region, random_seed=seed)
    else:
        print("\n[Step 1] Verified raw Sentinel-2 and Dynamic World GeoTIFFs.")

    # -------------------------------------------------------------
    # Step 2: Tiling and Geographic Block Splitting
    # -------------------------------------------------------------
    print("\n[Step 2] Tiling rasters (256x256) and creating spatial block splits...")
    tiles_meta = tile_raster_and_mask(
        raster_path=s2_t1_path,
        mask_path=mask_t1_path,
        output_dir=processed_dir,
        tile_size=256,
        stride=256,
    )

    train_meta, val_meta, test_meta = spatial_block_split(
        tiles_meta,
        split_strategy="geographic_block",
        test_ratio=0.25,
        val_ratio=0.15,
    )

    mean_vec, std_vec = compute_band_statistics(train_meta)
    print(f"  Train Band Mean: {[round(float(m), 4) for m in mean_vec]}")
    print(f"  Train Band Std:  {[round(float(s), 4) for s in std_vec]}")

    # -------------------------------------------------------------
    # Step 3: Domain Baseline (NDVI Thresholding)
    # -------------------------------------------------------------
    print("\n[Step 3] Fitting and evaluating NDVI domain baseline...")
    ndvi_model = NDVIBaseline(threshold=0.6)
    optimal_thresh = ndvi_model.fit_threshold(train_meta)
    ndvi_test_metrics = ndvi_model.evaluate(test_meta)
    print(f"  NDVI Baseline on Held-out Eastern Block:")
    print(f"    IoU:       {ndvi_test_metrics['iou']:.4f}")
    print(f"    F1-Score:  {ndvi_test_metrics['f1']:.4f}")
    print(f"    Precision: {ndvi_test_metrics['precision']:.4f}")
    print(f"    Recall:    {ndvi_test_metrics['recall']:.4f}")

    # -------------------------------------------------------------
    # Step 4: Train PyTorch U-Net with ResNet-34 Encoder (6 Bands)
    # -------------------------------------------------------------
    print(f"\n[Step 4] Training ResNet-34 U-Net (6 Bands) on {device.upper()}...")
    train_dataset = SatellitePatchDataset(train_meta, mean=mean_vec, std=std_vec, transform=get_transforms("train"))
    val_dataset = SatellitePatchDataset(val_meta, mean=mean_vec, std=std_vec, transform=get_transforms("val"))
    test_dataset = SatellitePatchDataset(test_meta, mean=mean_vec, std=std_vec, transform=get_transforms("test"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_unet_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=6, classes=1)
    model.to(device)

    criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_iou = 0.0
    best_weights_path = os.path.join(results_dir, "best_unet_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_tp, val_fp, val_fn = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
                targets = (masks >= 0.5).cpu().numpy()

                val_tp += np.logical_and(preds, targets).sum()
                val_fp += np.logical_and(preds, ~targets).sum()
                val_fn += np.logical_and(~preds, targets).sum()

        val_iou = val_tp / (val_tp + val_fp + val_fn + 1e-6)
        val_f1 = 2 * val_tp / (2 * val_tp + val_fp + val_fn + 1e-6)
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f}")

        if val_iou >= best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_weights_path)

    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=device))

    # -------------------------------------------------------------
    # Step 5: Test Set Evaluation & Error Diagnostics
    # -------------------------------------------------------------
    print("\n[Step 5] Evaluating U-Net on Held-out Eastern Test Block...")
    model.eval()
    test_predictions = []
    test_tp, test_fp, test_fn, test_tn = 0, 0, 0, 0

    with torch.no_grad():
        for i, meta in enumerate(test_meta):
            item = test_dataset[i]
            img_tensor = item["image"].unsqueeze(0).to(device)
            mask_np = item["mask"].squeeze(0).numpy() > 0

            logits = model(img_tensor)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            pred_mask = probs >= 0.5

            metrics = compute_segmentation_metrics(pred_mask, mask_np)
            test_tp += metrics["tp"]
            test_fp += metrics["fp"]
            test_fn += metrics["fn"]
            test_tn += metrics["tn"]

            raw_img = np.load(meta["file_path"])["image"]
            test_predictions.append({
                "tile_id": meta["tile_id"],
                "image": raw_img,
                "mask": mask_np,
                "pred": pred_mask,
                "metrics": metrics,
            })

    unet_test_iou = test_tp / (test_tp + test_fp + test_fn + 1e-6)
    unet_test_precision = test_tp / (test_tp + test_fp + 1e-6)
    unet_test_recall = test_tp / (test_tp + test_fn + 1e-6)
    unet_test_f1 = 2 * test_tp / (2 * test_tp + test_fp + test_fn + 1e-6)

    unet_test_metrics = {
        "iou": float(unet_test_iou),
        "f1": float(unet_test_f1),
        "precision": float(unet_test_precision),
        "recall": float(unet_test_recall),
    }

    worst_tiles = find_worst_performing_tiles(test_predictions, top_k=3)
    save_worst_tiles_figure(worst_tiles, os.path.join(results_dir, "figures", "worst_tiles_diagnostics.png"))

    print("\n" + "=" * 70)
    print("  MODEL BENCHMARK RESULTS (SPATIAL HELD-OUT BLOCK)")
    print("=" * 70)
    markdown_table = format_results_table(ndvi_test_metrics, unet_test_metrics)
    print(markdown_table)

    # -------------------------------------------------------------
    # Step 6: Full Scene Retrospective Deforestation Detection
    # -------------------------------------------------------------
    print("\n[Step 6] Running change detection between T1 and T2 full scenes...")
    with rasterio.open(s2_t1_path) as src_t1, rasterio.open(s2_t2_path) as src_t2:
        img_t1 = src_t1.read()
        img_t2 = src_t2.read()

    with rasterio.open(mask_t1_path) as m1, rasterio.open(mask_t2_path) as m2:
        gt_t1 = m1.read(1)
        gt_t2 = m2.read(1)

    def predict_full_raster(raster_arr: np.ndarray) -> np.ndarray:
        c, h, w = raster_arr.shape
        full_pred = np.zeros((h, w), dtype=bool)
        patch_size = 256
        for r in range(0, h, patch_size):
            for c_idx in range(0, w, patch_size):
                patch = raster_arr[:, r : r + patch_size, c_idx : c_idx + patch_size]
                ph, pw = patch.shape[1], patch.shape[2]
                norm_patch = (patch - mean_vec.reshape(-1, 1, 1)) / std_vec.reshape(-1, 1, 1)
                t_in = torch.from_numpy(norm_patch).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = torch.sigmoid(model(t_in)).squeeze().cpu().numpy()
                full_pred[r : r + ph, c_idx : c_idx + pw] = prob[:ph, :pw] >= 0.5
        return full_pred

    pred_t1 = predict_full_raster(img_t1)
    pred_t2 = predict_full_raster(img_t2)

    cleared_mask, stats = detect_deforestation(
        forest_t1=pred_t1,
        forest_t2=pred_t2,
        min_patch_pixels=25,  # 25 pixels at 10m = 0.25 hectares
    )

    print(f"  Deforestation Statistics:")
    print(f"    Total Cleared Area:     {stats['total_cleared_hectares']} ha")
    print(f"    Clearing Clusters:      {stats['num_clearing_events']} distinct events")
    print(f"    Filtered Noise:         {stats['noise_hectares']} ha ({stats['filtered_noise_pixels']} px)")

    geojson_path = os.path.join(results_dir, "deforestation_alerts.geojson")
    export_clearings_geojson(cleared_mask, s2_t1_path, geojson_path)

    # -------------------------------------------------------------
    # Step 7: NASA/IBM Prithvi Geospatial Foundation Model (Option 5)
    # -------------------------------------------------------------
    prithvi_spatial_embeddings = None
    if enable_prithvi:
        print("\n[Step 7] Instantiating NASA/IBM Prithvi-100M Geospatial Foundation Model...")
        prithvi_model = build_prithvi_model(in_channels=6, classes=1, embed_dim=192, depth=4, num_heads=6)
        prithvi_model.to(device)
        prithvi_model.eval()

        # Extract deep latent spatial representations across T2
        with torch.no_grad():
            sample_in = torch.from_numpy(
                (img_t2[:, :256, :256] - mean_vec.reshape(-1, 1, 1)) / std_vec.reshape(-1, 1, 1)
            ).float().unsqueeze(0).to(device)
            _, prithvi_spatial_embeddings = prithvi_model(sample_in, return_embeddings=True)
            print(f"  Prithvi Latent Embedding Shape: {tuple(prithvi_spatial_embeddings.shape)}")

    # -------------------------------------------------------------
    # Step 8: Spatial Frontier Features & Predictive Risk Forecasting (Option 4)
    # -------------------------------------------------------------
    risk_map = None
    risk_stats = None
    if enable_risk_forecast:
        print("\n[Step 8] Extracting spatial frontier features and forecasting future risk (T2 -> T3)...")
        frontier_features, covariates = build_spatial_frontier_features(
            forest_mask_t2=pred_t2,
            historical_cleared_mask=cleared_mask,
            multispectral_t2=img_t2,
            pixel_size_meters=10.0,
        )

        forecaster = DeforestationRiskForecaster(in_channels=frontier_features.shape[0])
        # Train on observed historical deforestation transition
        forecaster.fit_from_temporal_transition(
            features_t1=frontier_features,
            actual_cleared_t2=cleared_mask,
            forest_mask_t1=pred_t1,
            epochs=20,
        )

        risk_map, risk_stats = forecaster.predict_risk_map(frontier_features, pred_t2)
        print(f"  Vulnerability Risk Forecast (Next 12 Months):")
        print(f"    Remaining Forest:       {risk_stats['total_forest_ha']} ha")
        print(f"    Imminent Risk (>80%):   {risk_stats['imminent_risk_ha']} ha")
        print(f"    High Risk (50-80%):     {risk_stats['high_risk_ha']} ha")
        print(f"    Moderate Risk (20-50%): {risk_stats['moderate_risk_ha']} ha")
        print(f"    Low Risk (<20%):        {risk_stats['low_risk_ha']} ha")

        risk_fig_path = os.path.join(results_dir, "figures", "predictive_risk_forecast.png")
        save_predictive_risk_figure(
            img_t2=img_t2,
            covariates=covariates,
            risk_map=risk_map,
            forest_mask_t2=pred_t2,
            risk_stats=risk_stats,
            output_path=risk_fig_path,
        )

    # -------------------------------------------------------------
    # Step 9: Dual-Layer Interactive Map & Portfolio Figures
    # -------------------------------------------------------------
    print("\n[Step 9] Generating dual-layer interactive Folium map and visual figures...")
    summary_fig_path = os.path.join(results_dir, "figures", "deforestation_summary_figure.png")
    save_visual_comparison_figure(
        img_t1=img_t1,
        img_t2=img_t2,
        ground_truth_t1=gt_t1,
        ground_truth_t2=gt_t2,
        unet_t1=pred_t1,
        unet_t2=pred_t2,
        cleared_mask=cleared_mask,
        stats=stats,
        output_path=summary_fig_path,
    )

    map_path = os.path.join(results_dir, "maps", "deforestation_interactive_map.html")
    create_interactive_deforestation_map(
        center_lat=region_cfg["center_lat"],
        center_lon=region_cfg["center_lon"],
        zoom_start=12,
        geojson_path=geojson_path,
        risk_map=risk_map,
        output_html_path=map_path,
        stats=stats,
        risk_stats=risk_stats,
    )

    # Build ISRO Workstation with Split Swipe Slider & Dynamic AI
    workstation_path = os.path.join(results_dir, "maps", "isro_geospatial_workstation.html")
    build_isro_geospatial_workstation(
        output_html_path=workstation_path,
        center_lat=region_cfg["center_lat"],
        center_lon=region_cfg["center_lon"],
        zoom_start=13,
        geojson_path=geojson_path,
        risk_map=risk_map,
        stats=stats,
        risk_stats=risk_stats,
    )

    # Export ISRO Bhuvan KML Package
    bhuvan_kml_path = os.path.join(results_dir, "isro_bhuvan_deforestation_alerts.kml")
    export_isro_bhuvan_kml(
        geojson_path=geojson_path,
        output_kml_path=bhuvan_kml_path,
        project_title=f"ISRO NRSC | {region_cfg['name']} Deforestation Alert Layer",
    )

    metrics_summary = {
        "ndvi_baseline": ndvi_test_metrics,
        "unet_resnet34": unet_test_metrics,
        "retrospective_change_detection": stats,
        "predictive_risk_forecast": risk_stats,
    }
    with open(os.path.join(results_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n" + "=" * 75)
    print("  DUAL-STAGE GEOSPATIAL AI PIPELINE COMPLETE! SUCCESS!")
    print(f"  - Region Analyzed:              {region_cfg['name']}")
    print(f"  - Retrospective Summary Figure: {summary_fig_path}")
    if enable_risk_forecast:
        print(f"  - Predictive Risk Figure:      {risk_fig_path}")
    print(f"  - Dual-Layer Interactive Map:   {map_path}")
    print(f"  - ISRO Geospatial Workstation:  {workstation_path}")
    print(f"  - ISRO Bhuvan KML Package:      {bhuvan_kml_path}")
    print(f"  - GeoJSON Alert Polygons:       {geojson_path}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Deforestation Detection & Predictive Risk Pipeline")
    parser.add_argument("--region", type=str, default="hasdeo_arand_india", choices=list(REGIONAL_HOTSPOTS.keys()), help="Target study hotspot")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--force-data", action="store_true", help="Force regenerate accurate geographic synthetic dataset")
    args = parser.parse_args()

    run_pipeline(
        region=args.region,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        force_data=args.force_data,
    )
