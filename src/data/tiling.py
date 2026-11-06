"""
Tiling and Spatial Block Splitting Module
Cuts large satellite rasters into 256x256 tiles, computes band normalization,
and applies strict geographic block splitting to prevent spatial autocorrelation leakage.
"""

from typing import Tuple, List, Dict, Optional
import os
import json
import numpy as np
import rasterio


def tile_raster_and_mask(
    raster_path: str,
    mask_path: str,
    output_dir: str,
    tile_size: int = 256,
    stride: Optional[int] = None,
) -> List[Dict]:
    """Cuts multispectral GeoTIFF and corresponding binary forest mask into

    standardized tiles (e.g., 256x256).

    Args:
        raster_path: Path to multispectral GeoTIFF (6 bands).
        mask_path: Path to binary forest mask GeoTIFF (1 band).
        output_dir: Destination directory for .npy or .npz tile arrays.
        tile_size: Height and width of square tiles.
        stride: Stride between tiles (defaults to tile_size for non-overlapping).

    Returns:
        List of tile metadata dicts containing filepaths, pixel bounds, and geographic coordinates.
    """
    if stride is None:
        stride = tile_size

    tiles_dir = os.path.join(output_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    with rasterio.open(raster_path) as r_src, rasterio.open(mask_path) as m_src:
        raster = r_src.read()  # Shape: (C, H, W)
        mask = m_src.read(1)   # Shape: (H, W)
        transform = r_src.transform
        num_channels, height, width = raster.shape

    metadata_list = []
    tile_idx = 0

    for top in range(0, height - tile_size + 1, stride):
        for left in range(0, width - tile_size + 1, stride):
            bottom = top + tile_size
            right = left + tile_size

            # Extract patch
            img_patch = raster[:, top:bottom, left:right]
            mask_patch = mask[top:bottom, left:right]

            # Calculate geographic centroid using Affine transform
            center_col = left + tile_size / 2.0
            center_row = top + tile_size / 2.0
            geo_x, geo_y = rasterio.transform.xy(transform, center_row, center_col)

            tile_filename = f"tile_{tile_idx:04d}_r{top}_c{left}.npz"
            tile_path = os.path.join(tiles_dir, tile_filename)

            # Save as compressed NumPy archive
            np.savez_compressed(
                tile_path,
                image=img_patch.astype(np.float32),
                mask=mask_patch.astype(np.uint8),
            )

            meta = {
                "tile_id": tile_idx,
                "file_path": tile_path,
                "row_top": top,
                "row_bottom": bottom,
                "col_left": left,
                "col_right": right,
                "geo_x": geo_x,
                "geo_y": geo_y,
                "forest_ratio": float(np.mean(mask_patch)),
            }
            metadata_list.append(meta)
            tile_idx += 1

    metadata_path = os.path.join(output_dir, "tile_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_list, f, indent=2)

    print(f"[Tiling] Extracted {len(metadata_list)} tiles ({tile_size}x{tile_size}) into {tiles_dir}")
    return metadata_list


def spatial_block_split(
    metadata_list: List[Dict],
    split_strategy: str = "geographic_block",
    test_ratio: float = 0.25,
    val_ratio: float = 0.15,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Splits tiles geographically to guarantee out-of-region generalization.

    Never use random splits for satellite imagery: neighboring pixels have high
    spatial autocorrelation, which causes near-duplicate leakage between train and test.

    Strategies:
        'geographic_block': Withholds the entire Eastern quadrant / half as Held-Out Test,
                            North-West as Validation, and South-West as Train.
    """
    if not metadata_list:
        raise ValueError("Tile metadata list is empty.")

    # Sort tiles by geographic X coordinate (longitude / easting)
    all_x = [m["geo_x"] for m in metadata_list]
    all_y = [m["geo_y"] for m in metadata_list]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    x_range = max_x - min_x
    y_range = max_y - min_y

    train_tiles, val_tiles, test_tiles = [], [], []

    # Held-out Test: Eastern block (e.g., easternmost 30% of study area)
    test_threshold_x = min_x + x_range * (1.0 - test_ratio)
    # Validation: Northwestern block
    val_threshold_y = min_y + y_range * 0.5

    for m in metadata_list:
        if m["geo_x"] >= test_threshold_x:
            # Eastern block is strictly test
            m["split"] = "test"
            test_tiles.append(m)
        elif m["geo_y"] >= val_threshold_y and len(val_tiles) < len(metadata_list) * val_ratio:
            m["split"] = "val"
            val_tiles.append(m)
        else:
            m["split"] = "train"
            train_tiles.append(m)

    print(f"[Spatial Split] Total: {len(metadata_list)} tiles")
    print(f"  -> Train: {len(train_tiles)} tiles ({len(train_tiles)/len(metadata_list)*100:.1f}%) [Western Region]")
    print(f"  -> Val:   {len(val_tiles)} tiles ({len(val_tiles)/len(metadata_list)*100:.1f}%) [North-Western Region]")
    print(f"  -> Test:  {len(test_tiles)} tiles ({len(test_tiles)/len(metadata_list)*100:.1f}%) [Eastern Block Held-Out]")

    return train_tiles, val_tiles, test_tiles


def compute_band_statistics(train_tiles: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Computes channel-wise mean and standard deviation strictly on the

    TRAINING set to prevent data leakage.
    """
    means = []
    stds = []

    for tile in train_tiles:
        data = np.load(tile["file_path"])
        img = data["image"]  # (C, H, W)
        means.append(np.mean(img, axis=(1, 2)))
        stds.append(np.std(img, axis=(1, 2)))

    mean_vec = np.mean(means, axis=0)
    std_vec = np.mean(stds, axis=0) + 1e-6
    return mean_vec, std_vec
