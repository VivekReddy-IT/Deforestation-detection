"""
Change Detection and Deforestation Event Filtering Module
Detects newly cleared forest patches between T1 and T2 and filters out
small noisy artifacts (< 0.5 ha / 50 pixels at 10m resolution).
"""

from typing import Tuple, Dict, Any, Optional
import os
import json
import numpy as np
from scipy import ndimage
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping


def detect_deforestation(
    forest_t1: np.ndarray,
    forest_t2: np.ndarray,
    min_patch_pixels: int = 50,
    pixel_size_meters: float = 10.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Identifies newly deforested areas between two temporal dates.

    Args:
        forest_t1: Binary boolean mask of forest at baseline date T1.
        forest_t2: Binary boolean mask of forest at monitoring date T2.
        min_patch_pixels: Minimum contiguous cluster size to keep (default: 50 pixels = 0.5 ha).
        pixel_size_meters: Ground sampling distance (10m for Sentinel-2).

    Returns:
        Tuple of (filtered_cleared_mask, statistics_dict).
    """
    # 1. Raw change detection: Forest at T1 AND NOT Forest at T2
    raw_cleared = forest_t1.astype(bool) & (~forest_t2.astype(bool))

    # 2. Connected-component labeling
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    labeled_array, num_features = ndimage.label(raw_cleared, structure=structure)

    if num_features == 0:
        return np.zeros_like(raw_cleared, dtype=bool), {
            "total_cleared_pixels": 0,
            "total_cleared_hectares": 0.0,
            "num_clearing_events": 0,
            "filtered_noise_pixels": 0,
        }

    # 3. Sum pixel sizes for each contiguous component
    sizes = ndimage.sum(raw_cleared, labeled_array, range(1, num_features + 1))

    # 4. Filter out patches smaller than min_patch_pixels (< 0.5 ha)
    valid_component_indices = np.where(sizes >= min_patch_pixels)[0] + 1
    filtered_cleared = np.isin(labeled_array, valid_component_indices)

    # 5. Compute quantitative metrics
    ha_per_pixel = (pixel_size_meters * pixel_size_meters) / 10000.0  # 100 m2 = 0.01 ha
    total_pixels = int(np.sum(filtered_cleared))
    total_hectares = float(total_pixels * ha_per_pixel)
    noise_pixels = int(np.sum(raw_cleared) - total_pixels)
    num_events = len(valid_component_indices)

    stats = {
        "total_cleared_pixels": total_pixels,
        "total_cleared_hectares": round(total_hectares, 2),
        "num_clearing_events": num_events,
        "filtered_noise_pixels": noise_pixels,
        "noise_hectares": round(noise_pixels * ha_per_pixel, 2),
        "mean_event_hectares": round(total_hectares / max(1, num_events), 2),
    }

    return filtered_cleared, stats


def export_clearings_geojson(
    cleared_mask: np.ndarray,
    reference_geotiff_path: str,
    output_geojson_path: str,
) -> str:
    """Vectorizes binary deforestation raster into a GeoJSON feature collection

    preserving geographic CRS coordinates.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_geojson_path)), exist_ok=True)

    with rasterio.open(reference_geotiff_path) as src:
        transform = src.transform
        crs = src.crs

    features = []
    # Extract polygon geometries for pixels with value 1
    for geom, val in shapes(cleared_mask.astype(np.uint8), mask=cleared_mask, transform=transform):
        if val == 1:
            poly = shape(geom)
            features.append({
                "type": "Feature",
                "properties": {
                    "area_ha": round(poly.area / 10000.0, 3),
                    "alert": "Deforestation",
                },
                "geometry": mapping(poly),
            })

    geojson_obj = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": str(crs)},
        },
        "features": features,
    }

    with open(output_geojson_path, "w") as f:
        json.dump(geojson_obj, f)

    print(f"[GeoJSON] Exported {len(features)} deforestation polygons to {output_geojson_path}")
    return output_geojson_path
