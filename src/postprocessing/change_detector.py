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
    """Vectorizes binary deforestation raster into an RFC 7946 standard WGS84 GeoJSON
    feature collection enriched with ISRO NRSC-standard forest alert attributes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_geojson_path)), exist_ok=True)

    with rasterio.open(reference_geotiff_path) as src:
        transform = src.transform
        src_crs = src.crs

    # Setup reprojection transformer to EPSG:4326 (WGS 84 lon/lat)
    import pyproj
    from shapely.ops import transform as shapely_transform
    transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    raw_features = []
    # Extract polygon geometries for cleared pixels (value 1)
    for geom, val in shapes(cleared_mask.astype(np.uint8), mask=cleared_mask, transform=transform):
        if val == 1:
            poly_utm = shape(geom)
            area_ha = round(poly_utm.area / 10000.0, 3)
            # Minimum filter: keep patches >= 0.1 ha
            if area_ha < 0.05:
                continue

            # Reproject to WGS84 (lon, lat)
            poly_wgs84 = shapely_transform(transformer.transform, poly_utm)
            centroid_lon, centroid_lat = poly_wgs84.centroid.x, poly_wgs84.centroid.y
            centroid_x, centroid_y = poly_utm.centroid.x, poly_utm.centroid.y

            # Classification & impact metrics
            if area_ha >= 5.0:
                severity = "CRITICAL (Major Industrial/Mine Pit)"
                urgency = "IMMEDIATE_DISPATCH"
            elif area_ha >= 1.0:
                severity = "HIGH (Active Logging Front)"
                urgency = "PATROL_REQUIRED"
            else:
                severity = "MODERATE (Encroachment/Clear-cut)"
                urgency = "MONITORING"

            carbon_loss = round(area_ha * 310.0, 1)  # ~310 t CO2/ha in Sal/Tropical forest

            raw_features.append({
                "area_ha": area_ha,
                "area_acres": round(area_ha * 2.47105, 2),
                "severity": severity,
                "urgency": urgency,
                "centroid_lat": round(centroid_lat, 6),
                "centroid_lon": round(centroid_lon, 6),
                "utm_easting": round(centroid_x, 1),
                "utm_northing": round(centroid_y, 1),
                "carbon_loss_tco2": carbon_loss,
                "poly_wgs84": poly_wgs84,
            })

    # Sort descending by cleared area
    raw_features.sort(key=lambda x: x["area_ha"], reverse=True)

    features = []
    for idx, item in enumerate(raw_features):
        alert_id = f"ISRO-HAS-{idx+1:03d}"
        features.append({
            "type": "Feature",
            "id": alert_id,
            "properties": {
                "alert_id": alert_id,
                "area_ha": item["area_ha"],
                "area_acres": item["area_acres"],
                "severity": item["severity"],
                "urgency": item["urgency"],
                "alert": "Deforestation",
                "center_lat": item["centroid_lat"],
                "center_lon": item["centroid_lon"],
                "utm_coords": f"44N {int(item['utm_easting'])} {int(item['utm_northing'])}",
                "carbon_tco2": item["carbon_loss_tco2"],
                "detection_date": "2024-03-15",
                "division": "Hasdeo Arand Forest Division (Chhattisgarh)",
                "status": "PENDING_FIELD_VERIFICATION",
            },
            "geometry": mapping(item["poly_wgs84"]),
        })

    geojson_obj = {
        "type": "FeatureCollection",
        "name": "isro_hasdeo_arand_deforestation_alerts",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }

    with open(output_geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f, indent=2)

    print(f"[GeoJSON] Exported {len(features)} WGS84 deforestation alert features to {output_geojson_path}")
    return output_geojson_path

