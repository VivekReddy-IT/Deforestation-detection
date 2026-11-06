"""
Synthetic Study Area Generator with Regional Landscape Presets
Creates realistic multispectral Sentinel-2 GeoTIFFs (6 bands) and Dynamic World forest masks
for India (Hasdeo Arand, Western Ghats), Asia (Sumatra), and the Amazon.
"""

from typing import Tuple, Dict
import os
import numpy as np
from scipy import ndimage
import rasterio
from rasterio.transform import from_origin
from src.data.gee_download import REGIONAL_HOTSPOTS, get_hotspot_config


def generate_synthetic_study_area(
    output_dir: str = "data/raw",
    region_key: str = "hasdeo_arand_india",
    width: int = 1024,
    height: int = 1024,
    random_seed: int = 42,
) -> Dict[str, str]:
    """Generates two temporal scenes (T1 and T2) of a 10 km x 10 km region

    calibrated to the selected Asian or Global hotspot.

    Regions:
        - 'hasdeo_arand_india': Hasdeo Arand, Chhattisgarh (Central Indian Sal forest & coal corridor)
        - 'western_ghats_india': Western Ghats, Karnataka (Evergreen rainforest & linear roads)
        - 'sumatra_indonesia': Riau, Sumatra (Peat swamp & palm oil canals)
        - 'amazon_brazil': Rondônia, Brazil (Amazon fishbone logging)
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_seed)

    config = get_hotspot_config(region_key)
    print(f"[DataGen] Synthesizing study landscape: {config['name']}")
    print(f"  Description: {config['description']}")

    # 1. Base topography
    noise = np.random.randn(height // 8, width // 8)
    smooth_landscape = ndimage.zoom(noise, 8, order=1)[:height, :width]
    smooth_landscape = (smooth_landscape - smooth_landscape.min()) / (
        smooth_landscape.max() - smooth_landscape.min() + 1e-6
    )

    # 2. Forest mask at T1
    forest_t1 = smooth_landscape > 0.28

    # 3. Add winding river or water corridor
    x = np.linspace(0, 4 * np.pi, height)
    river_center = (width * 0.45 + np.sin(x) * (width * 0.15)).astype(int)
    river_mask = np.zeros((height, width), dtype=bool)
    for r in range(height):
        c = river_center[r]
        c_min, c_max = max(0, c - 8), min(width, c + 8)
        river_mask[r, c_min:c_max] = True

    forest_t1[river_mask] = False

    # 4. Generate new clearings at T2 (e.g. coal mining pit & linear haul track for Hasdeo Arand)
    new_clearing_mask = np.zeros((height, width), dtype=bool)

    # Main access artery (railway line / haul road)
    road_y = int(height * 0.60)
    new_clearing_mask[road_y - 4 : road_y + 4, int(width * 0.15) : int(width * 0.85)] = True

    # Clear-cut parcels & open-cast pits
    num_clearings = 20
    for _ in range(num_clearings):
        cx = np.random.randint(int(width * 0.2), int(width * 0.8))
        cy = road_y + np.random.randint(-160, 160)
        rx, ry = np.random.randint(25, 55), np.random.randint(25, 55)
        yy, xx = np.ogrid[:height, :width]
        dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        clearing_patch = dist <= 1.0
        new_clearing_mask = new_clearing_mask | clearing_patch

    new_clearing_mask[river_mask] = False
    forest_t2 = forest_t1 & (~new_clearing_mask)

    # 5. Multispectral Reflectance Generation (B2, B3, B4, B8, B11, B12)
    def create_multispectral_scene(forest_mask: np.ndarray) -> np.ndarray:
        scene = np.zeros((6, height, width), dtype=np.float32)
        tex = np.random.normal(0, 0.015, (height, width)).astype(np.float32)

        # For Indian Sal forests: dense green canopy, red lateritic soil in clearings
        scene[0] = np.where(forest_mask, 0.035, 0.085) + tex  # B2 Blue
        scene[1] = np.where(forest_mask, 0.065, 0.115) + tex  # B3 Green
        scene[2] = np.where(forest_mask, 0.030, 0.160) + tex  # B4 Red (Higher reflectance on red laterite soil)
        scene[3] = np.where(forest_mask, 0.590, 0.170) + tex * 2.0  # B8 NIR (High for Sal canopy)
        scene[4] = np.where(forest_mask, 0.105, 0.340) + tex  # B11 SWIR-1 (High on dry exposed mine soil)
        scene[5] = np.where(forest_mask, 0.045, 0.240) + tex  # B12 SWIR-2

        # Water absorption in river
        scene[0][river_mask] = 0.070
        scene[1][river_mask] = 0.060
        scene[2][river_mask] = 0.040
        scene[3][river_mask] = 0.015
        scene[4][river_mask] = 0.005
        scene[5][river_mask] = 0.002

        return np.clip(scene, 0.001, 0.999)

    scene_t1 = create_multispectral_scene(forest_t1)
    scene_t2 = create_multispectral_scene(forest_t2)

    # Geospatial transform based on selected region's exact coordinates
    pixel_size = 10.0  # 10m GSD
    crs = config.get("utm_epsg", "EPSG:32644")
    
    # Accurate UTM projection for the hotspot center
    try:
        import pyproj
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        center_x, center_y = transformer.transform(config["center_lon"], config["center_lat"])
        origin_x = center_x - (width * pixel_size) / 2.0
        origin_y = center_y + (height * pixel_size) / 2.0
    except Exception:
        origin_x = 500000.0
        origin_y = config["center_lat"] * 111000.0

    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)

    paths = {
        "s2_t1": os.path.join(output_dir, "sentinel2_t1.tif"),
        "s2_t2": os.path.join(output_dir, "sentinel2_t2.tif"),
        "mask_t1": os.path.join(output_dir, "dynamic_world_t1.tif"),
        "mask_t2": os.path.join(output_dir, "dynamic_world_t2.tif"),
    }

    for tag, arr, is_mask in [
        ("s2_t1", scene_t1, False),
        ("s2_t2", scene_t2, False),
        ("mask_t1", forest_t1.astype(np.uint8), True),
        ("mask_t2", forest_t2.astype(np.uint8), True),
    ]:
        file_path = paths[tag]
        if is_mask:
            with rasterio.open(
                file_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype=rasterio.uint8,
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(arr, 1)
        else:
            with rasterio.open(
                file_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=6,
                dtype=rasterio.float32,
                crs=crs,
                transform=transform,
            ) as dst:
                for b in range(6):
                    dst.write(arr[b], b + 1)

    cleared_ha = np.sum(forest_t1 & ~forest_t2) * 0.01
    print(f"  Geographic Origin: {config['name']} ({config['center_lat']}°N, {config['center_lon']}°E)")
    print(f"  T1 forest area: {np.sum(forest_t1) * 0.01:.2f} ha ({np.mean(forest_t1)*100:.1f}%)")
    print(f"  T2 forest area: {np.sum(forest_t2) * 0.01:.2f} ha ({np.mean(forest_t2)*100:.1f}%)")
    print(f"  Deforested between T1 and T2: {cleared_ha:.2f} ha")
    return paths
