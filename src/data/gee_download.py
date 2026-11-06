"""
Google Earth Engine Ingestion & Regional Hotspots Module
Fetches cloud-masked Sentinel-2 Level-2A imagery (6 bands: B2, B3, B4, B8, B11, B12)
and Google Dynamic World forest ("trees") ground truth masks.
Supports curated Asian and Global conservation hotspots including Hasdeo Arand (India),
Western Ghats (India), and Riau/Sumatra (Indonesia).
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import ee

# Curated regional hotspots with verified dry-season windows and coordinates
REGIONAL_HOTSPOTS = {
    "hasdeo_arand_india": {
        "name": "Hasdeo Arand, Chhattisgarh (India)",
        "description": "The 'Lungs of Chhattisgarh'. Contiguous Sal and Teak forest facing open-cast coal mining concessions (PEKB block).",
        "center_lat": 22.82,
        "center_lon": 82.78,
        # 10 km x 10 km bounding box [min_lon, min_lat, max_lon, max_lat]
        "bbox": [82.73, 22.77, 82.83, 22.87],
        # Pre-monsoon dry season (crystal clear skies, zero cloud cover)
        "t1_dates": ("2021-02-15", "2021-04-15"),
        "t2_dates": ("2024-02-15", "2024-04-15"),
        "utm_epsg": "EPSG:32644",  # UTM Zone 44N
    },
    "western_ghats_india": {
        "name": "Western Ghats, Karnataka (India)",
        "description": "UNESCO World Heritage biodiversity hotspot. Shola-rainforest canopy fragmented by linear road projects and coffee/rubber estates.",
        "center_lat": 12.35,
        "center_lon": 75.75,
        "bbox": [75.70, 12.30, 75.80, 12.40],
        # Dry winter window
        "t1_dates": ("2021-01-01", "2021-03-01"),
        "t2_dates": ("2024-01-01", "2024-03-01"),
        "utm_epsg": "EPSG:32643",  # UTM Zone 43N
    },
    "sumatra_indonesia": {
        "name": "Riau Province, Sumatra (Indonesia)",
        "description": "Tropical peat swamp forest conversion into industrial oil palm and acacia pulp plantations.",
        "center_lat": 0.55,
        "center_lon": 101.45,
        "bbox": [101.40, 0.50, 101.50, 0.60],
        "t1_dates": ("2021-06-01", "2021-08-31"),
        "t2_dates": ("2024-06-01", "2024-08-31"),
        "utm_epsg": "EPSG:32647",  # UTM Zone 47N
    },
    "amazon_brazil": {
        "name": "Rondônia, Amazon Basin (Brazil)",
        "description": "Iconic fishbone deforestation along BR-364 highway and cattle ranching corridors.",
        "center_lat": -10.0,
        "center_lon": -62.5,
        "bbox": [-62.55, -10.05, -62.45, -9.95],
        "t1_dates": ("2020-07-01", "2020-08-31"),
        "t2_dates": ("2023-07-01", "2023-08-31"),
        "utm_epsg": "EPSG:32720",  # UTM Zone 20S
    },
}


def get_hotspot_config(region_key: str = "hasdeo_arand_india") -> Dict[str, Any]:
    """Retrieves metadata and bounding parameters for a curated hotspot."""
    if region_key not in REGIONAL_HOTSPOTS:
        available = list(REGIONAL_HOTSPOTS.keys())
        raise KeyError(f"Region '{region_key}' not found. Available regions: {available}")
    return REGIONAL_HOTSPOTS[region_key]


def initialize_earth_engine(project_id: Optional[str] = None) -> bool:
    """Initializes Google Earth Engine."""
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        print("[GEE] Successfully initialized Google Earth Engine.")
        return True
    except Exception as e:
        print(f"[GEE] Earth Engine initialization notice: {e}")
        print("To authenticate GEE, run `earthengine authenticate` or sign up at https://earthengine.google.com")
        return False


def mask_s2_clouds_scl(image: ee.Image) -> ee.Image:
    """Masks clouds and shadows using Scene Classification Layer (SCL)."""
    scl = image.select("SCL")
    cloud_shadow_mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    optical_bands = image.select(["B2", "B3", "B4", "B8", "B11", "B12"]).divide(10000.0)
    return optical_bands.updateMask(cloud_shadow_mask)


def get_sentinel2_composite(
    roi: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud_cover: int = 25,
) -> ee.Image:
    """Builds a cloud-free median composite of Sentinel-2 L2A imagery (6 bands) over a region."""
    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover))
        .map(mask_s2_clouds_scl)
    )
    return s2_collection.median().clip(roi)


def get_dynamic_world_forest_mask(
    roi: ee.Geometry,
    start_date: str,
    end_date: str,
    tree_probability_threshold: float = 0.5,
) -> ee.Image:
    """Retrieves Dynamic World tree probability and creates a binary forest mask."""
    dw_collection = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
    )
    tree_prob = dw_collection.select("trees").mean().clip(roi)
    return tree_prob.gte(tree_probability_threshold).rename("forest_mask")


def export_geotiff(
    image: ee.Image,
    description: str,
    roi: ee.Geometry,
    folder: str = "Deforestation_Project",
    scale: int = 10,
) -> ee.batch.Task:
    """Submits an Earth Engine export task to Google Drive."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        scale=scale,
        region=roi,
        maxPixels=1e9,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"[GEE] Export task started: {description} (Status: {task.status().get('state')})")
    return task
