"""
Spatial Frontier Covariates & Landscape Geography Engine
Extracts distance-to-deforestation frontiers, transport network proximities,
and forest fragmentation metrics to power predictive vulnerability forecasting.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy import ndimage


def compute_euclidean_distance_transform(
    binary_mask: np.ndarray,
    pixel_size_meters: float = 10.0,
) -> np.ndarray:
    """Computes the exact distance in meters from every pixel to the nearest True feature in binary_mask.

    Uses Exact Euclidean Distance Transform (EDT).
    """
    if not np.any(binary_mask):
        # No features present; return high uniform distance
        return np.full(binary_mask.shape, 50000.0, dtype=np.float32)

    # distance_transform_edt computes distance to nearest zero pixel
    # So we invert the mask to find distance to nearest 1 (feature)
    inverted = ~binary_mask.astype(bool)
    distances_in_pixels = ndimage.distance_transform_edt(inverted)
    distances_in_meters = distances_in_pixels * pixel_size_meters
    return distances_in_meters.astype(np.float32)


def compute_canopy_edge_density(
    forest_mask: np.ndarray,
    filter_size: int = 5,
) -> np.ndarray:
    """Calculates forest fragmentation and edge density.

    Forest edges adjacent to cleared parcels suffer microclimate drying,
    fire leakage, and accessibility pressure.
    """
    forest_bool = forest_mask.astype(bool)
    # Edge is the difference between dilation and erosion of the forest canopy
    dilated = ndimage.binary_dilation(forest_bool, iterations=1)
    eroded = ndimage.binary_erosion(forest_bool, iterations=1)
    boundary = dilated ^ eroded

    # Smooth into a local edge density kernel
    kernel = np.ones((filter_size, filter_size), dtype=np.float32) / (filter_size * filter_size)
    edge_density = ndimage.convolve(boundary.astype(np.float32), kernel, mode="reflect")
    return edge_density


def extract_synthetic_or_detected_roads(
    multispectral_t2: np.ndarray,
    forest_mask_t2: np.ndarray,
) -> np.ndarray:
    """Detects linear transport arteries and logging corridors.

    Non-forest corridors with high SWIR reflectance and low NDVI correspond to dirt tracks.
    """
    # Channel 2: Red (B4), Channel 3: NIR (B8), Channel 4: SWIR-1 (B11)
    red = multispectral_t2[2]
    nir = multispectral_t2[3]
    swir = multispectral_t2[4]

    ndvi = (nir - red) / (nir + red + 1e-6)
    # Logging roads: non-forest, bare exposed soil (SWIR > 0.20, NDVI < 0.35)
    bare_ground = (~forest_mask_t2.astype(bool)) & (swir > 0.20) & (ndvi < 0.35)
    return bare_ground


def build_spatial_frontier_features(
    forest_mask_t2: np.ndarray,
    historical_cleared_mask: np.ndarray,
    multispectral_t2: np.ndarray,
    prithvi_embeddings: Optional[np.ndarray] = None,
    pixel_size_meters: float = 10.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Assembles a multi-covariate feature tensor representing spatial vulnerability to clearing.

    Covariates:
        1. Distance to nearest active logging clearing (meters)
        2. Distance to nearest transport / road corridor (meters)
        3. Forest edge density / fragmentation index
        4. Moisture Deficit Index (SWIR-1 / NIR ratio)
        5. NDVI vegetation density
        6. Optional deep spatial embeddings from Prithvi Foundation Model (channels 6..N)
    """
    H, W = forest_mask_t2.shape

    # 1. Distance to recent deforestation front
    dist_clearing = compute_euclidean_distance_transform(historical_cleared_mask, pixel_size_meters)

    # 2. Distance to access roads / tracks
    roads_mask = extract_synthetic_or_detected_roads(multispectral_t2, forest_mask_t2)
    dist_roads = compute_euclidean_distance_transform(roads_mask, pixel_size_meters)

    # 3. Canopy edge density
    edge_density = compute_canopy_edge_density(forest_mask_t2)

    # 4. Spectral indices: Normalized Moisture Index & NDVI
    red = multispectral_t2[2]
    nir = multispectral_t2[3]
    swir = multispectral_t2[4]
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndwi_moisture = (nir - swir) / (nir + swir + 1e-6)

    # Normalize distances using logarithmic scaling to stabilize regression gradients
    log_dist_clearing = np.log1p(dist_clearing) / 10.0
    log_dist_roads = np.log1p(dist_roads) / 10.0

    feature_list = [
        log_dist_clearing,
        log_dist_roads,
        edge_density,
        ndvi,
        ndwi_moisture,
    ]

    # Incorporate Prithvi Foundation Model latent embeddings if provided
    if prithvi_embeddings is not None:
        # prithvi_embeddings: shape (C_emb, H, W)
        for c in range(min(prithvi_embeddings.shape[0], 4)):
            emb_norm = (prithvi_embeddings[c] - np.mean(prithvi_embeddings[c])) / (np.std(prithvi_embeddings[c]) + 1e-6)
            feature_list.append(emb_norm)

    stacked_features = np.stack(feature_list, axis=0).astype(np.float32)

    covariates_dict = {
        "dist_clearing_meters": dist_clearing,
        "dist_roads_meters": dist_roads,
        "edge_density": edge_density,
        "ndvi": ndvi,
        "ndwi_moisture": ndwi_moisture,
    }

    return stacked_features, covariates_dict
