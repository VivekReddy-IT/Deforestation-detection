"""
Google Earth Engine (GEE) Dynamic Tile Service
Generates live, on-the-fly XYZ tile URL endpoints from Google Earth Engine's
distributed cloud computation engine for Sentinel-2, Dynamic World, and NDVI differencing.
"""

from typing import Dict, Any, Optional, Tuple
import os
import ee
from src.data.gee_download import initialize_earth_engine, REGIONAL_HOTSPOTS, get_hotspot_config


class EarthEngineTileService:
    """Manages dynamic Earth Engine computation graphs and exports live XYZ tile URLs

    compatible with Leaflet, Folium, MapLibre, QGIS, and ISRO Bhuvan.
    """

    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id
        self.is_authenticated = initialize_earth_engine(project_id)

    def get_dynamic_layers(
        self,
        region_key: str = "hasdeo_arand_india",
        t1_dates: Optional[Tuple[str, str]] = None,
        t2_dates: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Constructs live Earth Engine computation layers and returns their XYZ tile URL templates.

        Returns:
            Dictionary of layer names mapped to their tile URL templates and attribution.
        """
        hotspot = get_hotspot_config(region_key)
        bbox = hotspot["bbox"]
        t1 = t1_dates or hotspot["t1_dates"]
        t2 = t2_dates or hotspot["t2_dates"]

        if not self.is_authenticated:
            # Return high-resolution Fallback Tile Endpoints if GEE credentials not yet configured
            return self._get_fallback_layers(hotspot)

        try:
            roi = ee.Geometry.BBox(bbox[0], bbox[1], bbox[2], bbox[3])

            # 1. Cloud-masked Sentinel-2 Collections
            def mask_s2(img):
                scl = img.select("SCL")
                mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
                return img.select(["B2", "B3", "B4", "B8", "B11", "B12"]).updateMask(mask)

            s2_col_t1 = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(roi)
                .filterDate(t1[0], t1[1])
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                .map(mask_s2)
                .median()
                .clip(roi)
            )

            s2_col_t2 = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(roi)
                .filterDate(t2[0], t2[1])
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                .map(mask_s2)
                .median()
                .clip(roi)
            )

            # 2. Visualizations
            # True Color (TCI)
            rgb_vis = {"bands": ["B4", "B3", "B2"], "min": 300, "max": 2500, "gamma": 1.2}
            # False Color Infrared (CIR 8-4-3) - Vital for forestry canopy health
            cir_vis = {"bands": ["B8", "B4", "B3"], "min": 500, "max": 4000, "gamma": 1.1}

            # 3. Dynamic NDVI Difference Calculation (Computed on Google Cloud!)
            ndvi_t1 = s2_col_t1.normalizedDifference(["B8", "B4"]).rename("ndvi_t1")
            ndvi_t2 = s2_col_t2.normalizedDifference(["B8", "B4"]).rename("ndvi_t2")
            ndvi_diff = ndvi_t1.subtract(ndvi_t2).rename("delta_ndvi")
            ndvi_diff_vis = {
                "bands": ["delta_ndvi"],
                "min": -0.2,
                "max": 0.5,
                "palette": ["1a9850", "91cf60", "d9ef8b", "fee08b", "fc8d59", "d73027"],
            }

            # 4. Google Dynamic World 10m Tree Canopy Layer
            dw_col = (
                ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                .filterBounds(roi)
                .filterDate(t2[0], t2[1])
                .select("trees")
                .median()
                .clip(roi)
            )
            dw_vis = {
                "bands": ["trees"],
                "min": 0.0,
                "max": 0.8,
                "palette": ["ffffff", "d9f0a3", "78c679", "238443", "00441b"],
            }

            # Generate dynamic XYZ tile URL templates via getMapId
            map_rgb_t1 = s2_col_t1.getMapId(rgb_vis)
            map_rgb_t2 = s2_col_t2.getMapId(rgb_vis)
            map_cir_t2 = s2_col_t2.getMapId(cir_vis)
            map_ndvi = ndvi_diff.getMapId(ndvi_diff_vis)
            map_dw = dw_col.getMapId(dw_vis)

            return {
                "s2_t1_rgb": {
                    "name": f"🛰️ Sentinel-2 True Color ({t1[0][:4]} Baseline)",
                    "url": map_rgb_t1["tile_fetcher"].url_format,
                    "attribution": "Google Earth Engine | Copernicus Sentinel-2",
                    "type": "raster",
                },
                "s2_t2_rgb": {
                    "name": f"🛰️ Sentinel-2 True Color ({t2[0][:4]} Recent)",
                    "url": map_rgb_t2["tile_fetcher"].url_format,
                    "attribution": "Google Earth Engine | Copernicus Sentinel-2",
                    "type": "raster",
                },
                "s2_t2_cir": {
                    "name": "🌿 False Color Infrared (CIR 8-4-3 Forest Vigor)",
                    "url": map_cir_t2["tile_fetcher"].url_format,
                    "attribution": "Google Earth Engine | Copernicus Sentinel-2",
                    "type": "raster",
                },
                "delta_ndvi": {
                    "name": "📉 Live Delta-NDVI Forest Canopy Loss",
                    "url": map_ndvi["tile_fetcher"].url_format,
                    "attribution": "Google Earth Engine Live NDVI Differencing",
                    "type": "raster",
                },
                "dynamic_world_trees": {
                    "name": "🌍 Google Dynamic World 10m Tree Cover",
                    "url": map_dw["tile_fetcher"].url_format,
                    "attribution": "Google / World Resources Institute",
                    "type": "raster",
                },
            }

        except Exception as e:
            print(f"[GEE] Dynamic layer generation failed ({e}). Falling back to satellite basemap.")
            return self._get_fallback_layers(hotspot)

    def _get_fallback_layers(self, hotspot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Provides verified high-resolution satellite endpoints when GEE authentication is pending."""
        return {
            "esri_world_imagery": {
                "name": "🛰️ Esri High-Resolution Satellite (Sub-meter)",
                "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "attribution": "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics",
                "type": "raster",
                "active": True,
            },
            "esri_clarity": {
                "name": "🔬 Esri Clarity Imagery (Cloud-Free Archive)",
                "url": "https://clarity.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "attribution": "Tiles &copy; Esri Clarity",
                "type": "raster",
                "active": False,
            },
            "carto_dark": {
                "name": "🌑 CartoDB Dark Matter (High Contrast Mode)",
                "url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                "attribution": "&copy; CARTO &copy; OpenStreetMap contributors",
                "type": "raster",
                "active": False,
            },
        }
