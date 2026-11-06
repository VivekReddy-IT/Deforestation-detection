"""
VANASPATI-AI geospatial workstation builder.

Renders a standalone Leaflet HUD with split-swipe imagery, live cursor probe,
multi-sensor hover previews, and region-aware telemetry.
"""

from typing import Any, Dict, Optional
import json
import os
import base64
import io

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.visualization.hotspots import hotspot_by_code, workstation_region_payload

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "workstation.html")


def _risk_overlay_uri(risk_map: np.ndarray) -> str:
    norm_risk = np.clip(risk_map, 0.0, 1.0)
    cmap = plt.get_cmap("YlOrRd")
    colored = cmap(norm_risk)
    alpha = np.zeros_like(norm_risk, dtype=np.float32)
    high_risk = norm_risk >= 0.40
    alpha[high_risk] = (norm_risk[high_risk] - 0.40) / 0.60 * 0.85
    colored[..., 3] = alpha
    img_u8 = (colored * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_u8).save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _filter_geojson(raw: Dict[str, Any], min_patch_ha: float) -> Dict[str, Any]:
    features = [
        feat
        for feat in raw.get("features", [])
        if float(feat.get("properties", {}).get("area_ha", 0) or 0) >= min_patch_ha
    ]
    return {**raw, "features": features}


def build_isro_geospatial_workstation(
    output_html_path: str = "results/maps/isro_geospatial_workstation.html",
    center_lat: float = 22.82,
    center_lon: float = 82.78,
    zoom_start: int = 13,
    geojson_path: Optional[str] = "results/deforestation_alerts.geojson",
    risk_map: Optional[np.ndarray] = None,
    stats: Optional[Dict[str, Any]] = None,
    risk_stats: Optional[Dict[str, Any]] = None,
    initial_region: str = "hasdeo",
    min_patch_ha: float = 0.25,
) -> str:
    """Build the full-screen Web GIS workstation HTML file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)

    geojson_obj = None
    kml_text = ""
    if geojson_path and os.path.exists(geojson_path):
        with open(geojson_path, "r", encoding="utf-8") as handle:
            geojson_obj = _filter_geojson(json.load(handle), min_patch_ha)
        kml_path = os.path.join(
            os.path.dirname(os.path.abspath(geojson_path)),
            "isro_bhuvan_deforestation_alerts.kml",
        )
        if os.path.exists(kml_path):
            with open(kml_path, "r", encoding="utf-8") as handle:
                kml_text = handle.read()

    risk_data_uri = _risk_overlay_uri(risk_map) if risk_map is not None else ""
    delta = 0.045
    region = hotspot_by_code(initial_region)
    regions = workstation_region_payload()
    if stats:
        regions[initial_region]["clearedHa"] = stats.get(
            "total_cleared_hectares", regions[initial_region]["clearedHa"]
        )
        regions[initial_region]["clusters"] = stats.get(
            "num_clearing_events", regions[initial_region]["clusters"]
        )
    if risk_stats:
        regions[initial_region]["highRiskHa"] = risk_stats.get(
            "high_risk_ha", regions[initial_region]["highRiskHa"]
        )
        regions[initial_region]["monitoredHa"] = risk_stats.get(
            "total_forest_ha", regions[initial_region]["monitoredHa"]
        )

    if geojson_obj is None:
        delta_box = 0.04
        geojson_obj = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "alert_id": f"WATCH-{initial_region.upper()}",
                        "severity": "Watch window",
                        "area_ha": 0,
                        "status": "MONITORING",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [center_lon - delta_box, center_lat - delta_box],
                            [center_lon + delta_box, center_lat - delta_box],
                            [center_lon + delta_box, center_lat + delta_box],
                            [center_lon - delta_box, center_lat + delta_box],
                            [center_lon - delta_box, center_lat - delta_box],
                        ]],
                    },
                }
            ],
        }

    bootstrap = {
        "center": [center_lat, center_lon],
        "zoom": zoom_start,
        "initialRegion": initial_region,
        "geojson": geojson_obj,
        "kml": kml_text,
        "riskUri": risk_data_uri,
        "riskBounds": [
            [center_lat - delta, center_lon - delta],
            [center_lat + delta, center_lon + delta],
        ],
        "regions": regions,
        "division": region["division"],
    }

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as handle:
        template = handle.read()

    html_content = template.replace("__BOOTSTRAP_JSON__", json.dumps(bootstrap))

    with open(output_html_path, "w", encoding="utf-8") as handle:
        handle.write(html_content)

    print(f"[Workstation] Built Web GIS workstation: {output_html_path}")
    return output_html_path
