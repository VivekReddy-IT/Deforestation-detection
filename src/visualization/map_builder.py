"""
Interactive Geospatial Map Builder
Creates dynamic Leafmap / Folium interactive maps with split-panel comparisons,
historical deforestation alert overlays, and predictive future risk heatmaps.
"""

from typing import Dict, Any, Optional
import os
import json
import numpy as np
import folium
from folium import plugins
import matplotlib.cm as cm


def create_interactive_deforestation_map(
    center_lat: float = -9.5,
    center_lon: float = -63.0,
    zoom_start: int = 12,
    geojson_path: Optional[str] = None,
    risk_map: Optional[np.ndarray] = None,
    output_html_path: str = "results/maps/deforestation_interactive_map.html",
    stats: Optional[Dict[str, Any]] = None,
    risk_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Creates a browser-ready interactive map with historical alerts and predictive risk layers.

    Args:
        center_lat: Latitude centroid of study region.
        center_lon: Longitude centroid of study region.
        zoom_start: Initial map zoom level.
        geojson_path: Optional path to vectorized deforestation polygons.
        risk_map: Optional 2D float array of predicted deforestation probability.
        output_html_path: Output HTML file destination.
        stats: Dictionary of historical deforestation statistics.
        risk_stats: Dictionary of future predictive risk statistics.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)

    # Initialize Base Folium Map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=None,
    )

    # Add Satellite Basemap
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri High-Res Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap Streets",
        overlay=False,
        control=True,
    ).add_to(m)

    # 1. Historical Deforestation GeoJSON Vector Layer (Red)
    if geojson_path and os.path.exists(geojson_path):
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)

        has_features = len(geojson_data.get("features", [])) > 0
        tooltip = None
        if has_features:
            tooltip = folium.GeoJsonTooltip(
                fields=["area_ha", "alert"],
                aliases=["Cleared Area (ha):", "Alert Status:"],
                localize=True,
            )

        def style_function(feature):
            return {
                "fillColor": "#FF1744",
                "color": "#D50000",
                "weight": 2,
                "fillOpacity": 0.65,
            }

        folium.GeoJson(
            geojson_data,
            name="🔴 Historical Cleared Deforestation (Past)",
            style_function=style_function,
            tooltip=tooltip,
        ).add_to(m)

    # 2. Predictive Risk Heatmap Layer (Orange/Yellow)
    if risk_map is not None:
        # Bounding box roughly corresponding to 10km x 10km study area
        delta_deg = 0.045
        bounds = [
            [center_lat - delta_deg, center_lon - delta_deg],
            [center_lat + delta_deg, center_lon + delta_deg],
        ]

        # Convert continuous risk to RGBA colormap
        norm_risk = np.clip(risk_map, 0.0, 1.0)
        import matplotlib.pyplot as plt
        colormap = plt.get_cmap("YlOrRd")
        colored_risk = colormap(norm_risk)
        # Set alpha: transparent for low risk, vivid for high/imminent risk
        colored_risk[..., 3] = np.where(norm_risk > 0.15, norm_risk * 0.75, 0.0)

        risk_img_uint8 = (colored_risk * 255).astype(np.uint8)

        # Save temporary overlay image
        temp_img_dir = os.path.join(os.path.dirname(output_html_path), "assets")
        os.makedirs(temp_img_dir, exist_ok=True)
        temp_png_path = os.path.join(temp_img_dir, "risk_overlay.png")
        from PIL import Image
        Image.fromarray(risk_img_uint8).save(temp_png_path)

        folium.raster_layers.ImageOverlay(
            image=temp_png_path,
            bounds=bounds,
            name="🔥 Future Deforestation Risk Forecast (Next 12 Mos)",
            opacity=0.8,
            interactive=True,
            cross_origin=False,
            zindex=2,
        ).add_to(m)

    # Add Floating Information / Statistics Legend
    ha_text = f"{stats.get('total_cleared_hectares', 'N/A')} ha" if stats else "Reported"
    events_text = f"{stats.get('num_clearing_events', 'N/A')}" if stats else "Active"
    imminent_text = f"{risk_stats.get('imminent_risk_ha', 'N/A')} ha" if risk_stats else "Forecasted"

    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 310px; height: 210px; 
        background-color: rgba(255, 255, 255, 0.96);
        border: 2px solid #FF6F00;
        border-radius: 8px;
        z-index: 9999;
        font-family: Arial, sans-serif;
        font-size: 12px;
        padding: 12px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
    ">
        <h4 style="margin: 0 0 6px 0; color: #B71C1C;">🛰️ Dual Geospatial AI Intelligence</h4>
        
        <div style="margin-bottom: 6px;">
            <span style="background-color: #FF1744; width: 14px; height: 14px; display: inline-block; vertical-align: middle; margin-right: 6px; border-radius: 2px;"></span>
            <b>Historical Cleared:</b> <span style="color:#D50000; font-weight:bold;">{ha_text}</span> ({events_text} patches)
        </div>
        
        <div style="margin-bottom: 6px;">
            <span style="background-color: #FF6F00; width: 14px; height: 14px; display: inline-block; vertical-align: middle; margin-right: 6px; border-radius: 2px;"></span>
            <b>Imminent Future Risk (&gt;80%):</b> <span style="color:#E65100; font-weight:bold;">{imminent_text}</span>
        </div>
        
        <hr style="border: 0; border-top: 1px solid #ddd; margin: 6px 0;">
        <p style="margin: 2px 0; font-size: 11px; color: #444;"><b>Model 1:</b> NASA/IBM Prithvi ViT (6-Band)</p>
        <p style="margin: 2px 0; font-size: 11px; color: #444;"><b>Model 2:</b> Spatial Frontier Hazard Forecaster</p>
        <p style="margin: 2px 0; font-size: 10px; color: #777;">Toggle layers via top-right control icon.</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add Layer Control & Fullscreen button
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)

    m.save(output_html_path)
    print(f"[Map] Interactive dual-layer map saved to: {output_html_path}")
    return output_html_path
