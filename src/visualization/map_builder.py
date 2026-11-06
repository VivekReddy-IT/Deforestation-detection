"""
Interactive Geospatial Map Builder
Creates dynamic Leafmap / Folium interactive maps with Esri Satellite imagery,
clean vector basemaps, historical deforestation alert overlays, and self-contained
base64-encoded predictive future risk heatmaps without external tile blocking.
"""

from typing import Dict, Any, Optional
import os
import io
import json
import base64
import numpy as np
import folium
from folium import plugins
from PIL import Image
import matplotlib.pyplot as plt


def create_interactive_deforestation_map(
    center_lat: float = 22.82,
    center_lon: float = 82.78,
    zoom_start: int = 12,
    geojson_path: Optional[str] = None,
    risk_map: Optional[np.ndarray] = None,
    output_html_path: str = "results/maps/deforestation_interactive_map.html",
    stats: Optional[Dict[str, Any]] = None,
    risk_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """Creates a browser-ready interactive map with high-resolution satellite imagery

    and risk forecasting overlays, free of 403 tile usage blocks.

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
        prefer_canvas=True,
    )

    # 1. Primary Base Layer: Esri World Imagery (High-Resolution Satellite) - ALWAYS ACTIVE
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="🛰️ Esri High-Res Satellite",
        overlay=False,
        control=True,
        show=True,
        max_zoom=19,
    ).add_to(m)

    # 2. Secondary Base Layer: Esri World Street Map (Clean Topo & Roads, NO 403 blocks)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri World Street Map",
        name="🗺️ Esri Topo & Road Map",
        overlay=False,
        control=True,
        show=False,
        max_zoom=19,
    ).add_to(m)

    # 3. Third Base Layer: CartoDB Dark Matter (High-contrast dark mode for risk visualization)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="🌑 Dark Canvas (High Contrast)",
        overlay=False,
        control=True,
        show=False,
        subdomains="abcd",
        max_zoom=19,
    ).add_to(m)

    # 4. Historical Deforestation GeoJSON Vector Layer (Red with interactive popups)
    if geojson_path and os.path.exists(geojson_path):
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)

        features = geojson_data.get("features", [])
        has_features = len(features) > 0

        tooltip = None
        if has_features:
            tooltip = folium.GeoJsonTooltip(
                fields=["area_ha", "alert"],
                aliases=["Cleared Area (ha):", "Alert Level:"],
                localize=True,
                style="background-color: rgba(255, 255, 255, 0.95); font-weight: bold; border-radius: 4px; padding: 6px;",
            )

        def style_function(feature):
            return {
                "fillColor": "#FF1744",
                "color": "#D50000",
                "weight": 2.5,
                "fillOpacity": 0.70,
            }

        def highlight_function(feature):
            return {
                "fillColor": "#FFFF00",
                "color": "#FFEA00",
                "weight": 3.5,
                "fillOpacity": 0.90,
            }

        geojson_layer = folium.GeoJson(
            geojson_data,
            name="🔴 Historical Cleared Deforestation (Past Loss)",
            style_function=style_function,
            highlight_function=highlight_function if has_features else None,
            tooltip=tooltip,
        )
        geojson_layer.add_to(m)

    # 5. Predictive Risk Heatmap Layer (Embedded directly as base64 Data URI)
    if risk_map is not None:
        delta_deg = 0.045
        bounds = [
            [center_lat - delta_deg, center_lon - delta_deg],
            [center_lat + delta_deg, center_lon + delta_deg],
        ]

        norm_risk = np.clip(risk_map, 0.0, 1.0)
        colormap = plt.get_cmap("YlOrRd")
        colored_risk = colormap(norm_risk)

        # Smooth alpha transparency:
        # LOW and MODERATE background forest has alpha=0.0 (100% transparent!)
        # Only HIGH RISK (>0.45) and IMMINENT RISK (>0.75) glow in vivid fiery colors!
        alpha = np.zeros_like(norm_risk, dtype=np.float32)
        high_risk_mask = norm_risk >= 0.40
        alpha[high_risk_mask] = (norm_risk[high_risk_mask] - 0.40) / 0.60 * 0.85
        colored_risk[..., 3] = alpha

        risk_img_uint8 = (colored_risk * 255).astype(np.uint8)

        # Convert to self-contained Base64 Data URI so it never fails to load
        img_pil = Image.fromarray(risk_img_uint8)
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{img_base64}"

        folium.raster_layers.ImageOverlay(
            image=data_uri,
            bounds=bounds,
            name="🔥 Future Deforestation Risk Forecast (Next 12 Mos)",
            opacity=0.88,
            interactive=True,
            cross_origin=False,
            zindex=2,
        ).add_to(m)

    # 6. Sleek Glassmorphic Floating Legend
    ha_text = f"{stats.get('total_cleared_hectares', 'N/A')} ha" if stats else "Reported"
    events_text = f"{stats.get('num_clearing_events', 'N/A')}" if stats else "Active"
    high_text = f"{risk_stats.get('high_risk_ha', 'N/A')} ha" if risk_stats else "Forecasted"

    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 320px; 
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(8px);
        border: 2px solid #D50000;
        border-radius: 10px;
        z-index: 9999;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 12px;
        padding: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <h4 style="margin: 0; color: #B71C1C; font-size: 14px; font-weight: 700;">🛰️ Geospatial AI Monitor</h4>
            <span style="background: #FFEBEE; color: #C62828; font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: 600;">ACTIVE</span>
        </div>
        
        <div style="margin-bottom: 8px; padding: 6px; background: rgba(255, 23, 68, 0.08); border-left: 3px solid #D50000; border-radius: 3px;">
            <div style="display: flex; align-items: center; margin-bottom: 2px;">
                <span style="background-color: #FF1744; width: 12px; height: 12px; display: inline-block; margin-right: 6px; border-radius: 2px;"></span>
                <b>Historical Deforestation:</b>
            </div>
            <div style="padding-left: 18px; color: #555;">
                Cleared: <b style="color: #D50000;">{ha_text}</b> across <b>{events_text}</b> clusters
            </div>
        </div>
        
        <div style="margin-bottom: 8px; padding: 6px; background: rgba(255, 111, 0, 0.08); border-left: 3px solid #FF6F00; border-radius: 3px;">
            <div style="display: flex; align-items: center; margin-bottom: 2px;">
                <span style="background-color: #FF6F00; width: 12px; height: 12px; display: inline-block; margin-right: 6px; border-radius: 2px;"></span>
                <b>Forecasted High Risk (50-80%):</b>
            </div>
            <div style="padding-left: 18px; color: #555;">
                Vulnerable: <b style="color: #E65100;">{high_text}</b> next 12 months
            </div>
        </div>
        
        <div style="font-size: 11px; color: #666; margin-top: 6px; line-height: 1.4;">
            <b>Models:</b> NASA/IBM Prithvi-100M ViT + Spatial Hazard Net<br>
            <span style="color: #888;">Use the top-right layer control to toggle layers.</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add Layer Control & Fullscreen button
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)

    m.save(output_html_path)
    print(f"[Map] High-resolution interactive map saved to: {output_html_path}")
    return output_html_path
