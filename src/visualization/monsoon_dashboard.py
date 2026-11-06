"""Monsoon / wet-season panel. Isolated from deforestation detection."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.data.real_earth_obs import fetch_daily_climate, gibs_wms_url, summarize_wet_season


@st.cache_data(ttl="12h")
def load_climate(lat: float, lon: float, start: str, end: str, timezone: str) -> pd.DataFrame:
    return fetch_daily_climate(lat, lon, start, end, timezone)


def render_monsoon_panel(landscape: Dict[str, Any]) -> None:
    st.subheader("Monsoon and cloud-penetrating SAR")
    st.caption(
        "Optical sensors go dark under monsoon cloud. This panel uses Open-Meteo archive rainfall "
        "and NASA GIBS live snapshots (VIIRS, Harmonized Landsat-Sentinel, IMERG rain, Sentinel-1 OPERA SAR). "
        "Deforestation alerts and models are unchanged."
    )

    end = date.today() - timedelta(days=5)
    start = date(end.year - 1, 1, 1)
    wet_months = tuple(landscape.get("wet_months", (6, 9)))
    wet_label = landscape.get("wet_label", "Wet season")

    try:
        climate = load_climate(
            float(landscape["lat"]),
            float(landscape["lon"]),
            start.isoformat(),
            end.isoformat(),
            str(landscape["timezone"]),
        )
    except Exception as exc:
        st.error(f"Open-Meteo archive is unavailable right now: {exc}")
        return

    if climate.empty:
        st.warning("No climate rows returned for this window.")
        return

    stats = summarize_wet_season(climate, wet_months)
    monthly = stats["monthly"].copy()
    monthly["Month"] = monthly["Month"].astype(str)

    kpis = st.container(horizontal=True)
    kpis.metric("Wet-season rainfall", f"{stats['wet_mm']:,.0f} mm", border=True)
    kpis.metric("Dry-season rainfall", f"{stats['dry_mm']:,.0f} mm", border=True)
    kpis.metric(
        "Wet-season cloud cover",
        f"{stats['wet_cloud_pct']:.0f}%",
        delta=float(round(stats["wet_cloud_pct"] - stats["dry_cloud_pct"], 1)),
        delta_color="off",
        delta_description="vs dry months",
        border=True,
    )
    kpis.metric(
        "Peak daily rain",
        f"{stats['peak_mm']:.1f} mm",
        delta_description=stats["peak_date"],
        border=True,
    )

    st.caption(
        f"{wet_label} · {stats['wet_rainy_days']} wet-season rainy days (≥ 2.5 mm) vs "
        f"{stats['dry_rainy_days']} in the dry window · source: Open-Meteo historical API"
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("**Monthly rainfall**")
        st.bar_chart(
            monthly,
            x="Month",
            y="Rainfall mm",
            alt=f"Monthly rainfall at {landscape['name']}",
        )
    with chart_cols[1]:
        st.markdown("**Daily rainfall**")
        daily = climate.rename(columns={"time": "Date", "precipitation_sum": "Rain mm"})
        st.area_chart(
            daily,
            x="Date",
            y="Rain mm",
            alt=f"Daily precipitation at {landscape['name']}",
        )

    dry_date = landscape.get("dry_snapshot", "2025-03-12")
    wet_date = landscape.get("wet_snapshot", "2025-07-18")
    sar_date = landscape.get("sar_snapshot", "2025-07-18")
    lat, lon = float(landscape["lat"]), float(landscape["lon"])

    st.markdown("**Real NASA GIBS snapshots — dry optical vs monsoon optical vs SAR**")
    plates = [
        ("Dry-season VIIRS true color", "viirs", dry_date),
        ("Monsoon VIIRS (cloud deck)", "viirs", wet_date),
        ("Monsoon Sentinel-1 OPERA SAR", "sar", sar_date),
        ("IMERG precipitation rate", "imerg", wet_date),
        ("HLS Landsat / Sentinel-2", "hls", dry_date),
        ("MODIS Terra true color", "modis", dry_date),
    ]
    for row_start in range(0, len(plates), 3):
        cols = st.columns(3, border=True)
        for col, (title, layer, snap_date) in zip(cols, plates[row_start : row_start + 3]):
            with col:
                st.markdown(f"**{title}**")
                st.caption(f"{layer.upper()} · {snap_date} · NASA GIBS WMS")
                st.image(
                    gibs_wms_url(layer, lat, lon, snap_date),
                    width="stretch",
                    alt=f"{title} over {landscape['name']} on {snap_date}",
                )

    st.info(
        "Sentinel-1 C-band SAR keeps forest structure visible through monsoon cloud. "
        "Use it as a weather-independent watch layer. Clearing polygons still come only from the "
        "optical deforestation pipeline and are not rewritten here.",
        icon=":material/radar:",
    )
