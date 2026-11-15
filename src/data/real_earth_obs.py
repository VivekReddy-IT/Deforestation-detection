"""Live public Earth-observation helpers.

Uses Copernicus/NASA GIBS WMS snapshots and Open-Meteo archive climate.
No synthetic rasters and no changes to the deforestation detectors.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Tuple

import pandas as pd

GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

GIBS_LAYERS = {
    "viirs": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    "modis": "MODIS_Terra_CorrectedReflectance_TrueColor",
    "hls": "HLS_S30_Nadir_BRDF_Adjusted_Reflectance",
    "sar": "OPERA_L2_Radiometric_Terrain_Corrected_SAR_Sentinel-1",
    "imerg": "IMERG_Precipitation_Rate",
    "blue_marble": "BlueMarble_NextGeneration",
}

USER_AGENT = "CanopyWatch/2.0 (forest-sentinel; educational)"


def gibs_wms_url(
    layer_key: str,
    lat: float,
    lon: float,
    date: str,
    half_deg: float = 0.14,
    size: int = 768,
) -> str:
    """NASA GIBS WMS GetMap URL for a real satellite/radar/precip snapshot."""
    layer = GIBS_LAYERS.get(layer_key, layer_key)
    south, west = lat - half_deg, lon - half_deg
    north, east = lat + half_deg, lon + half_deg
    query = urllib.parse.urlencode(
        {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": layer,
            "CRS": "EPSG:4326",
            "BBOX": f"{south:.5f},{west:.5f},{north:.5f},{east:.5f}",
            "WIDTH": str(size),
            "HEIGHT": str(size),
            "FORMAT": "image/jpeg",
            "TIME": date,
            "TRANSPARENT": "TRUE",
        }
    )
    return f"{GIBS_WMS}?{query}"


def fetch_daily_climate(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timezone: str = "auto",
) -> pd.DataFrame:
    """Daily precipitation and cloud cover from the Open-Meteo archive (ERA5-backed)."""
    zone = timezone or "auto"
    query = urllib.parse.urlencode(
        {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": start_date,
            "end_date": end_date,
            "daily": "precipitation_sum,rain_sum,cloud_cover_mean,et0_fao_evapotranspiration",
            "timezone": zone,
        }
    )
    req = urllib.request.Request(
        f"{OPEN_METEO_ARCHIVE}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            payload: Dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            raise RuntimeError(parsed.get("reason") or parsed.get("error") or str(exc)) from exc
        except json.JSONDecodeError:
            raise RuntimeError(f"Open-Meteo HTTP {exc.code}: {detail[:180]}") from exc
    if payload.get("error"):
        raise RuntimeError(str(payload.get("reason") or "Open-Meteo rejected the request"))
    daily = payload.get("daily") or {}
    if not daily.get("time"):
        return pd.DataFrame()
    frame = pd.DataFrame(daily)
    frame["time"] = pd.to_datetime(frame["time"])
    for col in ("precipitation_sum", "rain_sum", "cloud_cover_mean", "et0_fao_evapotranspiration"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["month"] = frame["time"].dt.month
    return frame


def _in_wet_season(month: int, wet_months: Tuple[int, int]) -> bool:
    start, end = wet_months
    if start <= end:
        return start <= month <= end
    return month >= start or month <= end


def summarize_wet_season(frame: pd.DataFrame, wet_months: Tuple[int, int]) -> Dict[str, Any]:
    if frame.empty:
        return {}
    wet_mask = frame["month"].map(lambda month: _in_wet_season(int(month), wet_months))
    wet = frame.loc[wet_mask]
    dry = frame.loc[~wet_mask]
    wettest = frame.loc[frame["precipitation_sum"].idxmax()] if "precipitation_sum" in frame else None
    return {
        "wet_days": int(len(wet)),
        "dry_days": int(len(dry)),
        "wet_mm": float(wet["precipitation_sum"].sum()) if not wet.empty else 0.0,
        "dry_mm": float(dry["precipitation_sum"].sum()) if not dry.empty else 0.0,
        "wet_rainy_days": int((wet["precipitation_sum"] >= 2.5).sum()) if not wet.empty else 0,
        "dry_rainy_days": int((dry["precipitation_sum"] >= 2.5).sum()) if not dry.empty else 0,
        "wet_cloud_pct": float(wet["cloud_cover_mean"].mean()) if not wet.empty else 0.0,
        "dry_cloud_pct": float(dry["cloud_cover_mean"].mean()) if not dry.empty else 0.0,
        "peak_mm": float(wettest["precipitation_sum"]) if wettest is not None else 0.0,
        "peak_date": wettest["time"].date().isoformat() if wettest is not None else "",
        "monthly": (
            frame.groupby(frame["time"].dt.to_period("M"), sort=True)["precipitation_sum"]
            .sum()
            .rename("Rainfall mm")
            .reset_index()
            .rename(columns={"time": "Month"})
        ),
    }
