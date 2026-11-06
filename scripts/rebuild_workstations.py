"""Rebuild standalone workstation HTML maps for every hotspot."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization.hotspots import HOTSPOTS
from src.visualization.interactive_workstation import build_isro_geospatial_workstation


def main() -> None:
    metrics_path = ROOT / "results" / "metrics_summary.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    retro = metrics.get("retrospective_change_detection", {})
    risk = metrics.get("predictive_risk_forecast", {})

    for hotspot in HOTSPOTS.values():
        path = ROOT / "results" / "maps" / f"isro_workstation_{hotspot['code']}.html"
        geo = str(ROOT / "results" / "deforestation_alerts.geojson") if hotspot["code"] == "hasdeo" else None
        stats = {
            "total_cleared_hectares": hotspot["cleared_ha"],
            "num_clearing_events": hotspot["clusters"],
        }
        rstats = {
            "high_risk_ha": hotspot["high_risk_ha"],
            "total_forest_ha": hotspot["monitored_ha"],
        }
        if hotspot["code"] == "hasdeo":
            stats["total_cleared_hectares"] = retro.get("total_cleared_hectares", hotspot["cleared_ha"])
            stats["num_clearing_events"] = retro.get("num_clearing_events", hotspot["clusters"])
            rstats["high_risk_ha"] = risk.get("high_risk_ha", hotspot["high_risk_ha"])
            rstats["total_forest_ha"] = risk.get("total_forest_ha", hotspot["monitored_ha"])
        build_isro_geospatial_workstation(
            output_html_path=str(path),
            center_lat=hotspot["lat"],
            center_lon=hotspot["lon"],
            zoom_start=hotspot["zoom"],
            geojson_path=geo,
            stats=stats,
            risk_stats=rstats,
            initial_region=hotspot["code"],
        )

    build_isro_geospatial_workstation(
        output_html_path=str(ROOT / "results" / "maps" / "isro_geospatial_workstation.html"),
        geojson_path=str(ROOT / "results" / "deforestation_alerts.geojson"),
        stats={
            "total_cleared_hectares": retro.get("total_cleared_hectares", 7.74),
            "num_clearing_events": retro.get("num_clearing_events", 16),
        },
        risk_stats={
            "high_risk_ha": risk.get("high_risk_ha", 65.8),
            "total_forest_ha": risk.get("total_forest_ha", 10240),
        },
        initial_region="hasdeo",
    )


if __name__ == "__main__":
    main()
