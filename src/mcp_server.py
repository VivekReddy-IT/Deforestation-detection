"""Read-only CanopyWatch MCP server (stdio JSON-RPC).

Lists landscapes, alerts, monsoon stats, and pipeline metrics. No writes, no secrets.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.real_earth_obs import fetch_daily_climate, summarize_wet_season
from src.visualization.hotspots import HOTSPOTS, hotspot_by_code

PROTOCOL = "2024-11-05"
GEOJSON_PATH = ROOT / "results" / "deforestation_alerts.geojson"
METRICS_PATH = ROOT / "results" / "metrics_summary.json"


def _public_landscape(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "code": item["code"],
        "name": item["name"],
        "coords": item["coords"],
        "lat": item["lat"],
        "lon": item["lon"],
        "division": item["division"],
        "threat": item["threat"],
        "canopy": item["canopy"],
        "flag": item.get("flag"),
        "monitored_ha": item["monitored_ha"],
        "cleared_ha": item["cleared_ha"],
        "high_risk_ha": item["high_risk_ha"],
        "wet_label": item.get("wet_label"),
    }


TOOLS = [
    {
        "name": "list_landscapes",
        "description": "List CanopyWatch forest watch corridors (India and global).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "india_only": {"type": "boolean", "description": "If true, return only Indian landscapes."}
            },
        },
    },
    {
        "name": "get_landscape",
        "description": "Get one landscape by code (hasdeo, saranda, sundarbans, kaziranga, ghats, silentvalley, satpura, tadoba, namdapha, sumatra, amazon).",
        "inputSchema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "get_deforestation_alerts",
        "description": "Return Hasdeo pipeline GeoJSON alerts (read-only). Other landscapes have watch windows only.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_monsoon_summary",
        "description": "Open-Meteo archive rainfall summary for a landscape code.",
        "inputSchema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "get_pipeline_metrics",
        "description": "Held-out block metrics from the deforestation pipeline (unchanged by monsoon tools).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    if name == "list_landscapes":
        india_only = bool(arguments.get("india_only"))
        rows = []
        for item in HOTSPOTS.values():
            if india_only and item.get("flag") != "IN":
                continue
            rows.append(_public_landscape(item))
        return rows
    if name == "get_landscape":
        return _public_landscape(hotspot_by_code(str(arguments.get("code") or "hasdeo")))
    if name == "get_deforestation_alerts":
        if not GEOJSON_PATH.exists():
            return {"features": [], "note": "No GeoJSON on disk. Run scripts/run_pipeline.py."}
        return json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    if name == "get_monsoon_summary":
        land = hotspot_by_code(str(arguments.get("code") or "hasdeo"))
        end = date.today() - timedelta(days=5)
        start = date(end.year - 1, 1, 1)
        frame = fetch_daily_climate(land["lat"], land["lon"], start.isoformat(), end.isoformat(), "auto")
        stats = summarize_wet_season(frame, tuple(land.get("wet_months", (6, 9))))
        stats.pop("monthly", None)
        stats["landscape"] = land["name"]
        stats["wet_label"] = land.get("wet_label")
        return stats
    if name == "get_pipeline_metrics":
        if not METRICS_PATH.exists():
            return {}
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    raise ValueError(f"Unknown tool: {name}")


def _result(text: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _error(text: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def handle(message: Dict[str, Any]) -> Dict[str, Any] | None:
    method = message.get("method")
    msg_id = message.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "canopywatch", "version": "2.1.0"},
            },
        }
    if method == "notifications/initialized" or method == "initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        try:
            payload = _call_tool(name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": _result(json.dumps(payload, indent=2, default=str)),
            }
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": msg_id, "result": _error(str(exc))}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if msg_id is None:
        return None
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def _read_message() -> Dict[str, Any] | None:
    headers: Dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, _, value = line.decode("utf-8").partition(":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    raw = sys.stdin.buffer.read(length)
    return json.loads(raw.decode("utf-8"))


def _write_message(payload: Dict[str, Any]) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def main() -> None:
    while True:
        message = _read_message()
        if message is None:
            break
        reply = handle(message)
        if reply is not None:
            _write_message(reply)


if __name__ == "__main__":
    main()
