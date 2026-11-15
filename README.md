# CanopyWatch

CanopyWatch is a prototype forest-monitoring workstation. It combines satellite
imagery, experimental vegetation-change screening, and map review tools, with
Hasdeo Arand in Chhattisgarh as its initial analysis area.

**Prototype status:** Existing workstation hover readings, alert statistics,
model scores, and risk forecasts include simulated or unvalidated values. They
are not verified operational findings. Satellite imagery consists of periodic
observations, not continuous live video.

## Real Sentinel-2 analysis

The **Real Sentinel-2 analysis** tab uses Google Earth Engine's harmonized
Sentinel-2 surface reflectance collection to compare two user-selected periods
over a fixed Hasdeo window. It masks clouds and shadows, produces median
composites, and applies an adjustable NDVI rule to identify candidate changes.

Candidates are not confirmed deforestation. Clouds, seasonal variation, and
other vegetation changes can cause errors; the screening rule is not validated.
The app displays source scene dates in UTC, and date-range ends are exclusive.
Candidate polygons of at least 0.25 ha are generated at a 20 m analysis scale;
the review list and downloads contain at most the 100 largest patches. Review
statuses and notes last for the browser session, so download the GeoJSON or CSV
to retain them.

To use this tab, register a Google Cloud project for Earth Engine, enable the
Earth Engine API, authenticate locally with `earthengine authenticate`, and
enter your project ID. Access depends on your credentials and service access.

## Workstation features

- Compare Esri imagery with a split-swipe map and explore selected landscapes.
- Display Sentinel-2 cloudless, NASA GIBS VIIRS, MODIS, HLS, OPERA SAR and
  IMERG layers. Availability depends on location and date; rain data is coarse.
- Explore rainfall and satellite imagery in the monsoon tab.
- Download candidate GeoJSON and CSV, or use the existing experimental QGIS and
  Bhuvan exports. Older map alerts and hover values are not independently verified.

The repository also contains experimental NDVI, U-Net, transformer-inspired and
risk-model code. The existing data-generation pipeline can derive synthetic
spectral bands and labels from RGB imagery; its reported scores and risk values
should not be treated as independent validation or calibrated probabilities. The
transformer implementation does not load pretrained Prithvi weights.

## Run locally

On Windows, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8502
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8502
```

Open http://127.0.0.1:8502. The provided Streamlit configuration binds to
localhost. The optional read-only tool server runs with `python -m src.mcp_server`.

## Project layout

- `app.py`: Streamlit interface.
- `src/data/`: imagery ingestion, experimental Earth Engine analysis and datasets.
- `src/models/`, `src/baseline/` and `src/postprocessing/`: experimental detection,
  forecasting and export code.
- `src/visualization/`: map workstation and monsoon views.
- `scripts/`: pipeline and workstation-generation scripts.
- `qgis/`: QGIS project and processing script.
- `results/`: existing maps, figures, GeoJSON and KML outputs.

## Next steps

Validate candidate changes against independently reviewed examples, test review
and export workflows end to end, and measure false alerts before relying on
detection or developing a calibrated forecast.
