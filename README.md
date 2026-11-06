# 🛰️ VANASPATI-AI: Earth Observation Forest Canopy Sentinel
### Dual-Stage Spatio-Temporal Deforestation Detection & 12-Month Predictive Risk Forecasting Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Sentinel-2](https://img.shields.io/badge/Copernicus-Sentinel--2%20L2A-005BAA.svg?logo=esa&logoColor=white)](https://sentinels.copernicus.eu/)
[![Prithvi GFM](https://img.shields.io/badge/NASA%2FIBM-Prithvi--100M-black.svg?logo=huggingface&logoColor=yellow)](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)
[![QGIS](https://img.shields.io/badge/QGIS-3.22%2B%20LTS-589632.svg?logo=qgis&logoColor=white)](https://qgis.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **An operational, dual-stage Geospatial AI platform combining 6-band Sentinel-2 Level-2A surface reflectance, the NASA/IBM Prithvi-100M Foundation Model, and Convolutional Hazard Networks to detect illegal clearings in near-real-time and project next year's deforestation frontiers.**

---

## 📌 Executive Summary

Conventional forest monitoring systems are largely **retrospective**—logging tree loss months or years after canopies are cleared. Moreover, standard satellite deep learning pipelines suffer from severe **spatial autocorrelation leakage**, where overlapping test tiles artificially inflate evaluation metrics.

**VANASPATI-AI** addresses these limitations with an enterprise dual-stage pipeline:

1. **Stage 1 (Retrospective Change Detection & GFM Benchmarking):** Ingests 6-band **Sentinel-2 Level-2A** surface reflectance ($B_2, B_3, B_4, B_8, B_{11}, B_{12}$), benchmarks against a domain **NDVI baseline**, and conducts semantic segmentation using both **ResNet-34 U-Net** and the **NASA/IBM Prithvi-100M Spatio-Temporal Vision Transformer**. Morphological connected components filter out spatial noise ($< 0.25\text{ ha}$).
2. **Stage 2 (Predictive Risk Forecasting):** Extracts spatial frontier covariates (Euclidean distance transforms to active clearings, transport road corridors, canopy edge fragmentation) and trains a **Convolutional Hazard Network** to forecast a 12-month **Deforestation Vulnerability Heatmap**.
3. **Interactive Delivery & Desktop GIS:** Features a **Figma-grade Web Workstation** with an interactive **Before/After Split Swipe Curtain**, live dynamic AI cursor probe, direct online map streaming, turnkey **QGIS project and processing algorithm**, and **ISRO Bhuvan KML** export.

---

## 🗺️ System Architecture

```mermaid
flowchart TD
    subgraph Data ["1. Multispectral Remote Sensing"]
        S2["Sentinel-2 L2A BOA Reflectance<br>6 Bands: B2, B3, B4, B8, B11, B12"]
        DW["Google Dynamic World 10m<br>Ground Truth: Trees Class"]
        S2 & DW --> Tile["256x256 Tiling & Spatial Block Partition<br>(Eastern Block Isolated for Held-Out Test)"]
    end

    subgraph Retrospective ["2. Stage 1: Retrospective Detection & Foundation ViT"]
        Tile --> NDVI["NDVI Domain Baseline<br>(NIR - Red) / (NIR + Red)"]
        Tile --> UNet["ResNet-34 U-Net<br>(6 Multispectral Channels)"]
        Tile --> Prithvi["NASA/IBM Prithvi-100M<br>(Spatio-Temporal ViT)"]
        UNet & Prithvi --> Change["Bi-Temporal Differencing: T1 & NOT T2"]
        Change --> Filter["Morphological Connected Components<br>(Filter noise < 0.25 ha / 25 px)"]
        Filter --> Alerts["Vector Deforestation Polygons (RFC 7946 GeoJSON)"]
    end

    subgraph Predictive ["3. Stage 2: Predictive Frontier Hazard Modeling"]
        Filter --> DistFrontier["Euclidean Distance Transform<br>to Active Logging Fronts"]
        S2 --> Roads["Spectral Transport Extraction<br>(SWIR > 0.20 & Low NDVI)"]
        Roads --> DistRoads["Euclidean Distance to Haul Roads"]
        Prithvi --> Embeddings["Deep Spatial Embeddings<br>(Canopy Moisture & Structural Integrity)"]
        DistFrontier & DistRoads & Embeddings --> HazardNet["Convolutional Hazard Network<br>(Spatial Risk Modeling)"]
        HazardNet --> RiskMap["12-Month Vulnerability Heatmap<br>(Imminent, High, Moderate, Low Risk)"]
    end

    subgraph Delivery ["4. Enterprise Delivery & GIS Integration"]
        Alerts & RiskMap --> WebApp["Figma-Grade Streamlit Mission Workstation<br>(app.py)"]
        Alerts & RiskMap --> SwipeMap["Interactive Split Swipe Workstation<br>(Live AI Dynamic Probe)"]
        Alerts --> QGIS["Desktop QGIS (.qgs) & Custom Tool<br>(qgis/isro_deforestation_detector.py)"]
        Alerts --> Bhuvan["ISRO Bhuvan Geoportal KML<br>(results/isro_bhuvan_deforestation_alerts.kml)"]
    end
```

---

## 📊 Held-Out Spatial Block Benchmark

Models are strictly evaluated on an **independent held-out eastern geographic block** (zero spatial leakage across coordinates):

| Model Architecture | Input Channels | Forest IoU | Forest F1-Score | Precision | Recall | Target Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **NDVI Domain Baseline** | $B_4, B_8$ (2 Bands) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Spectral Reference Baseline |
| **ResNet-34 U-Net** | 6 Multispectral Bands | **0.9832** | **0.9915** | **0.9884** | **0.9946** | Deep Spatial Feature Extractor |
| **NASA/IBM Prithvi-100M ViT** | 6 Multispectral Bands | **0.9845** | **0.9922** | **0.9892** | **0.9951** | Spatio-Temporal GFM |

---

## 🔮 12-Month Deforestation Risk Accounting

| Vulnerability Risk Tier | Clearing Probability $P(\text{Loss})$ | Forest Area Affected | Operational Conservation Response |
| :--- | :---: | :---: | :--- |
| 🔴 **Imminent Risk** | $> 80\%$ | **0.00 ha** | Immediate drone patrol & ground verification dispatch |
| 🟠 **High Risk** | $50\% - 80\%$ | **100.73 ha** | High-priority satellite revisit corridor & barrier enforcement |
| 🟡 **Moderate Risk** | $20\% - 50\%$ | **10,285.52 ha** | Buffer zone stewardship & legal concession oversight |
| 🟢 **Low Risk** | $< 20\%$ | **0.00 ha** | Core interior undisturbed primary canopy |

---

## 🌟 Key Platform Features

### 1. Interactive Split-Screen Swipe Curtain & Live Dynamic AI Probe
* **Live Dynamic Geospatial AI on Cursor Move:** Move your mouse over any pixel on the map to compute ground coordinates, Forest Canopy Density (FCD %), $\Delta\text{NDVI}$, Hazard Probability, and a dynamic 6-band spectral response bar chart in real time.
* **Before / After Split Slider:** Drag the curtain horizontally across Hasdeo Arand to visually inspect **2021 Pre-clearing Sal Forest** vs **2024 Open-Cast Coal Pits & Haul Roads**.

### 2. Desktop QGIS Integration Suite (`qgis/`)
* **Turnkey QGIS Project File (`qgis/hasdeo_arand_monitoring.qgs`):**
  - Pre-projected in **EPSG:32644 (UTM 44N)** and **EPSG:4326**.
  - Configured with Sentinel-2 **False Color Infrared (CIR Bands 8-4-3)** and rule-based vector alert symbology:
    - 🔴 **CRITICAL Alert (> 5.0 ha):** Dark Red (`#D50000`)
    - 🟠 **HIGH Alert (1.0 - 5.0 ha):** Vivid Orange (`#FF6D00`)
    - 🟡 **MODERATE Alert (< 1.0 ha):** Yellow (`#FFD600`)
* **Custom QGIS Processing Algorithm (`qgis/isro_deforestation_detector.py`):**
  - An official `QgsProcessingAlgorithm` registered directly in the QGIS Processing Toolbox under **ISRO Geospatial AI**.

### 3. National Geoportal (ISRO Bhuvan) Packaging
* Exports deforestation polygons to **ISRO Bhuvan KML** (`results/isro_bhuvan_deforestation_alerts.kml`) conforming to NRSC Bhuvan standards with `ExtendedData` telemetry tables.

---

## 📂 Repository Organization

```text
deforestation-detection/
├── app.py                                   # Figma-grade Streamlit Web GIS Command Center
├── scripts/
│   └── run_pipeline.py                      # Automated end-to-end pipeline CLI
├── qgis/
│   ├── hasdeo_arand_monitoring.qgs          # Turnkey QGIS project file
│   ├── isro_deforestation_detector.py       # Custom QGIS Processing Toolbox algorithm
│   └── README.md                            # Desktop QGIS quickstart guide
├── src/
│   ├── data/
│   │   ├── gee_download.py                  # Earth Engine regional hotspots & cloud masking
│   │   ├── gee_tile_service.py              # Dynamic GEE tile URL streamer
│   │   ├── synthetic_data.py                # Accurately projected UTM multi-spectral generator
│   │   ├── tiling.py                        # Spatial block tiler with zero spatial leakage
│   │   └── dataset.py                       # PyTorch dataset with geometric augmentations
│   ├── models/
│   │   ├── unet.py                          # ResNet-34 6-channel multispectral U-Net
│   │   ├── prithvi.py                       # NASA/IBM Prithvi-100M Spatio-Temporal ViT
│   │   ├── risk_forecaster.py               # Convolutional Hazard Net (Frontier modeling)
│   │   └── loss.py                          # Soft-Dice + BCE compound loss
│   ├── postprocessing/
│   │   ├── change_detector.py               # Bi-temporal change detection & WGS84 GeoJSON export
│   │   ├── bhuvan_export.py                 # ISRO Bhuvan standard KML exporter
│   │   └── spatial_features.py              # Euclidean distance transforms to clearings/roads
│   ├── evaluation/
│   │   └── metrics.py                       # Held-out spatial block evaluation & worst-tile analysis
│   └── visualization/
│       ├── interactive_workstation.py       # Full-screen Web GIS (Split swipe & live probe)
│       ├── map_builder.py                   # Folium interactive dual-layer map
│       └── plot_results.py                  # 6-panel retrospective & 4-panel risk figures
└── results/
    ├── deforestation_alerts.geojson         # WGS84 RFC 7946 vector alert polygons
    ├── isro_bhuvan_deforestation_alerts.kml # ISRO NRSC Bhuvan KML package
    ├── metrics_summary.json                 # Quantitative benchmark metrics
    ├── figures/                             # Publication-quality high-res figures
    └── maps/
        ├── isro_geospatial_workstation.html # Full-screen Web GIS workstation
        └── deforestation_interactive_map.html# Interactive Folium map
```

---

## 🚀 Quickstart Guide

### 1. Set Up Environment
```bash
# Clone the repository
git clone https://github.com/VivekReddy-IT/deforestation-detection.git
cd deforestation-detection

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute the Pipeline
```bash
python scripts/run_pipeline.py --region hasdeo_arand_india --force-data --epochs 3 --batch-size 4
```

### 3. Launch the Web GIS Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` to access the interactive split-screen swipe curtain, live AI cursor probe, verified incident queue, and QGIS export center.

---

## 🔭 Future Scope & Enterprise Roadmap

- **Offline Edge Tile Caching for Remote Rangers:** Packaging pre-rendered tile caches (`/tiles/{z}/{x}/{y}.png` / MBTiles) into hand-held Android field units operating in zero-cellular jungle reserves.
- **Sentinel-1 SAR Penetration:** Integrating C-Band Synthetic Aperture Radar ($VV/VH$) to maintain uninterrupted deforestation detection during heavy monsoon cloud cover.
- **UAV Micro-Canopy Orthomosaics:** Incorporating 5cm drone imagery for legal timber enforcement and tree species classification.

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
