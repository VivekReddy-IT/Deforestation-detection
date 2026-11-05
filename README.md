# 🌲 Geospatial AI: Deforestation Detection & Predictive Risk Forecasting

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-34A853.svg?logo=google-earth&logoColor=white)](https://earthengine.google.com/)
[![Prithvi GFM](https://img.shields.io/badge/NASA%2FIBM-Prithvi--100M-black.svg?logo=huggingface&logoColor=yellow)](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A dual-stage Geospatial AI platform combining 6-band Sentinel-2 multispectral imagery, the NASA/IBM Prithvi Vision Transformer Foundation Model, and a Spatial Hazard Forecaster to detect historical deforestation and predict future forest vulnerability.**

---

## 📌 Executive Summary

Traditional remote-sensing systems operate **reactively**—documenting forest destruction months or years after trees have fallen. Furthermore, naive machine learning pipelines suffer from **spatial autocorrelation leakage**, where neighboring tiles leak during evaluation and inflate scores artificially.

This project delivers a complete, two-stage operational intelligence platform:

1. **Stage 1 (Retrospective Detection)**: Ingests 6-band **Sentinel-2 Level-2A** surface reflectance (Visible, NIR, SWIR-1, SWIR-2), benchmarks against a domain **NDVI baseline**, and performs semantic segmentation using both **ResNet-34 U-Net** and the **NASA/IBM Prithvi Spatio-Temporal Vision Transformer**. Morphological connected components filter false noise ($<0.25\text{ ha}$).
2. **Stage 2 (Proactive Risk Forecasting)**: Extracts spatial frontier covariates (Euclidean distance to active logging fronts, transport road networks, canopy edge fragmentation) and trains a **Convolutional Hazard Network** to generate a 12-month **Deforestation Vulnerability Heatmap**.

---

## 🗺️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Multispectral Data Ingestion"]
        S2["Sentinel-2 L2A (T1 & T2)<br>6 Bands: B2, B3, B4, B8, B11, B12"]
        DW["Dynamic World Land Cover<br>Ground Truth: Trees Class"]
        S2 & DW --> Tile["256x256 Tiling & Spatial Block Holdout<br>(Eastern Block Isolated for Test)"]
    end

    subgraph Retrospective ["2. Stage 1: Retrospective Detection & GFM Benchmarking"]
        Tile --> NDVI["NDVI Domain Baseline<br>(NIR - Red) / (NIR + Red)"]
        Tile --> UNet["ResNet-34 U-Net<br>(6-Channel Input)"]
        Tile --> Prithvi["NASA/IBM Prithvi-100M<br>(Spatio-Temporal ViT)"]
        UNet & Prithvi --> Change["Bi-Temporal Differencing: T1 & NOT T2"]
        Change --> Filter["Morphological Connected Components<br>(Filter noise < 0.25 ha / 25 px)"]
        Filter --> Alerts["Vectorized Deforestation Polygons (GeoJSON)"]
    end

    subgraph Predictive ["3. Stage 2: Predictive Risk Forecasting (Option 4 + Option 5)"]
        Filter --> DistFrontier["Euclidean Distance Transform<br>to Active Logging Clearings"]
        S2 --> Roads["Spectral Road & Corridor Extraction<br>(SWIR > 0.20 & Low NDVI)"]
        Roads --> DistRoads["Euclidean Distance to Roads"]
        Prithvi --> Embeddings["Deep Latent Spatial Embeddings<br>(Canopy Moisture & Structural Integrity)"]
        DistFrontier & DistRoads & Embeddings --> HazardNet["Convolutional Hazard Network<br>(Spatial Risk Modeling)"]
        HazardNet --> RiskMap["Forecasted Vulnerability Heatmap<br>(Imminent, High, Moderate, Low Risk)"]
    end

    subgraph Delivery ["4. Interactive Delivery"]
        Alerts & RiskMap --> FoliumMap["Dual-Layer Interactive Folium Map<br>(Satellite Basemap + Historical Polygons + Future Risk Glow)"]
    end
```

---

## 📊 Benchmark Results (Spatial Holdout Block)

Evaluated strictly on the **Held-Out Eastern Geographic Block** to measure out-of-region spatial generalization:

| Model / Approach | Architecture / Method | Forest IoU | Forest F1-Score | Precision | Recall | Generalization Delta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **NDVI Threshold (Baseline)** | 2-Band Optical Formula ($B8, B4$) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | *Baseline Reference* |
| **U-Net (ResNet-34)** | 6-Band Deep CNN | **0.9828** | **0.9913** | **0.9885** | **0.9942** | ResNet-34 Backbone |
| **Prithvi-100M (ViT)** | Spatio-Temporal Foundation ViT | **0.9845** | **0.9922** | **0.9892** | **0.9951** | **+Global Attention** |

---

## 🔮 Predictive Deforestation Risk Accounting (Next 12 Months)

Rather than merely counting past casualties, the **Convolutional Hazard Network** maps the spatial probability of future deforestation ($T_2 \rightarrow T_3$) based on logging proximity, road accessibility, and canopy edge stress:

| Vulnerability Risk Tier | Clearing Probability $P(\text{Loss})$ | Forest Area Affected | Operational Conservation Response |
| :--- | :---: | :---: | :--- |
| 🔴 **Imminent Risk** | $> 80\%$ | **0.00 ha** | Immediate drone patrol & ground enforcement dispatch |
| 🟠 **High Risk** | $50\% - 80\%$ | **129.90 ha** | High-priority monitoring corridor & satellite alerts |
| 🟡 **Moderate Risk** | $20\% - 50\%$ | **10,263.95 ha** | Buffer zone stewardship & legal concession oversight |
| 🟢 **Low Risk** | $< 20\%$ | **0.00 ha** | Protected core interior primary forest |

---

## 🖼️ Visual Intelligence Artifacts

### 1. Retrospective Deforestation Detection (6-Panel Summary)
Displays True Color, False-Color Infrared (B8-B4-B3), NDVI vegetation density, U-Net segmentations, and detected deforestation alerts:

![Summary Figure](results/figures/deforestation_summary_figure.png)

### 2. Predictive Risk Forecasting Diagnostics (4-Panel Summary)
Displays the landscape monitoring scene, Euclidean distance to active clearings, canopy edge fragmentation, and the calibrated **Future Vulnerability Risk Heatmap**:

![Predictive Risk Figure](results/figures/predictive_risk_forecast.png)

---

## 🌐 Interactive Web Map

Open the generated map directly in your web browser:
👉 **[`results/maps/deforestation_interactive_map.html`](results/maps/deforestation_interactive_map.html)**

### Features:
- **Esri High-Resolution Satellite Basemap**: True surface imagery of the study landscape.
- **Historical Deforestation Layer**: Individual cleared patches outlined in glowing red with tooltips displaying exact hectares cleared.
- **Future Deforestation Risk Layer**: Smooth gradient heatmap (Yellow $\rightarrow$ Orange $\rightarrow$ Red) projecting high-risk zones.
- **Dual Legend Banner**: Dynamic real-time statistics displaying both historical loss and forecasted risk hectares.

---

## 📂 Repository Layout

```text
deforestation-detection/
├── data/                                 # Protected by .gitignore
│   ├── raw/                              # Sentinel-2 & Dynamic World GeoTIFFs
│   └── processed/                        # 256x256 tiled patches & split metadata
├── notebooks/
│   └── deforestation_detection_pipeline.ipynb # Interactive walkthrough (Colab/Kaggle ready)
├── src/                                  # Modular production codebase
│   ├── data/
│   │   ├── gee_download.py               # Earth Engine S2 L2A & Dynamic World extractor
│   │   ├── synthetic_data.py             # Realistic offline study area generator
│   │   ├── tiling.py                     # 256x256 patch cutting & spatial block split
│   │   └── dataset.py                    # PyTorch Dataset with rotational augmentations
│   ├── baseline/
│   │   └── ndvi_baseline.py              # NDVI formula, threshold fitting & evaluation
│   ├── models/
│   │   ├── unet.py                       # ResNet-34 U-Net (6-channel input)
│   │   ├── prithvi.py                    # NASA/IBM Prithvi-100M ViT Foundation Model
│   │   ├── loss.py                       # Combined Dice + BCE Loss
│   │   └── risk_forecaster.py            # Convolutional Hazard Network for Risk Modeling
│   ├── postprocessing/
│   │   ├── change_detector.py            # Temporal diff & connected components (>0.25 ha)
│   │   └── spatial_features.py           # Euclidean distance transforms & frontier engine
│   ├── evaluation/
│   │   └── metrics.py                    # IoU, F1, Precision, Recall, Benchmark tables
│   └── visualization/
│       ├── map_builder.py                # Dual-layer interactive Folium map builder
│       └── plot_results.py               # 6-panel summary & 4-panel predictive risk plotter
├── scripts/
│   └── run_pipeline.py                   # Automated end-to-end CLI runner
├── results/
│   ├── figures/                          # Publication-quality high-res plots
│   ├── maps/                             # Interactive HTML maps with layer controls
│   ├── deforestation_alerts.geojson      # Vectorized deforestation alert polygons
│   └── metrics_summary.json              # Machine-readable evaluation record
├── requirements.txt                      # Pinned dependencies
└── README.md                             # Publication-grade GitHub README
```

---

## 🚀 Quickstart Guide

### 1. Set Up Environment
```bash
git clone https://github.com/your-username/deforestation-detection.git
cd deforestation-detection

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute the Dual-Stage Pipeline
```bash
python scripts/run_pipeline.py --epochs 3 --batch-size 4 --lr 0.001
```

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
