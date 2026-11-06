# ISRO NRSC & Forest Survey of India | QGIS Integration Suite

This directory contains turnkey Desktop GIS assets designed for GIS analysts at **ISRO (NRSC Bhuvan)** and State Forest Departments (Van Vibhag).

---

## 📁 Included Assets

1. **`hasdeo_arand_monitoring.qgs`**:
   - Turnkey QGIS Project File configured in **EPSG:32644 (WGS 84 / UTM Zone 44N)** and **EPSG:4326**.
   - Pre-loaded with:
     - 🌍 **Esri High-Resolution World Imagery** (Sub-meter basemap).
     - 🛰️ **Sentinel-2 False Color Infrared (CIR Bands 8-4-3)**: Visualizes vegetation vigor in deep scarlet and bare mining pits in cyan/grey.
     - 🔴 **ISRO Deforestation Alert Polygons**: Styled with rule-based symbology:
       - **CRITICAL Alert (> 5.0 ha)**: Dark Red fill (`#D50000`)
       - **HIGH Alert (1.0 - 5.0 ha)**: Vivid Orange fill (`#FF6D00`)
       - **MODERATE Alert (< 1.0 ha)**: Yellow fill (`#FFD600`)

2. **`isro_deforestation_detector.py`**:
   - Native QGIS Processing Toolbox Algorithm (`QgsProcessingAlgorithm`).
   - Enables running dual-stage Sentinel-2 change detection directly from the QGIS GUI.

---

## 🚀 Quickstart: Opening the Project in QGIS

1. Launch **QGIS Desktop** (v3.22 LTS or newer).
2. Go to **Project > Open...** (or press `Ctrl+O`).
3. Select `qgis/hasdeo_arand_monitoring.qgs`.
4. All satellite rasters, false-color band composites, and deforestation vector layers will automatically render with pre-configured color ramps!

---

## 🛠️ Adding the Custom Algorithm to the QGIS Processing Toolbox

1. In QGIS, open the **Processing Toolbox** panel (`Ctrl+Alt+T`).
2. Click the **Python icon** at the top of the toolbox and select **Add Script to Toolbox...**.
3. Navigate to and select `qgis/isro_deforestation_detector.py`.
4. The tool is now permanently registered under **ISRO Geospatial AI > ISRO NRSC: Sentinel-2 Deforestation Detector**.
5. Double-click the algorithm to run change detection on any new satellite imagery passes!
