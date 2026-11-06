# QGIS project

Open `hasdeo_arand_monitoring.qgs` in QGIS 3.22 or newer. It is set to UTM 44N and WGS84, with Esri imagery, Sentinel-2 CIR (8-4-3), and alert polygons:

- over 5 ha: red
- 1–5 ha: orange
- under 1 ha: yellow

To add the processing script: Processing Toolbox → Python → Add Script to Toolbox… → `isro_deforestation_detector.py`. It shows up under ISRO Geospatial AI.
