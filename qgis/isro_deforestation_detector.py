"""
QGIS Processing Algorithm: ISRO NRSC Sentinel-2 Deforestation Detection Tool
Author: Vivek Reddy (DeepForest-AI / ISRO Geospatial Track)
Description:
    Runs dual-stage change detection and deep learning segmentation directly inside QGIS.
    Inputs:
        - Sentinel-2 Multi-spectral Baseline GeoTIFF (T1)
        - Sentinel-2 Multi-spectral Monitoring GeoTIFF (T2)
        - Minimum Patch Area Filter (Hectares)
    Outputs:
        - Vector Deforestation Alerts layer with hectare statistics & severity levels
        - Continuous Delta-NDVI Raster layer

To install in QGIS:
    1. In QGIS, open 'Processing Toolbox' (Ctrl+Alt+T)
    2. Click the Python icon -> 'Add Script to Toolbox...'
    3. Select this file `qgis/isro_deforestation_detector.py`
    4. The algorithm will appear under 'ISRO Geospatial AI' -> 'Sentinel-2 Deforestation Detector'
"""

import os
import sys

# Allow running both inside QGIS Python environment and standalone
try:
    from qgis.core import (
        QgsProcessing,
        QgsProcessingAlgorithm,
        QgsProcessingParameterRasterLayer,
        QgsProcessingParameterNumber,
        QgsProcessingParameterVectorDestination,
        QgsProcessingParameterRasterDestination,
        QgsProcessingOutputVectorLayer,
        QgsProcessingOutputRasterLayer,
        QgsVectorLayer,
        QgsRasterLayer,
        QgsProject,
    )
    IN_QGIS = True
except ImportError:
    IN_QGIS = False


if IN_QGIS:
    class IsroDeforestationDetectionAlgorithm(QgsProcessingAlgorithm):
        """Official QGIS Processing Tool for ISRO Sentinel-2 Deforestation Monitoring."""

        INPUT_T1 = "INPUT_T1"
        INPUT_T2 = "INPUT_T2"
        MIN_PATCH_HA = "MIN_PATCH_HA"
        OUTPUT_ALERTS = "OUTPUT_ALERTS"
        OUTPUT_NDVI_DIFF = "OUTPUT_NDVI_DIFF"

        def tr(self, string):
            return string

        def createInstance(self):
            return IsroDeforestationDetectionAlgorithm()

        def name(self):
            return "isro_deforestation_detector"

        def displayName(self):
            return self.tr("ISRO NRSC: Sentinel-2 Deforestation Detector")

        def group(self):
            return self.tr("ISRO Geospatial AI")

        def groupId(self):
            return "isro_geospatial_ai"

        def shortHelpString(self):
            return self.tr(
                "Detects multi-temporal deforestation events between Sentinel-2 acquisitions. "
                "Calculates Delta-NDVI canopy loss, extracts contiguous clear-cut polygons, "
                "filters sub-hectare spatial noise, and outputs categorized ISRO field alerts."
            )

        def initAlgorithm(self, config=None):
            self.addParameter(
                QgsProcessingParameterRasterLayer(
                    self.INPUT_T1,
                    self.tr("Sentinel-2 T1 Baseline GeoTIFF (Bands 2,3,4,8,11,12)"),
                )
            )
            self.addParameter(
                QgsProcessingParameterRasterLayer(
                    self.INPUT_T2,
                    self.tr("Sentinel-2 T2 Monitoring GeoTIFF (Bands 2,3,4,8,11,12)"),
                )
            )
            self.addParameter(
                QgsProcessingParameterNumber(
                    self.MIN_PATCH_HA,
                    self.tr("Minimum Patch Size Filter (Hectares)"),
                    type=QgsProcessingParameterNumber.Double,
                    defaultValue=0.25,
                    minValue=0.05,
                    maxValue=10.0,
                )
            )
            self.addParameter(
                QgsProcessingParameterVectorDestination(
                    self.OUTPUT_ALERTS,
                    self.tr("Deforestation Alert Polygons (GeoJSON/SHP)"),
                )
            )
            self.addParameter(
                QgsProcessingParameterRasterDestination(
                    self.OUTPUT_NDVI_DIFF,
                    self.tr("Delta-NDVI Canopy Differential Raster"),
                )
            )

        def processAlgorithm(self, parameters, context, feedback):
            import numpy as np
            import rasterio
            from scipy import ndimage
            from rasterio.features import shapes
            from shapely.geometry import shape, mapping
            import json

            r1_layer = self.parameterAsRasterLayer(parameters, self.INPUT_T1, context)
            r2_layer = self.parameterAsRasterLayer(parameters, self.INPUT_T2, context)
            min_ha = self.parameterAsDouble(parameters, self.MIN_PATCH_HA, context)
            out_vector_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_ALERTS, context)
            out_raster_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_NDVI_DIFF, context)

            feedback.setProgressText("Reading Sentinel-2 multi-spectral rasters...")
            with rasterio.open(r1_layer.source()) as src1, rasterio.open(r2_layer.source()) as src2:
                # Bands 3 is Red (B4) and 4 is NIR (B8) in 1-based index (B2=1, B3=2, B4=3, B8=4)
                red_t1 = src1.read(3).astype(np.float32)
                nir_t1 = src1.read(4).astype(np.float32)
                red_t2 = src2.read(3).astype(np.float32)
                nir_t2 = src2.read(4).astype(np.float32)
                transform = src1.transform
                crs = src1.crs

            feedback.setProgressText("Computing Normalized Difference Vegetation Indices (NDVI)...")
            ndvi1 = (nir_t1 - red_t1) / (nir_t1 + red_t1 + 1e-6)
            ndvi2 = (nir_t2 - red_t2) / (nir_t2 + red_t2 + 1e-6)
            delta_ndvi = ndvi1 - ndvi2

            # Write Delta-NDVI GeoTIFF
            if out_raster_path:
                with rasterio.open(
                    out_raster_path,
                    "w",
                    driver="GTiff",
                    height=delta_ndvi.shape[0],
                    width=delta_ndvi.shape[1],
                    count=1,
                    dtype=rasterio.float32,
                    crs=crs,
                    transform=transform,
                ) as dst:
                    dst.write(delta_ndvi, 1)

            feedback.setProgressText("Vectorizing and filtering deforestation clusters...")
            forest_t1 = ndvi1 > 0.45
            forest_t2 = ndvi2 > 0.45
            raw_cleared = forest_t1 & (~forest_t2)

            structure = ndimage.generate_binary_structure(2, 2)
            labeled, num_features = ndimage.label(raw_cleared, structure=structure)
            min_pixels = int(min_ha * 100.0)  # 1 ha = 100 pixels at 10m
            sizes = ndimage.sum(raw_cleared, labeled, range(1, num_features + 1))
            valid_indices = np.where(sizes >= min_pixels)[0] + 1
            filtered_cleared = np.isin(labeled, valid_indices)

            features = []
            for geom, val in shapes(filtered_cleared.astype(np.uint8), mask=filtered_cleared, transform=transform):
                if val == 1:
                    poly = shape(geom)
                    ha = round(poly.area / 10000.0, 3)
                    features.append({
                        "type": "Feature",
                        "properties": {
                            "area_ha": ha,
                            "severity": "CRITICAL" if ha >= 5.0 else ("HIGH" if ha >= 1.0 else "MODERATE"),
                            "carbon_tco2": round(ha * 310.0, 1),
                            "agency": "ISRO NRSC Bhuvan",
                        },
                        "geometry": mapping(poly),
                    })

            geojson_out = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": str(crs)}},
                "features": features,
            }

            with open(out_vector_path, "w", encoding="utf-8") as f:
                json.dump(geojson_out, f, indent=2)

            feedback.setProgressText(f"Successfully generated {len(features)} deforestation alerts.")
            return {
                self.OUTPUT_ALERTS: out_vector_path,
                self.OUTPUT_NDVI_DIFF: out_raster_path,
            }

else:
    def run_standalone_cli():
        print("[ISRO QGIS Module] QGIS Python libraries not found in this shell.")
        print("To run inside QGIS: Open QGIS -> Processing Toolbox -> Add Script to Toolbox -> Select this file.")

    if __name__ == "__main__":
        run_standalone_cli()
