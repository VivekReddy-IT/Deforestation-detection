"""Experimental Sentinel-2 change screening using Google Earth Engine."""
import ee

BOUNDS = [82.73, 22.77, 82.83, 22.87]
COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'


def analyze(project, baseline, recent, baseline_ndvi=0.4):
    """Return previews and source metadata; date ranges use an exclusive end."""
    ee.Initialize(project=project)
    roi = ee.Geometry.Rectangle(BOUNDS)

    def composite(period):
        collection = (ee.ImageCollection(COLLECTION).filterBounds(roi)
                      .filterDate(*period).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)))
        info = collection.aggregate_array('system:time_start').getInfo()
        if not info:
            raise ValueError(f'No Sentinel-2 scenes found for {period[0]} to {period[1]}.')

        def mask(image):
            scl = image.select('SCL')
            valid = scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
            return image.select(['B4', 'B3', 'B2', 'B8']).divide(10000).updateMask(valid)

        return collection.map(mask).median().clip(roi), info

    before, before_dates = composite(baseline)
    after, after_dates = composite(recent)
    ndvi_before = before.normalizedDifference(['B8', 'B4'])
    ndvi_after = after.normalizedDifference(['B8', 'B4'])
    # A simple screening rule, not an independently validated loss detector.
    candidate = ndvi_before.gte(baseline_ndvi).And(ndvi_after.lt(0.4)).And(ndvi_before.subtract(ndvi_after).gte(0.2)).selfMask()
    params = {'region': roi, 'dimensions': 768, 'format': 'png'}
    rgb = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3, **params}
    vectors = candidate.toInt().rename('candidate').reduceToVectors(
        geometry=roi, crs='EPSG:32644', scale=20, geometryType='polygon',
        eightConnected=True, maxPixels=1000000,
    )
    def add_area(feature):
        return feature.set('area_ha', feature.geometry().area(maxError=1).divide(10000))
    vectors = vectors.map(add_area).filter(ee.Filter.gte('area_ha', 0.25)).sort('area_ha', False)
    total = vectors.size().getInfo()
    features = vectors.limit(100).getInfo()['features']
    for index, feature in enumerate(features, 1):
        feature['properties'].update({'candidate_id': f'CW-{index:03}', 'review': 'Unreviewed',
                                      'baseline_start': baseline[0], 'baseline_end_exclusive': baseline[1],
                                      'recent_start': recent[0], 'recent_end_exclusive': recent[1],
                                      'source': COLLECTION, 'baseline_ndvi_threshold': baseline_ndvi})
    return {'before': before.getThumbURL(rgb), 'after': after.getThumbURL(rgb),
            'change': candidate.visualize(palette=['ff3b30']).getThumbURL(params),
            'before_dates': before_dates, 'after_dates': after_dates,
            'candidates': {'type': 'FeatureCollection', 'features': features}, 'candidate_count': total}
