# recom.pdf revision: MK/Theil-Sen, optional rainfall residuals, NDMI and explicit assumptions.
# Set RAINFALL_CSV to a CSV with year,rainfall_mm (or year,raster_path).
# Short records / absent rainfall yield provisional screening, not confirmed degradation.
# First run refreshes annual composites if the new NDMI cache is absent.
# Original map/DEM/building pipeline and editable cost formulas are retained.
"""
Springshed intervention planner: use-case geometry and location-rule screening.
Run: python springshed_intervention_planner_v3_fixed_new.py --kml /path/to/aoi.kml
OUTPUT_DIR and OPENTOPOGRAPHY_API_KEY retain their existing configuration.
A local-metre bounding rectangle is expanded 25% left/right and 25% top/bottom.
All generated files use the input KML stem as a filename prefix.
OUTPUT_DIR is the output root: AOI.kml writes to OUTPUT_DIR/AOI/;
repeat runs use AOI_002/, AOI_003/, etc. Existing results are preserved.
Each run has its own data cache; previous runs are not reused automatically.
The buffered KML is named AOI_buffered_aoi_25pct.kml inside that run folder.
Satellite, indices and terrain analysis cover this rectangle. Recommendations
are constrained by exact whole-footprint containment in the original AOI,
including holes. Annual caches are isolated by AOI/buffer identity.
Optional DEM_INPUT is a buffered native-resolution elevation raster in metres.
Optional SITING_MANIFEST supplies surveyed suitability rasters and approved fence
routes. Example manifest: {"layers":{"recharge_zone":"recharge.tif",
"stable_ground":"stable.tif"},"fence_boundary":"fence_routes.geojson"}.
Paths are relative to the manifest. Raster values: 1=suitable, 0=unsuitable,
nodata=unknown. Additional survey layer names are listed in RULES/COMMON_CHECKS.

All 16 interventions are evaluated against their use cases before ranking.
SCT/CCT and terrace corridors follow extracted DEM contours; check-dam axes
cross DEM drainage paths; roof pits search ground near buildings; fencing uses
explicit supplied protection routes. Whole footprints are checked, not only
centre pixels. Scientific use-case sources are recorded in SITING_SOURCES.
Numerical settings and original slope tables are planning assumptions.

Must have/High >=70, Good to have/Medium >=40, Better to have/Low below40.
Unknown physical/field checks cap scores at69. Known failed checks reject sites.
No eligible locations = Not recommended. Contours derived from coarse DEMs are
survey candidates, not verified construction lines. Missing geological data,
channel hydraulics and permissions are never treated as passed checks.

Outputs: ranked Excel/CSV, per-location rule audit, rejection counts, actual
layout GeoJSON, layout PNG, DEM/slope/drainage GeoTIFFs, existing NDVI/NDWI/NDMI
maps and rainfall-adjustment diagnostics. Costs remain editable assumed BOQs.
CCT/terrace/dyke previously zero dimensions now use explicit fixed survey units;
these are not approved designs. All candidate types now compete in one score-first
allocation, rejecting intersecting footprints and conflicting buffer spacing.
NDVI, NDWI and NDMI for every year are aligned to a common metric DEM grid
BEFORE trend/suitability analysis. --dem-subdivide-factor defaults to 5.
This interpolation aligns pixels; it does not increase source-data accuracy.
Roof pits require --buildings verified_buildings.geojson (Polygon/MultiPolygon).
Without supplied building footprints, automatic image-colour detections are used
only for exclusion; roof pits are omitted with an explicit rejection reason.
RAINFALL_CSV remains optional (year,rainfall_mm or year,raster_path).
Dependencies are those of the original script plus contourpy (Matplotlib dependency).
"""

import csv
import os
import time
import warnings
from io import BytesIO
from pathlib import Path

import cv2
import fiona
import geopandas as gpd
import mercantile
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
import pyproj
import pystac_client
import planetary_computer
import rasterio
import requests
import rioxarray
import xarray as xr
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from osgeo import gdal
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import shapes as rio_shapes
from rasterio.mask import mask
from rasterio.transform import from_bounds, xy as transform_xy
from rasterio.warp import calculate_default_transform, reproject
from scipy import ndimage as ndi
from scipy.stats import linregress
from shapely.geometry import shape as shp_shape
from shapely.ops import transform as shp_transform

warnings.filterwarnings("ignore")
fiona.drvsupport.supported_drivers["KML"] = "rw"

os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "15")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")


def union_of(gdf):
    return gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# The original AOI is supplied via required --kml argument.
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_DIR", "./"))
OUTPUT_DIR = OUTPUT_ROOT
AOI_OUTPUT_NAME = ""


def output_file(directory, filename):
    """Keep descriptive suffixes while identifying every product by input KML."""
    name = Path(filename).name
    return Path(directory) / (f"{AOI_OUTPUT_NAME}_{name}" if AOI_OUTPUT_NAME else name)


def configure_output_paths(kml_path):
    """Reserve a fresh AOI folder atomically; never overwrite an earlier run."""
    global AOI_OUTPUT_NAME, OUTPUT_DIR, MOSAIC_TIFF, MOSAIC_PNG, HOUSE_MASK_TIFF
    global PLAN_OUTPUT_DIR, YEARLY_GEOTIFF_DIR, YEARLY_QUICKLOOK_DIR
    import re
    stem = Path(kml_path).stem
    AOI_OUTPUT_NAME = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', stem).strip(' .') or 'AOI'
    OUTPUT_ROOT.expanduser().mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        folder = AOI_OUTPUT_NAME if number == 1 else f"{AOI_OUTPUT_NAME}_{number:03d}"
        candidate = OUTPUT_ROOT.expanduser() / folder
        try:
            candidate.mkdir()  # Exclusive creation also protects concurrent runs.
            break
        except FileExistsError:
            number += 1
    OUTPUT_DIR = candidate.resolve()
    MOSAIC_TIFF = output_file(OUTPUT_DIR, 'google_satellite_latest.tif')
    MOSAIC_PNG = output_file(OUTPUT_DIR, 'google_satellite_latest.png')
    HOUSE_MASK_TIFF = output_file(OUTPUT_DIR, 'house_layer_mask.tif')
    PLAN_OUTPUT_DIR = OUTPUT_DIR / 'intervention_plan'
    YEARLY_GEOTIFF_DIR = OUTPUT_DIR / 'yearly_outputs' / 'geotiff'
    YEARLY_QUICKLOOK_DIR = OUTPUT_DIR / 'yearly_outputs' / 'quicklook'
    print(f'Output folder: {OUTPUT_DIR}')


MOSAIC_TIFF = output_file(OUTPUT_DIR, "google_satellite_latest.tif")
MOSAIC_PNG = output_file(OUTPUT_DIR, "google_satellite_latest.png")
HOUSE_MASK_TIFF = output_file(OUTPUT_DIR, "house_layer_mask.tif")
ZOOM_LEVEL = 19
GOOGLE_SAT_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
WGS84_CRS = rasterio.crs.CRS.from_epsg(4326)  # target CRS for map display - lon/lat, matching the slope map
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

api_key = os.environ.get("OPENTOPOGRAPHY_API_KEY")
START_YEAR = 2018
END_YEAR = 2025
CLOUD_COVER_MAX = 20
MAX_SCENES_PER_YEAR = 5
MIN_SCENES_PER_YEAR = 2
NDVI_BARE_THRESHOLD = 0.35
YEARLY_GEOTIFF_DIR = OUTPUT_DIR / "yearly_outputs" / "geotiff"
YEARLY_QUICKLOOK_DIR = OUTPUT_DIR / "yearly_outputs" / "quicklook"
CONTOUR_INTERVAL = 3
PLAN_OUTPUT_DIR = OUTPUT_DIR / "intervention_plan"

VEG_PRIORITY_CLASSES = {
    1: ("Priority planting (bare & declining)", "red"),
    2: ("Planting needed (bare, stable/improving)", "orange"),
    3: ("Early intervention (vegetated, declining)", "gold"),
    4: ("Stable / healthy", "forestgreen"),
    5: ("Structure - excluded", "dimgray"),
}

# ---------------------------------------------------------------------------
# The 16 interventions per Interventions_matrix.pdf. slope/ndvi/ndwi are
# (min, max); length/width/depth are (min, max) meters; buffer_ha is the
# PDF's (min, max) required buffer AREA in hectares between two units of the
# same intervention; ndvi/ndwi = None means the PDF gives no meaningful
# range (only #10, which is building-sited rather than vegetation-sited).
# ---------------------------------------------------------------------------
INTERVENTIONS_V2 = {
    1: dict(name="Staggered Contour Trench (SCT)", slope=(10, 45), ndvi=(-0.1, 0.3), ndwi=(-0.5, -0.1), length=(2, 3), width=(0.45, 1), depth=(0.3, 0.75), buffer_ha=(0.5, 2), house_required=False),
    2: dict(name="Continuous Contour Trench (CCT)", slope=(0, 10), ndvi=(-0.1, 0.25), ndwi=(-0.5, -0.1), length=(0, 0), width=(0.3, 1), depth=(0.3, 0.75), buffer_ha=(1, 5), house_required=False),
    3: dict(name="Percolation Pit", slope=(0, 50), ndvi=(-0.1, 0.35), ndwi=(-0.4, -0.05), length=(0.3, 1), width=(0.3, 1), depth=(0.3, 1), buffer_ha=(0.1, 0.5), house_required=False),
    4: dict(name="Percolation / Small Dugout Pond", slope=(2, 15), ndvi=(-0.05, 0.3), ndwi=(-0.2, 0.1), length=(10, 20), width=(10, 15), depth=(0.9, 1.5), buffer_ha=(1, 5), house_required=False),
    5: dict(name="Subsurface Check Dam", slope=(1, 5), ndvi=(-0.05, 0.25), ndwi=(0.05, 0.3), length=(0, 0), width=(2, 6), depth=(1, 3), buffer_ha=(2, 10), house_required=False),
    6: dict(name="Loose Boulder Check Dam (LBCD)", slope=(5, 20), ndvi=(0, 0.3), ndwi=(0, 0.25), length=(3, 10), width=(1, 2), depth=(0.5, 1.5), buffer_ha=(0.5, 3), house_required=False),
    7: dict(name="Gully Plug", slope=(5, 25), ndvi=(-0.05, 0.3), ndwi=(-0.1, 0.2), length=(2, 5), width=(0.5, 1.5), depth=(0.5, 1), buffer_ha=(0.2, 1), house_required=False),
    8: dict(name="Gabion Check Dam", slope=(5, 15), ndvi=(-0.05, 0.3), ndwi=(0.05, 0.3), length=(5, 15), width=(1.5, 3), depth=(1, 2.5), buffer_ha=(2, 10), house_required=False),
    9: dict(name="Brush Wood Check Dam", slope=(2, 10), ndvi=(0.1, 0.4), ndwi=(-0.1, 0.15), length=(2, 6), width=(0.5, 1), depth=(0.5, 1), buffer_ha=(0.1, 0.5), house_required=False),
    10: dict(name="Roof Rainwater Recharge Pit", slope=(0, 5), ndvi=None, ndwi=None, length=(1, 2), width=(1, 2), depth=(1.5, 2.5), buffer_ha=(0, 0), house_required=True),
    11: dict(name="Deep Recharge Pit", slope=(10, 30), ndvi=(-0.1, 0.3), ndwi=(-0.4, -0.1), length=(1.2, 1.5), width=(1.2, 1.5), depth=(1.2, 1.5), buffer_ha=(0.5, 2), house_required=False),
    12: dict(name="Deep Recharge Pit with Fruit Trees", slope=(10, 30), ndvi=(0.1, 0.35), ndwi=(-0.4, -0.1), length=(1.2, 1.5), width=(1.2, 1.5), depth=(1.2, 1.5), buffer_ha=(0.5, 2), house_required=False),
    13: dict(name="Barbed Wire Fencing", slope=(0, 50), ndvi=(-1, 1), ndwi=(-1, 1), length=(0, 0), width=(0, 0), depth=(1.2, 1.5), buffer_ha=(0, 0), house_required=False),
    14: dict(name="Bio Fencing", slope=(0, 45), ndvi=(0.2, 0.6), ndwi=(-0.3, 0), length=(0, 0), width=(0.5, 1), depth=(0.3, 0.5), buffer_ha=(0, 0), house_required=False),
    15: dict(name="Fruit Plantation", slope=(0, 35), ndvi=(0.15, 0.45), ndwi=(-0.3, 0), length=(0.3, 1.2), width=(0.3, 1.2), depth=(0.3, 1.2), buffer_ha=(1, 5), house_required=False),
    16: dict(name="Bench Terrace", slope=(16, 33), ndvi=(0.1, 0.4), ndwi=(-0.3, -0.05), length=(0, 0), width=(0, 0), depth=(0.7, 1.5), buffer_ha=(0.5, 3), house_required=False),
}

# Rows 10/13/14 give buffer_ha = (0, 0) - the PDF places no spacing
# requirement on them (roof pits are naturally one-per-building; fencing/
# bio-fencing are continuous lines, not discrete spaced points). A hard
# floor keeps the NMS pass from placing physically-overlapping units.
MIN_BUFFER_HA_FLOOR = 0.01  # ~ 4 m radius
MIN_FOOTPRINT_M2_FLOOR = 0.25
MAX_NMS_CANDIDATES = 8000  # bounds the O(n^2) spacing pass; see plan_interventions()

_tab20 = plt.get_cmap("tab20").colors
INTERVENTION_COLORS = {iid: _tab20[i % 20] for i, iid in enumerate(INTERVENTIONS_V2)}
INTERVENTION_MARKERS = {iid: m for iid, m in zip(
    INTERVENTIONS_V2, ["^", "v", "<", ">", "s", "D", "d", "P", "*", "h", "p", "H", "1", "2", "o", "3"]
)}


# Changes below implement recom.pdf only. Practitioner slope/dimension tables,
# costs, mapping, DEM and building pipeline are retained from the attached v3.
MIN_TREND_YEARS = 3
ROBUST_TREND_YEARS = 10  # reporting rule, not a guarantee of robust inference
TREND_ALPHA = 0.05
APPLY_MATRIX_INDEX_FILTERS = False  # unvalidated index ranges are preferences
BUFFER_AREA_INTERPRETATION = os.environ.get('BUFFER_AREA_INTERPRETATION','legacy_radius')
RAINFALL_CSV = os.environ.get('RAINFALL_CSV','')
METHODOLOGY_REPORT = {}
# Planning thresholds, not validated construction or ecological thresholds.
PRIORITY_HIGH_SCORE = 70.0
PRIORITY_MEDIUM_SCORE = 40.0
ROOF_SCREENING_SCORE = 50.0  # neutral: roof suitability data are incomplete


def priority_category(score, eligible=True):
    if not eligible:
        return 'Not recommended', 'Not applicable'
    if not 0 <= PRIORITY_MEDIUM_SCORE < PRIORITY_HIGH_SCORE <= 100:
        raise ValueError('Require 0 <= medium threshold < high threshold <= 100')
    if not np.isfinite(score):
        return 'Not assessed', 'Not applicable'
    if score >= PRIORITY_HIGH_SCORE:
        return 'Must have', 'High'
    if score >= PRIORITY_MEDIUM_SCORE:
        return 'Good to have', 'Medium'
    return 'Better to have', 'Low'



def sen_mk(values, years):
    """Theil-Sen slope per actual year and tie-corrected Mann-Kendall p.
    Classical MK assumes independent observations; no autocorrelation or
    multiple-testing correction is claimed. Short series remain exploratory.
    """
    from scipy.stats import norm, theilslopes
    values=np.asarray(values,dtype=float); years=np.asarray(years,dtype=float)
    ok=np.isfinite(values)&np.isfinite(years)
    v=values[ok]; t=years[ok]; n=len(v)
    if n<MIN_TREND_YEARS: return np.nan,np.nan,n
    order=np.argsort(t); t=t[order]; v=v[order]
    if len(np.unique(t))!=n: raise ValueError('Duplicate observation years')
    if np.ptp(v)<1e-10: return 0.,1.,n
    slope=float(theilslopes(v,t)[0])
    score=sum(np.sign(v[i+1:]-v[i]).sum() for i in range(n-1))
    _,counts=np.unique(v,return_counts=True)
    variance=(n*(n-1)*(2*n+5)-sum(c*(c-1)*(2*c+5) for c in counts))/18.
    z=(score-np.sign(score))/np.sqrt(variance) if variance>0 else 0.
    return slope,float(2*norm.sf(abs(z))),n


def load_rainfall(years, base):
    """Optional CSV: year,rainfall_mm OR year,raster_path, exactly one per row.
    rainfall_mm is an explicitly AOI-uniform predictor. raster_path is an
    annual/seasonal rainfall total GeoTIFF in mm, relative to the CSV folder.
    Supply a consistent rainfall/antecedent window relevant to April-June.
    Missing years remain NaN; precipitation is never fabricated or interpolated.
    """
    if not RAINFALL_CSV: return None
    path=Path(RAINFALL_CSV)
    data=np.full((len(years),)+np.asarray(base.values).squeeze().shape,np.nan)
    lookup={int(y):i for i,y in enumerate(years)}; seen=set()
    with path.open(newline='',encoding='utf-8-sig') as f:
        for record in csv.DictReader(f):
            year=int(record['year'])
            if year in seen: raise ValueError(f'Duplicate rainfall year {year}')
            seen.add(year)
            if year not in lookup: continue
            value=record.get('rainfall_mm','').strip(); raster=record.get('raster_path','').strip()
            if value and raster: raise ValueError('Use rainfall_mm or raster_path, not both')
            if raster:
                with rioxarray.open_rasterio(path.parent/raster,masked=True) as layer:
                    rain=layer.rio.reproject_match(base,resampling=Resampling.bilinear).values.squeeze()
                if np.any(np.isfinite(rain)&(rain<0)): raise ValueError('Rainfall must be non-negative mm')
                data[lookup[year]]=rain
            elif value:
                rain=float(value)
                if not np.isfinite(rain) or rain<0: raise ValueError('Invalid rainfall_mm')
                data[lookup[year]]=rain
    return data


def analyze_index(stack, years, rainfall=None, adjust=False):
    """Raw MK/Sen and optional linear rainfall-residual MK/Sen per pixel.
    Correction codes: 0=not requested, 1=rain missing, 2=too few paired years,
    3=constant rain, 4=rain/time confounded (|r|>=0.9), 5=weak/nonpositive fit,
    6=RESTREND-style adjustment applied. Failed fits retain raw evidence with
    explicit attribution uncertainty; residual trends are not proof of causes.
    """
    stack=np.asarray(stack,dtype=float); years=np.asarray(years,dtype=float)
    if stack.ndim!=3 or len(years)!=stack.shape[0] or len(np.unique(years))!=len(years):
        raise ValueError('Expected time x y x x array with unique matching years')
    if rainfall is not None and rainfall.shape!=stack.shape: raise ValueError('Rainfall/index shape mismatch')
    shape=stack.shape[1:]
    out={k:np.full(shape,np.nan) for k in ('raw_slope','raw_p','slope','p','n','paired_n','rain_r2')}
    out['correction']=np.zeros(shape,dtype=np.uint8)
    for row,col in np.ndindex(shape):
        v=stack[:,row,col]
        raw,p,n=sen_mk(v,years)
        out['raw_slope'][row,col]=raw; out['raw_p'][row,col]=p
        out['slope'][row,col]=raw; out['p'][row,col]=p; out['n'][row,col]=n
        if not adjust: continue
        code=1
        if rainfall is not None:
            rain=rainfall[:,row,col]; ok=np.isfinite(v)&np.isfinite(rain)
            count=int(ok.sum()); out['paired_n'][row,col]=count
            code=2
            if count>=ROBUST_TREND_YEARS:
                code=3
                if np.ptp(rain[ok])>1e-8:
                    code=4
                    if abs(np.corrcoef(rain[ok],years[ok])[0,1])<.9:
                        fit=linregress(rain[ok],v[ok]); code=5
                        out['rain_r2'][row,col]=fit.rvalue**2
                        if fit.slope>0 and fit.pvalue<=TREND_ALPHA:
                            residual=v[ok]-(fit.intercept+fit.slope*rain[ok])
                            slope,p,_=sen_mk(residual,years[ok])
                            out['slope'][row,col]=slope; out['p'][row,col]=p; out['n'][row,col]=count
                            code=6
        out['correction'][row,col]=code
    # Negative, nonsignificant slopes remain provisional leads, not confirmed decline.
    out['decline']=np.isfinite(out['slope'])&(out['slope']<0)
    out['significant']=out['decline']&(out['p']<=TREND_ALPHA)
    out['long_record']=out['n']>=ROBUST_TREND_YEARS
    return out


def build_evidence(ndvi_stack, ndwi_stack, ndmi_stack, years, rainfall=None):
    vi=analyze_index(ndvi_stack,years,rainfall,adjust=True)
    wi=analyze_index(ndwi_stack,years)
    mi=analyze_index(ndmi_stack,years,rainfall,adjust=True)
    good=np.isfinite(ndwi_stack); count=good.sum(axis=0)
    history=np.divide(((ndwi_stack>0)&good).sum(axis=0),count,
                      out=np.zeros(count.shape,float),where=count>0)
    # Green-NIR water decline is valid only where historically water-like.
    water_decline=wi['decline']&(history>=.25)
    land_decline=vi['decline']|mi['decline']
    significant=vi['significant']|mi['significant']|(wi['significant']&(history>=.25))
    supported=((vi['significant']&vi['long_record']&(vi['correction']==6))|
               (mi['significant']&mi['long_record']&(mi['correction']==6))|
               (wi['significant']&wi['long_record']&(history>=.25)))
    score=40*vi['decline'].astype(float)+40*mi['decline']+20*water_decline
    score+=20*significant
    return dict(ndvi=vi,ndwi=wi,ndmi=mi,water_history=history,
                degrading=land_decline|water_decline,score=score,
                supported=supported,years=list(map(int,years)))


def export_methodology(evidence, base, out_dir):
    """Full precision diagnostics, including short records and failed adjustment."""
    import json
    out_dir=Path(PLAN_OUTPUT_DIR if out_dir is None else out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    summary=dict(years=evidence['years'],rainfall_supplied=bool(RAINFALL_CSV),
                 min_years=MIN_TREND_YEARS,long_record_threshold=ROBUST_TREND_YEARS,
                 buffer_interpretation=BUFFER_AREA_INTERPRETATION,
                 matrix_index_filters=APPLY_MATRIX_INDEX_FILTERS)
    for name in ('ndvi','ndwi','ndmi'):
        stats=evidence[name]
        summary[name]=dict(valid_pixels=int(np.isfinite(stats['slope']).sum()),
            short_record_pixels=int(((stats['n']>=MIN_TREND_YEARS)&~stats['long_record']).sum()),
            adjusted_pixels=int((stats['correction']==6).sum()),
            negative_pixels=int(stats['decline'].sum()),significant_negative_pixels=int(stats['significant'].sum()))
        for key in ('raw_slope','raw_p','slope','p','n','paired_n','rain_r2','correction'):
            layer=base.copy(data=stats[key].astype('float32')).rio.write_nodata(np.nan)
            layer.rio.to_raster(output_file(out_dir, f'{name}_{key}.tif'),compress='deflate')
    (output_file(out_dir, 'methodology_diagnostics.json')).write_text(json.dumps(summary,indent=2))
    METHODOLOGY_REPORT.clear(); METHODOLOGY_REPORT.update(summary)
    return summary


def add_methodology_sheet(wb, results):
    ws=wb.create_sheet('Methodology Review')
    ws.append(['Issue in recom.pdf','Implemented change / limitation'])
    rows=[
        ('Trend estimator','Theil-Sen slope per actual year; tie-corrected classical Mann-Kendall significance. No claim of autocorrelation or multiple-testing correction.'),
        ('Rainfall correction','RESTREND-style linear rainfall residuals only with >=10 paired years, positive significant fit, varying rainfall and |rain/time r|<0.9. Otherwise retain raw evidence and flag attribution uncertainty.'),
        ('Rainfall input','Set RAINFALL_CSV to year,rainfall_mm or year,raster_path. Totals must use a consistent local rainfall/antecedent window appropriate for April-June. Missing rainfall is never fabricated.'),
        ('Short records','3-9 years allow provisional screening, not robust attribution. >=10 years is a reporting threshold, not a guarantee. Increasing START_YEAR cannot create missing Sentinel-2 observations.'),
        ('Moisture index','NDMI=(B08-B11)/(B08+B11), with SWIR aligned to the NIR grid. Green-NIR NDWI remains open-water evidence only. Old NDWI thresholds are never reused for NDMI.'),
        ('Unvalidated thresholds','Original matrix slope/dimension values retained as practitioner assumptions. NDVI and NDWI ranges are soft preferences by default; APPLY_MATRIX_INDEX_FILTERS restores strict legacy filtering.'),
        ('SCT versus CCT','Original slope ordering retained. recom.pdf explicitly identifies this as field confirmation rather than a proven code error.'),
        ('Buffer interpretation','Default legacy_radius retains earlier sqrt(area_m2/pi) centre spacing. Optional allocation_circle: centre spacing=2*sqrt(area_m2/pi), at least footprint diameter. Both are explicit planning interpretations, not hydraulic design.'),
        ('Cost rates','Editable assumed rates and Excel formulas retained; previously zero CCT/terrace/dyke dimensions now use the explicitly documented planning segments. Rates, dimensions and generic quantities remain planning allowances, not sourced SOR or approved estimates.'),
        ('Score assumptions','40 points for negative NDVI, 40 for negative NDMI, 20 for historically water-like NDWI decline, 20 for any significant decline, 5 each for original index-range matches. Mean site score ranks alternatives; ties resolve by ID. These weights, NDWI>0 history fraction >=25%, and rainfall screening cutoffs are configurable heuristics, not validated intervention thresholds.'),
        ('Recommendation scope','Ranked first-pass priorities. Provisional negative trends can support survey, but do not establish land degradation, hydraulic connectivity, or construction suitability.'),
        ('RESTREND source','Wessels et al., Can human-induced land degradation be distinguished from effects of rainfall variability? https://researchspace.csir.co.za/server/api/core/bitstreams/93dbe38e-9c03-4b44-a0fa-7a1a43107369/content'),
        ('NDMI source','USGS: https://www.usgs.gov/landsat-missions/normalized-difference-moisture-index'),
    ]
    rows.append(('Priority categories', f'Must have / High >= {PRIORITY_HIGH_SCORE}; Good to have / Medium >= {PRIORITY_MEDIUM_SCORE}; Better to have / Low below medium. No eligible sites = Not recommended, rank 99. Labels express planning urgency, not construction approval.'))
    rows.append(('Ranking method', 'Spectral need score = evidence-plus-preference / 130 * 100. Final location score = 75% need plus 25% available-check completion, capped at69 when field checks are pending. Intervention score = mean selected location score. Locations ranked within each intervention, highest first; equal scores retain deterministic pixel order. Intervention ties use ID. No category quotas: categories may be empty. When capped scores tie, uncapped score breaks ties before coordinates/ID. Evidence confidence remains separate.'))
    rows.append(('Roof ranking', f'Qualifying roof candidates use a neutral {ROOF_SCREENING_SCORE}/100 score, independent of vegetation decline, pending roof area/runoff and adjacent-ground assessment. Roof locations now use nearby open-ground footprints, not the building centroid.'))
    for row in rows: ws.append(row)
    for key,value in METHODOLOGY_REPORT.items(): ws.append([str(key),str(value)])
    ws.append(['Intervention','Rank / evidence / remaining assumptions'])
    for r in sorted(results,key=lambda r:r.get('rank',99)):
        ws.append([r['name'],f"Rank {r.get('rank')}; {r.get('evidence_status')}; score {r.get('priority_score')}; {r.get('review_note')}"])
    from openpyxl.styles import Alignment
    ws.column_dimensions['A'].width=34; ws.column_dimensions['B'].width=115
    ws.freeze_panes='B2'
    for row in ws:
        ws.row_dimensions[row[0].row].height=65
        for cell in row: cell.alignment=Alignment(wrap_text=True,vertical='top')
    for cell in ws[1]: cell.font=Font(bold=True)


def decline_threshold(slope_map, default=-1e-3):
    """Decline means a negative slope, never the lowest positive quartile."""
    return 0.0


def _unit_footprint_m2(spec):
    l = (spec["length"][0] + spec["length"][1]) / 2.0
    w = (spec["width"][0] + spec["width"][1]) / 2.0
    return max(l * w, MIN_FOOTPRINT_M2_FLOOR)


def _buffer_radius_m(spec):
    ha = spec["buffer_ha"][0] or MIN_BUFFER_HA_FLOOR
    radius = float(np.sqrt(ha * 10000.0 / np.pi))
    if BUFFER_AREA_INTERPRETATION == 'legacy_radius':
        return radius
    if BUFFER_AREA_INTERPRETATION != 'allocation_circle':
        raise ValueError('BUFFER_AREA_INTERPRETATION must be allocation_circle or legacy_radius')
    return max(2 * radius, 2 * np.sqrt(_unit_footprint_m2(spec) / np.pi))


def filter_by_buffer(y_indices, x_indices, min_buffer_pixels):
    """Greedy spatial non-maximum suppression: keep a candidate, drop every
    later candidate within min_buffer_pixels of it. O(n^2) - bounded by the
    MAX_NMS_CANDIDATES cap in plan_interventions(). Adapted from Inter_v2.py."""
    if len(y_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    coords = np.column_stack((x_indices, y_indices)).astype("float64")
    keep_mask = np.ones(len(coords), dtype=bool)
    selected = []
    for i in range(len(coords)):
        if keep_mask[i]:
            selected.append(i)
            dists = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
            keep_mask[dists < min_buffer_pixels] = False
    selected = np.array(selected)
    return y_indices[selected], x_indices[selected]


def pixel_size_m_from_da(da):
    transform_ = da.rio.transform()
    res_x, res_y = abs(transform_.a), abs(transform_.e)
    crs = da.rio.crs
    if crs and crs.is_geographic:
        res_x *= 111320.0
        res_y *= 111320.0
    return (res_x + res_y) / 2.0


# ---------------- Research-informed spatial siting, September 2026 ----------------
# These sources support use cases. Numeric GIS screening settings below are
# transparent planning assumptions unless explicitly attributed in RULES.
SITING_SOURCES = {
 'contour': ('FAO, Watershed management: land treatment', 'https://www.fao.org/4/t0115e/T0115E0k.htm'),
 'gully': ('FAO, Gully control, specific treatment measures', 'https://www.fao.org/4/ad082e/ad082e03.htm'),
 'recharge': ('CGWB, Manual on Artificial Recharge of Ground Water, sections 6-7', 'https://cgwb.gov.in/cgwbpnm/public/uploads/documents/1679997242339591060file.pdf'),
 'roof': ('Indian Railways CAMTECH, Rain water harvesting handbook', 'https://indianrailways.gov.in/railwayboard/uploads/directorate/eff_res/camtech/Civil%20Engineering/YearWise//Handbook%20on%20Rain%20water%20harvesting%281%29.pdf'),
 'terrace': ('FAO, Continuous types of terraces', 'https://www.fao.org/4/ad083e/ad083e07.htm'),
 'trees': ('FAO, Agroforestry module', 'https://www.fao.org/sustainable-forest-management-toolbox/modules/agroforestry/en'),
 'fence': ('NRCS, Fence 382, published 2021 practice standard (not local law)', 'https://www.nrcs.usda.gov/sites/default/files/2022-09/Fence_382_NHCP_CPS_2021.pdf'),
 'spring': ('ICIMOD, Hydrogeological model for spring revival', 'https://www.icimod.org/hydrogeological-model-of-godavari-landscape-to-support-spring-revival-and-springshed-management/'),
 'flow': ('Barnes et al., Priority-Flood', 'https://arxiv.org/abs/1511.04463'),
}
# Boolean survey rasters: 1=verified suitable, 0=verified unsuitable, nodata=unknown.
# Never interpret an absent input or its nodata as a passed field check.
# JSON manifest: {"layers": {"stable_ground": "stable.tif", ...},
#                "fence_boundary": "approved_protection_boundary.geojson"}
# Each layer represents suitability for the named check, not raw measurements.
SITING_MANIFEST = os.environ.get('SITING_MANIFEST', '')
DEM_INPUT = os.environ.get('DEM_INPUT', '')  # metres elevation; buffered native DEM
SITING = dict(contour_interval_m=2.0, contour_tolerance_m=0.25,
              contour_grade_pct=2.0, cct_segment_m=30.0, terrace_segment_m=20.0,
              terrace_width_m=3.0, sct_gap_m=2.5, fence_segment_m=10.0, dyke_segment_m=20.0,
              stream_threshold_ha=0.25, pond_min_catchment_ha=0.5,
              pond_stream_distance_m=50.0, building_setback_m=5.0,
              roof_search_m=30.0, bank_relief_m=0.20, max_candidates=8000,
              provisional_score_cap=69.0)
COMMON_CHECKS = ('stable_ground', 'land_permission', 'utility_clearance')
RECHARGE_CHECKS = ('recharge_zone', 'infiltration_suitable', 'groundwater_clearance', 'clean_runoff')
RULES = {
 1: ('contour', 'Interrupted trenches on contours; alternate row phase; exclude channels.', RECHARGE_CHECKS + ('soil_depth_suitable',), ('contour','spring')),
 2: ('contour', 'Continuous contour segments; no channels or buildings along footprint.', RECHARGE_CHECKS + ('soil_depth_suitable','overflow_design'), ('contour','spring')),
 3: ('pit', 'Percolation on open ground outside channels with permeable material.', RECHARGE_CHECKS + ('soil_depth_suitable',), ('recharge','spring')),
 4: ('pond', 'Runoff-receiving ground; available footprint and infiltration; spillway needed.', RECHARGE_CHECKS + ('soil_depth_suitable','overflow_design','runoff_available'), ('recharge','spring')),
 5: ('channel', 'Valley/channel proxy only; groundwater flow and impermeable base must be mapped.', ('recharge_zone','impermeable_base','aquifer_storage','downstream_impact','groundwater_flow_alignment'), ('recharge','spring')),
 6: ('channel', 'Cross-gully boulder structure; FAO gully catchment <=20 ha; verify foundation and flood capacity.', ('channel_verified','foundation_suitable','spillway_design','downstream_impact'), ('gully',)),
 7: ('channel', 'Cross-gully plug; verify gully type, bed erosion and safe overflow.', ('channel_verified','foundation_suitable','spillway_design','gully_plug_design'), ('gully',)),
 8: ('channel', 'Cross-channel gabion; bank anchorage and hydraulic design.', ('channel_verified','foundation_suitable','spillway_design','downstream_impact'), ('gully',)),
 9: ('channel', 'Temporary brushwood control of small gullies; FAO catchment <=1 ha.', ('channel_verified','foundation_suitable','spillway_design','temporary_control_suitable'), ('gully',)),
 10: ('roof', 'Connect roof to separate nearby open-ground pit; inspect gutters and filters.', RECHARGE_CHECKS + ('roof_runoff_suitable','foundation_setback_verified'), ('roof','recharge')),
 11: ('pit', 'Deep pit requires permeable strata and groundwater separation.', RECHARGE_CHECKS + ('deep_excavation_suitable',), ('recharge','spring')),
 12: ('pit', 'Deep pit plus fruit trees; combine recharge and species/root-zone checks.', RECHARGE_CHECKS + ('deep_excavation_suitable','species_suitable'), ('recharge','trees')),
 13: ('fence', 'Use a supplied protection boundary; do not scatter posts across the AOI.', ('protection_needed','wildlife_access_design'), ('fence',)),
 14: ('fence', 'Live fence on supplied protection boundary; compatible local species.', ('protection_needed','wildlife_access_design','species_suitable'), ('fence','trees')),
 15: ('tree', 'Plant on suitable open land; species, rooting depth and water demand matter.', ('soil_depth_suitable','species_suitable','planting_land_suitable'), ('trees',)),
 16: ('contour', 'Contour corridor for terrace survey; cut/fill, soil and outlet design required.', ('soil_depth_suitable','terrace_design','overflow_design','planting_land_suitable'), ('terrace',)),
}
# Fill previously zero line dimensions with explicit, editable planning units.
# Cost quantities will therefore match each fixed-length proposed segment.
def apply_siting_dimensions():
    for iid, length, width in ((2,SITING['cct_segment_m'],None),
                               (5,SITING['dyke_segment_m'],None), (16,SITING['terrace_segment_m'],SITING['terrace_width_m']),
                               (13,SITING['fence_segment_m'],0.2),(14,SITING['fence_segment_m'],None)):
        INTERVENTIONS_V2[iid]['length'] = (length,length)
        if width is not None: INTERVENTIONS_V2[iid]['width'] = (width,width)


def route_dem(dem, dx, dy):
    """Priority-flood parent routing; steepest D8 on slopes, parent route on flats.
    Reverse flood order accumulates contributing area including the cell itself.
    Domain-edge/void inflow uncertainty propagates downstream. No inferred area
    outside the supplied buffered DEM, and no claim of measured channel flow.
    """
    import heapq
    z=np.asarray(dem,float); h,w=z.shape; valid=np.isfinite(z)
    filled=z.copy(); parent=np.full(z.size,-1,int); seen=np.zeros_like(valid)
    edge=valid & ~ndi.binary_erosion(valid,structure=np.ones((3,3)),border_value=0)
    heap=[]; order=[]
    for r,c in zip(*np.where(edge)):
        seen[r,c]=True; heapq.heappush(heap,(z[r,c],int(r*w+c)))
    neighbors=[(r,c) for r in (-1,0,1) for c in (-1,0,1) if r or c]
    while heap:
        height,i=heapq.heappop(heap); order.append(i); r,c=divmod(i,w)
        for dr,dc in neighbors:
            rr,cc=r+dr,c+dc
            if 0<=rr<h and 0<=cc<w and valid[rr,cc] and not seen[rr,cc]:
                seen[rr,cc]=True; j=rr*w+cc; parent[j]=i
                filled[rr,cc]=max(height,z[rr,cc]); heapq.heappush(heap,(filled[rr,cc],j))
    # Replace flood parents by steepest strictly lower neighbor, where available.
    for i in order:
        r,c=divmod(i,w); best=0.
        for dr,dc in neighbors:
            rr,cc=r+dr,c+dc
            if 0<=rr<h and 0<=cc<w and valid[rr,cc]:
                drop=(filled[r,c]-filled[rr,cc])/np.hypot(dr*dy,dc*dx)
                if drop>best: best=drop; parent[i]=rr*w+cc
    area=np.where(valid,dx*dy,0.).ravel(); uncertain=edge.ravel().copy()
    # Strictly lower neighbors precede their donors in the flood ordering.
    for i in reversed(order):
        j=parent[i]
        if j>=0: area[j]+=area[i]; uncertain[j]|=uncertain[i]
    vx=np.full(z.size,np.nan); vy=vx.copy()
    for i in order:
        j=parent[i]
        if j>=0:
            r,c=divmod(i,w); rr,cc=divmod(j,w)
            vx[i]=(cc-c)*dx; vy[i]=-(rr-r)*dy
    return dict(accumulation_ha=area.reshape(z.shape)/10000., filled_dem=filled,
                flow_x=vx.reshape(z.shape),flow_y=vy.reshape(z.shape),
                edge_inflow_unknown=uncertain.reshape(z.shape).astype(float))


def assert_analysis_grid(layer, reference, label='layer'):
    """Reject half-pixel offsets as well as CRS, shape and resolution mismatches."""
    if layer.dims != reference.dims or layer.shape != reference.shape or layer.rio.crs != reference.rio.crs:
        raise ValueError(f'{label}: not on the DEM analysis grid (dimensions/shape/CRS)')
    if not np.allclose(tuple(layer.rio.transform()), tuple(reference.rio.transform()), rtol=0, atol=1e-8):
        raise ValueError(f'{label}: transform/origin differs from DEM analysis grid')
    for axis in ('x','y'):
        if not np.allclose(layer[axis], reference[axis], rtol=0, atol=1e-8):
            raise ValueError(f'{label}: pixel centres differ from DEM grid')


def align_to_dem_grid(layer, reference, label='layer', categorical=False):
    if 'band' in layer.dims:
        layer=layer.squeeze('band',drop=True)
    layer=layer.transpose('y','x').astype('float32').drop_encoding().rio.write_nodata(np.nan)
    method=Resampling.nearest if categorical else Resampling.bilinear
    aligned=layer.rio.reproject_match(reference,resampling=method,nodata=np.nan)
    # Use identical coordinate labels to prevent xarray expanding an almost-equal grid.
    aligned=aligned.assign_coords(x=reference.x,y=reference.y).transpose('y','x')
    aligned=aligned.where(np.isfinite(reference)).rio.write_nodata(np.nan)
    assert_analysis_grid(aligned,reference,label)
    return aligned


def prepare_dem_analysis_grid(dem_path, analysis_gdf, factor, out_dir):
    """Project native DEM once, subdivide its pixels, and freeze the master grid."""
    import json
    from affine import Affine
    if not isinstance(factor,int) or factor < 1:
        raise ValueError('DEM subdivision factor must be a positive integer')
    with rioxarray.open_rasterio(dem_path,masked=True) as src:
        native=src.squeeze('band',drop=True).rio.reproject(analysis_gdf.estimate_utm_crs(),resampling=Resampling.bilinear).load()
    native_path=output_file(out_dir,'dem_native_metric.tif')
    native.rio.to_raster(native_path)
    tr=native.rio.transform()*Affine.scale(1/factor,1/factor)
    reference=native.rio.reproject(native.rio.crs,shape=(native.shape[0]*factor,native.shape[1]*factor),
        transform=tr,resampling=Resampling.bilinear,nodata=np.nan)
    reference=reference.rio.clip(list(analysis_gdf.to_crs(reference.rio.crs).geometry),drop=False)
    reference=reference.transpose('y','x').drop_encoding().rio.write_nodata(np.nan)
    reference.rio.to_raster(output_file(out_dir,'dem_analysis_grid.tif'))
    report=dict(crs=str(reference.rio.crs),shape=list(reference.shape),transform=list(reference.rio.transform()),
                native_resolution_m=list(map(abs,native.rio.resolution())),
                analysis_resolution_m=list(map(abs,reference.rio.resolution())),dem_subdivide_factor=factor,
                continuous_resampling='bilinear',categorical_resampling='nearest; vector buildings rasterized conservatively',
                note='Subdividing DEM/index pixels does not add terrain, spectral or building-detection accuracy.')
    output_file(out_dir,'analysis_grid.json').write_text(json.dumps(report,indent=2))
    print(f'Common DEM analysis grid: {reference.shape}, {reference.rio.resolution()} m, {reference.rio.crs}')
    return native_path,reference


def prepare_building_layer(reference, automatic_mask_path, verified_path=None):
    """Preserve vector geometry; colour detections alone cannot authorize roof pits."""
    from rasterio.features import rasterize
    from shapely.ops import unary_union
    records=[]
    if verified_path:
        buildings=gpd.read_file(verified_path)
        if buildings.crs is None: raise ValueError('Building footprints require a declared CRS')
        buildings=buildings.to_crs(reference.rio.crs)
        ids=set()
        for index,row in buildings.iterrows():
            geom=row.geometry
            if geom is None or geom.is_empty: continue
            if geom.geom_type not in ('Polygon','MultiPolygon') or not geom.is_valid:
                raise ValueError('Buildings must contain valid Polygon/MultiPolygon footprints')
            bid=str(row.get('building_id',index))
            if bid in ids: raise ValueError(f'Duplicate building_id: {bid}')
            ids.add(bid);records.append(dict(id=bid,geometry=geom))
        exclusion_geometries=[r['geometry'] for r in records]
        source='User-supplied building footprints; field verification still required'
    else:
        exclusion_geometries=[]
        if automatic_mask_path and Path(automatic_mask_path).is_file():
            with rasterio.open(automatic_mask_path) as src:
                arr=src.read(1)
                valid=src.dataset_mask()>0
                positive=(arr>0)&valid
                convert=pyproj.Transformer.from_crs(src.crs,reference.rio.crs,always_xy=True).transform
                for geom,value in rio_shapes(positive.astype('uint8'),mask=positive,transform=src.transform):
                    exclusion_geometries.append(shp_transform(convert,shp_shape(geom)))
        source='Automatic colour mask: exclusions only, not accepted as roof sources'
    union=unary_union(exclusion_geometries)
    occupied=rasterize([(g,1) for g in exclusion_geometries],out_shape=reference.shape,
                       transform=reference.rio.transform(),all_touched=True,dtype='uint8') if exclusion_geometries else np.zeros(reference.shape,'uint8')
    return occupied.astype(bool),dict(building_sources=records,building_geometry=union,building_source_note=source,
                                    building_footprints_supplied=bool(verified_path))


def allocate_nonoverlapping_candidates(pools, ctx=None):
    """Global score-first greedy allocation, using true footprints across all types.
    This is a ranked greedy plan, not a claim of globally optimal design.
    """
    from shapely.strtree import STRtree
    accepted=[]; chosen={iid:[] for iid in pools}; buildings=set(); tree=None; indexed=0; max_spacing=0.
    all_candidates=[]
    for iid,state in pools.items():
        for item in state['candidates']:
            item['iid']=iid;item['spacing']=state['spacing'];all_candidates.append(item)
    all_candidates.sort(key=lambda x:(-x['details']['location_score'],-x['details']['raw_location_score'],x['iid'],x['row'],x['col']))
    lookup={}
    occupied=np.zeros(ctx['valid'].shape,bool) if ctx is not None else None
    for item in all_candidates:
        state=pools[item['iid']];bid=item['details'].get('building_id')
        if bid is not None and bid in buildings:
            state['rejected']['alternative_for_same_building']+=1;continue
        if len(accepted)-indexed>=64:
            tree=STRtree([x['footprint'] for x in accepted]);indexed=len(accepted)
            lookup={id(x['footprint']):i for i,x in enumerate(accepted)}
        neighbours=list(accepted[indexed:])
        if tree is not None:
            query=item['footprint'].buffer(max(max_spacing,item['spacing'],1e-6))
            for hit in tree.query(query):
                index=int(hit) if isinstance(hit,(int,np.integer)) else lookup[id(hit)]
                neighbours.append(accepted[index])
        reason=None
        for old in neighbours:
            if item['footprint'].intersects(old['footprint']):
                reason='global_footprint_overlap';break
            spacing=max(item['spacing'],old['spacing'])
            if spacing>0 and item['footprint'].distance(old['footprint'])<spacing:
                reason='global_buffer_spacing';break
        cells=footprint_cells(ctx,item['footprint']) if ctx is not None else None
        if ctx is not None and cells is None: reason='global_footprint_outside_grid'
        if cells is not None and occupied[cells].any(): reason='global_analysis_cell_conflict'
        if reason:
            state['rejected'][reason]+=1;continue
        if cells is not None: occupied[cells]=True
        accepted.append(item);chosen[item['iid']].append(item)
        max_spacing=max(max_spacing,item['spacing'])
        if bid is not None: buildings.add(bid)
    return chosen


def build_siting_context(dem_path, base, house_mask):
    """Derive routing and slope on a buffered native-resolution metric DEM.
    Then align to the frozen DEM analysis grid. Upsampling never improves accuracy.
    """
    import json
    crs=pyproj.CRS(base.rio.crs)
    if not crs.is_projected or any(abs(a.unit_conversion_factor-1)>1e-8 for a in crs.axis_info[:2]):
        raise ValueError('Siting requires a projected metre grid (e.g. local UTM).')
    if dem_path is None or not Path(dem_path).exists():
        raise ValueError('Siting requires a valid DEM. Set DEM_INPUT or enable the DEM download.')
    with rioxarray.open_rasterio(dem_path,masked=True) as src:
        src=src.squeeze('band',drop=True)
        native=src.load() if src.rio.crs == base.rio.crs else src.rio.reproject(crs,resampling=Resampling.bilinear)
        native=native.drop_encoding().rio.write_nodata(np.nan)
    tr=native.rio.transform(); dx,dy=abs(tr.a),abs(tr.e)
    z=np.asarray(native.values,float)
    if min(z.shape)<3: raise ValueError('Buffered DEM must contain at least 3 x 3 cells')
    gy,gx=np.gradient(z,dy,dx)
    layers=route_dem(z,dx,dy)
    layers.update(dem=z,slope_pct=100*np.hypot(gx,gy),aspect=np.mod(np.degrees(np.arctan2(-gx,gy)),360))
    aligned={}
    for key,arr in layers.items():
        method=Resampling.bilinear if key in ('dem','slope_pct') else Resampling.nearest
        layer=align_to_dem_grid(native.copy(data=arr.astype('float32')),base,key,categorical=method==Resampling.nearest)
        aligned[key]=layer.values
    aligned.update(base=base,transform=base.rio.transform(),crs=base.rio.crs,
                   native_resolution_m=max(dx,dy),survey={},boundary_lines=[],manifest={},building_mask_available=house_mask is not None)
    shape=base.shape
    if house_mask is not None and np.asarray(house_mask).shape != shape:
        raise ValueError('Building mask must use the common DEM grid')
    aligned['house']=np.zeros(shape,bool) if house_mask is None else np.asarray(house_mask,bool)
    aligned['valid']=np.isfinite(base.values)&np.isfinite(aligned['dem'])&np.isfinite(aligned['slope_pct'])
    if not aligned['valid'].any(): raise ValueError('No valid DEM/index overlap within AOI.')
    if SITING_MANIFEST:
        path=Path(SITING_MANIFEST); manifest=json.loads(path.read_text()); aligned['manifest']=manifest
        allowed=set(COMMON_CHECKS)|{k for rule in RULES.values() for k in rule[2]}|{'channel_verified'}
        for key,filename in manifest.get('layers',{}).items():
            if key not in allowed: raise ValueError(f'Unknown survey layer: {key}')
            with rioxarray.open_rasterio(path.parent/filename,masked=True) as layer:
                a=layer.squeeze('band',drop=True).rio.reproject_match(base,resampling=Resampling.nearest).values
            if np.any(np.isfinite(a)&~np.isin(a,[0,1])): raise ValueError(f'{key} must contain 0, 1, nodata')
            aligned['survey'][key]=a
        if manifest.get('fence_boundary'):
            boundary=gpd.read_file(path.parent/manifest['fence_boundary'])
            if boundary.crs is None: raise ValueError('Fence boundary requires a declared CRS')
            for geom in boundary.to_crs(crs).geometry:
                if geom.geom_type not in ('LineString','MultiLineString'):
                    raise ValueError('Supply explicitly approved fence LineString/MultiLineString routes, not an AOI polygon')
                aligned['boundary_lines'].extend(list(geom.geoms) if hasattr(geom,'geoms') else [geom])
    inferred=(aligned['accumulation_ha']>=SITING['stream_threshold_ha'])&aligned['valid']
    verified=aligned['survey'].get('channel_verified',np.full(shape,np.nan))
    aligned['stream']=np.where(np.isfinite(verified),verified==1,inferred)&aligned['valid']
    t=aligned['transform']; pixel=(abs(t.e),abs(t.a))
    aligned['stream_distance_m']=(ndi.distance_transform_edt(~aligned['stream'],sampling=pixel)
                                  if aligned['stream'].any() else np.full(shape,np.inf))
    aligned['building_distance_m']=(ndi.distance_transform_edt(~aligned['house'],sampling=pixel)
                                    if aligned['house'].any() else np.full(shape,np.inf))
    return aligned


def xy_indices(ctx, coords):
    pts=np.asarray(coords,float); tr=ctx['transform']; inv=~tr
    cols=inv.a*pts[:,0]+inv.b*pts[:,1]+inv.c
    rows=inv.d*pts[:,0]+inv.e*pts[:,1]+inv.f
    return rows,cols


def sample_dem(ctx, coords):
    rr,cc=xy_indices(ctx,coords)
    return ndi.map_coordinates(ctx['dem'],[rr-.5,cc-.5],order=1,mode='constant',cval=np.nan,prefilter=False)


def footprint_cells(ctx, geom):
    """All touched cells, cropped window. Reject any footprint outside grid."""
    from rasterio.features import rasterize
    from affine import Affine
    bounds=geom.bounds; rr,cc=xy_indices(ctx,[[bounds[0],bounds[1]],[bounds[2],bounds[3]]])
    r0,c0=int(np.floor(min(rr))),int(np.floor(min(cc)))
    r1,c1=int(np.floor(max(rr)))+1,int(np.floor(max(cc)))+1
    h,w=ctx['valid'].shape
    if r0<0 or c0<0 or r1>h or c1>w or r0>=r1 or c0>=c1: return None
    a=rasterize([(geom,1)],out_shape=(r1-r0,c1-c0),transform=ctx['transform']*Affine.translation(c0,r0),all_touched=True)
    r,c=np.where(a==1)
    if not len(r): return None
    return r+r0,c+c0


def contour_candidates(ctx, iid):
    """Actual level-line polylines, interrupted/staggered for SCT.
    Row phase alternates by elevation band. Complex/disconnected contours still
    need survey to verify true staggered overlap; no raster-derived survey claim.
    """
    import contourpy
    from shapely.geometry import LineString
    from shapely.ops import substring
    z=ctx['dem']; tr=ctx['transform']; h,w=z.shape
    if tr.b or tr.d: raise ValueError('Contour extraction requires a north-up raster')
    x=tr.c+(np.arange(w)+.5)*tr.a; y=tr.f+(np.arange(h)+.5)*tr.e
    generator=contourpy.contour_generator(x=x,y=y,z=np.ma.masked_invalid(z))
    interval=SITING['contour_interval_m']
    if interval<=0: raise ValueError('Contour interval must be positive')
    length=float(np.mean(INTERVENTIONS_V2[iid]['length']))
    spacing=max(length+(SITING['sct_gap_m'] if iid==1 else 0),_buffer_radius_m(INTERVENTIONS_V2[iid]))
    first=np.ceil(np.nanmin(z)/interval)*interval
    for number,level in enumerate(np.arange(first,np.nanmax(z),interval)):
        for points in generator.lines(float(level)):
            if len(points)<2: continue
            # Canonical direction gives deterministic alternating row offsets.
            if tuple(points[0])>tuple(points[-1]): points=points[::-1]
            line=LineString(points)
            phase=spacing/2 if iid==1 and number%2 else 0.
            for start in np.arange(phase,max(phase,line.length-length)+1e-8,spacing):
                if start+length>line.length+1e-8: continue
                segment=substring(line,float(start),float(start+length))
                yield segment,dict(contour_elevation_m=float(level),contour_row=number,
                                   layout='Staggered contour segment' if iid==1 else 'Contour survey segment')


def site_rule_mask(ctx, iid):
    mode=RULES[iid][0]; slope=ctx['slope_pct']; lo,hi=INTERVENTIONS_V2[iid]['slope']
    mask=ctx['valid']&~ctx['house']&(slope>=lo)&(slope<=hi)
    if 'recommendation_mask' in ctx: mask &= ctx['recommendation_mask']
    mask &= ctx['building_distance_m']>=SITING['building_setback_m']
    if mode=='channel': mask &= ctx['stream']
    else: mask &= ~ctx['stream']
    if mode=='pond':
        mask &= (ctx['stream_distance_m']<=SITING['pond_stream_distance_m'])&(ctx['accumulation_ha']>=SITING['pond_min_catchment_ha'])
    if iid==6: mask &= ctx['accumulation_ha']<=20.0  # FAO boulder check-dam gully case
    if iid==9: mask &= ctx['accumulation_ha']<=1.0  # FAO brushwood small-gully case
    # Known-unsuitable field cells are excluded; unknowns remain provisional.
    for key in COMMON_CHECKS+RULES[iid][2]:
        layer=ctx['survey'].get(key)
        if layer is not None: mask &= ~(np.isfinite(layer)&(layer==0))
    return mask


def evaluate_geometry(ctx,iid,geom,details,base_score):
    """Check complete footprint, then report PASS/UNKNOWN rather than invent data."""
    from shapely.geometry import Point
    mode=RULES[iid][0]; spec=INTERVENTIONS_V2[iid]
    width=float(np.mean(spec['width']))
    footprint=geom.buffer(max(width/2,.1)) if geom.geom_type=='LineString' else geom
    # Exact vector test prevents crossing boundaries or holes even when a centre
    # pixel passes. Do not clip failed lines into misleading partial structures.
    if 'recommendation_geometry' in ctx and not ctx['recommendation_geometry'].covers(footprint):
        return None,'footprint_outside_original_AOI'
    cells=footprint_cells(ctx,footprint)
    if cells is None: return None,'footprint_outside_grid'
    rr,cc=cells
    if not np.all(ctx['valid'][rr,cc]): return None,'footprint_nodata_or_outside_AOI'
    if np.any(ctx['house'][rr,cc]): return None,'footprint_intersects_building'
    buildings=ctx.get('building_geometry')
    if buildings is not None and not buildings.is_empty:
        if footprint.intersects(buildings): return None,'exact_building_intersection'
        if footprint.distance(buildings)<SITING['building_setback_m']: return None,'exact_building_setback'
    if np.any(ctx['building_distance_m'][rr,cc]<SITING['building_setback_m']): return None,'building_setback'
    if mode!='channel' and np.any(ctx['stream'][rr,cc]): return None,'footprint_crosses_drainage'
    # Channel-bank slopes can differ from channel-bed slope. Other footprints
    # must meet the original practitioner slope limits over their whole area.
    if mode!='channel':
        lo,hi=spec['slope']
        if np.any((ctx['slope_pct'][rr,cc]<lo)|(ctx['slope_pct'][rr,cc]>hi)): return None,'footprint_slope'
    pending=[]; checked=['AOI/footprint','terrain slope','drainage position']
    if ctx.get('building_mask_available',True): checked.append('mapped building exclusion')
    else: pending.append('building mask unavailable')
    for key in COMMON_CHECKS+RULES[iid][2]:
        layer=ctx['survey'].get(key)
        values=layer[rr,cc] if layer is not None else np.array([np.nan])
        if np.any(values==0): return None,'field_failed:'+key
        if np.all(values==1): checked.append(key)
        else: pending.append(key)
    center=geom.interpolate(.5,normalized=True) if geom.geom_type=='LineString' else geom.centroid
    r,c=xy_indices(ctx,[[center.x,center.y]]); r,c=int(r[0]),int(c[0])
    survey_scale=ctx['native_resolution_m']
    if geom.geom_type=='LineString':
        n=max(3,int(np.ceil(geom.length/max(.25,min(abs(ctx['transform'].a),abs(ctx['transform'].e))/2))))
        coords=np.array([[p.x,p.y] for p in (geom.interpolate(float(t),normalized=True) for t in np.linspace(0,1,n))])
        zz=sample_dem(ctx,coords)
        if not np.all(np.isfinite(zz)): return None,'alignment_dem_nodata'
        details['alignment_length_m']=round(geom.length,3)
        if mode=='contour':
            variation=float(np.ptp(zz)); ds=np.linalg.norm(np.diff(coords,axis=0),axis=1)
            grade=float(np.max(np.abs(np.diff(zz))/np.maximum(ds,1e-8))*100)
            if variation>SITING['contour_tolerance_m'] or grade>SITING['contour_grade_pct']: return None,'contour_alignment'
            details.update(elevation_range_m=round(variation,3),max_alignment_grade_pct=round(grade,3))
            checked.append('DEM contour alignment')
            pending.append('field contour levels and outlets')
            if iid==1: pending.append('field staggered-row overlap')
        if mode=='channel':
            v=np.array([ctx['flow_x'][r,c],ctx['flow_y'][r,c]])
            axis=coords[-1]-coords[0]
            if not np.isfinite(v).all() or np.linalg.norm(v)==0: return None,'flow_direction_unknown'
            deviation=float(np.degrees(np.arcsin(np.clip(abs(np.dot(axis,v))/(np.linalg.norm(axis)*np.linalg.norm(v)),0,1))))
            if deviation>10: return None,'not_cross_channel'
            details['cross_flow_deviation_deg']=round(deviation,2)
            checked.append('perpendicular to DEM flow')
            if iid!=5 and min(zz[0],zz[-1])-zz[len(zz)//2]<SITING['bank_relief_m']:
                # A coarse DEM cannot resolve small banks: keep only as provisional.
                if survey_scale<=min(geom.length/5, max(width/2,.1)):
                    return None,'no_bank_anchorage_in_resolved_DEM'
                pending.append('channel banks not resolved by DEM')
            pending.append('hydraulic spacing/backwater and flood design')
    else: details['alignment_length_m']=0.
    if mode in ('channel','pond') and ctx['edge_inflow_unknown'][r,c]>0:
        pending.append('catchment extends beyond DEM/void boundary')
    channel_survey=ctx['survey'].get('channel_verified')
    if channel_survey is None or not np.all(np.isfinite(channel_survey[rr,cc])):
        pending.append('drainage inferred from DEM; verify channels')
    # Raster interpolation does not establish metre-scale construction accuracy.
    shortest=max(.1,min(float(np.mean(spec['length'])),max(width,.1)))
    if survey_scale>shortest/3: pending.append('DEM too coarse for construction footprint')
    pending=list(dict.fromkeys(pending))
    compliance=100*len(checked)/(len(checked)+len(pending))
    raw_score=float(np.clip(.75*base_score+.25*compliance,0,100))
    score=min(raw_score,SITING['provisional_score_cap']) if pending else raw_score
    category,priority=priority_category(score)
    details.update(location_score=round(score,2),raw_location_score=round(raw_score,2),
        location_category=category,location_priority=priority,rule_check_score=round(compliance,2),
        spatial_status='Provisional - field checks pending' if pending else 'Available spatial checks passed',
        passed_checks='; '.join(checked),pending_checks='; '.join(pending),
        slope_pct=round(float(ctx['slope_pct'][r,c]),2),
        contributing_area_ha=round(float(ctx['accumulation_ha'][r,c]),3),
        native_dem_resolution_m=round(float(survey_scale),2),
        evidence_status='Field validation required' if pending else 'Spatial screening passed; confirm design',
        source_urls=' | '.join(SITING_SOURCES[key][1] for key in RULES[iid][3]))
    return dict(geometry=geom,footprint=footprint,center=center,row=r,col=c,details=details),None


def plan_interventions(latest_ndvi_da, latest_ndwi_da, ndvi_slope_map, ndwi_slope_map,
                       slope_pct_arr, house_mask_bool, out_dir=None, evidence=None, siting=None):
    """Generate geometry, reject known rule failures, rank surviving survey sites.
    Requires spatial context; no silent fallback to the former pixel-only model.
    """
    from shapely.geometry import Point,LineString,box
    from shapely.ops import substring
    from collections import Counter
    if siting is None: raise ValueError('Pass build_siting_context(...) as siting; pixel-only siting is disabled.')
    apply_siting_dimensions(); ctx=siting; out_dir=Path(PLAN_OUTPUT_DIR if out_dir is None else out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    vi=np.asarray(latest_ndvi_da.values); wi=np.asarray(latest_ndwi_da.values)
    need=evidence['score'] if evidence is not None else 40*(np.asarray(ndvi_slope_map)<0)
    degrading=evidence['degrading'] if evidence is not None else np.asarray(ndvi_slope_map)<0
    results=[]; points_by_id={}; audits=[]; pools={}
    assert_analysis_grid(latest_ndvi_da,ctx["base"],"NDVI")
    assert_analysis_grid(latest_ndwi_da,ctx["base"],"NDWI")
    for iid,spec in INTERVENTIONS_V2.items():
        mode=RULES[iid][0]; mask=site_rule_mask(ctx,iid); rejected=Counter(); candidates=[]
        # Channel conservation and fencing have use cases independent of a local
        # spectral decline (erosion in an upstream catchment or protection need).
        if mode not in ('channel','roof','fence'): mask &= degrading
        if APPLY_MATRIX_INDEX_FILTERS:
            for arr,key in ((vi,'ndvi'),(wi,'ndwi')):
                if spec[key] is not None:
                    lo,hi=spec[key]; mask &= (arr>=lo)&(arr<=hi)
        preference=np.zeros(mask.shape)
        for arr,key in ((vi,'ndvi'),(wi,'ndwi')):
            if spec[key] is not None:
                lo,hi=spec[key]; preference+=5*((arr>=lo)&(arr<=hi))
        scores=np.clip((need+preference)/130*100,0,100)
        if mode in ('channel','fence'): scores=np.maximum(scores,40.)
        if mode=='roof': scores=np.full(mask.shape,ROOF_SCREENING_SCORE)
        length=float(np.mean(spec['length'])); width=float(np.mean(spec['width']))
        def pixel_center(r,c):
            x,y=ctx['transform']*(float(c)+.5,float(r)+.5); return Point(x,y)
        def shapes():
            if mode=='contour':
                yield from contour_candidates(ctx,iid); return
            if mode=='fence':
                if not ctx['boundary_lines']: rejected['missing_approved_fence_boundary']+=1
                for line in ctx['boundary_lines']:
                    for d in np.arange(0,max(0,line.length-SITING['fence_segment_m'])+1e-8,SITING['fence_segment_m']):
                        if d+SITING['fence_segment_m']<=line.length:
                            yield substring(line,float(d),float(d+SITING['fence_segment_m'])),dict(layout='Approved boundary segment')
                return
            if mode=='roof':
                sources=ctx.get('building_sources',[])
                if not sources:
                    rejected['missing_supplied_building_footprints']+=1;return
                to_wgs=pyproj.Transformer.from_crs(ctx['crs'],'EPSG:4326',always_xy=True).transform
                from shapely.ops import nearest_points
                for source in sources:
                    building=source['geometry'];b=source['id']
                    if 'recommendation_geometry' in ctx and not building.intersects(ctx['recommendation_geometry']):
                        rejected['roof_source_outside_original_AOI']+=1;continue
                    rr,cc=np.where(mask); nearby=[]
                    for r,c in zip(rr,cc):
                        p=pixel_center(r,c);distance=p.distance(building)
                        if not (SITING['building_setback_m']<=distance<=SITING['roof_search_m']):continue
                        geom=box(p.x-length/2,p.y-width/2,p.x+length/2,p.y+width/2)
                        if geom.distance(building)<SITING['building_setback_m']:continue
                        nearby.append((distance,float(ctx['slope_pct'][r,c]),int(r),int(c),geom))
                    if not nearby: rejected['building_no_nearby_ground']+=1;continue
                    for distance,_,r,c,geom in sorted(nearby,key=lambda x:x[:4])[:50]:
                        roof_point,pit_point=nearest_points(building,geom)
                        lon,lat=to_wgs(roof_point.x,roof_point.y)
                        plon,plat=to_wgs(pit_point.x,pit_point.y)
                        yield geom,dict(layout='Ground pit connected to supplied building footprint',building_id=b,
                            roof_distance_m=round(distance,2),roof_source_longitude=lon,roof_source_latitude=lat,
                            pit_connection_longitude=plon,pit_connection_latitude=plat,
                            building_source=ctx.get('building_source_note','Supplied footprints'))
                return
            rr,cc=np.where(mask); order=np.argsort(-scores[rr,cc],kind='stable')
            if len(order)>SITING['max_candidates']: rejected['candidate_cap_unexamined']+=len(order)-SITING['max_candidates']
            for j in order[:SITING['max_candidates']]:
                r,c=rr[j],cc[j]; p=pixel_center(r,c)
                if mode=='channel':
                    flow=np.array([ctx['flow_x'][r,c],ctx['flow_y'][r,c]])
                    if not np.isfinite(flow).all() or np.linalg.norm(flow)==0:
                        rejected['flow_direction_unknown']+=1; continue
                    axis=np.array([-flow[1],flow[0]])/np.linalg.norm(flow)*length/2
                    yield LineString([(p.x-axis[0],p.y-axis[1]),(p.x+axis[0],p.y+axis[1])]),dict(layout='Cross-channel survey axis')
                else:
                    yield box(p.x-length/2,p.y-width/2,p.x+length/2,p.y+width/2),dict(layout='Ground footprint')
        examined=0
        for geom,details in shapes():
            examined+=1
            if examined>SITING['max_candidates']:
                rejected['geometry_cap_reached']+=1; break
            center=geom.interpolate(.5,normalized=True) if geom.geom_type=='LineString' else geom.centroid
            rr,cc=xy_indices(ctx,[[center.x,center.y]]); r,c=int(rr[0]),int(cc[0])
            if not (0<=r<mask.shape[0] and 0<=c<mask.shape[1]) or not mask[r,c]:
                rejected['center_use_case_or_slope']+=1; continue
            item,reason=evaluate_geometry(ctx,iid,geom,details,float(scores[r,c]))
            if reason: rejected[reason]+=1; continue
            # Keep statistical uncertainty independent from physical compliance.
            trend_supported=evidence is not None and bool(evidence['supported'][r,c])
            item['details']['trend_status']='Supported trend' if trend_supported else 'Provisional / no local trend requirement'
            item['details']['trend_supported']=trend_supported
            if mode not in ('roof','channel','fence') and not trend_supported:
                d=item['details']; d['location_score']=min(d['location_score'],SITING['provisional_score_cap'])
                d['location_category'],d['location_priority']=priority_category(d['location_score'])
            candidates.append(item)
        if not mask.any(): rejected['no_pixels_pass_use_case_slope_or_need']+=1
        if examined==0 and mode=='contour': rejected['no_contour_segment_of_required_length']+=1
        spacing=_buffer_radius_m(spec) if mode not in ('roof','fence') else 0.
        pools[iid]=dict(candidates=candidates,rejected=rejected,spacing=spacing,
                        examined=examined,eligible_pixels=int(mask.sum()))
    allocations=allocate_nonoverlapping_candidates(pools,ctx)
    for iid,spec in INTERVENTIONS_V2.items():
        state=pools[iid];selected=allocations[iid];rejected=state['rejected']
        spacing=state['spacing'];examined=state['examined']
        metadata=[]; geoms=[]; rows=[]; cols=[]
        for rank,item in enumerate(selected,1):
            details=item['details']; details.update(site_id=f'I{iid:02d}-S{rank:05d}',location_rank=rank)
            # Fractional row/col recover the geometry centre exactly in exports.
            rr,cc=xy_indices(ctx,[[item['center'].x,item['center'].y]])
            rows.append(rr[0]-.5); cols.append(cc[0]-.5)
            metadata.append(details); geoms.append(item['geometry'])
        count=len(selected); mean=round(float(np.mean([d['location_score'] for d in metadata])),2) if count else 0.
        category,level=priority_category(mean,count>0)
        record=dict(id=iid,name=spec['name'],eligible_pixels=state["eligible_pixels"],recommended_units=count,
                    unit_footprint_m2=round(_unit_footprint_m2(spec),2),
                    total_area_m2=round(sum(item['footprint'].area for item in selected),2),
                    buffer_spacing_m=round(spacing,2),house_required=bool(spec['house_required']),
                    priority_score=mean,raw_priority_score=round(float(np.mean([d['raw_location_score'] for d in metadata])),2) if count else 0.,
                    priority_category=category,priority_level=level,
                    supported_sites=sum(d['trend_supported'] for d in metadata),
                    high_priority_sites=sum(d['location_priority']=='High' for d in metadata),
                    medium_priority_sites=sum(d['location_priority']=='Medium' for d in metadata),
                    low_priority_sites=sum(d['location_priority']=='Low' for d in metadata),
                    evidence_status='No eligible geometry' if not count else 'Survey candidates; inspect location audit',
                    review_note='Coordinated score-first allocation; no intersecting footprints or conflicting buffers across intervention types. Field checks remain mandatory.',
                    rejection_summary='; '.join(f'{k}: {v}' for k,v in sorted(rejected.items())),
                    _site_priorities=metadata,_site_geometries=geoms)
        results.append(record); points_by_id[iid]=(np.array(rows),np.array(cols))
        audits.append(dict(id=iid,name=spec['name'],use_case=RULES[iid][1],examined=examined,
                           selected=count,rejections=record['rejection_summary']))
    for rank,r in enumerate(sorted(results,key=lambda r:(-bool(r['recommended_units']),-r['priority_score'],-r['raw_priority_score'],r['id'])),1):
        r['rank']=rank if r['recommended_units'] else 99
    with (output_file(out_dir, 'intervention_plan.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in results[0] if not k.startswith('_')],extrasaction='ignore')
        writer.writeheader();writer.writerows(results)
    export_site_priority_csv(results,points_by_id,latest_ndvi_da,output_file(out_dir, 'intervention_locations_ranked.csv'))
    export_siting_outputs(ctx,results,audits,out_dir)
    for r in sorted(results,key=lambda r:(r['rank'],r['id'])):
        print(f"{r['rank']:>2} {r['name']}: {r['priority_category']} / {r['priority_level']}; {r['recommended_units']} sites")
    return results,points_by_id,output_file(out_dir, 'intervention_plan.csv')


def export_siting_outputs(ctx,results,audits,out_dir):
    import json
    from shapely.geometry import mapping
    to_wgs=pyproj.Transformer.from_crs(ctx['crs'],'EPSG:4326',always_xy=True).transform
    features=[]
    for r in results:
        for geom,details in zip(r['_site_geometries'],r['_site_priorities']):
            features.append(dict(type='Feature',geometry=mapping(shp_transform(to_wgs,geom)),
                                 properties=dict(intervention_id=r['id'],intervention=r['name'],**details)))
    roof_sources=[];roof_links=[]
    from shapely.geometry import LineString
    for source in ctx.get('building_sources',[]):
        roof_sources.append(dict(type='Feature',geometry=mapping(shp_transform(to_wgs,source['geometry'])),
                                 properties=dict(building_id=source['id'])))
    for r in results:
        if r['id']!=10:continue
        for d in r['_site_priorities']:
            coords=[(d['roof_source_longitude'],d['roof_source_latitude']),
                    (d['pit_connection_longitude'],d['pit_connection_latitude'])]
            roof_links.append(dict(type='Feature',geometry=mapping(LineString(coords)),
                                   properties=dict(site_id=d['site_id'],building_id=d['building_id'],
                                                   note='Source association only; gutter/pipe routing requires design')))
    output_file(out_dir,'roof_source_buildings.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=roof_sources),indent=2))
    output_file(out_dir,'roof_to_pit_connections.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=roof_links),indent=2))
    (output_file(out_dir, 'intervention_layouts.geojson')).write_text(json.dumps(dict(type='FeatureCollection',features=features),indent=2))
    with (output_file(out_dir, 'siting_rejections.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(audits[0]));writer.writeheader();writer.writerows(audits)
    from rasterio.features import rasterize
    priority_dir=out_dir/'location_priority_tifs';priority_dir.mkdir(exist_ok=True)
    for r in results:
        items=[]
        for geom,d in zip(r['_site_geometries'],r['_site_priorities']):
            footprint=geom.buffer(max(.1,float(np.mean(INTERVENTIONS_V2[r['id']]['width']))/2)) if geom.geom_type=='LineString' else geom
            items.append((footprint,{'High':1,'Medium':2,'Low':3}[d['location_priority']]))
        data=rasterize(items,out_shape=ctx['valid'].shape,transform=ctx['transform'],fill=0,all_touched=True,dtype='uint8') if items else np.zeros(ctx['valid'].shape,dtype='uint8')
        if 'recommendation_mask' in ctx: data[~ctx['recommendation_mask']]=0
        layer=ctx['base'].copy(data=data).rio.write_nodata(0)
        layer.rio.to_raster(output_file(priority_dir, f"intervention_{r['id']:02d}_priority.tif"),compress='deflate')
    (output_file(priority_dir, 'legend.json')).write_text(json.dumps({'0':'No selected footprint / nodata','1':'Must have / High','2':'Good to have / Medium','3':'Better to have / Low'}))
    for name in ('dem','slope_pct','accumulation_ha','stream','edge_inflow_unknown'):

        data=np.where(ctx['valid'],ctx[name],np.nan).astype('float32')
        ctx['base'].copy(data=data).rio.write_nodata(np.nan).rio.to_raster(output_file(out_dir, f'siting_{name}.tif'),compress='deflate')
    if 'recommendation_mask' in ctx:
        ctx['base'].copy(data=ctx['recommendation_mask'].astype('uint8')).rio.write_nodata(0).rio.to_raster(output_file(out_dir, 'original_aoi_recommendation_mask.tif'))
    (output_file(out_dir, 'siting_settings.json')).write_text(json.dumps(dict(settings=SITING,sources=SITING_SOURCES,
        native_resolution_m=ctx['native_resolution_m'],manifest=ctx['manifest'],
        note='Field checks pending are not passes. DEM routes are proxies. All numbers in settings are planning assumptions.'),indent=2))


def add_siting_sheets(wb,results):
    from openpyxl.styles import Alignment
    rules=wb.create_sheet('Siting Rules and Sources')
    rules.append(['ID','Intervention','Use case / spatial rule','Required field checks','Sources'])
    for iid,rule in RULES.items():
        rules.append([iid,INTERVENTIONS_V2[iid]['name'],rule[1],'; '.join(COMMON_CHECKS+rule[2]),
                      '\n'.join(SITING_SOURCES[k][1] for k in rule[3])])
    rules.append(['','Analysis boundary','25% of AOI bounding-box width on left/right; 25% of height above/below, in local metres. Downloads and analysis use the expanded rectangle.','Entire recommended footprint must remain inside the original polygon, excluding holes.',''])
    rules.append(['','Numeric assumptions',str(SITING),'Original slope ranges are practitioner assumptions; retain pending local review.',''])
    rules.append(['','Interpretation','DEM alignment is not survey validation. SCT phase is a layout heuristic; CCT/terraces are fixed survey segments, not completed hydraulic networks.','Missing checks cap score at 69 (Medium); known failed checks reject.',''])
    audit=wb.create_sheet('Location Rule Audit')
    fields=['site_id','location_rank','location_score','raw_location_score','location_category','location_priority',
            'layout','alignment_length_m','contour_elevation_m','contour_row','elevation_range_m',
            'max_alignment_grade_pct','cross_flow_deviation_deg','building_id','roof_distance_m',
            'building_source','roof_source_longitude','roof_source_latitude',
            'slope_pct','contributing_area_ha','native_dem_resolution_m',
            'spatial_status','passed_checks','pending_checks','trend_status','source_urls']
    audit.append(['Intervention']+fields)
    for r in results:
        for d in r['_site_priorities']: audit.append([r['name']]+[d.get(k,'') for k in fields])
    rejects=wb.create_sheet('Rejected Locations Summary'); rejects.append(['ID','Intervention','Selected','Rejection reasons'])
    for r in results: rejects.append([r['id'],r['name'],r['recommended_units'],r['rejection_summary']])
    for ws in (rules,audit,rejects):
        ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions
        for cell in ws[1]: cell.font=Font(bold=True)
        for col in ws.columns: ws.column_dimensions[col[0].column_letter].width=40
        for row in ws.iter_rows(min_row=2):
            ws.row_dimensions[row[0].row].height=75
            for cell in row: cell.alignment=Alignment(wrap_text=True,vertical='top')


def _points_to_geo(points, transform_):
    y_idx, x_idx = points
    if len(y_idx) == 0:
        return np.array([]), np.array([])
    xs, ys = transform_xy(transform_, y_idx.tolist(), x_idx.tolist())
    return np.asarray(xs), np.asarray(ys)


def _points_to_latlon(points, transform_, src_crs):
    """Pixel row/col -> (latitude, longitude) in WGS84 decimal degrees,
    matching the reference budget sheet's 'Latitude'/'Longitude' columns."""
    xs, ys = _points_to_geo(points, transform_)
    if len(xs) == 0:
        return np.array([]), np.array([])
    if src_crs and not src_crs.is_geographic:
        to_wgs84 = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        lon, lat = to_wgs84.transform(xs, ys)
    else:
        lon, lat = xs, ys
    return np.asarray(lat), np.asarray(lon)


def _format_dims(spec):
    """'LxWxD' in meters (rounded, zero dimensions dropped) - matches the
    reference sheet's 'Measurements/unit' column, e.g. '1.2x1.2x1.2'."""
    l = (spec["length"][0] + spec["length"][1]) / 2.0
    w = (spec["width"][0] + spec["width"][1]) / 2.0
    d = (spec["depth"][0] + spec["depth"][1]) / 2.0
    parts = [round(v, 2) for v in (l, w, d) if v > 0]
    return "x".join(f"{p:g}" for p in parts) if parts else "-"


def build_site_list(results, points_by_id, latest_ndvi_da):
    """One row per recommended intervention SITE (not per type) - S.No,
    name, qty, unit, latitude, longitude, dimensions - the same granularity
    as the reference budget sheet's per-site table."""
    transform_ = latest_ndvi_da.rio.transform()
    crs = latest_ndvi_da.rio.crs
    rows = []
    sno = 1
    for r in results:
        iid = r["id"]
        spec = INTERVENTIONS_V2[iid]
        lat_arr, lon_arr = _points_to_latlon(points_by_id.get(iid, (np.array([]), np.array([]))), transform_, crs)
        dims = _format_dims(spec)
        metadata = r.get('_site_priorities', [])
        if len(metadata) != len(lat_arr):
            raise ValueError(f"Missing or mismatched location rankings for intervention {iid}; rerun plan_interventions")
        for position, (lat, lon) in enumerate(zip(lat_arr, lon_arr)):
            rows.append(dict(
                sno=sno, name=spec["name"], qty=1, unit="No.",
                latitude=round(float(lat), 6), longitude=round(float(lon), 6),
                measurements=dims, intervention_id=iid,
                intervention_rank=r['rank'], intervention_score=r['priority_score'],
                intervention_category=r['priority_category'], intervention_priority=r['priority_level'],
                **metadata[position],
            ))
            sno += 1
    return rows


def export_site_priority_csv(results, points_by_id, latest_ndvi_da, path):
    rows = build_site_list(results, points_by_id, latest_ndvi_da)
    # Always emit a useful header, even when no sites are eligible.
    fields = ['sno', 'site_id', 'intervention_id', 'name', 'latitude', 'longitude',
              'intervention_rank', 'intervention_score', 'intervention_category', 'intervention_priority',
              'location_rank', 'location_score', 'location_category', 'location_priority',
              'evidence_status', 'qty', 'unit', 'measurements']
    fields += sorted({key for row in rows for key in row if key not in fields})
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return Path(path)


# ---------------------------------------------------------------------------
# ASSUMED cost model (per user request) - generic per-unit line-item rates,
# in the style of the reference budget PDF's "Estimate of X" tables, but NOT
# real Meghalaya MPWD Schedule-of-Rates figures. These are placeholder
# planning-stage rates only; replace RATES_ASSUMED with local, current SOR
# values before using this for an actual budget submission.
# ---------------------------------------------------------------------------
RATES_ASSUMED = {
    "clearing_sqm": 4.0,
    "excavation_cum": 213.0,
    "filling_cum": 141.0,
    "stone_pitching_cum": 2200.0,
    "pcc_masonry_cum": 7500.0,
    "gabion_cum": 1900.0,
    "brushwood_cum": 350.0,
    "fencing_post_wire_per_m": 180.0,
    "bio_fencing_per_m": 60.0,
    "plant_supply_each": 80.0,
    "planting_labor_each": 15.0,
    "roofing_accessories_lumpsum": 9000.0,
}

# id -> which "structural fill" material this intervention's core structure
# uses, once the common clearing/excavation/backfill lines are done.
_STRUCTURE_MATERIAL = {5: "pcc_masonry_cum", 6: "stone_pitching_cum", 7: "stone_pitching_cum",
                       8: "gabion_cum", 9: "brushwood_cum", 16: "stone_pitching_cum", 10: "roofing_accessories_lumpsum"}
_STRUCTURE_LABEL = {"pcc_masonry_cum": "Plain cement concrete (foundation/apron)",
                    "stone_pitching_cum": "Stone/boulder pitching",
                    "gabion_cum": "Gabion wire-mesh box with stone fill",
                    "brushwood_cum": "Brushwood/stake structure",
                    "roofing_accessories_lumpsum": "Roofing/gutter/downpipe accessories (lump sum)"}


def build_line_items(iid):
    """A short, generic per-unit BOQ for one intervention - ASSUMED
    quantities/rates, not a certified estimate. Returns a list of
    (particulars, qty, unit, rate) tuples."""
    spec = INTERVENTIONS_V2[iid]
    items = []

    if iid in (13, 14):  # fencing - linear, not a footprint/volume structure
        run_m = SITING['fence_segment_m']  # same fixed segment length as exported boundary geometry
        if iid == 13:
            items.append(("Barbed wire fencing incl. posts, wire & labour (per running m)",
                          run_m, "M", RATES_ASSUMED["fencing_post_wire_per_m"]))
        else:
            items.append(("Bio-fencing saplings incl. planting (per running m, 1 per m)",
                          run_m, "M", RATES_ASSUMED["bio_fencing_per_m"]))
        return items

    footprint = _unit_footprint_m2(spec)
    depth = (spec["depth"][0] + spec["depth"][1]) / 2.0
    excavation_vol = round(footprint * depth, 2)

    items.append(("Clearing grass & site preparation", round(footprint, 2), "Sqm", RATES_ASSUMED["clearing_sqm"]))
    items.append(("Earthwork excavation (manual, ordinary soil)", excavation_vol, "Cum", RATES_ASSUMED["excavation_cum"]))

    if iid in _STRUCTURE_MATERIAL:
        mat_key = _STRUCTURE_MATERIAL[iid]
        if mat_key == "roofing_accessories_lumpsum":
            items.append((_STRUCTURE_LABEL[mat_key], 1, "LS", RATES_ASSUMED[mat_key]))
        else:
            items.append((_STRUCTURE_LABEL[mat_key], round(excavation_vol * 0.8, 2), "Cum", RATES_ASSUMED[mat_key]))
        backfill_vol = round(excavation_vol * 0.2, 2)
    else:
        backfill_vol = round(excavation_vol * 0.5, 2)
    items.append(("Earthwork filling / backfill (available excavated earth)", backfill_vol, "Cum", RATES_ASSUMED["filling_cum"]))

    if iid in (12, 15):  # plantation-type interventions
        items.append(("Supply of fruit sapling with transport", 1, "No.", RATES_ASSUMED["plant_supply_each"]))
        items.append(("Planting labour", 1, "No.", RATES_ASSUMED["planting_labor_each"]))

    return items


def apply_intervention_template(wb, results):
    """Keep template BOQs as unit estimates; link project quantities and costs.

    Counts are model locations, not material counts or hectares. Linear estimates
    use validated alignment lengths. Area-based estimates require a user input:
    the planner's spacing/buffer area is NOT a measured treatment area.
    """
    from openpyxl.styles import Alignment
    from openpyxl.comments import Comment
    from openpyxl.workbook.properties import CalcProperties
    mapping = {
        1: ('Staggered Contour Trenches ', 'J8', 'No.', 1),
        2: ('Continuous Contour Trench', 'J8', 'm', 1),
        3: ('Percolation Pit', 'J8', 'No.', 1),
        4: (' Percolation Pond', 'J9', 'No.', 1),
        5: ('Sub surface Check dam', 'J11', 'm', 5),
        6: ('Loose Boulder Checkdam', 'J12', 'm', 5),
        7: ('Gully Plug', 'J10', 'm', 1),
        8: ('Gabion Check dam', 'J14', 'm', 5),
        9: ('Brushwood check dam', 'J8', 'm', 1),
        10: ('Roof Rainwater Recharge Pit', 'J17', 'No.', 1),
        11: ('Deep Recharge Pit', 'J9', 'No.', 1),
        12: ('Deep Recharge pit with Fruit Pl', 'J11', 'No.', 1),
        13: ('Barbed Wire Fencing', 'F10', 'ha', 1),
        14: ('Bio-Fencing', 'F6', 'ha', 1),
        15: ('Fruit Plantation', 'J10', 'No.', 1),
        16: (' Bench Terracing ', 'F22', 'ha', .5),
    }
    cct = wb.copy_worksheet(wb['Staggered Contour Trenches '])
    cct.title = 'Continuous Contour Trench'
    cct['A1'] = 'Continuous Contour Trench: assumed estimate per metre, adapted from SCT'
    for row in (5, 6):
        cct[f'C{row}'] = 1
        cct[f'D{row}'] = 1
        cct[f'E{row}'] = float(np.mean(INTERVENTIONS_V2[2]['width']))
    cct['F6'] = float(np.mean(INTERVENTIONS_V2[2]['depth']))
    # Remove example project counts: these two BOQs become per-structure costs.
    for name in ('Staggered Contour Trenches ', 'Percolation Pit'):
        wb[name]['C5'] = wb[name]['C6'] = 1
        wb[name]['A1'] = name.strip() + ': estimate per structure'
    # The template's fixed "Say" amount did not respond to rate changes.
    wb['Brushwood check dam']['J8'] = '=ROUND(J7,-1)'
    wb[' Bench Terracing ']['D18'] = '=(D9/D5/D7)*0.3*D5*1.5*D2'
    # Original template costs are retained as editable assumptions, not certified SOR.
    note = wb.create_sheet('Template Notes')
    for row in [
        ['Template-based recommendation workbook'],
        ['Edit blue/yellow input cells and template unit rates; Excel recalculates formulas on opening.'],
        ['P2 on each intervention sheet is the recommended location count. Component quantities in the unit BOQ are NOT location counts.'],
        ['P3 is the cost quantity: structures, alignment metres, or editable treated hectares. P6 is the project estimate in Rs.'],
        ['Percolation / Small Dugout Pond uses the Percolation Pond template only; Small Pond is an unused alternative.'],
        ['CCT is adapted from SCT per metre using model width/depth; dimensions and template rates require local review.'],
        ['Linear check-dam estimates scale the template cost by alignment length / template span (5 m or 1 m); this is a preliminary proportional estimate.'],
        ['For area-priced fencing and terracing enter actual treated hectares in P3. Location counts and alignment lengths cannot determine fenced hectares.'],
        ['Template dimensions remain reference assumptions; site dimensions in Site List can differ and must be reconciled before construction.'],
        ['Unused template sheets are hidden to preserve formulas and source material; they are not recommended work.'],
        ['No recommended sites produces zero project totals. Missing quantities for recommended work leave costs blank and are flagged in Cost Summary.'],
        ['Brushwood fixed rounding replaced with a formula; bench retaining volume now includes treatment area.'],
    ]:
        note.append(row)
    note.column_dimensions['A'].width = 115
    for row in note:
        row[0].alignment = Alignment(wrap_text=True, vertical='top')
        note.row_dimensions[row[0].row].height = 32
    # Replace the generic estimate summary with the template-derived project costs.
    costs = wb['Cost Summary']
    costs.delete_rows(1, costs.max_row)
    costs.append(['ID', 'Intervention', 'Recommended locations', 'Cost quantity', 'Unit',
                  'Rate per cost unit (Rs)', 'Project estimate (Rs)', 'Status', 'Template sheet'])
    extra_names = {'Small Pond', 'Plantation', 'Grass Plantation', 'Direct Seed Sowing', 'ANR'}
    for name in extra_names:
        sheet = wb[name]
        sheet['O1'] = 'NOT RECOMMENDED: reference template only'
        sheet['O2'], sheet['P2'] = 'Recommended locations', 0
        sheet.sheet_state = 'hidden'
    for index, r in enumerate(results, start=2):
        iid = r['id']
        name, total, unit, basis = mapping[iid]
        sheet = wb[name]
        count = int(r['recommended_units'])
        if count < 0 or count != r['recommended_units']:
            raise ValueError(f'Invalid recommendation count for {name}')
        metadata = r.get('_site_priorities', [])
        lengths = [float(p.get('alignment_length_m', 0) or 0) for p in metadata]
        valid_lengths = len(lengths) == count and all(np.isfinite(v) and v > 0 for v in lengths)
        length = sum(lengths) if valid_lengths else None
        quantity = count if unit == 'No.' else (0 if count == 0 else (length if unit == 'm' else None))
        sheet['O1'] = 'RECOMMENDED PROJECT QUANTITIES / COST (Rs)'
        sheet['O2'], sheet['P2'] = 'Recommended locations (No.)', f'=Summary!C{index}'
        sheet['O3'], sheet['P3'] = f'Cost quantity ({unit})', ('=P2' if unit == 'No.' else quantity)
        sheet['O4'], sheet['P4'] = 'Template cost basis (' + unit + ')', basis
        sheet['O5'], sheet['P5'] = 'Cost per template basis (Rs)', '=' + total
        sheet['O6'], sheet['P6'] = 'Recommended project cost (Rs)', '=IF(P2=0,0,IF(P3="","",P3/P4*P5))'
        sheet['O7'], sheet['P7'] = 'Quantity status', '=IF(P2=0,"Not recommended",IF(P3="","ENTER COST QUANTITY","Estimated; review assumptions"))'
        sheet['O8'], sheet['P8'] = 'BOQ scope', 'Reference unit/basis only; project total is P6'
        sheet['O9'], sheet['P9'] = 'Known alignment length (m)', length if length is not None else 'Unavailable'
        if iid == 7:
            sheet['C15'], sheet['F15'] = '=IF(P3="","",P3)', '=P6'
        elif iid == 6:
            sheet['C17'], sheet['D17'], sheet['F17'] = '=IF(P3="","",P3/P4)', '5 m equivalents', '=P6'
        sheet['P3'].comment = Comment(
            'Cost quantity, not component quantity. For ha enter surveyed treatment area. '
            'For metres use total validated alignment length. Do not enter a spacing buffer area.', 'Planner')
        for rownum in range(1, 10):
            sheet[f'O{rownum}'].font = Font(bold=True)
        sheet['P3'].fill = PatternFill('solid', fgColor='FFF2CC')
        sheet['P3'].font = Font(color='0000FF')
        sheet.column_dimensions['O'].width = 43
        sheet.column_dimensions['P'].width = 48
        sheet['P6'].number_format = '#,##0.00'
        sheet.sheet_state = 'visible' if count else 'hidden'
        quoted = "'" + name.replace("'", "''") + "'"
        costs.append([iid, r['name'], f'={quoted}!P2',
                      f'=IF({quoted}!P3="","",{quoted}!P3)', unit,
                      f'={quoted}!P5/{quoted}!P4',
                      f'=IF({quoted}!P6="","",{quoted}!P6)', f'={quoted}!P7', name])
    end = len(results) + 1
    costs.append(['', 'Priced subtotal (Rs)', '', '', '', '', f'=SUM(G2:G{end})'])
    costs.append(['', 'Budget completeness', '', '', '', '', '',
                  f'=IF(COUNTIF(H2:H{end},"ENTER COST QUANTITY")>0,"INCOMPLETE: enter missing quantities","All recommendations priced")'])
    for cell in costs[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill('solid', fgColor='D9E1F2')
    for col, width in zip('ABCDEFGHI', [8, 44, 24, 20, 10, 26, 26, 45, 34]):
        costs.column_dimensions[col].width = width
    costs.freeze_panes = 'C2'
    costs.auto_filter.ref = f'A1:I{end}'
    # Keep the old generic BOQ only as a hidden reference, never as project costing.
    wb['Unit Cost Estimates'].sheet_state = 'hidden'
    wb.active = wb.sheetnames.index('Summary')
    wb.calculation = CalcProperties(calcId=0, fullCalcOnLoad=True, forceFullCalc=True, calcMode='auto')


def export_intervention_plan_excel(results, points_by_id, latest_ndvi_da, out_path, template_path=None):
    """Export the supplied template with project counts, formula costs and siting audits."""
    from openpyxl import load_workbook
    template_path = Path(template_path) if template_path else Path(__file__).with_name('Interventions.xlsx')
    if template_path.resolve() == Path(out_path).resolve():
        raise ValueError('Output must not overwrite the intervention template')
    site_rows = build_site_list(results, points_by_id, latest_ndvi_da)

    wb = load_workbook(template_path)
    header_font = Font(name="Arial", bold=True)
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    body_font = Font(name="Arial")
    assumed_font = Font(name="Arial", color="0000FF")  # blue = hardcoded input/assumption
    assumed_fill = PatternFill("solid", fgColor="FFFF00")  # yellow = cell the user should review/edit
    title_font = Font(name="Arial", bold=True, size=11)

    ws = wb.create_sheet("Site List")
    headers = ["S.No", "Name of Work", "Qty", "Unit", "Latitude", "Longitude", "Measurements/unit (L x W x D, m)"]
    headers += ['Site ID', 'Intervention Rank', 'Intervention Score (0-100)',
                'Intervention Category', 'Intervention Priority', 'Location Rank within Intervention',
                'Location Score (0-100)', 'Location Category', 'Location Priority', 'Evidence Status']
    ws.append(headers)
    for cell in ws[1]:
        cell.font, cell.fill = header_font, header_fill
    for row in site_rows:
        ws.append([row["sno"], row["name"], row["qty"], row["unit"],
                   row["latitude"], row["longitude"], row["measurements"], row['site_id'],
                   row['intervention_rank'], row['intervention_score'], row['intervention_category'],
                   row['intervention_priority'], row['location_rank'], row['location_score'],
                   row['location_category'], row['location_priority'], row['evidence_status']])
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = body_font
    for col_cells in ws.columns:
        width = max((len(str(c.value)) for c in col_cells if c.value is not None), default=10)
        ws.column_dimensions[col_cells[0].column_letter].width = min(max(width + 2, 10), 42)

    ws2 = wb.create_sheet("Summary")
    ws2.append(["S.No", "Name of Work", "Qty", "Unit", "Total Area (m2)", "Eligible Pixels", "Min Spacing (m)", 'Rank', 'Score (0-100)', 'Category', 'Priority',
                'High Priority Sites', 'Medium Priority Sites', 'Low Priority Sites', 'Evidence Status'])
    for cell in ws2[1]:
        cell.font, cell.fill = header_font, header_fill
    for i, r in enumerate(results, start=1):
        ws2.append([i, r["name"], r["recommended_units"], "No.",
                   r["total_area_m2"], r["eligible_pixels"], r["buffer_spacing_m"], r['rank'],
                   r['priority_score'], r['priority_category'], r['priority_level'],
                   r['high_priority_sites'], r['medium_priority_sites'], r['low_priority_sites'], r['evidence_status']])
    total_units = sum(r["recommended_units"] for r in results)
    total_area = round(sum(r["total_area_m2"] for r in results), 1)
    ws2.append(["", "Total", total_units, "", total_area, "", ""])
    for row in ws2.iter_rows(min_row=2, max_row=ws2.max_row - 1):
        for cell in row:
            cell.font = body_font
    for cell in ws2[ws2.max_row]:
        cell.font = header_font
    for col_cells in ws2.columns:
        width = max((len(str(c.value)) for c in col_cells if c.value is not None), default=10)
        ws2.column_dimensions[col_cells[0].column_letter].width = min(max(width + 2, 10), 42)

    # --- "Unit Cost Estimates": one per-unit BOQ block per intervention ---
    ws3 = wb.create_sheet("Unit Cost Estimates")
    ws3.append(["ASSUMED PLACEHOLDER RATES - replace with local Schedule of Rates (SOR) before use as an actual budget"])
    ws3["A1"].font = Font(name="Arial", bold=True, italic=True, size=11, color="CC0000")
    ws3.append([])
    unit_total_row = {}  # iid -> row number of that block's "Unit Total (Rs)" row, for Cost Summary to reference
    for r in results:
        iid = r["id"]
        block_title_row = ws3.max_row + 1
        ws3.append([f"Estimate of {r['name']} (per unit)"])
        ws3.cell(row=block_title_row, column=1).font = title_font
        header_row = ws3.max_row + 1
        ws3.append(["S.No", "Particulars", "Qty", "Unit", "Rate (Rs)", "Amount (Rs)"])
        for cell in ws3[header_row]:
            cell.font, cell.fill = header_font, header_fill

        items = build_line_items(iid)
        first_item_row = ws3.max_row + 1
        for j, (particulars, qty, unit, rate) in enumerate(items, start=1):
            row_num = ws3.max_row + 1
            ws3.append([j, particulars, qty, unit, rate, f"=C{row_num}*E{row_num}"])
            ws3.cell(row=row_num, column=5).font = assumed_font
            ws3.cell(row=row_num, column=5).fill = assumed_fill
            for col in (1, 2, 3, 4, 6):
                ws3.cell(row=row_num, column=col).font = body_font
        last_item_row = ws3.max_row

        total_row = ws3.max_row + 1
        ws3.append(["", "Unit Total (Rs)", "", "", "", f"=SUM(F{first_item_row}:F{last_item_row})"])
        for cell in ws3[total_row]:
            cell.font = header_font
        unit_total_row[iid] = total_row
        ws3.append([])  # spacer between blocks

    for col_letter, width in zip("ABCDEF", (6, 46, 8, 8, 12, 14)):
        ws3.column_dimensions[col_letter].width = width

    # --- "Cost Summary": counterpart to the reference's final Total table ---
    ws4 = wb.create_sheet("Cost Summary")
    ws4.append(["S.No", "Name of Work", "Qty", "Unit", "Unit Rate (Rs)", "Amount (Rs)"])
    for cell in ws4[1]:
        cell.font, cell.fill = header_font, header_fill
    first_data_row = ws4.max_row + 1
    for i, r in enumerate(results, start=1):
        summary_row = i + 1  # Summary sheet row for this same intervention (same id order)
        row_num = ws4.max_row + 1
        ws4.append([
            i, r["name"], f"=Summary!C{summary_row}", "No.",
            f"='Unit Cost Estimates'!F{unit_total_row[r['id']]}",
            f"=C{row_num}*E{row_num}",
        ])
        for cell in ws4[row_num]:
            cell.font = body_font
    last_data_row = ws4.max_row
    grand_total_row = ws4.max_row + 1
    ws4.append(["", "Grand Total (Rs)", "", "", "", f"=SUM(F{first_data_row}:F{last_data_row})"])
    say_row = ws4.max_row + 1
    ws4.append(["", "Say (Rs)", "", "", "", f"=ROUND(F{grand_total_row},-2)"])
    for cell in ws4[grand_total_row] + ws4[say_row]:
        cell.font = header_font
    ws4.append([])
    ws4.append(["Note: 'Unit Rate' pulls from the ASSUMED per-unit line-item build-up on the "
                "'Unit Cost Estimates' sheet - edit the yellow-highlighted Rate cells there with real "
                "local SOR figures and every total here recalculates automatically."])
    ws4["A" + str(ws4.max_row)].font = Font(name="Arial", italic=True, size=9)
    for col_letter, width in zip("ABCDEF", (6, 42, 8, 8, 14, 16)):
        ws4.column_dimensions[col_letter].width = width

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    add_methodology_sheet(wb, results)
    add_siting_sheets(wb, results)
    # A sorted view leaves Summary row order intact for cost-formula references.
    ranked = wb.create_sheet('Intervention Rankings')
    ranked.append(['Rank', 'ID', 'Intervention', 'Score (0-100)', 'Category', 'Priority',
                   'Sites', 'High Sites', 'Medium Sites', 'Low Sites', 'Evidence Status'])
    for r in sorted(results, key=lambda r: (r['rank'], r['id'])):
        ranked.append([r['rank'], r['id'], r['name'], r['priority_score'],
                       r['priority_category'], r['priority_level'], r['recommended_units'],
                       r['high_priority_sites'], r['medium_priority_sites'], r['low_priority_sites'], r['evidence_status']])
    for sheet, priority_col in ((ws, 16), (ws2, 11), (ranked, 6)):
        sheet.freeze_panes = 'C2'
        last = sheet.max_row - 1 if sheet == ws2 else sheet.max_row
        sheet.auto_filter.ref = f'A1:{sheet.cell(1, sheet.max_column).column_letter}{max(1, last)}'
        for row in sheet.iter_rows(min_row=2, max_row=last):
            level = row[priority_col - 1].value
            color = {'High': 'F4CCCC', 'Medium': 'FFF2CC', 'Low': 'D9EAD3'}.get(level, 'E7E6E6')
            row[priority_col - 1].fill = PatternFill('solid', fgColor=color)
        for cell in sheet[1]:
            cell.font, cell.fill = header_font, header_fill
        for column in sheet.columns:
            sheet.column_dimensions[column[0].column_letter].width = min(48, max(12, max(len(str(c.value or '')) for c in column) + 2))
    apply_intervention_template(wb, results)
    wb.save(out_path)
    print(f"\u2705 Intervention plan (Excel) saved to {out_path}: "
          f"{len(site_rows)} sites across {len(results)} intervention types "
          "(template BOQs, project costs, rankings and siting audits)")
    return out_path


def create_intervention_plan_map(results, points_by_id, latest_ndvi_da, gdf, out_dir=None):
    """Combined overview map (all recommended sites, one figure) plus one
    individual map per intervention that has at least one recommended unit,
    all on the downloaded satellite mosaic, displayed in true lon/lat (not
    the NDVI grid's native UTM-like CRS) with the same cartographic
    furniture as the DEM slope map: non-scientific axis tick labels, a
    north arrow, a dynamic scale bar, and a source attribution. Axis limits
    are pinned to the reprojected mosaic's own bounds AFTER every point/
    boundary is plotted, so the satellite image reliably fills the frame
    instead of matplotlib's default autoscale potentially stretching the
    view to whatever scatter point happens to sit furthest out."""
    transform_ = latest_ndvi_da.rio.transform()
    native_crs = latest_ndvi_da.rio.crs
    if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    out_dir = Path(PLAN_OUTPUT_DIR if out_dir is None else out_dir)
    individual_dir = out_dir / "individual_interventions"
    individual_dir.mkdir(parents=True, exist_ok=True)

    print("\U0001f5fa\ufe0f  Generating intervention site-plan maps...")

    fig, ax = plt.subplots(figsize=(11, 11))
    extent = add_local_satellite_basemap(ax, target_crs=WGS84_CRS)
    gdf.plot(ax=ax, facecolor="none", edgecolor="yellow", linewidth=1.5, zorder=3)

    legend_elements = []
    for r in results:
        iid = r["id"]
        lat, lon = _points_to_latlon(points_by_id.get(iid, (np.array([]), np.array([]))), transform_, native_crs)
        if len(lat) == 0:
            continue
        color, marker = INTERVENTION_COLORS[iid], INTERVENTION_MARKERS[iid]
        ax.scatter(lon, lat, c=[color], marker=marker, s=30, edgecolors="black", linewidth=0.4, zorder=4)
        legend_elements.append(mlines.Line2D(
            [], [], marker=marker, linestyle="None", markersize=8,
            markerfacecolor=color, markeredgecolor="black",
            label=f"{iid}. {r['name']} ({r['recommended_units']})",
        ))

    ax.set_title("Recommended Intervention Sites\n(slope + NDVI/NDWI degradation + house masking + buffer spacing)", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if extent:
        add_map_furniture(ax, extent)
    if legend_elements:
        ax.legend(handles=legend_elements, loc="lower right", fontsize=7, framealpha=0.9)
    combined_path = output_file(out_dir, "intervention_plan_overview.png")
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\u2705 Saved {combined_path}")

    for r in results:
        iid = r["id"]
        lat, lon = _points_to_latlon(points_by_id.get(iid, (np.array([]), np.array([]))), transform_, native_crs)
        fig, ax = plt.subplots(figsize=(9, 8))
        extent = add_local_satellite_basemap(ax, target_crs=WGS84_CRS)
        gdf.plot(ax=ax, facecolor="none", edgecolor="yellow", linewidth=1.2, zorder=3)
        if len(lat) > 0:
            ax.scatter(lon, lat, c=[INTERVENTION_COLORS[iid]], marker=INTERVENTION_MARKERS[iid],
                       s=45, edgecolors="black", linewidth=0.5, zorder=4,
                       label=f"Planned units: {r['recommended_units']}\n"
                             f"Footprint: {r['unit_footprint_m2']} m2 each\n"
                             f"Total area: {r['total_area_m2']} m2\n"
                             f"Min spacing: {r['buffer_spacing_m']} m")
            ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
        else:
            ax.text(0.5, 0.5, "No eligible sites found inside AOI", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="white",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
        ax.set_title(f"Intervention {iid}: {r['name']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if extent:
            add_map_furniture(ax, extent)
        fname = f"{iid:02d}_{r['name'].lower().replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(output_file(individual_dir, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"\u2705 Saved {len(results)} individual maps to {individual_dir}")
    return combined_path


# ---------------------------------------------------------------------------
# Stage: satellite mosaic + house/structure mask (from HouseMask.py)
# ---------------------------------------------------------------------------

def fetch_tile(x, y, z):
    url = GOOGLE_SAT_URL.format(x=x, y=y, z=z)
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
    if response.status_code != 200:
        raise ConnectionError(f"Failed to fetch tile {z}/{x}/{y}: HTTP {response.status_code}")
    return Image.open(BytesIO(response.content)).convert("RGB")


def download_google_satellite_from_kml(kml_path, output_tiff_path, output_png_path, zoom):
    output_tiff_path.parent.mkdir(parents=True, exist_ok=True)

    print("Reading KML geometry...")
    gdf = gpd.read_file(kml_path, driver="KML")
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    aoi_geom = union_of(gdf)
    minx, miny, maxx, maxy = aoi_geom.bounds
    print(f"AOI Bounding Box (WGS84): [{minx:.6f}, {miny:.6f}, {maxx:.6f}, {maxy:.6f}]")

    tiles = list(mercantile.tiles(minx, miny, maxx, maxy, zoom))
    print(f"Covering AOI with {len(tiles)} tiles at Zoom Level {zoom}...")

    min_tile_x = min(t.x for t in tiles)
    max_tile_x = max(t.x for t in tiles)
    min_tile_y = min(t.y for t in tiles)
    max_tile_y = max(t.y for t in tiles)

    tile_size = 256
    canvas_w = (max_tile_x - min_tile_x + 1) * tile_size
    canvas_h = (max_tile_y - min_tile_y + 1) * tile_size
    stitched_image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    print("Downloading tiles...")
    n_failed = 0
    for idx, tile in enumerate(tiles, start=1):
        try:
            img = fetch_tile(tile.x, tile.y, tile.z).convert("RGBA")
            offset_x = (tile.x - min_tile_x) * tile_size
            offset_y = (tile.y - min_tile_y) * tile_size
            stitched_image.paste(img, (offset_x, offset_y))
            print(f"[{idx}/{len(tiles)}] Tile ({tile.x}, {tile.y}, {tile.z}) downloaded.", end="\r")
        except Exception as e:
            n_failed += 1
            print(f"\nWarning: Could not fetch tile {tile}: {e}")
    if n_failed:
        print(f"\n{n_failed}/{len(tiles)} tiles failed and were left transparent.")

    print("\nStitching complete. Georeferencing tile bounds...")
    ul_bounds = mercantile.xy_bounds(min_tile_x, min_tile_y, zoom)
    lr_bounds = mercantile.xy_bounds(max_tile_x, max_tile_y, zoom)
    west, north = ul_bounds.left, ul_bounds.top
    east, south = lr_bounds.right, lr_bounds.bottom
    transform_ = from_bounds(west, south, east, north, canvas_w, canvas_h)

    image_array = np.array(stitched_image)
    temp_mosaic_path = output_file(output_tiff_path.parent, "temp_stitched_mosaic.tif")

    with rasterio.open(
        temp_mosaic_path, "w", driver="GTiff", height=canvas_h, width=canvas_w,
        count=4, dtype=image_array.dtype, crs="EPSG:3857", transform=transform_,
    ) as dst:
        for i in range(4):
            dst.write(image_array[:, :, i], i + 1)

    print("Clipping raster with transparency to exact KML geometry...")
    gdf_mercator = gdf.to_crs("EPSG:3857")
    aoi_geom_mercator = union_of(gdf_mercator)

    with rasterio.open(temp_mosaic_path) as src:
        out_image, out_transform = mask(src, [aoi_geom_mercator], crop=True, filled=True, nodata=0)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2],
            "transform": out_transform, "count": 4, "nodata": 0,
        })
        with rasterio.open(output_tiff_path, "w", **out_meta) as dst:
            dst.write(out_image)

    png_array = np.transpose(out_image, (1, 2, 0))
    Image.fromarray(png_array, mode="RGBA").save(output_png_path, "PNG")

    if temp_mosaic_path.exists():
        os.remove(temp_mosaic_path)

    print("\n\u2713 Satellite mosaic complete!")
    print(f"  GeoTIFF: {output_tiff_path.resolve()}")
    print(f"  PNG:     {output_png_path.resolve()}")


def generate_house_mask_tiff(input_tiff_path, output_mask_path):
    with rasterio.open(input_tiff_path) as src:
        profile = src.profile.copy()
        rgba = src.read()
        transform_ = src.transform
        crs = src.crs

    if rgba.shape[0] < 3:
        raise ValueError("Input raster must have at least 3 bands (RGB).")

    rgb = np.transpose(rgba[:3], (1, 2, 0)).astype(np.uint8)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if rgba.shape[0] >= 4:
        aoi_mask = (rgba[3] > 10).astype(np.uint8) * 255
    else:
        aoi_mask = np.full(img.shape[:2], 255, dtype=np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 30, 30]), np.array([85, 255, 255]))
    brown_mask = cv2.inRange(hsv, np.array([10, 40, 40]), np.array([25, 255, 255]))
    bright_mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 110, 255]))
    red_roof_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 60, 60]), np.array([12, 255, 255])),
        cv2.inRange(hsv, np.array([165, 60, 60]), np.array([180, 255, 255])),
    )

    background_mask = cv2.bitwise_or(green_mask, brown_mask)
    roof_color_mask = cv2.bitwise_or(bright_mask, red_roof_mask)
    raw_mask = cv2.bitwise_and(roof_color_mask, cv2.bitwise_not(background_mask))
    raw_mask = cv2.bitwise_and(raw_mask, aoi_mask)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_small)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_large)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(raw_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 <= area <= 2500:
            cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    mask_profile = profile.copy()
    mask_profile.update({"driver": "GTiff", "count": 1, "dtype": "uint8", "nodata": 0, "crs": crs, "transform": transform_})
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_mask_path, "w", **mask_profile) as dst:
        dst.write(final_mask, 1)
    print(f"\u2713 House mask GeoTIFF saved: {output_mask_path.resolve()}")


def read_house_mask_as_bool(house_mask_path, like_da):
    # Generic categorical alignment. Main pipeline preserves source polygons and
    # rasterizes them conservatively; neither approach uses max-pooling as roofs.
    with rioxarray.open_rasterio(house_mask_path, masked=False) as layer:
        matched=align_to_dem_grid(layer,like_da,'house mask',categorical=True)
        return np.isfinite(matched.values)&(matched.values>0)


def read_slope_as_array_on_grid(slope_tif_path, like_da):
    slope_da = rioxarray.open_rasterio(slope_tif_path, masked=True)
    slope_matched = slope_da.rio.reproject_match(like_da, resampling=Resampling.bilinear)
    arr = np.asarray(slope_matched.values, dtype="float32")
    if arr.ndim == 3:
        arr = arr.squeeze()
    return arr


# ---------------------------------------------------------------------------
# Stage: Sentinel-2 NDVI/NDWI multi-year composites (from NDVI_NDWI_mask_v3.py)
# ---------------------------------------------------------------------------

def get_indices_from_item(item, kml_geom):
    green_url = item.assets["B03"].href
    red_url = item.assets["B04"].href
    nir_url = item.assets["B08"].href
    scl_url = item.assets["SCL"].href
    swir_url = item.assets["B11"].href

    green = rioxarray.open_rasterio(green_url, masked=True, chunks={"x": 1024, "y": 1024})
    red = rioxarray.open_rasterio(red_url, masked=True, chunks={"x": 1024, "y": 1024})
    nir = rioxarray.open_rasterio(nir_url, masked=True, chunks={"x": 1024, "y": 1024})
    scl = rioxarray.open_rasterio(scl_url, masked=True, chunks={"x": 1024, "y": 1024})

    swir = rioxarray.open_rasterio(swir_url, masked=True, chunks={"x": 1024, "y": 1024})

    wgs84 = pyproj.CRS("EPSG:4326")
    raster_crs = pyproj.CRS(red.rio.crs)
    project = pyproj.Transformer.from_crs(wgs84, raster_crs, always_xy=True).transform
    kml_geom_projected = shp_transform(project, kml_geom)

    green_clipped = green.rio.clip([kml_geom_projected]).astype("float32")
    red_clipped = red.rio.clip([kml_geom_projected]).astype("float32")
    nir_clipped = nir.rio.clip([kml_geom_projected]).astype("float32")
    scl_clipped = scl.rio.clip([kml_geom_projected])
    swir_clipped = swir.rio.clip([kml_geom_projected]).astype("float32").rio.reproject_match(nir_clipped, resampling=Resampling.bilinear)

    if red_clipped.rio.width == 0 or red_clipped.rio.height == 0:
        raise ValueError("AOI clip produced an empty raster (KML boundary may not "
                          "overlap this scene, or is smaller than one pixel).")

    scl_matched = scl_clipped.rio.reproject_match(red_clipped, resampling=Resampling.nearest)
    invalid_mask = scl_matched.isin([0, 1, 3, 8, 9, 10, 11])
    green_clipped = green_clipped.where(~invalid_mask)
    red_clipped = red_clipped.where(~invalid_mask)
    nir_clipped = nir_clipped.where(~invalid_mask)

    swir_clipped = swir_clipped.where(~invalid_mask)
    ndmi = (nir_clipped - swir_clipped) / (nir_clipped + swir_clipped)
    ndmi = ndmi.where(np.isfinite(ndmi)).clip(-1, 1)
    ndvi = (nir_clipped - red_clipped) / (nir_clipped + red_clipped)
    ndwi = (green_clipped - nir_clipped) / (green_clipped + nir_clipped)
    ndvi = ndvi.where(np.isfinite(ndvi)).clip(-1, 1)
    ndwi = ndwi.where(np.isfinite(ndwi)).clip(-1, 1)
    return ndvi.compute(), ndwi.compute(), ndmi.compute()


def build_year_composite(year, catalog, gdf_bounds, kml_geom):
    start_date, end_date = f"{year}-04-01", f"{year}-06-30"
    search = catalog.search(
        collections=["sentinel-2-l2a"], bbox=gdf_bounds, datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": CLOUD_COVER_MAX}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    )
    items = list(search.item_collection())[:MAX_SCENES_PER_YEAR]
    if len(items) < MIN_SCENES_PER_YEAR:
        return None, None, None, 0, []

    ndvi_list, ndwi_list, ndmi_list, dates = [], [], [], []
    for item in items:
        try:
            ndvi_da, ndwi_da, ndmi_da = get_indices_from_item(item, kml_geom)
            if not ndvi_list:
                # Keep the complete buffered grid even when the first scene is
                # only one of several tiles; unobserved cells remain nodata.
                project = pyproj.Transformer.from_crs('EPSG:4326', ndvi_da.rio.crs, always_xy=True).transform
                bounds = shp_transform(project, kml_geom).bounds
                ndvi_da = ndvi_da.rio.pad_box(*bounds, constant_values=np.nan)
                ndwi_da = ndwi_da.rio.reproject_match(ndvi_da)
                ndmi_da = ndmi_da.rio.reproject_match(ndvi_da)
            if len(ndvi_list) > 0:
                ndvi_da = ndvi_da.rio.reproject_match(ndvi_list[0])
                ndwi_da = ndwi_da.rio.reproject_match(ndwi_list[0])
                ndmi_da = ndmi_da.rio.reproject_match(ndvi_list[0])
            ndvi_list.append(ndvi_da.squeeze())
            ndwi_list.append(ndwi_da.squeeze())
            ndmi_list.append(ndmi_da.squeeze())
            dates.append(item.datetime.strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"      (skipped one scene dated {item.datetime}: {e})")
            continue

    if len(ndvi_list) < MIN_SCENES_PER_YEAR:
        return None, None, None, 0, []

    ndvi_composite = xr.concat(ndvi_list, dim="scene").median(dim="scene", skipna=True)
    ndwi_composite = xr.concat(ndwi_list, dim="scene").median(dim="scene", skipna=True)
    ndvi_composite = ndvi_composite.rio.write_crs(ndvi_list[0].rio.crs)
    ndwi_composite = ndwi_composite.rio.write_crs(ndwi_list[0].rio.crs)
    ndmi_composite = xr.concat(ndmi_list, dim="scene").median(dim="scene", skipna=True).rio.write_crs(ndvi_list[0].rio.crs)
    return ndvi_composite, ndwi_composite, ndmi_composite, len(ndvi_list), dates


def save_year_outputs(da, year, band_label, make_plot=True):
    YEARLY_GEOTIFF_DIR.mkdir(parents=True, exist_ok=True)
    YEARLY_QUICKLOOK_DIR.mkdir(parents=True, exist_ok=True)

    tif_path = output_file(YEARLY_GEOTIFF_DIR, f"{band_label}_{year}.tif")
    da.rio.to_raster(str(tif_path), dtype="float32", driver="GTiff")

    if not make_plot:
        return tif_path

    png_path = output_file(YEARLY_QUICKLOOK_DIR, f"{band_label}_{year}.png")
    cmap = "RdYlGn" if band_label == "NDVI" else "Blues"
    fig, ax = plt.subplots(figsize=(6, 6))
    add_local_satellite_basemap(ax, da.rio.crs)
    left,bottom,right,top = da.rio.bounds()
    im = ax.imshow(da.values, cmap=cmap, vmin=-1, vmax=1,
                   extent=(left,right,bottom,top), alpha=0.60)
    ax.set_title(f"{band_label} - Peak Summer {year}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=band_label)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      saved {tif_path.name} and {png_path.name}")


def load_cached_year_composite(year, band_label):
    """Loads a previously-saved NDVI/NDWI composite GeoTIFF for `year`, if
    one already exists from an earlier run, so a rerun doesn't repeat the
    STAC search + Sentinel-2 band download for years already fetched.
    Returns the DataArray (2D, CRS preserved from the saved file), or None
    if no cached file is on disk for this year/band."""
    tif_path = output_file(YEARLY_GEOTIFF_DIR, f"{band_label}_{year}.tif")
    if not tif_path.exists():
        return None
    da = rioxarray.open_rasterio(tif_path, masked=True)
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)
    return da.astype("float32")


def compute_slope(y, x):
    slope, p, n = sen_mk(y, x)
    return slope if np.isfinite(p) and p <= TREND_ALPHA else (0.0 if n >= MIN_TREND_YEARS else np.nan)


def convert_slope_to_grayscale_mask(slope_map, base_da, output_filename, label=None, year_range=None):
    valid_mask = ~np.isnan(slope_map)
    slope_values = slope_map[valid_mask]
    grayscale_array = np.zeros_like(slope_map, dtype=np.uint8)
    p_min = p_max = None

    if len(slope_values) > 0:
        p_min = np.percentile(slope_values, 2)
        p_max = np.percentile(slope_values, 98)
        if p_max != p_min:
            clipped = np.clip(slope_map[valid_mask], p_min, p_max)
            normalized = (p_max - clipped) / (p_max - p_min) * 254.0 + 1.0
            grayscale_array[valid_mask] = normalized.astype(np.uint8)
        else:
            grayscale_array[valid_mask] = 128

    mask_da = xr.DataArray(grayscale_array, coords={"y": base_da.y, "x": base_da.x}, dims=["y", "x"], attrs=base_da.attrs)
    mask_da.rio.write_crs(base_da.rio.crs, inplace=True)
    mask_da.rio.write_nodata(0, inplace=True)
    mask_da.rio.to_raster(output_filename, dtype="uint8", driver="GTiff")
    print(f"Successfully exported georeferenced mask: {output_filename}")

    label = label or Path(output_filename).stem.replace("_mask", "")
    title_range = f" ({year_range[0]}-{year_range[1]})" if year_range else ""
    png_path = str(Path(output_filename).with_suffix(".png"))
    display = np.ma.masked_where(~valid_mask, slope_map)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("lightgray")

    fig, ax = plt.subplots(figsize=(8, 8))
    add_local_satellite_basemap(ax, base_da.rio.crs)
    left,bottom,right,top = base_da.rio.bounds()
    extent = (left,right,bottom,top)
    if p_min is not None and p_max != p_min:
        im = ax.imshow(display, cmap=cmap, vmin=p_min, vmax=p_max, extent=extent, alpha=0.60)
    else:
        im = ax.imshow(display, cmap=cmap, extent=extent, alpha=0.60)
    ax.set_title(f"{label} rate of change{title_range}", fontweight="bold")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{label}/year  (green = improving, red = declining, gray = no data)")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\u2705 Quicklook PNG saved: {png_path}")


# ---------------------------------------------------------------------------
# Stage: generic 5-class vegetation-priority overlay (kept as a coarse,
# complementary view alongside the detailed per-intervention plan above)
# ---------------------------------------------------------------------------

def classify_vegetation_priority(
    ndvi_slope_map, latest_ndvi_da, house_mask_bool=None,
    ndvi_low_thresh=NDVI_BARE_THRESHOLD, decline_slope_thresh=None,
    out_path="vegetation_priority.tif",
):
    ndvi_now = np.asarray(latest_ndvi_da.values)
    if ndvi_now.ndim == 3:
        ndvi_now = ndvi_now.squeeze()

    if decline_slope_thresh is None:
        decline_slope_thresh = decline_threshold(ndvi_slope_map)

    declining = (ndvi_slope_map < decline_slope_thresh) & ~np.isnan(ndvi_slope_map)
    bare = ndvi_now < ndvi_low_thresh

    cls = np.full(ndvi_now.shape, 4, dtype="uint8")
    cls[~bare & declining] = 3
    cls[bare & ~declining] = 2
    cls[bare & declining] = 1
    cls[np.isnan(ndvi_now) | np.isnan(ndvi_slope_map)] = 0
    if house_mask_bool is not None:
        if house_mask_bool.shape != cls.shape:
            raise ValueError(
                f"house_mask_bool shape {house_mask_bool.shape} does not match "
                f"NDVI grid shape {cls.shape} - reproject it onto the NDVI grid first."
            )
        cls[house_mask_bool] = 5

    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 1, "nodata": 0,
        "height": cls.shape[0], "width": cls.shape[1],
        "transform": latest_ndvi_da.rio.transform(), "crs": latest_ndvi_da.rio.crs,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(cls, 1)
    print(f"\u2705 Vegetation intervention-priority map saved to {out_path} "
          f"(decline threshold used: {decline_slope_thresh:.5f} NDVI/yr)")
    return out_path


_LOCAL_BASEMAP_CACHE = {}  # (mosaic_path, crs_string) -> (display_array, extent) - avoids
                           # re-reprojecting the same mosaic for every one of the 17 maps drawn per run


def add_local_satellite_basemap(ax, target_crs, mosaic_path=None):
    """Plots the already-downloaded satellite mosaic (MOSAIC_TIFF, from
    Stage 0) as the basemap instead of fetching tiles live from Google -
    avoids the repeat 'Connection reset by peer' / 400 errors that live
    tile fetching hit during plotting, since the mosaic was already
    downloaded once for the house mask. Reprojected result is cached so the
    16 individual per-intervention maps don't each redo the same warp."""
    mosaic_path = MOSAIC_TIFF if mosaic_path is None else mosaic_path
    if not Path(mosaic_path).exists():
        print(f"  \u26a0\ufe0f local satellite mosaic not found at {mosaic_path}; plotting without a basemap.")
        return
    cache_key = (str(mosaic_path), target_crs.to_string() if target_crs else "EPSG:4326")
    cached = _LOCAL_BASEMAP_CACHE.get(cache_key)
    if cached is None:
        with rasterio.open(mosaic_path) as src:
            if src.crs == target_crs:
                rgba = src.read()
                left, bottom, right, top = src.bounds
            else:
                dst_transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                rgba = np.zeros((src.count, height, width), dtype=src.dtypes[0])
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i), destination=rgba[i - 1],
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=dst_transform, dst_crs=target_crs,
                        resampling=Resampling.bilinear,
                    )
                left, bottom, right, top = rasterio.transform.array_bounds(height, width, dst_transform)
        display = np.transpose(rgba, (1, 2, 0))  # (bands, H, W) -> (H, W, bands), RGB or RGBA
        cached = (display, [left, right, bottom, top])
        _LOCAL_BASEMAP_CACHE[cache_key] = cached
    display, extent = cached
    ax.imshow(display, extent=extent, zorder=0)
    return extent


def add_map_furniture(ax, extent, source_label="Google Satellite (downloaded mosaic)"):
    """North arrow, dynamic scale bar, non-scientific lon/lat tick labels,
    and a source attribution - the same cartographic elements
    create_visualizations() already draws on the slope map, applied here to
    the intervention maps too. White-on-black-halo styling so it stays
    legible over a busy satellite image (the slope map's plain black works
    fine there because it sits on a matplotlib colormap, not a photo)."""
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)

    halo = [pe.withStroke(linewidth=3, foreground="black")]
    ax.text(0.95, 0.95, "N", transform=ax.transAxes, ha="center", va="center",
            weight="bold", fontsize=12, color="white", path_effects=halo, zorder=5)
    ax.annotate("", xy=(0.95, 0.93), xytext=(0.95, 0.85),
                arrowprops=dict(arrowstyle="->", color="white", linewidth=1.5),
                transform=ax.transAxes, zorder=5)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    avg_lat = (y_min + y_max) / 2
    width_m = (x_max - x_min) * (111320 * np.cos(np.radians(avg_lat)))
    for threshold, scale_len, scale_lbl in [
        (20000, 10000, "10 km"), (5000, 2000, "2 km"), (2000, 1000, "1 km"),
        (1000, 500, "500 m"), (200, 100, "100 m"), (0, 20, "20 m"),
    ]:
        if width_m > threshold:
            break
    scale_ratio = scale_len / width_m if width_m > 0 else 0.1
    ax.plot([0.05, 0.05 + scale_ratio], [0.05, 0.05], transform=ax.transAxes,
            color="white", linewidth=3, path_effects=halo, zorder=5)
    ax.text(0.05 + scale_ratio / 2, 0.07, scale_lbl, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="white", path_effects=halo, zorder=5)
    ax.text(0.98, 0.01, f"Basemap: {source_label}", transform=ax.transAxes, ha="right",
            fontsize=7, color="white", path_effects=halo, zorder=5)


def create_vegetation_priority_map(class_path, gdf, save_name="vegetation_priority_map.png"):
    print("\U0001f5fa\ufe0f  Generating vegetation intervention-priority map...")
    with rasterio.open(class_path) as src:
        arr = src.read(1)
        transform_ = src.transform
        crs = src.crs
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    fig, ax = plt.subplots(figsize=(10, 10))
    add_local_satellite_basemap(ax, target_crs=crs)

    counts = {}
    for code, (label, color) in VEG_PRIORITY_CLASSES.items():
        cls_mask = arr == code
        counts[code] = int(cls_mask.sum())
        if not cls_mask.any():
            continue
        geoms = [shp_shape(geom) for geom, _ in rio_shapes(arr, mask=cls_mask, transform=transform_)]
        gpd.GeoDataFrame(geometry=geoms, crs=crs).plot(
            ax=ax, facecolor=color, edgecolor=color, alpha=0.45, linewidth=0, zorder=2
        )

    gdf.plot(ax=ax, facecolor="none", edgecolor="yellow", linewidth=1.5, zorder=3)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    for axis in [ax.xaxis, ax.yaxis]:
        f = ScalarFormatter(useOffset=False)
        f.set_scientific(False)
        axis.set_major_formatter(f)
    ax.set_title("Vegetation Intervention Priority\n(multi-year NDVI trend + current level, structures excluded)", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    legend_elements = [
        mlines.Line2D([], [], marker="s", linestyle="None", markersize=12,
                      markerfacecolor=color, markeredgecolor=color, alpha=0.7,
                      label=f"{label} ({counts.get(code, 0)} px)")
        for code, (label, color) in VEG_PRIORITY_CLASSES.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\u2705 Saved {save_name}")
    return save_name


# ---------------------------------------------------------------------------
# Stage: DEM download + slope/aspect (from NDVI_NDWI_mask_v3.py)
# ---------------------------------------------------------------------------

def dem_service_bounds(bounds, minimum_side_m=300.0):
    """Pad only the download envelope to meet the service's ~250 m minimum.
    The 300 m target adds a margin. Analysis and recommendation geometry are
    unchanged; main() clips the returned DEM to the analysis rectangle.
    """
    west,south,east,north=map(float,bounds)
    if not np.isfinite([west,south,east,north]).all() or not (-180<=west<east<=180 and -90<south<north<90):
        raise ValueError('DEM request requires ordered, non-polar WGS84 bounds.')
    if not np.isfinite(minimum_side_m) or minimum_side_m<=250:
        raise ValueError('DEM download minimum side must exceed 250 metres.')
    geod=pyproj.Geod(ellps='WGS84')
    for _ in range(10):
        width=min(abs(geod.inv(west,lat,east,lat)[2]) for lat in (south,north))
        height=abs(geod.inv(west,south,west,north)[2])
        if width>=minimum_side_m and height>=minimum_side_m:
            return [west,south,east,north]
        if min(width,height)<=0: raise ValueError('DEM bounding box has no measurable extent.')
        if height<minimum_side_m:
            margin=(north-south)*(minimum_side_m*1.001/height-1)/2
            south-=margin;north+=margin
        if width<minimum_side_m:
            margin=(east-west)*(minimum_side_m*1.001/width-1)/2
            west-=margin;east+=margin
        if not (-180<=west<east<=180 and -90<south<north<90):
            raise ValueError('Minimum DEM envelope crosses the dateline/pole; supply DEM_INPUT for this AOI.')
    raise ValueError('Could not construct a service-compatible DEM download envelope.')


def download_dem(bounds, api_key, dem_type="COP30", output="dem.tif"):
    """Keep the dataset; pad undersized download bounds and report service errors.
    Do not silently replace a rejected SRTM request with a different DEM source.
    """
    import json,tempfile
    try: west,south,east,north=map(float,bounds)
    except (TypeError,ValueError):raise ValueError('DEM bounds must be west, south, east, north in decimal degrees.')
    if not np.isfinite([west,south,east,north]).all() or not (-180<=west<east<=180 and -90<=south<north<=90):
        raise ValueError('Invalid DEM bounds: expected ordered WGS84 west/south/east/north.')
    key=str(api_key or '').strip()
    if not key:
        raise RuntimeError('OPENTOPOGRAPHY_API_KEY is empty. Set it in this terminal, or supply an existing DEM through DEM_INPUT.')
    analysis_bounds=[west,south,east,north]
    west,south,east,north=dem_service_bounds(analysis_bounds)
    padded=[west,south,east,north]!=analysis_bounds
    if padded:
        print('DEM service minimum: download envelope expanded to at least 300 m per side; analysis rectangle unchanged.')
    params={'demtype':dem_type,'south':south,'north':north,'west':west,'east':east,
            'outputFormat':'GTiff','API_Key':key}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    report_path=output.with_suffix('.download_error.json')
    output.with_suffix('.request.json').write_text(json.dumps(dict(
        dataset=dem_type,analysis_bounds_wgs84=analysis_bounds,
        download_bounds_wgs84=[west,south,east,north],minimum_download_side_m=300,
        padded_for_service=padded,post_download='Clip to unchanged 25% analysis rectangle'),indent=2),encoding='utf-8')
    print(f'DEM request: {dem_type}; west={west:.8f}, south={south:.8f}, east={east:.8f}, north={north:.8f} (API key hidden)')
    def fail(message,status=None):
        clean=safe_dem_error(message,key)
        report=dict(service='OpenTopography globaldem',http_status=status,
                    dataset=dem_type,bounds_wgs84=[west,south,east,north],analysis_bounds_wgs84=analysis_bounds,detail=clean)
        report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise RuntimeError(f'DEM download failed'+(f' (HTTP {status})' if status is not None else '')+
            f': {clean}\nDiagnostic report: {report_path}\n'
            'Check the service message for API-key validity, dataset access, quota or request errors. '
            'Alternatively set DEM_INPUT to an existing elevation GeoTIFF covering the buffered rectangle.') from None
    try:
        response=requests.get('https://portal.opentopography.org/API/globaldem',params=params,timeout=(15,120))
    except requests.exceptions.RequestException as exc:
        fail(str(exc))
    if response.status_code!=200:
        fail(response.text,status=response.status_code)
    if response.content[:4] not in (b'II\x2a\x00',b'MM\x00\x2a',b'II\x2b\x00',b'MM\x00\x2b'):
        fail('Service returned HTTP 200 but no GeoTIFF. '+response.text,status=200)
    # Validate before replacing any existing output; never save HTML as a DEM.
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent,suffix='.tif',delete=False) as f:
            temporary=Path(f.name);f.write(response.content)
        with rasterio.open(temporary) as dataset:
            if dataset.crs is None or dataset.count<1 or min(dataset.width,dataset.height)<2:
                raise ValueError('Returned raster has missing CRS, bands or usable dimensions')
        temporary.replace(output)
    except Exception as exc:
        if temporary is not None:temporary.unlink(missing_ok=True)
        fail('Invalid/unwritable downloaded raster: '+str(exc),status=200)
    report_path.unlink(missing_ok=True)
    print(f'Downloaded {dem_type} to {output}')
    return str(output)


def upscale_dem(input_path, subdivide_factor=5, output_path="upscaled_dem.tif"):
    print(f"\U0001f680 Resampling {input_path} by factor {subdivide_factor}...")
    with rasterio.open(input_path) as src:
        new_height = int(src.height * subdivide_factor)
        new_width = int(src.width * subdivide_factor)
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width), (src.height / new_height)
        )
        profile = src.profile.copy()
        profile.update({"transform": new_transform, "height": new_height, "width": new_width})
        data = src.read(out_shape=(src.count, new_height, new_width), resampling=Resampling.bilinear)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
    print(f"\u2705 Upscale complete: {output_path}")
    return output_path


def calculate_slope_aspect(dem_file, gdf, slope_file="slope.tif", aspect_file="aspect.tif"):
    """Metre-based gradients; map products returned on the input DEM grid.
    The siting engine separately derives gradients from the native buffered DEM.
    """
    with rioxarray.open_rasterio(dem_file, masked=True) as src:
        reference = src.squeeze('band', drop=True).load()
    metric = reference.rio.reproject(gdf.estimate_utm_crs(), resampling=Resampling.bilinear)
    tr = metric.rio.transform()
    gy, gx = np.gradient(metric.values, abs(tr.e), abs(tr.a))
    slope = 100 * np.hypot(gx, gy)
    aspect = np.mod(np.degrees(np.arctan2(-gx, gy)), 360)
    aspect = np.where(slope > 1e-8, aspect, np.nan)
    for data, path, method in ((slope, slope_file, Resampling.bilinear), (aspect, aspect_file, Resampling.nearest)):
        metric.copy(data=data.astype('float32')).rio.write_nodata(np.nan).rio.reproject_match(reference, resampling=method).rio.to_raster(path)
    return slope_file, aspect_file


def create_visualizations(dem_path, slope_path, aspect_path, gdf, interval=CONTOUR_INTERVAL):
    print(f"\U0001f5bc\ufe0f Generating plots with {interval}m elevation contours on Slope Analysis...")
    with rasterio.open(dem_path) as src:
        elev_data = src.read(1)
        elev_data = np.ma.masked_invalid(np.ma.masked_equal(elev_data, src.nodata))
        contour_grid = np.flipud(elev_data)
        levels = np.arange(np.floor(elev_data.min() / interval) * interval, elev_data.max() + interval, interval)

    layers = [
        (dem_path, "Elevation (m)", "terrain", "dem_plot.png", "DEM Elevation", None, None, False),
        (slope_path, "Slope (%)", "RdYlGn_r", "slope_contour_plot.png", f"Slope & {interval}m Contours", 0, 100, True),
        (aspect_path, "Aspect (\u00b0)", "twilight", "aspect_plot.png", "Aspect Analysis", 0, 360, False),
    ]

    for (path, cbar_label, cmap, save_name, title, vmin, vmax, draw_contours) in layers:
        save_name = output_file(OUTPUT_DIR, save_name)
        with rasterio.open(path) as src:
            data = src.read(1)
            data = np.ma.masked_invalid(np.ma.masked_equal(data, src.nodata))
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

            fig, ax = plt.subplots(figsize=(10, 8))
            add_local_satellite_basemap(ax, src.crs)
            im = ax.imshow(data, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax, origin="upper", alpha=0.60)
            for axis in [ax.xaxis, ax.yaxis]:
                formatter = ScalarFormatter(useOffset=False)
                formatter.set_scientific(False)
                axis.set_major_formatter(formatter)
            if draw_contours:
                cs = ax.contour(contour_grid, levels=levels, extent=extent, colors="black", linewidths=0.5, alpha=0.7)
                ax.clabel(cs, inline=True, fontsize=8, fmt="%1.0f m")

            cbar = fig.colorbar(im, ax=ax, shrink=0.7)
            cbar.set_label(cbar_label)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Easting (m)" if src.crs.is_projected else "Longitude")
            ax.set_ylabel("Northing (m)" if src.crs.is_projected else "Latitude")
            gdf.to_crs(src.crs).plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.5)
            ax.text(0.95, 0.95, "N", transform=ax.transAxes, ha="center", weight="bold", fontsize=12)
            ax.annotate("", xy=(0.95, 0.93), xytext=(0.95, 0.85), arrowprops=dict(arrowstyle="->"), transform=ax.transAxes)
            ax.text(0.98, 0.01, "Source: CartoDEM", transform=ax.transAxes, ha="right", fontsize=8)

            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            avg_lat = (y_min + y_max) / 2
            width_m = (x_max - x_min) if src.crs.is_projected else (x_max - x_min) * (111320 * np.cos(np.radians(avg_lat)))
            if width_m > 20000:
                scale_len, scale_lbl = 10000, "10 km"
            elif width_m > 5000:
                scale_len, scale_lbl = 2000, "2 km"
            elif width_m > 2000:
                scale_len, scale_lbl = 1000, "1 km"
            elif width_m > 1000:
                scale_len, scale_lbl = 500, "500 m"
            else:
                scale_len, scale_lbl = 100, "100 m"
            scale_ratio = scale_len / width_m
            ax.plot([0.05, 0.05 + scale_ratio], [0.05, 0.05], transform=ax.transAxes, color="black", linewidth=3)
            ax.text(0.05 + (scale_ratio / 2), 0.07, scale_lbl, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, fontweight="bold")

            plt.savefig(save_name, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved {save_name}")

    return "Plots Saved"


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

ANALYSIS_BUFFER_FRACTION = 0.25


def create_siting_layout_map(ctx, results, out_dir):
    out_dir = Path(out_dir)
    fig,ax=plt.subplots(figsize=(11,9)); t=ctx['transform']; h,w=ctx['dem'].shape
    extent=(t.c,t.c+w*t.a,t.f+h*t.e,t.f)
    add_local_satellite_basemap(ax, ctx['crs'])
    ax.imshow(np.ma.masked_where(~ctx['valid'],ctx['dem']),extent=extent,cmap='terrain',alpha=.6)
    colors={'High':'#b2182b','Medium':'#e08214','Low':'#2166ac'}
    for r in results:
        for geom,d in zip(r['_site_geometries'],r['_site_priorities']):
            xy=np.array(geom.coords if geom.geom_type=='LineString' else geom.exterior.coords)
            ax.plot(xy[:,0],xy[:,1],color=colors[d['location_priority']],linewidth=1)
    if 'recommendation_geometry' in ctx:
        original=ctx['recommendation_geometry']
        polygons=list(original.geoms) if original.geom_type=='MultiPolygon' else [original]
        for polygon in polygons:
            for ring in [polygon.exterior]+list(polygon.interiors):
                coords=np.array(ring.coords);ax.plot(coords[:,0],coords[:,1],color='black',linewidth=1.8)
    ax.set_title('Rule-screened layouts inside original AOI (black); buffered analysis background')
    ax.set_xlabel('Easting (m)');ax.set_ylabel('Northing (m)');ax.set_aspect('equal')
    ax.legend(handles=[mlines.Line2D([],[],color=c,label=k) for k,c in colors.items()])
    fig.savefig(output_file(out_dir, 'siting_layouts_ranked.png'),dpi=180,bbox_inches='tight');plt.close(fig)


def safe_dem_error(text, api_key):
    """Readable service diagnostics, with credentials and request URLs removed."""
    import re,html
    from urllib.parse import quote,quote_plus
    from html.parser import HTMLParser
    text=html.unescape(str(text))
    # Redact before parsing as well as afterward (HTML may split text nodes).
    secrets={str(api_key or ''),str(api_key or '').strip()}
    for secret in list(secrets):
        if secret: secrets.update((quote(secret,safe=''),quote_plus(secret)))
    for secret in sorted(secrets,key=len,reverse=True):
        if secret: text=text.replace(secret,'[REDACTED]')
    class TextOnly(HTMLParser):
        def __init__(self): super().__init__();self.parts=[];self.hidden=0
        def handle_starttag(self,tag,attrs):
            if tag in ('script','style'):self.hidden+=1
        def handle_endtag(self,tag):
            if tag in ('script','style'):self.hidden=max(0,self.hidden-1)
        def handle_data(self,data):
            if not self.hidden:self.parts.append(data)
    parser=TextOnly();parser.feed(text);text=' '.join(parser.parts)
    for secret in sorted(secrets,key=len,reverse=True):
        if secret:text=text.replace(secret,'[REDACTED]')
    text=re.sub(r'https?://\S+','[request URL omitted]',text)
    text=re.sub(r'(?i)(api[_ -]?key\s*[:=]\s*)[^\s&,;]+',r'\1[REDACTED]',text)
    return re.sub(r'\s+',' ',text).strip()[:1200] or 'No readable response body.'


def analysis_geometries(original_wgs84, metric_crs, fraction=ANALYSIS_BUFFER_FRACTION):
    from shapely.geometry import box
    if original_wgs84.is_empty or not original_wgs84.is_valid or original_wgs84.geom_type not in ('Polygon','MultiPolygon'):
        raise ValueError('AOI must be a valid Polygon or MultiPolygon (holes are preserved).')
    if not np.isfinite(fraction) or fraction<0: raise ValueError('Buffer fraction must be non-negative')
    to_m=pyproj.Transformer.from_crs('EPSG:4326',metric_crs,always_xy=True).transform
    to_wgs=pyproj.Transformer.from_crs(metric_crs,'EPSG:4326',always_xy=True).transform
    original=shp_transform(to_m,original_wgs84)
    left,bottom,right,top=original.bounds; width=right-left; height=top-bottom
    if width<=0 or height<=0: raise ValueError('AOI has zero width or height')
    expanded=box(left-fraction*width,bottom-fraction*height,right+fraction*width,top+fraction*height)
    return original,expanded,shp_transform(to_wgs,expanded)


def save_polygon_kml(geometry_wgs84, path, name):
    """Write standard WGS84 polygon KML, including multipolygons and holes."""
    import xml.etree.ElementTree as ET
    namespace='http://www.opengis.net/kml/2.2';ET.register_namespace('',namespace)
    def tag(name):return '{'+namespace+'}'+name
    root=ET.Element(tag('kml'));doc=ET.SubElement(root,tag('Document'))
    placemark=ET.SubElement(doc,tag('Placemark'));ET.SubElement(placemark,tag('name')).text=name
    parts=list(geometry_wgs84.geoms) if geometry_wgs84.geom_type=='MultiPolygon' else [geometry_wgs84]
    container=ET.SubElement(placemark,tag('MultiGeometry')) if len(parts)>1 else placemark
    for polygon in parts:
        item=ET.SubElement(container,tag('Polygon'))
        for label,ring in [('outerBoundaryIs',polygon.exterior)]+[('innerBoundaryIs',r) for r in polygon.interiors]:
            boundary=ET.SubElement(item,tag(label));linear=ET.SubElement(boundary,tag('LinearRing'))
            ET.SubElement(linear,tag('coordinates')).text=' '.join(f'{x:.12f},{y:.12f},0' for x,y,*rest in ring.coords)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    return path


def prepare_analysis_aoi(gdf, out_dir):
    import hashlib,json
    from shapely.geometry import mapping
    if gdf.crs is None: raise ValueError('AOI requires a declared CRS')
    original_wgs=union_of(gdf.to_crs('EPSG:4326'));metric_crs=gdf.estimate_utm_crs()
    original,expanded,expanded_wgs=analysis_geometries(original_wgs,metric_crs)
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    kml=save_polygon_kml(expanded_wgs,output_file(out_dir, 'buffered_aoi_25pct.kml'),'Buffered analysis AOI - 25 percent on each side')
    save_polygon_kml(original_wgs,output_file(out_dir, 'original_recommendation_aoi.kml'),'Original AOI - recommendations only')
    identity=dict(original_wkb=original_wgs.wkb_hex,metric_crs=str(metric_crs),fraction=ANALYSIS_BUFFER_FRACTION,
                  season='April-June',cloud_cover=CLOUD_COVER_MAX,max_scenes=MAX_SCENES_PER_YEAR,min_scenes=MIN_SCENES_PER_YEAR)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    left,bottom,right,top=original.bounds
    report=dict(buffer_fraction_each_side=ANALYSIS_BUFFER_FRACTION,
                left_right_buffer_m=(right-left)*ANALYSIS_BUFFER_FRACTION,
                top_bottom_buffer_m=(top-bottom)*ANALYSIS_BUFFER_FRACTION,
                original_bounds_metric=list(original.bounds),analysis_bounds_metric=list(expanded.bounds),
                metric_crs=str(metric_crs),cache_key=key,
                recommendation_rule='Entire construction footprint covered by original polygon; holes excluded.')
    (output_file(out_dir, 'analysis_aoi_metadata.json')).write_text(json.dumps(report,indent=2))
    features=[dict(type='Feature',properties={'role':'original_recommendation_aoi'},geometry=mapping(original_wgs)),
              dict(type='Feature',properties={'role':'buffered_analysis_aoi'},geometry=mapping(expanded_wgs))]
    (output_file(out_dir, 'analysis_boundaries.geojson')).write_text(json.dumps(dict(type='FeatureCollection',features=features)))
    return gpd.GeoDataFrame(geometry=[expanded_wgs],crs='EPSG:4326'),kml,original_wgs,key


def restrict_recommendations_to_aoi(ctx, original_wgs84):
    """Retain buffered terrain/hydrology; add separate original-AOI eligibility."""
    from rasterio.features import geometry_mask
    project=pyproj.Transformer.from_crs('EPSG:4326',ctx['crs'],always_xy=True).transform
    original=shp_transform(project,original_wgs84)
    ctx['recommendation_geometry']=original
    ctx['recommendation_mask']=geometry_mask([original],out_shape=ctx['valid'].shape,
        transform=ctx['transform'],invert=True,all_touched=False)
    if not np.any(ctx['recommendation_mask']&ctx['valid']):
        raise ValueError('No valid analysis pixels inside original AOI.')
    return ctx


def parse_arguments(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Analyse a 25%-expanded rectangle; recommend interventions only inside the original KML.')
    parser.add_argument('--kml', required=True, type=Path, help='Input AOI polygon KML file')
    parser.add_argument('--buildings', type=Path, help='Field-checked building footprint polygons (GeoJSON, GPKG or KML); required for roof-pit sources')
    parser.add_argument('--dem-subdivide-factor', type=int, default=5, help='Common analysis grid subdivision of native metric DEM pixels (default 5; interpolation only)')
    parser.add_argument('--template', type=Path, default=Path(__file__).with_name('Interventions.xlsx'),
                        help='Interventions.xlsx reference template (default: beside this script)')
    args = parser.parse_args(argv)
    if args.dem_subdivide_factor < 1: parser.error('--dem-subdivide-factor must be >=1')
    if args.buildings:
        args.buildings=args.buildings.expanduser().resolve()
        if not args.buildings.is_file(): parser.error(f'Building footprints not found: {args.buildings}')
    args.template = args.template.expanduser().resolve()
    if not args.template.is_file(): parser.error(f'Excel template does not exist: {args.template}; supply --template')
    args.kml = args.kml.expanduser().resolve()
    if not args.kml.is_file(): parser.error(f'KML file does not exist: {args.kml}')
    if args.kml.suffix.lower() != '.kml': parser.error('--kml must be a .kml file')
    return args


def main(argv=None):
    global YEARLY_GEOTIFF_DIR, YEARLY_QUICKLOOK_DIR
    args = parse_arguments(argv)
    configure_output_paths(args.kml)
    print('1. Reading original recommendation AOI from --kml...')
    gdf = gpd.read_file(args.kml, driver='KML')
    if gdf.crs is None: raise ValueError('KML must define a geographic coordinate reference system')
    gdf = gdf.to_crs('EPSG:4326')

    print('2. Creating rectangular analysis KML: 25% width/height on each side...')
    analysis_gdf, analysis_kml, original_wgs, cache_key = prepare_analysis_aoi(gdf, OUTPUT_DIR)
    gdf_bounds = analysis_gdf.total_bounds
    kml_geom = union_of(analysis_gdf)
    YEARLY_GEOTIFF_DIR = OUTPUT_DIR / 'yearly_outputs' / ('buffer25_' + cache_key) / 'geotiff'
    YEARLY_QUICKLOOK_DIR = OUTPUT_DIR / 'yearly_outputs' / ('buffer25_' + cache_key) / 'quicklook'

    print('3. Acquiring rectangular Google satellite, DEM and NDVI/NDWI/NDMI data...')
    _LOCAL_BASEMAP_CACHE.clear()
    if Path(MOSAIC_TIFF).is_file() and Path(MOSAIC_PNG).is_file():
        print(f"   [\u2713] {Path(MOSAIC_TIFF).name} / {Path(MOSAIC_PNG).name} already downloaded - "
              "using cached satellite mosaic, skipping tile download.")
    else:
        download_google_satellite_from_kml(analysis_kml, MOSAIC_TIFF, MOSAIC_PNG, ZOOM_LEVEL)

    dem_output_path = output_file(OUTPUT_DIR, 'dem_buffer25_' + cache_key + '.tif')
    if DEM_INPUT:
        source_dem = DEM_INPUT
    elif Path(dem_output_path).is_file():
        print(f"   [\u2713] {Path(dem_output_path).name} already downloaded - using cached DEM, skipping download.")
        source_dem = str(dem_output_path)
    else:
        source_dem = download_dem(gdf_bounds, api_key, output=str(dem_output_path))
    if not source_dem or not Path(source_dem).is_file():
        raise ValueError('DEM acquisition failed. Supply DEM_INPUT or check the DEM service/API key.')
    # Download services accept a geographic bounding box. Clip that DEM to the
    # same exact rectangular analysis polygon used for satellite/index clipping.
    raw_dem = output_file(OUTPUT_DIR, 'dem_analysis_rectangle_' + cache_key + '.tif')
    with rioxarray.open_rasterio(source_dem, masked=True) as src:
        native = src.squeeze('band', drop=True)
        native.rio.clip(list(analysis_gdf.to_crs(native.rio.crs).geometry)).rio.to_raster(raw_dem)
    metric_dem_path,analysis_grid=prepare_dem_analysis_grid(raw_dem,analysis_gdf,args.dem_subdivide_factor,OUTPUT_DIR)
    catalog = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
        modifier=planetary_computer.sign_inplace)
    annual_ndvis, annual_ndwis, annual_ndmis, years = [], [], [], []
    print(f"   Building cloud-masked peak-summer (April-June) median composites per year, {START_YEAR}-{END_YEAR}...")
    for year in range(START_YEAR, END_YEAR + 1):
        cached_ndvi = load_cached_year_composite(year, "NDVI")
        cached_ndwi = load_cached_year_composite(year, "NDWI")
        cached_ndmi = load_cached_year_composite(year, "NDMI")
        use_cache = all(v is not None for v in (cached_ndvi, cached_ndwi, cached_ndmi))
        if use_cache:
            print(f"   [\u2713] {year}: NDVI_{year}.tif / NDWI_{year}.tif already downloaded - "
                  "using cached composite, skipping STAC search & Sentinel-2 download.")
            ndvi_comp, ndwi_comp, ndmi_comp = cached_ndvi, cached_ndwi, cached_ndmi
        else:
            time.sleep(1)
            try:
                ndvi_comp, ndwi_comp, ndmi_comp, n_used, dates = build_year_composite(year, catalog, gdf_bounds, kml_geom)
            except Exception as e:
                print(f"   [-] {year}: Skipped due to API error. ({e})")
                continue
            if ndvi_comp is None:
                print(f"   [-] {year}: Skipped - fewer than {MIN_SCENES_PER_YEAR} clear scenes found.")
                continue
            print(f"   [+] {year}: Composited from {n_used} scene(s): {', '.join(dates)}")

        ndvi_comp=align_to_dem_grid(ndvi_comp,analysis_grid,f'NDVI {year}')
        ndwi_comp=align_to_dem_grid(ndwi_comp,analysis_grid,f'NDWI {year}')
        ndmi_comp=align_to_dem_grid(ndmi_comp,analysis_grid,f'NDMI {year}')
        if not use_cache:  # refresh all three indices together when NDMI cache is absent
            save_year_outputs(ndvi_comp, year, "NDVI", make_plot=False)
            save_year_outputs(ndwi_comp, year, "NDWI", make_plot=False)
            save_year_outputs(ndmi_comp, year, "NDMI", make_plot=False)
        annual_ndvis.append(ndvi_comp)
        annual_ndwis.append(ndwi_comp)
        annual_ndmis.append(ndmi_comp)
        years.append(year)

    if len(annual_ndvis) < 3:
        raise ValueError(
            f"Only {len(annual_ndvis)} valid year(s) available. At least 3 are required "
            "for the rate-of-change regression. Widen the year range, raise CLOUD_COVER_MAX, "
            "or lower MIN_SCENES_PER_YEAR."
        )


    print('4. Analysing terrain, structures and multi-year trends over the rectangle...')
    generate_house_mask_tiff(MOSAIC_TIFF, HOUSE_MASK_TIFF)
    house_mask_bool,building_context=prepare_building_layer(analysis_grid,HOUSE_MASK_TIFF,args.buildings)
    analysis_grid.copy(data=house_mask_bool.astype('uint8')).rio.write_nodata(0).rio.to_raster(
        output_file(OUTPUT_DIR,'building_exclusion_analysis_grid.tif'))
    print(building_context['building_source_note'])
    stacked_ndvi = xr.concat(annual_ndvis, dim='time').values
    stacked_ndwi = xr.concat(annual_ndwis, dim='time').values
    stacked_ndmi = xr.concat(annual_ndmis, dim='time').values
    rainfall = load_rainfall(years, annual_ndvis[0])
    evidence = build_evidence(stacked_ndvi, stacked_ndwi, stacked_ndmi, years, rainfall)
    ndvi_slope_map = evidence['ndvi']['slope']
    ndwi_slope_map = evidence['ndwi']['slope']
    export_methodology(evidence, annual_ndvis[0], PLAN_OUTPUT_DIR / 'trend_diagnostics')
    priority_path = classify_vegetation_priority(ndvi_slope_map, annual_ndvis[-1], house_mask_bool=house_mask_bool,
        out_path=output_file(OUTPUT_DIR, 'vegetation_priority.tif'))
    # Derive terrain at native resolution, then align all terrain products to the
    # SAME master grid used by every annual spectral layer and the building mask.
    cropped_dem=output_file(OUTPUT_DIR,'dem_analysis_grid.tif')
    siting=build_siting_context(metric_dem_path,analysis_grid,house_mask_bool)
    siting.update(building_context)
    slope_path=output_file(OUTPUT_DIR,'slope.tif');aspect_path=output_file(OUTPUT_DIR,'aspect.tif')
    for key,path in (('slope_pct',slope_path),('aspect',aspect_path)):
        analysis_grid.copy(data=siting[key].astype('float32')).rio.write_nodata(np.nan).rio.to_raster(path)

    print('5. Ranking eligible intervention footprints inside the original KML boundary...')
    restrict_recommendations_to_aoi(siting, original_wgs)
    results, points_by_id, csv_path = plan_interventions(
        annual_ndvis[-1], annual_ndwis[-1], ndvi_slope_map, ndwi_slope_map,
        siting['slope_pct'], house_mask_bool, evidence=evidence, siting=siting)
    excel_path = export_intervention_plan_excel(results, points_by_id, annual_ndvis[-1],
                                              output_file(PLAN_OUTPUT_DIR, 'intervention_recommendation.xlsx'), template_path=args.template)

    print('6. Generating maps on the rectangular Google satellite basemap...')
    for i, year in enumerate(years):
        for label, layers in (('NDVI', annual_ndvis), ('NDWI', annual_ndwis), ('NDMI', annual_ndmis)):
            save_year_outputs(layers[i], year, label)
    year_range = (years[0], years[-1])
    for label in ('NDVI', 'NDWI', 'NDMI'):
        convert_slope_to_grayscale_mask(evidence[label.lower()]['slope'], annual_ndvis[0],
                                       str(output_file(OUTPUT_DIR, f'{label}_mask.tif')), label=label, year_range=year_range)
    create_vegetation_priority_map(priority_path, analysis_gdf,
        save_name=output_file(OUTPUT_DIR, 'vegetation_priority_map.png'))
    create_visualizations(cropped_dem, slope_path, aspect_path, analysis_gdf)
    create_siting_layout_map(siting, results, PLAN_OUTPUT_DIR)
    create_intervention_plan_map(results, points_by_id, annual_ndvis[-1], gdf)
    print(f'Complete. Buffered KML: {analysis_kml}; original boundary: {output_file(OUTPUT_DIR, "original_recommendation_aoi.kml")}')
    print(f'Ranked results: {csv_path} and {excel_path}')


if __name__ == "__main__":
    main()
