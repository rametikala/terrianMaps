"""
Terrain-only generator: downloads the DEM (OpenTopography COP30) and produces
three maps for an AOI KML: DEM elevation, slope with contours, and aspect.

Run:  python terrain_maps.py --kml /path/to/aoi.kml [--no-basemap] [--interval 3]
Env:  OPENTOPOGRAPHY_API_KEY (required unless DEM_INPUT is set), OUTPUT_DIR,
      DEM_INPUT (optional existing elevation GeoTIFF in metres covering the AOI).

The AOI is expanded 25% left/right and top/bottom (as in the full pipeline) so
edge slopes are not distorted. Outputs go to OUTPUT_DIR/<AOI>/ :
  <AOI>_dem_plot.png, <AOI>_slope_contour_plot.png, <AOI>_aspect_plot.png
  plus slope.tif, aspect.tif, cropped_dem.tif (GeoTIFFs).
"""

import os
import warnings
from io import BytesIO
from pathlib import Path

import fiona
import geopandas as gpd
import mercantile
import numpy as np
import pyproj
import rasterio
import requests
import rioxarray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from PIL import Image
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject
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
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_DIR", "./"))
OUTPUT_DIR = OUTPUT_ROOT
AOI_OUTPUT_NAME = ""
MOSAIC_TIFF = None
MOSAIC_PNG = None
ZOOM_LEVEL = 19
GOOGLE_SAT_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
api_key = os.environ.get("OPENTOPOGRAPHY_API_KEY")
DEM_INPUT = os.environ.get("DEM_INPUT", "")
CONTOUR_INTERVAL = 5
ANALYSIS_BUFFER_FRACTION = 0.25
_LOCAL_BASEMAP_CACHE = {}


def output_file(directory, filename):
    name = Path(filename).name
    return Path(directory) / (f"{AOI_OUTPUT_NAME}_{name}" if AOI_OUTPUT_NAME else name)


def configure_output_paths(kml_path):
    """Reserve a fresh AOI folder; never overwrite an earlier run."""
    global AOI_OUTPUT_NAME, OUTPUT_DIR, MOSAIC_TIFF, MOSAIC_PNG
    import re
    stem = Path(kml_path).stem
    AOI_OUTPUT_NAME = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', stem).strip(' .') or 'AOI'
    OUTPUT_ROOT.expanduser().mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        folder = AOI_OUTPUT_NAME if number == 1 else f"{AOI_OUTPUT_NAME}_{number:03d}"
        candidate = OUTPUT_ROOT.expanduser() / folder
        try:
            candidate.mkdir()
            break
        except FileExistsError:
            number += 1
    OUTPUT_DIR = candidate.resolve()
    MOSAIC_TIFF = output_file(OUTPUT_DIR, 'google_satellite_latest.tif')
    MOSAIC_PNG = output_file(OUTPUT_DIR, 'google_satellite_latest.png')
    print(f'Output folder: {OUTPUT_DIR}')


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
        elev_data = np.ma.masked_equal(elev_data, src.nodata)
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
            data = np.ma.masked_equal(data, src.nodata)
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
            ax.text(0.98, 0.01, "Source: COP30", transform=ax.transAxes, ha="right", fontsize=8)

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
    """Create the 25%-expanded rectangular AOI (WGS84) and save it as KML."""
    if gdf.crs is None:
        raise ValueError('AOI requires a declared CRS')
    original_wgs = union_of(gdf.to_crs('EPSG:4326'))
    metric_crs = gdf.estimate_utm_crs()
    _, _, expanded_wgs = analysis_geometries(original_wgs, metric_crs)
    kml = save_polygon_kml(expanded_wgs, output_file(out_dir, 'buffered_aoi_25pct.kml'),
                           'Buffered analysis AOI - 25 percent on each side')
    return gpd.GeoDataFrame(geometry=[expanded_wgs], crs='EPSG:4326'), kml


def parse_arguments(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Download DEM and generate DEM, slope+contour and aspect maps.')
    parser.add_argument('--kml', required=True, type=Path, help='Input AOI polygon KML file')
    parser.add_argument('--interval', type=float, default=CONTOUR_INTERVAL, help='Contour interval in metres (default 3)')
    parser.add_argument('--no-basemap', action='store_true', help='Skip Google satellite download; plot on a plain background')
    args = parser.parse_args(argv)
    args.kml = args.kml.expanduser().resolve()
    if not args.kml.is_file():
        parser.error(f'KML file does not exist: {args.kml}')
    if args.kml.suffix.lower() != '.kml':
        parser.error('--kml must be a .kml file')
    return args


def main(argv=None):
    args = parse_arguments(argv)
    configure_output_paths(args.kml)

    print('1. Reading AOI...')
    gdf = gpd.read_file(args.kml, driver='KML')
    if gdf.crs is None:
        raise ValueError('KML must define a geographic coordinate reference system')
    gdf = gdf.to_crs('EPSG:4326')

    print('2. Creating rectangular analysis AOI (25% on each side)...')
    analysis_gdf, analysis_kml = prepare_analysis_aoi(gdf, OUTPUT_DIR)
    bounds = analysis_gdf.total_bounds

    if not args.no_basemap:
        print('3a. Downloading Google satellite basemap...')
        try:
            download_google_satellite_from_kml(analysis_kml, MOSAIC_TIFF, MOSAIC_PNG, ZOOM_LEVEL)
        except Exception as e:
            print(f'   Basemap download failed ({e}); continuing without basemap.')

    print('3b. Acquiring DEM...')
    source_dem = DEM_INPUT or download_dem(bounds, api_key, output=str(output_file(OUTPUT_DIR, 'dem_download.tif')))
    if not source_dem or not Path(source_dem).is_file():
        raise ValueError('DEM acquisition failed. Supply DEM_INPUT or check the DEM service/API key.')

    raw_dem = output_file(OUTPUT_DIR, 'dem_analysis_rectangle.tif')
    with rioxarray.open_rasterio(source_dem, masked=True) as src:
        native = src.squeeze('band', drop=True)
        native.rio.clip(list(analysis_gdf.to_crs(native.rio.crs).geometry)).rio.to_raster(raw_dem)

    print('4. Computing slope and aspect...')
    high_res_dem = upscale_dem(raw_dem, subdivide_factor=5, output_path=output_file(OUTPUT_DIR, 'upscaled_dem.tif'))
    cropped_dem = output_file(OUTPUT_DIR, 'cropped_dem.tif')
    with rasterio.open(high_res_dem) as src:
        out_img, out_trans = mask(src, analysis_gdf.to_crs(src.crs).geometry, crop=True)
        meta = src.meta.copy()
        meta.update(height=out_img.shape[1], width=out_img.shape[2], transform=out_trans)
        with rasterio.open(cropped_dem, 'w', **meta) as dst:
            dst.write(out_img)
    slope_path, aspect_path = calculate_slope_aspect(cropped_dem, analysis_gdf,
        slope_file=output_file(OUTPUT_DIR, 'slope.tif'), aspect_file=output_file(OUTPUT_DIR, 'aspect.tif'))

    print('5. Generating DEM, slope+contour and aspect maps...')
    create_visualizations(cropped_dem, slope_path, aspect_path, analysis_gdf, interval=args.interval)
    print(f'Complete. Outputs in {OUTPUT_DIR}')


if __name__ == "__main__":
    main()
