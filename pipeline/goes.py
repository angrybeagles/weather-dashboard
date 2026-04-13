"""
GOES-16/18 satellite imagery pipeline.

Fetches Cloud & Moisture Imagery (CMI) products from AWS Open Data,
decodes NetCDF, and produces map-ready arrays.
"""

import logging
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import xarray as xr

from config import CONUS_BOUNDS, DATA_DIR, GOES_BUCKET, GOES_CHANNELS, GOES_CONUS_PRODUCT

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "goes"
CACHE_DIR.mkdir(exist_ok=True)


def _find_latest_goes_file(
    channel: int,
    satellite: str = "18",  # Updated to GOES-18
    product: str | None = None,
    scan_minutes_back: int = 120,
) -> str | None:
    """
    Search the GOES S3 bucket for the most recent file for a given channel.

    Returns the S3 key of the latest file, or None if not found.
    """
    import s3fs

    if product is None:
        product = GOES_CONUS_PRODUCT

    fs = s3fs.S3FileSystem(anon=True)
    bucket = f"noaa-goes{satellite}"
    now = datetime.now(timezone.utc)

    # Search backwards through time directories
    for minutes_ago in range(0, scan_minutes_back, 5):
        scan_time = now - timedelta(minutes=minutes_ago)
        day_of_year = scan_time.timetuple().tm_yday
        hour = scan_time.hour

        prefix = (
            f"{bucket}/{product}/{scan_time.year}/{day_of_year:03d}/{hour:02d}"
        )

        try:
            files = fs.ls(prefix)
            # Filter for our channel
            channel_str = f"M6C{channel:02d}"
            channel_files = [f for f in files if channel_str in f]
            if channel_files:
                # Return the most recent (last alphabetically = latest scan)
                return sorted(channel_files)[-1]
        except FileNotFoundError:
            continue

    return None


def fetch_goes_channel(
    channel_name: str = "ir",
    satellite: str = "18",  # Updated to GOES-18
) -> xr.DataArray | None:
    """
    Fetch the latest GOES imagery for a named channel.

    Parameters
    ----------
    channel_name : str
        One of: 'visible', 'shortwave_ir', 'water_vapor', 'ir'
    satellite : str
        '16' (GOES-East) or '18' (GOES-West)

    Returns
    -------
    xr.DataArray or None
        2D array with lat/lon coordinates, or None on failure.
    """
    import s3fs

    channel_num = GOES_CHANNELS.get(channel_name)
    if channel_num is None:
        logger.error("Unknown channel: %s", channel_name)
        return None

    s3_key = _find_latest_goes_file(channel_num, satellite)
    if s3_key is None:
        logger.warning("No GOES file found for channel %s", channel_name)
        return None

    logger.info("Fetching GOES: %s", s3_key)

    try:
        fs = s3fs.S3FileSystem(anon=True)
        with fs.open(s3_key, "rb") as f:
            ds = xr.open_dataset(BytesIO(f.read()), engine="h5netcdf")

        # The CMI variable contains the actual data
        cmi = ds["CMI"]

        # GOES uses fixed-grid projection (geostationary).
        # Convert x/y to lat/lon using the projection info.
        proj_info = ds["goes_imager_projection"]

        # Satellite height, longitude, sweep
        h = proj_info.attrs["perspective_point_height"]
        lon_0 = proj_info.attrs["longitude_of_projection_origin"]
        sweep = proj_info.attrs.get("sweep_angle_axis", "x")

        # x and y are scanning angles in radians
        x = ds["x"].values * h
        y = ds["y"].values * h

        # Convert geostationary projection to lat/lon
        # Using simplified conversion (accurate to ~0.01 deg for CONUS)
        from numpy import arctan, arctan2, cos, sin, sqrt

        req = 6378137.0  # equatorial radius
        rpol = 6356752.31414  # polar radius
        e = 0.0818191910435

        xx, yy = np.meshgrid(x, y)

        a = sin(xx / h) ** 2 + (
            cos(xx / h) ** 2
            * (cos(yy / h) ** 2 + (req**2 / rpol**2) * sin(yy / h) ** 2)
        )
        b = -2 * h * cos(xx / h) * cos(yy / h)
        c = h**2 - req**2

        discriminant = b**2 - 4 * a * c
        # Mask invalid (off-earth) pixels
        valid = discriminant >= 0
        rs = np.where(valid, (-b - sqrt(np.maximum(discriminant, 0))) / (2 * a), np.nan)

        sx = rs * cos(xx / h) * cos(yy / h)
        sy = -rs * sin(xx / h)
        sz = rs * cos(xx / h) * sin(yy / h)

        lat = np.degrees(
            arctan((req**2 / rpol**2) * sz / sqrt((h - sx) ** 2 + sy**2))
        )
        lon = np.degrees(arctan2(sy, h - sx)) + lon_0

        # Create DataArray with lat/lon
        result = xr.DataArray(
            cmi.values,
            dims=["y", "x"],
            coords={"latitude": (["y", "x"], lat), "longitude": (["y", "x"], lon)},
            attrs={
                "channel": channel_name,
                "channel_number": channel_num,
                "satellite": f"GOES-{satellite}",
                "scan_time": str(ds.attrs.get("time_coverage_start", "unknown")),
                "units": cmi.attrs.get("units", ""),
            },
        )

        return result

    except Exception as e:
        logger.error("Failed to fetch GOES channel %s: %s", channel_name, e)
        return None


def goes_to_plotly_image(
    da: xr.DataArray,
    bounds: dict | None = None,
    colorscale: str = "gray_r",
) -> dict:
    """
    Convert a GOES DataArray to a dict suitable for plotly's mapbox image layer.

    Returns dict with keys: 'source' (base64 PNG), 'coordinates' (corner coords).
    """
    from PIL import Image
    import base64

    if bounds is None:
        bounds = CONUS_BOUNDS

    # Use full-resolution source pixels. Stride-decimating in scan-angle
    # space produces geographically non-uniform point density (sparsest
    # near the satellite limb), which makes griddata(nearest) draw large
    # voronoi cells / vertical bands over the west coast for GOES-East.
    # Cast to float32 to halve the working buffer.
    lat = da.coords["latitude"].values.astype(np.float32, copy=False)
    lon = da.coords["longitude"].values.astype(np.float32, copy=False)
    data = da.values.astype(np.float32, copy=False)

    # Mask to CONUS bounds
    mask = (
        (lat >= bounds["lat_min"])
        & (lat <= bounds["lat_max"])
        & (lon >= bounds["lon_min"])
        & (lon <= bounds["lon_max"])
    )

    # Output grid (kept at 800x1200 — final image resolution)
    lat_bins = np.linspace(bounds["lat_min"], bounds["lat_max"], 800, dtype=np.float32)
    lon_bins = np.linspace(bounds["lon_min"], bounds["lon_max"], 1200, dtype=np.float32)

    from scipy.interpolate import griddata

    valid = ~np.isnan(data) & mask
    if valid.sum() < 100:
        logger.warning("Insufficient valid pixels for GOES rendering")
        return {}

    points = np.column_stack([lon[valid], lat[valid]])
    values = data[valid]

    lon_grid, lat_grid = np.meshgrid(lon_bins, lat_bins)
    gridded = griddata(points, values, (lon_grid, lat_grid), method="nearest")
    del points, values, lat, lon, data, mask, valid, lon_grid, lat_grid

    # Normalize to 0-255
    vmin, vmax = np.nanpercentile(gridded[~np.isnan(gridded)], [2, 98])
    normalized = np.clip((gridded - vmin) / (vmax - vmin + 1e-6), 0, 1)
    img_array = (normalized * 255).astype(np.uint8)

    # Flip vertically (lat increases upward)
    img_array = np.flipud(img_array)

    img = Image.fromarray(img_array, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "source": f"data:image/png;base64,{b64}",
        "coordinates": [
            [bounds["lon_min"], bounds["lat_max"]],  # top-left
            [bounds["lon_max"], bounds["lat_max"]],  # top-right
            [bounds["lon_max"], bounds["lat_min"]],  # bottom-right
            [bounds["lon_min"], bounds["lat_min"]],  # bottom-left
        ],
    }
