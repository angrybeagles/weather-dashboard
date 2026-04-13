"""
NEXRAD radar data pipeline.

Fetches NEXRAD Level III composite reflectivity from the MRMS
(Multi-Radar Multi-Sensor) dataset on AWS, which provides a
pre-mosaicked CONUS-wide radar composite — much more practical
than stitching individual radar sites.

Also supports fetching individual site data from the NEXRAD Level II
archive on AWS for higher-resolution local views.

AWS buckets:
  - s3://noaa-mrms-pds          — MRMS composites (CONUS mosaic)
  - s3://noaa-nexrad-level2     — Individual radar volumes
"""

import logging
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
import os

import numpy as np

from config import CONUS_BOUNDS, DATA_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "nexrad"
CACHE_DIR.mkdir(exist_ok=True)

# MRMS CONUS composite grid specs (0.01° resolution)
MRMS_LAT_MIN = 20.0
MRMS_LAT_MAX = 55.0
MRMS_LON_MIN = -130.0
MRMS_LON_MAX = -60.0
MRMS_RESOLUTION = 0.01  # degrees


def fetch_mrms_composite(
    product: str = "MergedReflectivityQCComposite_00.50",
    max_age_minutes: int = 60,
) -> dict | None:
    """
    Fetch the latest MRMS composite reflectivity from AWS.

    The MRMS system produces a seamless CONUS-wide radar mosaic that
    combines all ~160 NEXRAD sites with gap-filling algorithms. This
    is the same product that drives NWS radar displays.

    Parameters
    ----------
    product : str
        MRMS product name. Default is composite reflectivity.
    max_age_minutes : int
        How far back to search for a valid file.

    Returns
    -------
    dict with keys:
        'data': 2D numpy array (dBZ values)
        'lats': 1D array of latitudes
        'lons': 1D array of longitudes
        'valid_time': datetime
        'source': str
    Or None on failure.
    """
    import gzip
    import s3fs
    import tempfile

    fs = s3fs.S3FileSystem(anon=True)
    bucket = "noaa-mrms-pds"
    now = datetime.now(timezone.utc)

    # MRMS files are organized by product/level/time
    # Path pattern: CONUS/{product}/{YYYYMMDD}/MRMS_{product}_{YYYYMMDD}-{HHMMSS}.grib2.gz
    # Look for any recent files instead of exact timestamps
    date_str = now.strftime("%Y%m%d")
    prefix = f"{bucket}/CONUS/{product}/{date_str}"

    try:
        # Get all files for today
        all_files = fs.glob(f"{prefix}/MRMS_{product}_{date_str}*.grib2.gz")
        if not all_files:
            # Try yesterday if no files for today
            yesterday = now - timedelta(days=1)
            date_str = yesterday.strftime("%Y%m%d")
            prefix = f"{bucket}/CONUS/{product}/{date_str}"
            all_files = fs.glob(f"{prefix}/MRMS_{product}_{date_str}*.grib2.gz")
        
        if not all_files:
            logger.warning("No MRMS files found for recent dates")
            return None
            
        # Filter for files within the age limit
        recent_files = []
        for f in all_files:
            # Extract timestamp from filename
            fname = f.split('/')[-1].replace('.grib2.gz', '')
            time_part = fname.split('_')[-1]  # YYYYMMDD-HHMMSS
            try:
                file_time = datetime.strptime(time_part, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
                age_minutes = (now - file_time).total_seconds() / 60
                if age_minutes <= max_age_minutes:
                    recent_files.append((f, file_time))
            except ValueError:
                continue
                
        if not recent_files:
            logger.warning("No recent MRMS files found within %d minutes", max_age_minutes)
            return None
            
        # Get the most recent file
        latest_file, latest_time = max(recent_files, key=lambda x: x[1])
        logger.info("Fetching MRMS: %s", latest_file)

        with fs.open(latest_file, "rb") as f:
            compressed = f.read()

        # Decompress and read GRIB2
        raw = gzip.decompress(compressed)

        # Save to temporary file for cfgrib (it needs seekable file access)
        with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as tmp_file:
            tmp_file.write(raw)
            tmp_path = tmp_file.name

        try:
            # Parse with cfgrib
            import xarray as xr

            ds = xr.open_dataset(
                tmp_path,
                engine="cfgrib",
                backend_kwargs={"indexpath": ""},
            )

            # Get the reflectivity variable
            data_vars = list(ds.data_vars)
            if not data_vars:
                return None

            refl = ds[data_vars[0]].values

            # Build coordinate arrays
            lats = ds.latitude.values if "latitude" in ds.coords else np.linspace(
                MRMS_LAT_MAX, MRMS_LAT_MIN, refl.shape[0]
            )
            lons = ds.longitude.values if "longitude" in ds.coords else np.linspace(
                MRMS_LON_MIN, MRMS_LON_MAX, refl.shape[1]
            )

            # Convert longitude from 0-360 to -180-180 if needed
            if np.any(lons > 180):
                lons = np.where(lons > 180, lons - 360, lons)

            # Parse valid time from filename
            fname = Path(latest_file).stem  # remove .gz
            time_part = fname.split("_")[-1].replace(".grib2", "")
            try:
                valid_time = datetime.strptime(
                    time_part, "%Y%m%d-%H%M%S"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                valid_time = now

            return {
                "data": refl,
                "lats": lats,
                "lons": lons,
                "valid_time": valid_time,
                "source": latest_file,
            }
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

    except Exception as e:
        logger.warning("MRMS fetch failed: %s", e)
        return None


def fetch_nexrad_sites() -> list[dict]:
    """
    Return metadata for all NEXRAD radar sites in CONUS.
    Useful for displaying site locations on the map.
    """
    # Hardcoded subset of major NEXRAD sites with coordinates
    # Full list: https://www.roc.noaa.gov/WSR88D/Maps.aspx
    sites = [
        {"id": "KATX", "name": "Seattle", "lat": 48.1946, "lon": -122.4957},
        {"id": "KRTX", "name": "Portland", "lat": 45.7150, "lon": -122.9656},
        {"id": "KMTX", "name": "Salt Lake City", "lat": 41.2628, "lon": -112.4480},
        {"id": "KFTG", "name": "Denver", "lat": 39.7866, "lon": -104.5458},
        {"id": "KFWS", "name": "Dallas-Ft Worth", "lat": 32.5730, "lon": -97.3031},
        {"id": "KHGX", "name": "Houston", "lat": 29.4719, "lon": -95.0789},
        {"id": "KLOT", "name": "Chicago", "lat": 41.6044, "lon": -88.0847},
        {"id": "KDTX", "name": "Detroit", "lat": 42.6999, "lon": -83.4718},
        {"id": "KOKX", "name": "New York City", "lat": 40.8655, "lon": -72.8638},
        {"id": "KBOX", "name": "Boston", "lat": 41.9558, "lon": -71.1369},
        {"id": "KLWX", "name": "Washington DC", "lat": 38.9753, "lon": -77.4778},
        {"id": "KFFC", "name": "Atlanta", "lat": 33.3636, "lon": -84.5658},
        {"id": "KAMX", "name": "Miami", "lat": 25.6111, "lon": -80.4128},
        {"id": "KTBW", "name": "Tampa Bay", "lat": 27.7056, "lon": -82.4017},
        {"id": "KLIX", "name": "New Orleans", "lat": 30.3367, "lon": -89.8256},
        {"id": "KPUX", "name": "Pueblo", "lat": 38.4595, "lon": -104.1816},
        {"id": "KSGF", "name": "Springfield MO", "lat": 37.2353, "lon": -93.4006},
        {"id": "KLSX", "name": "St. Louis", "lat": 38.6987, "lon": -90.6828},
        {"id": "KMSP", "name": "Minneapolis", "lat": 44.8488, "lon": -93.5654},
        {"id": "KMPX", "name": "Chanhassen", "lat": 44.8488, "lon": -93.5654},
        {"id": "KPBZ", "name": "Pittsburgh", "lat": 40.5317, "lon": -80.2179},
        {"id": "KILN", "name": "Cincinnati", "lat": 39.4203, "lon": -83.8217},
        {"id": "KIND", "name": "Indianapolis", "lat": 39.7075, "lon": -86.2803},
        {"id": "KMKX", "name": "Milwaukee", "lat": 42.9678, "lon": -88.5506},
        {"id": "KPDT", "name": "Pendleton OR", "lat": 45.6906, "lon": -118.8529},
        {"id": "KSFX", "name": "Pocatello ID", "lat": 43.1058, "lon": -112.6861},
        {"id": "KMSX", "name": "Missoula MT", "lat": 47.0411, "lon": -113.9864},
        {"id": "KBIS", "name": "Bismarck ND", "lat": 46.7710, "lon": -100.7605},
        {"id": "KABR", "name": "Aberdeen SD", "lat": 45.4558, "lon": -98.4132},
        {"id": "KUEX", "name": "Grand Island NE", "lat": 40.3206, "lon": -98.4418},
        {"id": "KICT", "name": "Wichita KS", "lat": 37.6546, "lon": -97.4431},
        {"id": "KTLX", "name": "Oklahoma City", "lat": 35.3331, "lon": -97.2778},
        {"id": "KGRK", "name": "Austin TX", "lat": 30.7217, "lon": -97.3828},
        {"id": "KEPZ", "name": "El Paso", "lat": 31.8731, "lon": -106.6981},
        {"id": "KIWA", "name": "Phoenix", "lat": 33.2891, "lon": -111.6700},
        {"id": "KESX", "name": "Las Vegas", "lat": 35.7013, "lon": -114.8918},
        {"id": "KMUX", "name": "San Francisco", "lat": 37.1551, "lon": -121.8983},
        {"id": "KVTX", "name": "Los Angeles", "lat": 34.4116, "lon": -119.1795},
        {"id": "KNKX", "name": "San Diego", "lat": 32.9189, "lon": -117.0419},
        {"id": "KBGM", "name": "Binghamton NY", "lat": 42.1997, "lon": -75.9847},
        {"id": "KCLE", "name": "Cleveland", "lat": 41.4131, "lon": -81.8597},
        {"id": "KBUF", "name": "Buffalo", "lat": 42.9489, "lon": -78.7369},
        {"id": "KRAX", "name": "Raleigh", "lat": 35.6658, "lon": -78.4900},
        {"id": "KCAE", "name": "Columbia SC", "lat": 33.9487, "lon": -81.1184},
        {"id": "KJAX", "name": "Jacksonville", "lat": 30.4847, "lon": -81.7019},
        {"id": "KBMX", "name": "Birmingham", "lat": 33.1722, "lon": -86.7697},
        {"id": "KNQA", "name": "Memphis", "lat": 35.3447, "lon": -89.8734},
        {"id": "KLZK", "name": "Little Rock", "lat": 34.8364, "lon": -92.2622},
    ]
    return sites


def radar_to_plotly_data(
    radar: dict,
    subsample: int = 4,
) -> dict | None:
    """
    Convert MRMS radar composite to arrays ready for Plotly scattermapbox.

    Parameters
    ----------
    radar : dict
        Output from fetch_mrms_composite().
    subsample : int
        Subsample factor for performance.

    Returns
    -------
    dict with 'lats', 'lons', 'values', 'valid_time' — filtered to
    only show pixels with reflectivity > threshold.
    """
    if radar is None:
        return None

    data = radar["data"]
    lats = radar["lats"]
    lons = radar["lons"]

    # Build 2D coordinate arrays if 1D
    if lats.ndim == 1 and lons.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
    else:
        lat_grid, lon_grid = lats, lons

    # Subsample
    data_sub = data[::subsample, ::subsample]
    lat_sub = lat_grid[::subsample, ::subsample]
    lon_sub = lon_grid[::subsample, ::subsample]

    # Filter: CONUS bounds + significant reflectivity
    mask = (
        (lat_sub >= CONUS_BOUNDS["lat_min"])
        & (lat_sub <= CONUS_BOUNDS["lat_max"])
        & (lon_sub >= CONUS_BOUNDS["lon_min"])
        & (lon_sub <= CONUS_BOUNDS["lon_max"])
        & ~np.isnan(data_sub)
        & (data_sub > 10)  # Only show meaningful returns
    )

    if mask.sum() == 0:
        return None

    return {
        "lats": lat_sub[mask].ravel(),
        "lons": lon_sub[mask].ravel(),
        "values": data_sub[mask].ravel(),
        "valid_time": radar["valid_time"],
    }
