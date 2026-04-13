"""
Configuration for the CONUS Weather Dashboard.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# NWS API
# ---------------------------------------------------------------------------
# The NWS API requires a descriptive User-Agent. Put your contact info here.
NWS_USER_AGENT = "(WeatherDashboard, your-email@example.com)"
NWS_API_BASE = "https://api.weather.gov"

# ---------------------------------------------------------------------------
# GOES-16 / AWS
# ---------------------------------------------------------------------------
GOES_BUCKET = "noaa-goes18"  # Updated to GOES-18 (GOES-16 was decommissioned)
GOES_PRODUCT = "ABI-L2-CMIPF"  # Cloud & Moisture Imagery — Full Disk
GOES_CONUS_PRODUCT = "ABI-L2-CMIPC"  # CONUS sector (more frequent)
GOES_CHANNELS = {
    "visible": 2,       # 0.64 µm — true-color visible
    "shortwave_ir": 7,  # 3.9 µm — fog/low cloud detection
    "water_vapor": 9,   # 6.9 µm — mid-level water vapor
    "ir": 13,           # 10.3 µm — clean longwave IR window
}

# ---------------------------------------------------------------------------
# HRRR / Model data
# ---------------------------------------------------------------------------
HRRR_VARIABLES = {
    "TMP:2 m": "temperature_2m",
    "UGRD:10 m": "u_wind_10m",
    "VGRD:10 m": "v_wind_10m",
    "APCP:surface": "total_precip",
    "CAPE:surface": "cape",
    "REFC:entire atmosphere": "composite_reflectivity",
    "DPT:2 m": "dewpoint_2m",
    "PRES:surface": "surface_pressure",
    "VIS:surface": "visibility",
    "GUST:surface": "wind_gust",
}

# Forecast hours to fetch (0 = analysis, then hourly out to 18)
HRRR_FHOURS = list(range(0, 19))

# ---------------------------------------------------------------------------
# CONUS bounding box (for clipping / map extent)
# ---------------------------------------------------------------------------
CONUS_BOUNDS = {
    "lat_min": 24.0,
    "lat_max": 50.0,
    "lon_min": -125.0,
    "lon_max": -66.0,
}

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
DASH_HOST = "0.0.0.0"
DASH_PORT = 8050
DASH_DEBUG = True

# Map defaults
MAP_CENTER = {"lat": 39.0, "lon": -96.0}
MAP_ZOOM = 4

# Refresh intervals (seconds)
HRRR_REFRESH_INTERVAL = 3600       # 1 hour
GOES_REFRESH_INTERVAL = 300        # 5 minutes
ALERTS_REFRESH_INTERVAL = 120      # 2 minutes
OBS_REFRESH_INTERVAL = 900         # 15 minutes
