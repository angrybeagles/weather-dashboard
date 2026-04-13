# CONUS Weather Dashboard

A local weather dashboard built with Plotly Dash that pulls real-time data from
NOAA's operational models and satellite feeds. Designed to give you better
situational awareness than typical consumer weather sites.

## Data Sources

| Layer | Source | Resolution | Update Cadence |
|-------|--------|-----------|----------------|
| Forecast fields (temp, wind, precip, CAPE) | HRRR via Herbie | 3 km | Hourly |
| Satellite imagery | GOES-16/18 (AWS) | 2 km (CONUS) | 5 min |
| Severe weather alerts | NWS API | County-level | Real-time |
| Surface observations | NWS API (METAR) | Station-level | Hourly |

## Setup

### 1. Install system dependencies

```bash
# Ubuntu/Debian
sudo apt install libeccodes-dev libgeos-dev libproj-dev

# macOS
brew install eccodes geos proj
```

### 2. Create virtual environment and install Python packages

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
python app.py
```

Open `http://localhost:8050` in your browser.

## Architecture

```
weather-dashboard/
├── app.py                  # Dash application entry point
├── config.py               # Configuration constants
├── requirements.txt
├── data/                   # Cached data files (auto-populated)
├── assets/                 # Dash static assets (CSS)
│   └── style.css
├── components/
│   ├── __init__.py
│   ├── map_layer.py        # Map rendering with Plotly
│   ├── alerts_panel.py     # NWS alerts sidebar
│   └── controls.py         # Layer toggles, time slider
├── pipeline/
│   ├── __init__.py
│   ├── hrrr.py             # HRRR model data fetcher
│   ├── goes.py             # GOES satellite imagery fetcher
│   ├── nws.py              # NWS alerts + observations
│   └── scheduler.py        # Background data refresh
└── README.md
```

## Key Libraries

- **herbie** — HRRR/GFS GRIB2 download and subsetting
- **xarray + cfgrib** — GRIB2 decoding into labeled arrays
- **s3fs** — Direct access to NOAA data on AWS Open Data
- **plotly + dash** — Interactive maps and dashboard framework
- **cartopy** — Map projections and geographic features
- **apscheduler** — Background data refresh scheduling

## Notes

- First run will download ~200-500 MB of HRRR data depending on selected
  variables. Subsequent runs use cached data and only fetch new cycles.
- GOES imagery is fetched on-demand for the latest scan; historical scans
  can be browsed via the time slider.
- The NWS API has no key requirement but does ask for a User-Agent header
  with contact info — set yours in `config.py`.
