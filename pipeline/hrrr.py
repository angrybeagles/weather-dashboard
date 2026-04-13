"""
HRRR model data pipeline.

Uses the `herbie` library to fetch HRRR GRIB2 data from AWS Open Data,
decodes it with cfgrib/xarray, and caches processed fields as NetCDF.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from config import CONUS_BOUNDS, DATA_DIR, HRRR_FHOURS, HRRR_VARIABLES

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "hrrr"
CACHE_DIR.mkdir(exist_ok=True)


def _latest_hrrr_cycle() -> datetime:
    """
    Determine the most recent HRRR cycle likely to be fully available.
    HRRR runs hourly; products typically appear ~50 min after cycle time.
    We back off by 2 hours to be safe.
    """
    now = datetime.now(timezone.utc)
    cycle = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
    return cycle.replace(tzinfo=None)


def fetch_hrrr(
    cycle: datetime | None = None,
    fhours: list[int] | None = None,
    variables: dict[str, str] | None = None,
) -> dict[str, xr.Dataset]:
    """
    Fetch HRRR fields for the given cycle and forecast hours.

    Parameters
    ----------
    cycle : datetime, optional
        Model initialization time (UTC). Defaults to latest available.
    fhours : list[int], optional
        Forecast hours to retrieve. Defaults to config HRRR_FHOURS.
    variables : dict, optional
        Mapping of GRIB2 search strings to friendly names.
        Defaults to config HRRR_VARIABLES.

    Returns
    -------
    dict[str, xr.Dataset]
        Keyed by friendly variable name, each Dataset has dims
        (latitude, longitude) with a 'valid_time' coordinate.
    """
    from herbie import Herbie

    if cycle is None:
        cycle = _latest_hrrr_cycle()
    if fhours is None:
        fhours = HRRR_FHOURS
    if variables is None:
        variables = HRRR_VARIABLES

    cycle_str = cycle.strftime("%Y%m%d_%H")
    results: dict[str, list[xr.DataArray]] = {v: [] for v in variables.values()}

    for fhr in fhours:
        # Check if all variables are cached for this forecast hour
        all_cached = True
        cached_data = {}
        for friendly_name in variables.values():
            var_cache_file = CACHE_DIR / f"hrrr_{cycle_str}_f{fhr:02d}_{friendly_name}.nc"
            if not var_cache_file.exists():
                all_cached = False
                break
            try:
                cached_data[friendly_name] = xr.open_dataarray(var_cache_file)
            except Exception as e:
                logger.warning("Failed to load cached %s fhr=%d: %s", friendly_name, fhr, e)
                all_cached = False
                break

        if all_cached:
            logger.info("Cache hit: %s fhr=%d", cycle_str, fhr)
            for friendly_name, da in cached_data.items():
                results[friendly_name].append(da)
            continue

        logger.info("Fetching HRRR cycle=%s fhr=%d", cycle_str, fhr)
        try:
            H = Herbie(cycle, model="hrrr", product="sfc", fxx=fhr)
        except Exception as e:
            logger.warning("Could not init Herbie for fhr=%d: %s", fhr, e)
            continue

        merged_arrays = {}
        for search_str, friendly_name in variables.items():
            try:
                ds = H.xarray(search_str, remove_grib=True)
                # Find the data variable (herbie names vary)
                data_vars = [
                    v for v in ds.data_vars if v not in ("gribfile_projection",)
                ]
                if not data_vars:
                    continue
                da = ds[data_vars[0]].rename(friendly_name)

                # Add valid_time as a coordinate
                valid_time = cycle + timedelta(hours=fhr)
                da = da.assign_coords(valid_time=valid_time)

                merged_arrays[friendly_name] = da
                results[friendly_name].append(da)
            except Exception as e:
                logger.warning(
                    "Failed to fetch %s for fhr=%d: %s", search_str, fhr, e
                )

        # Cache this forecast hour as NetCDF - save each variable separately
        for friendly_name, da in merged_arrays.items():
            try:
                var_cache_file = CACHE_DIR / f"hrrr_{cycle_str}_f{fhr:02d}_{friendly_name}.nc"
                da.to_netcdf(var_cache_file)
            except Exception as e:
                logger.warning("Failed to cache %s fhr=%d: %s", friendly_name, fhr, e)

    # Concatenate across forecast hours
    output = {}
    for name, arrays in results.items():
        if arrays:
            try:
                output[name] = xr.concat(arrays, dim="valid_time").sortby(
                    "valid_time"
                )
            except Exception as e:
                logger.warning("Failed to concat %s: %s", name, e)

    return output


def get_wind_speed_direction(
    u: xr.DataArray, v: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute wind speed (m/s) and meteorological direction from U/V."""
    speed = np.sqrt(u**2 + v**2)
    direction = (270 - np.degrees(np.arctan2(v, u))) % 360
    speed.name = "wind_speed_10m"
    direction.name = "wind_direction_10m"
    return speed, direction


def kelvin_to_fahrenheit(da: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Fahrenheit."""
    return (da - 273.15) * 9 / 5 + 32


def kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Celsius."""
    return da - 273.15


def ms_to_mph(da: xr.DataArray) -> xr.DataArray:
    """Convert m/s to mph."""
    return da * 2.23694


def ms_to_knots(da: xr.DataArray) -> xr.DataArray:
    """Convert m/s to knots."""
    return da * 1.94384


def clean_cache(max_age_hours: int = 24) -> None:
    """Remove cached HRRR files older than max_age_hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    for f in CACHE_DIR.glob("hrrr_*.nc"):
        try:
            # Parse cycle time from filename
            parts = f.stem.split("_")
            file_time = datetime.strptime(
                f"{parts[1]}_{parts[2]}", "%Y%m%d_%H"
            ).replace(tzinfo=timezone.utc)
            if file_time < cutoff:
                f.unlink()
                logger.info("Cleaned cache: %s", f.name)
        except (ValueError, IndexError):
            pass
