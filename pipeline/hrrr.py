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
        # Single combined NetCDF Dataset per forecast hour (Phase 2)
        cache_file = CACHE_DIR / f"hrrr_{cycle_str}_f{fhr:02d}.nc"
        if cache_file.exists():
            try:
                # Lazy-load via dask so per-fhr arrays stay on disk until
                # actually materialized downstream (Phase 3).
                cached_ds = xr.open_dataset(cache_file, chunks={})
                if all(name in cached_ds.data_vars for name in variables.values()):
                    logger.info("Cache hit: %s fhr=%d", cycle_str, fhr)
                    for friendly_name in variables.values():
                        results[friendly_name].append(cached_ds[friendly_name])
                    continue
                cached_ds.close()
            except Exception as e:
                logger.warning("Failed to load cache %s: %s", cache_file.name, e)

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

                # Drop scalar non-dim coords (heightAboveGround, surface,
                # etc.) — they conflict between variables (2m vs 10m) when
                # combined into a single Dataset and across fhrs on concat.
                drop_coords = [
                    c for c in da.coords
                    if c not in da.dims and c not in ("latitude", "longitude", "valid_time")
                ]
                if drop_coords:
                    da = da.reset_coords(drop_coords, drop=True)

                merged_arrays[friendly_name] = da
                results[friendly_name].append(da)
            except Exception as e:
                logger.warning(
                    "Failed to fetch %s for fhr=%d: %s", search_str, fhr, e
                )

        # Cache as a single combined Dataset (Phase 2: 1 file per fhr, not 10)
        if merged_arrays:
            try:
                xr.Dataset(merged_arrays).to_netcdf(cache_file)
            except Exception as e:
                logger.warning("Failed to cache fhr=%d: %s", fhr, e)

    # Concatenate across forecast hours, keeping each fhr as its own
    # dask chunk so the full stack never materializes in memory until
    # an explicit .values / .compute() (Phase 3).
    output = {}
    for name, arrays in results.items():
        if arrays:
            try:
                stacked = xr.concat(
                    arrays,
                    dim="valid_time",
                    coords="minimal",
                    compat="override",
                ).sortby("valid_time")
                output[name] = stacked.chunk({"valid_time": 1})
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


def latest_cached_cycle() -> datetime | None:
    """Return the most recent cycle datetime for which on-disk files exist."""
    cycles = set()
    for f in CACHE_DIR.glob("hrrr_*_f*.nc"):
        parts = f.stem.split("_")
        if len(parts) < 4:
            continue
        try:
            t = datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H")
            cycles.add(t)
        except ValueError:
            continue
    return max(cycles) if cycles else None


def load_hrrr_window(
    cycle: datetime | None = None,
    center_fhr: int = 0,
    radius: int = 2,
    variables: list[str] | None = None,
) -> dict[str, xr.DataArray]:
    """
    Open the per-fhr NetCDF cache files within [center_fhr-radius, center_fhr+radius]
    and stack on valid_time — restricted to the requested variables.

    Files that don't exist are skipped silently (they'll be populated by the next
    full fetch_hrrr cycle). Returned DataArrays are dask-chunked on valid_time so
    the working set is bounded by the window size, not the full 19-fhr horizon.
    """
    if cycle is None:
        cycle = latest_cached_cycle()
        if cycle is None:
            return {}
    if variables is None:
        friendly_names = list(HRRR_VARIABLES.values())
    else:
        friendly_names = list(variables)

    cycle_str = cycle.strftime("%Y%m%d_%H")
    lo = max(min(HRRR_FHOURS), center_fhr - radius)
    hi = min(max(HRRR_FHOURS), center_fhr + radius)

    collected: dict[str, list[xr.DataArray]] = {n: [] for n in friendly_names}
    for fhr in range(lo, hi + 1):
        cache_file = CACHE_DIR / f"hrrr_{cycle_str}_f{fhr:02d}.nc"
        if not cache_file.exists():
            continue
        try:
            ds = xr.open_dataset(cache_file, chunks={})
        except Exception as e:
            logger.warning("Failed to open %s: %s", cache_file.name, e)
            continue
        for name in friendly_names:
            if name in ds.data_vars:
                collected[name].append(ds[name])

    output: dict[str, xr.DataArray] = {}
    for name, arrays in collected.items():
        if not arrays:
            continue
        try:
            stacked = xr.concat(
                arrays, dim="valid_time", coords="minimal", compat="override"
            ).sortby("valid_time")
            output[name] = stacked.chunk({"valid_time": 1})
        except Exception as e:
            logger.warning("Failed to concat window for %s: %s", name, e)
    return output


def clean_cache(max_age_hours: int = 6) -> None:
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
