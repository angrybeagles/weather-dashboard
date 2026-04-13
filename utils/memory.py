"""Memory accounting helpers for the DataStore and process RSS."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def array_bytes(obj: Any) -> int:
    """Return in-memory byte size of numpy/xarray objects. Returns 0 for other types."""
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, (xr.DataArray, xr.Dataset)):
        return int(obj.nbytes)
    return 0


def _container_bytes(values) -> int:
    """Sum array_bytes over an iterable of values, including nested numpy arrays in dicts."""
    total = 0
    for v in values:
        n = array_bytes(v)
        if n:
            total += n
        elif isinstance(v, dict):
            total += _container_bytes(v.values())
    return total


def _rough_pyobj_bytes(seq) -> int:
    """Rough upper-bound sizing for lists of small dicts (alerts, observations)."""
    if not seq:
        return 0
    sample = seq[0] if isinstance(seq, list) else seq
    per = sys.getsizeof(sample)
    if isinstance(sample, dict):
        per += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in sample.items())
    return per * (len(seq) if hasattr(seq, "__len__") else 1)


def store_bytes_breakdown(store) -> dict[str, int]:
    """Per-cache byte breakdown for the global DataStore."""
    nexrad_bytes = 0
    if getattr(store, "nexrad_data", None):
        nexrad_bytes = _container_bytes(store.nexrad_data.values())

    return {
        "hrrr": _container_bytes(getattr(store, "hrrr_data", {}).values())
        + _container_bytes(getattr(store, "hrrr_window", {}).values()),
        "goes": _container_bytes(getattr(store, "goes_data", {}).values()),
        "nexrad": nexrad_bytes,
        "alerts": _rough_pyobj_bytes(getattr(store, "alerts", [])),
        "observations": _rough_pyobj_bytes(getattr(store, "observations", [])),
    }


def disk_cache_bytes(cache_dir: Path) -> int:
    """Sum on-disk NetCDF cache size."""
    if not cache_dir.exists():
        return 0
    return sum(p.stat().st_size for p in cache_dir.glob("*.nc"))


def process_rss_mb() -> float:
    """Resident set size of the current process, in MB."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0
