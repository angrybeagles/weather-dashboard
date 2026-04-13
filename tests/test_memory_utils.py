"""Tests for utils/memory.py sizing helpers."""

import numpy as np
import pytest
import xarray as xr

from utils.memory import (
    array_bytes,
    disk_cache_bytes,
    process_rss_mb,
    store_bytes_breakdown,
)


def test_array_bytes_numpy():
    arr = np.zeros((100, 100), dtype=np.float32)
    assert array_bytes(arr) == 100 * 100 * 4


def test_array_bytes_dataarray():
    da = xr.DataArray(np.zeros((10, 20), dtype=np.float64))
    assert array_bytes(da) == 10 * 20 * 8


def test_array_bytes_unknown_type_returns_zero():
    assert array_bytes("hello") == 0
    assert array_bytes(None) == 0
    assert array_bytes(42) == 0


def test_store_breakdown_empty_store():
    class _S:
        hrrr_data = {}
        hrrr_window = {}
        goes_data = {}
        nexrad_data = None
        alerts = []
        observations = []

    b = store_bytes_breakdown(_S())
    assert set(b.keys()) == {"hrrr", "goes", "nexrad", "alerts", "observations"}
    assert all(v == 0 for v in b.values())


def test_store_breakdown_counts_hrrr():
    class _S:
        hrrr_data = {"temp": xr.DataArray(np.zeros((50, 50), dtype=np.float32))}
        hrrr_window = {}
        goes_data = {}
        nexrad_data = None
        alerts = []
        observations = []

    assert store_bytes_breakdown(_S())["hrrr"] == 50 * 50 * 4


def test_process_rss_mb_returns_positive():
    rss = process_rss_mb()
    assert rss >= 0.0


def test_disk_cache_bytes_missing_dir(tmp_path):
    missing = tmp_path / "nope"
    assert disk_cache_bytes(missing) == 0
