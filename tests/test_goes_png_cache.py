"""Tests for GOES PNG memoization."""

import numpy as np
import xarray as xr

from pipeline import goes as goes_mod
from pipeline.goes import clear_png_cache, goes_to_plotly_image


def _tiny_goes_da(channel="ir", scan_time="2026-01-01T00:00:00"):
    # Minimal synthetic DataArray with a CONUS-ish lat/lon 2D grid
    y, x = 40, 60
    lats = np.tile(np.linspace(50, 25, y).reshape(-1, 1), (1, x)).astype(np.float32)
    lons = np.tile(np.linspace(-125, -67, x).reshape(1, -1), (y, 1)).astype(np.float32)
    data = np.random.RandomState(0).rand(y, x).astype(np.float32) * 300
    return xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"latitude": (["y", "x"], lats), "longitude": (["y", "x"], lons)},
        attrs={"channel": channel, "scan_time": scan_time},
    )


def setup_function(_):
    clear_png_cache()


def test_png_cache_returns_same_object_for_identical_key():
    da = _tiny_goes_da()
    first = goes_to_plotly_image(da)
    second = goes_to_plotly_image(da)
    assert first is second


def test_png_cache_invalidates_on_scan_time_change():
    first = goes_to_plotly_image(_tiny_goes_da(scan_time="t1"))
    second = goes_to_plotly_image(_tiny_goes_da(scan_time="t2"))
    assert first is not second


def test_png_cache_bounded_to_max():
    for i in range(10):
        goes_to_plotly_image(_tiny_goes_da(scan_time=f"t{i}"))
    assert len(goes_mod._PNG_CACHE) <= goes_mod._PNG_CACHE_MAX


def test_clear_png_cache_by_channel():
    goes_to_plotly_image(_tiny_goes_da(channel="ir", scan_time="a"))
    goes_to_plotly_image(_tiny_goes_da(channel="visible", scan_time="a"))
    removed = clear_png_cache(channel="ir")
    assert removed == 1
    assert all(k[0] != "ir" for k in goes_mod._PNG_CACHE)
