"""Tests for DataStore eviction API."""

import numpy as np
import xarray as xr

from pipeline.scheduler import DataStore


def _da(shape=(10, 10)):
    return xr.DataArray(np.zeros(shape, dtype=np.float32))


def test_evict_hrrr():
    s = DataStore()
    s.hrrr_data = {"temp": _da()}
    s.evict_hrrr()
    assert s.hrrr_data == {}


def test_evict_goes_all_channels():
    s = DataStore()
    s.goes_data = {"ir": _da(), "visible": _da()}
    s.evict_goes()
    assert s.goes_data == {}


def test_evict_goes_single_channel():
    s = DataStore()
    s.goes_data = {"ir": _da(), "visible": _da()}
    s.evict_goes("ir")
    assert "ir" not in s.goes_data
    assert "visible" in s.goes_data


def test_evict_nexrad():
    s = DataStore()
    s.nexrad_data = {"lats": np.zeros(10), "lons": np.zeros(10)}
    s.evict_nexrad()
    assert s.nexrad_data is None


def test_evict_alerts_and_observations():
    s = DataStore()
    s.alerts = [{"id": "a"}]
    s.observations = [{"station": "k"}]
    s.evict_alerts()
    s.evict_observations()
    assert s.alerts == []
    assert s.observations == []


def test_evict_goes_invalidates_png_cache():
    from pipeline import goes as goes_mod

    goes_mod._PNG_CACHE[("ir", "t1", 0, 0, 0, 0)] = {"source": "x", "coordinates": []}
    goes_mod._PNG_CACHE[("visible", "t1", 0, 0, 0, 0)] = {"source": "y", "coordinates": []}

    s = DataStore()
    s.goes_data = {"ir": _da(), "visible": _da()}
    s.evict_goes("ir")

    assert not any(k[0] == "ir" for k in goes_mod._PNG_CACHE)
    assert any(k[0] == "visible" for k in goes_mod._PNG_CACHE)

    # cleanup
    goes_mod._PNG_CACHE.clear()
