"""Tests for HRRR window loading from on-disk cache."""

from datetime import datetime

import numpy as np
import xarray as xr

from pipeline import hrrr as hrrr_mod
from pipeline.hrrr import latest_cached_cycle, load_hrrr_window


def _write_fhr(cache_dir, cycle_str, fhr, vars_):
    ds = xr.Dataset(
        {
            name: xr.DataArray(
                np.zeros((5, 5), dtype=np.float32),
                dims=["y", "x"],
                coords={
                    "latitude": (["y", "x"], np.zeros((5, 5))),
                    "longitude": (["y", "x"], np.zeros((5, 5))),
                    "valid_time": datetime(2026, 1, 1, fhr),
                },
            )
            for name in vars_
        }
    )
    path = cache_dir / f"hrrr_{cycle_str}_f{fhr:02d}.nc"
    ds.to_netcdf(path)


def test_load_hrrr_window_limits_to_radius(tmp_path, monkeypatch):
    monkeypatch.setattr(hrrr_mod, "CACHE_DIR", tmp_path)
    cycle = datetime(2026, 1, 1, 12)
    cycle_str = cycle.strftime("%Y%m%d_%H")
    for fhr in range(0, 10):
        _write_fhr(tmp_path, cycle_str, fhr, ["temperature_2m", "u_wind_10m"])

    window = load_hrrr_window(
        cycle=cycle, center_fhr=5, radius=2, variables=["temperature_2m"]
    )
    assert "temperature_2m" in window
    assert len(window["temperature_2m"].valid_time) == 5  # fhrs 3,4,5,6,7
    assert "u_wind_10m" not in window


def test_load_hrrr_window_missing_files_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(hrrr_mod, "CACHE_DIR", tmp_path)
    cycle = datetime(2026, 1, 1, 12)
    cycle_str = cycle.strftime("%Y%m%d_%H")
    _write_fhr(tmp_path, cycle_str, 5, ["temperature_2m"])

    window = load_hrrr_window(
        cycle=cycle, center_fhr=5, radius=3, variables=["temperature_2m"]
    )
    assert len(window["temperature_2m"].valid_time) == 1


def test_load_hrrr_window_empty_when_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(hrrr_mod, "CACHE_DIR", tmp_path)
    assert load_hrrr_window(cycle=datetime(2026, 1, 1), center_fhr=0) == {}


def test_latest_cached_cycle_picks_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(hrrr_mod, "CACHE_DIR", tmp_path)
    _write_fhr(tmp_path, "20260101_06", 0, ["temperature_2m"])
    _write_fhr(tmp_path, "20260101_12", 0, ["temperature_2m"])
    assert latest_cached_cycle() == datetime(2026, 1, 1, 12)


def test_latest_cached_cycle_none_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(hrrr_mod, "CACHE_DIR", tmp_path)
    assert latest_cached_cycle() is None
