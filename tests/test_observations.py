"""Tests for NWS observation parsing — full field coverage + None handling."""

from pipeline.nws import _parse_obs_properties


SAMPLE_PROPS_FULL = {
    "temperature": {"value": 20.0, "unitCode": "wmoUnit:degC"},
    "dewpoint": {"value": 10.0},
    "windSpeed": {"value": 16.0},  # km/h
    "windDirection": {"value": 270},
    "barometricPressure": {"value": 101325},  # Pa
    "visibility": {"value": 16093.4},  # m
    "textDescription": "Mostly Cloudy",
    "timestamp": "2026-04-13T12:00:00+00:00",
}


def test_parse_full_observation():
    obs = _parse_obs_properties("KSEA", SAMPLE_PROPS_FULL, [-122.3, 47.5], "Seattle")
    assert obs["station_id"] == "KSEA"
    assert obs["station_name"] == "Seattle"
    assert obs["lat"] == 47.5
    assert obs["lon"] == -122.3
    assert obs["temperature_f"] == 68.0
    assert obs["temperature_c"] == 20.0
    assert obs["dewpoint_f"] == 50.0
    assert obs["wind_speed_mph"] == round(16.0 * 0.621371, 1)
    assert obs["wind_direction"] == 270
    assert obs["pressure_mb"] == 1013.2
    assert obs["visibility_mi"] == 10.0
    assert obs["description"] == "Mostly Cloudy"


def test_pressure_mb_present_in_bulk_shape():
    """Regression: bulk fetcher used to omit pressure_mb entirely so the map
    hover always read 'Pressure: — mb'. Now every parsed obs has the key."""
    obs = _parse_obs_properties("KDEN", SAMPLE_PROPS_FULL, [-104.6, 39.8])
    assert "pressure_mb" in obs
    assert obs["pressure_mb"] == 1013.2


def test_missing_wind_returns_none_not_zero():
    """Regression: previously reported 0 mph when wind data missing; should be None."""
    props = dict(SAMPLE_PROPS_FULL)
    props["windSpeed"] = {"value": None}
    obs = _parse_obs_properties("KDEN", props, [-104.6, 39.8])
    assert obs["wind_speed_mph"] is None


def test_missing_optional_fields_are_none():
    obs = _parse_obs_properties(
        "KXYZ",
        {"temperature": {"value": None}},
        [-100.0, 40.0],
    )
    assert obs["temperature_f"] is None
    assert obs["dewpoint_f"] is None
    assert obs["pressure_mb"] is None
    assert obs["visibility_mi"] is None
    assert obs["wind_direction"] is None


def test_handles_missing_property_blocks():
    """Some stations return missing/null sub-blocks instead of {'value': null}."""
    props = {"temperature": None, "windSpeed": None, "barometricPressure": None}
    obs = _parse_obs_properties("KXYZ", props, [-100.0, 40.0])
    assert obs["temperature_f"] is None
    assert obs["wind_speed_mph"] is None
    assert obs["pressure_mb"] is None


def test_handles_empty_geometry():
    obs = _parse_obs_properties("KXYZ", SAMPLE_PROPS_FULL, [])
    assert obs["lat"] is None
    assert obs["lon"] is None
