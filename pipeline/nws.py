"""
NWS API pipeline for alerts and surface observations.

Endpoints used:
  - /alerts/active          — active weather alerts (watches, warnings, advisories)
  - /stations/observations  — latest METAR observations
  - /points/{lat},{lon}     — forecast metadata for a point
"""

import logging
from datetime import datetime, timezone
from typing import Any

import requests

from config import NWS_API_BASE, NWS_USER_AGENT

logger = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": NWS_USER_AGENT,
        "Accept": "application/geo+json",
    }
)

# ---------------------------------------------------------------------------
# Alert severity / urgency ordering for display
# ---------------------------------------------------------------------------
SEVERITY_ORDER = {
    "Extreme": 0,
    "Severe": 1,
    "Moderate": 2,
    "Minor": 3,
    "Unknown": 4,
}

ALERT_COLORS = {
    "Extreme": "#FF0000",
    "Severe": "#FF6600",
    "Moderate": "#FFCC00",
    "Minor": "#00CC66",
    "Unknown": "#999999",
}


def fetch_active_alerts(
    area: str | None = None,
    severity: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch active NWS alerts.

    Parameters
    ----------
    area : str, optional
        State code(s) like "WA" or "WA,OR,CA". None = all CONUS.
    severity : list[str], optional
        Filter by severity: 'Extreme', 'Severe', 'Moderate', 'Minor'.

    Returns
    -------
    list of dicts with keys:
        id, event, severity, urgency, headline, description,
        areas, onset, expires, sender, color
    """
    params = {"status": "actual", "message_type": "alert"}
    if area:
        params["area"] = area
    if severity:
        params["severity"] = ",".join(severity)

    try:
        resp = SESSION.get(f"{NWS_API_BASE}/alerts/active", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Failed to fetch alerts: %s", e)
        return []

    alerts = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        sev = props.get("severity", "Unknown")
        alerts.append(
            {
                "id": props.get("id", ""),
                "event": props.get("event", "Unknown"),
                "severity": sev,
                "urgency": props.get("urgency", "Unknown"),
                "headline": props.get("headline", ""),
                "description": props.get("description", ""),
                "instruction": props.get("instruction", ""),
                "areas": props.get("areaDesc", ""),
                "onset": props.get("onset", ""),
                "expires": props.get("expires", ""),
                "sender": props.get("senderName", ""),
                "color": ALERT_COLORS.get(sev, "#999999"),
                "geometry": feature.get("geometry"),
            }
        )

    # Sort by severity
    alerts.sort(key=lambda a: SEVERITY_ORDER.get(a["severity"], 99))
    return alerts


def fetch_observations_by_state(state: str = "WA") -> list[dict[str, Any]]:
    """
    Fetch latest observations for all stations in a state.

    Note: The NWS API doesn't provide a direct 'all stations in state' endpoint
    efficiently, so we use /stations with state filter and then batch-fetch
    latest observations.
    """
    try:
        resp = SESSION.get(
            f"{NWS_API_BASE}/stations",
            params={"state": state, "limit": 200},
            timeout=15,
        )
        resp.raise_for_status()
        stations = resp.json().get("features", [])
    except Exception as e:
        logger.error("Failed to fetch stations for %s: %s", state, e)
        return []

    observations = []
    for station in stations[:50]:  # Limit to avoid rate limiting
        station_id = station["properties"].get("stationIdentifier", "")
        try:
            obs_resp = SESSION.get(
                f"{NWS_API_BASE}/stations/{station_id}/observations/latest",
                timeout=10,
            )
            if obs_resp.status_code != 200:
                continue
            obs = obs_resp.json().get("properties", {})
            coords = station.get("geometry", {}).get("coordinates", [None, None]) or [
                None,
                None,
            ]
            observations.append(
                _parse_obs_properties(
                    station_id,
                    obs,
                    coords,
                    station_name=station["properties"].get("name", ""),
                )
            )
        except Exception:
            continue

    return observations


def _parse_obs_properties(
    sid: str,
    props: dict,
    coords: list,
    station_name: str = "",
) -> dict[str, Any]:
    """Parse a raw NWS observation properties dict into our normalized shape."""

    def _val(key: str):
        entry = props.get(key) or {}
        return entry.get("value") if isinstance(entry, dict) else None

    temp_c = _val("temperature")
    dewpoint_c = _val("dewpoint")
    wind_speed_kmh = _val("windSpeed")
    wind_dir = _val("windDirection")
    pressure_pa = _val("barometricPressure")
    visibility_m = _val("visibility")

    lon = coords[0] if len(coords) >= 2 else None
    lat = coords[1] if len(coords) >= 2 else None

    return {
        "station_id": sid,
        "station_name": station_name,
        "lon": lon,
        "lat": lat,
        "temperature_f": round(temp_c * 9 / 5 + 32, 1) if temp_c is not None else None,
        "temperature_c": round(temp_c, 1) if temp_c is not None else None,
        "dewpoint_f": round(dewpoint_c * 9 / 5 + 32, 1)
        if dewpoint_c is not None
        else None,
        "wind_speed_mph": round(wind_speed_kmh * 0.621371, 1)
        if wind_speed_kmh is not None
        else None,
        "wind_direction": wind_dir,
        "pressure_mb": round(pressure_pa / 100, 1) if pressure_pa is not None else None,
        "visibility_mi": round(visibility_m / 1609.34, 1)
        if visibility_m is not None
        else None,
        "description": props.get("textDescription", ""),
        "timestamp": props.get("timestamp", ""),
    }


def fetch_observations_bulk(
    station_ids: list[str],
) -> list[dict[str, Any]]:
    """
    Fetch latest observations for a list of station IDs.
    More efficient when you already know which stations you want.
    """
    observations = []
    for sid in station_ids:
        try:
            resp = SESSION.get(
                f"{NWS_API_BASE}/stations/{sid}/observations/latest",
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            props = data.get("properties", {})
            coords = data.get("geometry", {}).get("coordinates", [None, None]) or [
                None,
                None,
            ]
            observations.append(_parse_obs_properties(sid, props, coords))
        except Exception:
            continue

    return observations


def fetch_point_forecast(lat: float, lon: float) -> dict[str, Any]:
    """
    Fetch the NWS text forecast for a specific lat/lon.
    Two-step: /points -> /forecast.
    """
    try:
        resp = SESSION.get(
            f"{NWS_API_BASE}/points/{lat:.4f},{lon:.4f}", timeout=10
        )
        resp.raise_for_status()
        forecast_url = resp.json()["properties"]["forecast"]

        forecast_resp = SESSION.get(forecast_url, timeout=10)
        forecast_resp.raise_for_status()
        periods = forecast_resp.json()["properties"]["periods"]

        return {
            "periods": [
                {
                    "name": p["name"],
                    "temperature": p["temperature"],
                    "temperatureUnit": p["temperatureUnit"],
                    "windSpeed": p["windSpeed"],
                    "windDirection": p["windDirection"],
                    "shortForecast": p["shortForecast"],
                    "detailedForecast": p["detailedForecast"],
                }
                for p in periods
            ]
        }
    except Exception as e:
        logger.error("Failed to fetch forecast for %.2f,%.2f: %s", lat, lon, e)
        return {"periods": []}
