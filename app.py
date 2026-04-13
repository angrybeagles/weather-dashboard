"""
CONUS Weather Dashboard — Main Application

A Plotly Dash application that displays real-time weather data from
HRRR model output, GOES satellite imagery, NEXRAD radar composites,
NWS alerts, and surface observations on an interactive CONUS map.

Features:
  - Toggleable data layers (temp, wind, radar, reflectivity, obs, alerts)
  - Forecast hour time slider (HRRR 0-18h)
  - Click-on-map for NWS 7-day point forecast + HRRR meteogram
  - NEXRAD MRMS composite radar mosaic
  - Background data refresh via APScheduler
"""

import logging

from dash import ALL, Dash, Input, Output, State, callback, ctx, dcc, html, no_update

from components.alerts_panel import build_alerts_panel
from components.controls import (
    build_layer_controls,
    build_satellite_selector,
    build_status_bar,
    build_time_slider,
)
from components.forecast_detail import build_meteogram, build_point_forecast_panel
from components.memory_stats import build_memory_panel, render_rows
from components.map_layer import (
    add_alerts_layer,
    add_nexrad_radar_layer,
    add_observations_layer,
    add_radar_sites_layer,
    add_reflectivity_layer,
    add_satellite_layer,
    add_temperature_layer,
    add_wind_layer,
    create_base_map,
)
from config import DASH_DEBUG, DASH_HOST, DASH_PORT
from pipeline.scheduler import start_scheduler, store

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dashboard")

# ---------------------------------------------------------------------------
# Initialize Dash app
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    title="CONUS Weather",
    update_title=None,
    suppress_callback_exceptions=True,
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    className="dashboard-container",
    children=[
        # ---- Header ----
        html.Div(
            className="dashboard-header",
            children=[
                html.H1(
                    [html.Span("wx"), " CONUS Weather Dashboard"],
                    className="dashboard-title",
                ),
                html.Div(id="status-bar"),
            ],
        ),
        # ---- Left Panel: Controls ----
        html.Div(
            className="left-panel",
            children=[
                build_layer_controls(),
                build_memory_panel(),
                build_time_slider(),
                build_satellite_selector(),
                html.Hr(style={"borderColor": "#2a3548", "margin": "16px 0"}),
                html.Div(id="point-forecast-panel"),
            ],
        ),
        # ---- Center: Map ----
        html.Div(
            className="map-container",
            children=[
                dcc.Graph(
                    id="weather-map",
                    config={
                        "scrollZoom": True,
                        "displayModeBar": False,
                        "doubleClick": "reset",
                    },
                    style={"height": "100%", "width": "100%"},
                ),
            ],
        ),
        # ---- Right Panel: Alerts + Meteogram ----
        html.Div(
            className="right-panel",
            children=[
                html.Div(id="alerts-panel"),
                html.Hr(style={"borderColor": "#2a3548", "margin": "16px 0"}),
                html.Div(id="meteogram-panel"),
            ],
        ),
        # ---- Footer ----
        html.Div(className="dashboard-footer", id="footer-status"),
        # ---- Auto-refresh timer ----
        dcc.Interval(
            id="refresh-interval",
            interval=30_000,  # 30 seconds
            n_intervals=0,
        ),
        # ---- Hidden store for clicked coordinates ----
        dcc.Store(id="clicked-coords", data=None),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    Output("weather-map", "figure"),
    Input("layer-toggles", "value"),
    Input("forecast-hour-slider", "value"),
    Input("satellite-channel", "value"),
    Input("refresh-interval", "n_intervals"),
)
def update_map(active_layers, forecast_hour, sat_channel, _n):
    """Rebuild the map figure based on active layers and forecast hour."""
    fig = create_base_map()

    # --- NEXRAD real radar (MRMS composite) ---
    if "nexrad" in active_layers and store.nexrad_data is not None:
        fig = add_nexrad_radar_layer(fig, store.nexrad_data)
        from pipeline.nexrad import fetch_nexrad_sites

        fig = add_radar_sites_layer(fig, fetch_nexrad_sites())

    # --- HRRR model layers ---
    hrrr = store.hrrr_data

    if hrrr and "temperature_2m" in hrrr and "temperature" in active_layers:
        try:
            temp_da = hrrr["temperature_2m"]
            if (
                "valid_time" in temp_da.dims
                and len(temp_da.valid_time) > forecast_hour
            ):
                temp_slice = temp_da.isel(valid_time=forecast_hour)
            elif "valid_time" in temp_da.dims:
                temp_slice = temp_da.isel(valid_time=-1)
            else:
                temp_slice = temp_da

            from pipeline.hrrr import kelvin_to_fahrenheit

            temps_f = kelvin_to_fahrenheit(temp_slice).values
            lats = temp_slice.latitude.values
            lons = temp_slice.longitude.values
            fig = add_temperature_layer(fig, lats, lons, temps_f)
        except Exception as e:
            logger.warning("Temperature layer error: %s", e)

    if (
        hrrr
        and "u_wind_10m" in hrrr
        and "v_wind_10m" in hrrr
        and "wind" in active_layers
    ):
        try:
            from pipeline.hrrr import get_wind_speed_direction, ms_to_mph

            u = hrrr["u_wind_10m"]
            v = hrrr["v_wind_10m"]
            if "valid_time" in u.dims and len(u.valid_time) > forecast_hour:
                u = u.isel(valid_time=forecast_hour)
                v = v.isel(valid_time=forecast_hour)
            elif "valid_time" in u.dims:
                u = u.isel(valid_time=-1)
                v = v.isel(valid_time=-1)

            speed, direction = get_wind_speed_direction(u, v)
            speed_mph = ms_to_mph(speed).values
            fig = add_wind_layer(
                fig,
                u.latitude.values,
                u.longitude.values,
                speed_mph,
                direction.values,
            )
        except Exception as e:
            logger.warning("Wind layer error: %s", e)

    if hrrr and "composite_reflectivity" in hrrr and "reflectivity" in active_layers:
        try:
            refl = hrrr["composite_reflectivity"]
            if "valid_time" in refl.dims and len(refl.valid_time) > forecast_hour:
                refl = refl.isel(valid_time=forecast_hour)
            elif "valid_time" in refl.dims:
                refl = refl.isel(valid_time=-1)
            fig = add_reflectivity_layer(
                fig, refl.latitude.values, refl.longitude.values, refl.values
            )
        except Exception as e:
            logger.warning("Reflectivity layer error: %s", e)

    # --- Satellite (GOES) ---
    if "satellite_ir" in active_layers:
        try:
            from pipeline.goes import goes_to_plotly_image

            store.ensure_goes(sat_channel)
            sat_da = store.goes_data.get(sat_channel)
            if sat_da is not None:
                img = goes_to_plotly_image(sat_da)
                if img:
                    fig = add_satellite_layer(fig, img)
        except Exception as e:
            logger.warning("Satellite layer error: %s", e)
    elif store.goes_data:
        # Satellite layer off: drop all channels to free memory.
        store.evict_goes()

    # --- Observations ---
    if "observations" in active_layers and store.observations:
        fig = add_observations_layer(fig, store.observations)

    # --- Alerts ---
    if "alerts" in active_layers and store.alerts:
        fig = add_alerts_layer(fig, store.alerts)

    return fig


@callback(
    Output("clicked-coords", "data"),
    Input("weather-map", "clickData"),
    prevent_initial_call=True,
)
def capture_click(click_data):
    """Store the clicked lat/lon coordinates."""
    if click_data is None:
        return no_update
    try:
        point = click_data["points"][0]
        return {"lat": point["lat"], "lon": point["lon"]}
    except (KeyError, IndexError):
        return no_update


@callback(
    Output("point-forecast-panel", "children"),
    Input("clicked-coords", "data"),
    prevent_initial_call=True,
)
def show_point_forecast(coords):
    """Fetch and display NWS forecast for the clicked point."""
    if coords is None:
        return no_update

    lat, lon = coords["lat"], coords["lon"]
    try:
        from pipeline.nws import fetch_point_forecast

        forecast = fetch_point_forecast(lat, lon)
        return build_point_forecast_panel(forecast, lat, lon)
    except Exception as e:
        logger.warning("Point forecast error: %s", e)
        return html.P(
            f"Could not load forecast for {lat:.2f}, {lon:.2f}",
            className="forecast-empty",
        )


@callback(
    Output("meteogram-panel", "children"),
    Input("clicked-coords", "data"),
    prevent_initial_call=True,
)
def show_meteogram(coords):
    """Build HRRR meteogram for the clicked point."""
    if coords is None:
        return no_update

    return build_meteogram(store.hrrr_data, coords["lat"], coords["lon"])


@callback(
    Output("alerts-panel", "children"),
    Input("refresh-interval", "n_intervals"),
)
def update_alerts_panel(_n):
    """Refresh the alerts sidebar."""
    return build_alerts_panel(store.alerts)


@callback(
    Output("footer-status", "children"),
    Input("refresh-interval", "n_intervals"),
)
def update_status(_n):
    """Refresh the status bar."""
    return build_status_bar(store.last_updated)


@callback(
    Output("memory-stats-content", "children", allow_duplicate=True),
    Input({"type": "mem-clear", "cache": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def on_memory_clear(n_clicks_list):
    """Dispatch Clear button to the matching evict_* method."""
    triggered = ctx.triggered_id
    if not triggered or not any(n_clicks_list or []):
        return no_update
    cache = triggered.get("cache")
    evictor = {
        "hrrr": store.evict_hrrr,
        "goes": lambda: store.evict_goes(None),
        "nexrad": store.evict_nexrad,
        "alerts": store.evict_alerts,
        "observations": store.evict_observations,
    }.get(cache)
    if evictor:
        evictor()

    from pipeline.hrrr import CACHE_DIR as HRRR_CACHE_DIR
    from utils.memory import (
        disk_cache_bytes,
        process_rss_mb,
        store_bytes_breakdown,
    )

    return render_rows(
        store_bytes_breakdown(store),
        process_rss_mb(),
        disk_cache_bytes(HRRR_CACHE_DIR) / (1024 * 1024),
    )


@callback(
    Output("memory-stats-content", "children"),
    Input("refresh-interval", "n_intervals"),
)
def update_memory_panel(_n):
    """Refresh per-cache sizes, process RSS, and on-disk cache total."""
    from pipeline.hrrr import CACHE_DIR as HRRR_CACHE_DIR
    from utils.memory import (
        disk_cache_bytes,
        process_rss_mb,
        store_bytes_breakdown,
    )

    breakdown = store_bytes_breakdown(store)
    rss_mb = process_rss_mb()
    disk_mb = disk_cache_bytes(HRRR_CACHE_DIR) / (1024 * 1024)
    return render_rows(breakdown, rss_mb, disk_mb)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting CONUS Weather Dashboard")
    scheduler = start_scheduler()

    try:
        app.run(host=DASH_HOST, port=DASH_PORT, debug=DASH_DEBUG)
    finally:
        scheduler.shutdown(wait=False)
