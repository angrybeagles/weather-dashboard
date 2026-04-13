"""
Map layer rendering for the Dash dashboard.

Builds Plotly mapbox figures with toggleable weather data layers.
"""

import numpy as np
import plotly.graph_objects as go

from config import CONUS_BOUNDS, MAP_CENTER, MAP_ZOOM


def _hex_to_rgba(color: str, alpha: float = 1.0) -> str:
    """Convert a #RRGGBB hex color (or passthrough rgb/rgba) to rgba() string."""
    if color.startswith("rgba") or color.startswith("rgb"):
        return color
    c = color.lstrip("#")
    if len(c) != 6:
        return f"rgba(128,128,128,{alpha})"
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def create_base_map(style: str = "carto-darkmatter") -> go.Figure:
    """Create the base CONUS map figure."""
    fig = go.Figure()

    fig.update_layout(
        mapbox=dict(
            style=style,
            center=MAP_CENTER,
            zoom=MAP_ZOOM,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=700,
    )

    return fig


def add_temperature_layer(
    fig: go.Figure,
    lats: np.ndarray,
    lons: np.ndarray,
    temps_f: np.ndarray,
    visible: bool = True,
) -> go.Figure:
    """
    Add temperature contour/heatmap layer to the map.

    Uses scattermapbox with color-coded markers for a grid of points.
    For performance, we subsample the HRRR grid.
    """
    # Subsample for performance (every 5th point)
    step = 5
    lat_sub = lats[::step, ::step].ravel()
    lon_sub = lons[::step, ::step].ravel()
    temp_sub = temps_f[::step, ::step].ravel()

    # Filter to CONUS and valid data
    mask = (
        (lat_sub >= CONUS_BOUNDS["lat_min"])
        & (lat_sub <= CONUS_BOUNDS["lat_max"])
        & (lon_sub >= CONUS_BOUNDS["lon_min"])
        & (lon_sub <= CONUS_BOUNDS["lon_max"])
        & ~np.isnan(temp_sub)
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=lat_sub[mask],
            lon=lon_sub[mask],
            mode="markers",
            marker=dict(
                size=6,
                color=temp_sub[mask],
                colorscale="RdYlBu_r",
                cmin=0,
                cmax=110,
                opacity=0.6,
                colorbar=dict(
                    title=dict(text="°F", font=dict(color="white")),
                    x=0.01,
                    y=0.5,
                    len=0.4,
                    bgcolor="rgba(20,20,20,0.8)",
                    tickfont=dict(color="white"),
                ),
            ),
            text=[f"{t:.0f}°F" for t in temp_sub[mask]],
            hovertemplate="%{text}<extra>Temperature</extra>",
            visible=visible,
            name="Temperature",
        )
    )

    return fig


def add_wind_layer(
    fig: go.Figure,
    lats: np.ndarray,
    lons: np.ndarray,
    speed_mph: np.ndarray,
    direction: np.ndarray,
    visible: bool = True,
) -> go.Figure:
    """Add wind barb layer to the map."""
    step = 8
    lat_sub = lats[::step, ::step].ravel()
    lon_sub = lons[::step, ::step].ravel()
    spd_sub = speed_mph[::step, ::step].ravel()
    dir_sub = direction[::step, ::step].ravel()

    mask = (
        (lat_sub >= CONUS_BOUNDS["lat_min"])
        & (lat_sub <= CONUS_BOUNDS["lat_max"])
        & (lon_sub >= CONUS_BOUNDS["lon_min"])
        & (lon_sub <= CONUS_BOUNDS["lon_max"])
        & ~np.isnan(spd_sub)
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=lat_sub[mask],
            lon=lon_sub[mask],
            mode="markers",
            marker=dict(
                size=np.clip(spd_sub[mask] / 3, 3, 15),
                color=spd_sub[mask],
                colorscale="Viridis",
                cmin=0,
                cmax=60,
                opacity=0.7,
                colorbar=dict(
                    title=dict(text="mph", font=dict(color="white")),
                    x=0.06,
                    y=0.5,
                    len=0.4,
                    bgcolor="rgba(20,20,20,0.8)",
                    tickfont=dict(color="white"),
                ),
            ),
            text=[
                f"{s:.0f} mph from {d:.0f}°"
                for s, d in zip(spd_sub[mask], dir_sub[mask])
            ],
            hovertemplate="%{text}<extra>Wind</extra>",
            visible=visible,
            name="Wind",
        )
    )

    return fig


def add_reflectivity_layer(
    fig: go.Figure,
    lats: np.ndarray,
    lons: np.ndarray,
    refl_dbz: np.ndarray,
    visible: bool = True,
) -> go.Figure:
    """Add composite reflectivity (radar-like) layer."""
    step = 3
    lat_sub = lats[::step, ::step].ravel()
    lon_sub = lons[::step, ::step].ravel()
    refl_sub = refl_dbz[::step, ::step].ravel()

    mask = (
        (lat_sub >= CONUS_BOUNDS["lat_min"])
        & (lat_sub <= CONUS_BOUNDS["lat_max"])
        & (lon_sub >= CONUS_BOUNDS["lon_min"])
        & (lon_sub <= CONUS_BOUNDS["lon_max"])
        & ~np.isnan(refl_sub)
        & (refl_sub > 5)  # Only show where there's actual precipitation signal
    )

    if mask.sum() == 0:
        return fig

    # NWS-style reflectivity colorscale
    nws_colors = [
        [0.0, "rgba(0,0,0,0)"],       # < 5 dBZ transparent
        [0.07, "#04e9e7"],              # 5 dBZ
        [0.13, "#019ff4"],              # 10
        [0.20, "#0300f4"],              # 15
        [0.27, "#02fd02"],              # 20
        [0.33, "#01c501"],              # 25
        [0.40, "#008e00"],              # 30
        [0.47, "#fdf802"],              # 35
        [0.53, "#e5bc00"],              # 40
        [0.60, "#fd9500"],              # 45
        [0.67, "#fd0000"],              # 50
        [0.73, "#d40000"],              # 55
        [0.80, "#bc0000"],              # 60
        [0.87, "#f800fd"],              # 65
        [0.93, "#9854c6"],              # 70
        [1.0, "#ffffff"],               # 75+
    ]

    fig.add_trace(
        go.Scattermapbox(
            lat=lat_sub[mask],
            lon=lon_sub[mask],
            mode="markers",
            marker=dict(
                size=5,
                color=refl_sub[mask],
                colorscale=nws_colors,
                cmin=0,
                cmax=75,
                opacity=0.8,
                colorbar=dict(
                    title=dict(text="dBZ", font=dict(color="white")),
                    x=0.11,
                    y=0.5,
                    len=0.4,
                    bgcolor="rgba(20,20,20,0.8)",
                    tickfont=dict(color="white"),
                ),
            ),
            text=[f"{r:.0f} dBZ" for r in refl_sub[mask]],
            hovertemplate="%{text}<extra>Reflectivity</extra>",
            visible=visible,
            name="Reflectivity",
        )
    )

    return fig


def add_observations_layer(
    fig: go.Figure,
    observations: list[dict],
    visible: bool = True,
) -> go.Figure:
    """Add surface observation station markers."""
    valid_obs = [o for o in observations if o.get("lat") and o.get("lon")]
    if not valid_obs:
        return fig

    fig.add_trace(
        go.Scattermapbox(
            lat=[o["lat"] for o in valid_obs],
            lon=[o["lon"] for o in valid_obs],
            mode="markers+text",
            marker=dict(size=10, color="#00ffcc", opacity=0.9),
            text=[
                f"{o.get('temperature_f', '—')}°"
                if o.get("temperature_f") is not None
                else "—"
                for o in valid_obs
            ],
            textfont=dict(size=9, color="white"),
            textposition="top center",
            customdata=[
                f"{o.get('station_id','')}: {o.get('description','')}<br>"
                f"Wind: {o.get('wind_speed_mph','—')} mph<br>"
                f"Pressure: {o.get('pressure_mb','—')} mb"
                for o in valid_obs
            ],
            hovertemplate="%{customdata}<extra>%{text}</extra>",
            visible=visible,
            name="Observations",
        )
    )

    return fig


def add_alerts_layer(
    fig: go.Figure,
    alerts: list[dict],
    visible: bool = True,
) -> go.Figure:
    """
    Add alert polygons/markers to the map.
    NWS alerts sometimes have polygon geometry, sometimes just area names.
    """
    geo_alerts = [a for a in alerts if a.get("geometry")]
    if not geo_alerts:
        return fig

    for alert in geo_alerts[:20]:  # Limit for performance
        geom = alert["geometry"]
        if geom.get("type") == "Polygon":
            coords = geom["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]

            fig.add_trace(
                go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode="lines",
                    line=dict(width=2, color=alert["color"]),
                    fill="toself",
                    fillcolor=_hex_to_rgba(alert["color"], alpha=0.2),
                    text=alert["headline"],
                    hovertemplate="%{text}<extra></extra>",
                    visible=visible,
                    name=alert["event"],
                    showlegend=False,
                )
            )

    return fig


def add_nexrad_radar_layer(
    fig: go.Figure,
    radar_data: dict,
    visible: bool = True,
) -> go.Figure:
    """
    Add MRMS composite reflectivity (real radar) layer to the map.

    Parameters
    ----------
    radar_data : dict
        Output from nexrad.radar_to_plotly_data() with 'lats', 'lons', 'values'.
    """
    if radar_data is None:
        return fig

    lats = radar_data["lats"]
    lons = radar_data["lons"]
    vals = radar_data["values"]

    # NWS-style reflectivity colorscale
    nws_colors = [
        [0.0, "rgba(0,0,0,0)"],
        [0.07, "#04e9e7"],
        [0.13, "#019ff4"],
        [0.20, "#0300f4"],
        [0.27, "#02fd02"],
        [0.33, "#01c501"],
        [0.40, "#008e00"],
        [0.47, "#fdf802"],
        [0.53, "#e5bc00"],
        [0.60, "#fd9500"],
        [0.67, "#fd0000"],
        [0.73, "#d40000"],
        [0.80, "#bc0000"],
        [0.87, "#f800fd"],
        [0.93, "#9854c6"],
        [1.0, "#ffffff"],
    ]

    fig.add_trace(
        go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="markers",
            marker=dict(
                size=4,
                color=vals,
                colorscale=nws_colors,
                cmin=0,
                cmax=75,
                opacity=0.85,
                colorbar=dict(
                    title=dict(text="dBZ", font=dict(color="white", size=10)),
                    x=0.01,
                    y=0.5,
                    len=0.35,
                    bgcolor="rgba(20,20,20,0.8)",
                    tickfont=dict(color="white", size=9),
                ),
            ),
            text=[f"{v:.0f} dBZ" for v in vals],
            hovertemplate="%{text}<extra>Radar</extra>",
            visible=visible,
            name="NEXRAD Radar",
        )
    )

    return fig


def add_satellite_layer(
    fig: go.Figure,
    goes_image: dict,
    opacity: float = 0.55,
    visible: bool = True,
) -> go.Figure:
    """
    Add a GOES satellite image overlay to the mapbox figure.

    Parameters
    ----------
    goes_image : dict
        Output of ``pipeline.goes.goes_to_plotly_image`` with ``source`` (data URI)
        and ``coordinates`` (4 corner [lon, lat] pairs).
    """
    if not goes_image or "source" not in goes_image:
        return fig

    layout = fig.layout
    existing = list(layout.mapbox.layers) if layout.mapbox.layers else []
    existing.append(
        {
            "sourcetype": "image",
            "source": goes_image["source"],
            "coordinates": goes_image["coordinates"],
            "opacity": opacity,
            "below": "traces",
        }
    )
    fig.update_layout(mapbox=dict(layers=existing))
    # Name-only trace so toggling/visibility & tests can assert presence.
    fig.add_trace(
        go.Scattermapbox(
            lat=[None],
            lon=[None],
            mode="markers",
            marker=dict(size=0, opacity=0),
            name="Satellite",
            visible=visible,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    return fig


def add_radar_sites_layer(
    fig: go.Figure,
    sites: list[dict],
    visible: bool = True,
) -> go.Figure:
    """Add small markers for each NEXRAD radar site location."""
    fig.add_trace(
        go.Scattermapbox(
            lat=[s["lat"] for s in sites],
            lon=[s["lon"] for s in sites],
            mode="markers",
            marker=dict(
                size=5,
                color="rgba(0,255,204,0.3)",
                symbol="circle",
            ),
            text=[f"{s['id']} — {s['name']}" for s in sites],
            hovertemplate="%{text}<extra>Radar Site</extra>",
            visible=visible,
            name="Radar Sites",
            showlegend=False,
        )
    )

    return fig
