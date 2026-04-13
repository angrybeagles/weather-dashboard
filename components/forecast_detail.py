"""
Point forecast detail panel — shown when user clicks a location on the map.

Fetches the NWS 7-day forecast for the clicked lat/lon and displays
it in a compact card format alongside HRRR model meteogram data.
"""

from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_point_forecast_panel(
    forecast: dict,
    lat: float,
    lon: float,
) -> html.Div:
    """
    Build the forecast detail panel for a clicked point.

    Parameters
    ----------
    forecast : dict
        Output from nws.fetch_point_forecast() with 'periods' key.
    lat, lon : float
        Coordinates of the clicked point.
    """
    periods = forecast.get("periods", [])
    if not periods:
        return html.Div(
            className="forecast-panel",
            children=[
                html.P(
                    f"No forecast available for {lat:.2f}°N, {lon:.2f}°W",
                    className="forecast-empty",
                ),
            ],
        )

    # Header
    header = html.Div(
        className="forecast-header",
        children=[
            html.H4("Point Forecast", className="panel-title"),
            html.Span(
                f"{lat:.2f}°N, {abs(lon):.2f}°W",
                className="forecast-coords",
            ),
        ],
    )

    # Period cards (first 6 periods = ~3 days)
    period_cards = []
    for p in periods[:8]:
        temp = p.get("temperature", "—")
        unit = p.get("temperatureUnit", "F")
        wind = p.get("windSpeed", "—")
        wind_dir = p.get("windDirection", "")
        short = p.get("shortForecast", "")

        # Icon based on forecast text
        icon = _forecast_icon(short)

        period_cards.append(
            html.Div(
                className="forecast-period",
                children=[
                    html.Div(p.get("name", ""), className="period-name"),
                    html.Div(icon, className="period-icon"),
                    html.Div(
                        f"{temp}°{unit}",
                        className="period-temp",
                    ),
                    html.Div(short, className="period-short"),
                    html.Div(
                        f"{wind_dir} {wind}",
                        className="period-wind",
                    ),
                ],
            )
        )

    periods_row = html.Div(className="forecast-periods", children=period_cards)

    return html.Div(
        className="forecast-panel",
        children=[header, periods_row],
    )


def _forecast_icon(short_forecast: str) -> str:
    """Map NWS short forecast text to an emoji icon."""
    text = short_forecast.lower()
    if any(w in text for w in ["thunder", "tstorm"]):
        return "⛈"
    if any(w in text for w in ["snow", "blizzard", "flurr"]):
        return "🌨"
    if any(w in text for w in ["rain", "shower", "drizzle"]):
        return "🌧"
    if "fog" in text:
        return "🌫"
    if any(w in text for w in ["cloud", "overcast"]):
        return "☁"
    if "partly" in text:
        return "⛅"
    if any(w in text for w in ["sun", "clear"]):
        return "☀"
    if "wind" in text:
        return "💨"
    if "hot" in text:
        return "🌡"
    return "🌤"


def build_meteogram(
    hrrr_data: dict,
    lat: float,
    lon: float,
) -> dcc.Graph | html.Div:
    """
    Build a meteogram (time-series plot) for a point using HRRR data.

    Shows temperature, wind speed, and precipitation over the forecast
    horizon at the nearest HRRR grid point.

    Parameters
    ----------
    hrrr_data : dict
        Store's hrrr_data with xarray DataArrays keyed by variable name.
    lat, lon : float
        Target coordinates.

    Returns
    -------
    dcc.Graph with the meteogram figure, or an empty Div on failure.
    """
    import numpy as np

    if not hrrr_data:
        return html.Div("HRRR data not yet loaded", className="meteogram-empty")

    try:
        from pipeline.hrrr import kelvin_to_fahrenheit, ms_to_mph

        # Extract nearest grid point for each variable
        def nearest_point(da):
            """Find nearest grid point to lat/lon."""
            if "latitude" in da.coords and "longitude" in da.coords:
                lat_vals = da.latitude.values
                lon_vals = da.longitude.values

                if lat_vals.ndim == 2:
                    # 2D coordinate arrays — find nearest cell
                    dist = (lat_vals - lat) ** 2 + (lon_vals - lon) ** 2
                    idx = np.unravel_index(np.argmin(dist), dist.shape)
                    if "valid_time" in da.dims:
                        return da.isel(y=idx[0], x=idx[1])
                    return da.isel(y=idx[0], x=idx[1])
            return None

        # Build time series
        times = []
        temps_f = []
        winds_mph = []
        precip = []

        temp_da = hrrr_data.get("temperature_2m")
        u_da = hrrr_data.get("u_wind_10m")
        v_da = hrrr_data.get("v_wind_10m")
        precip_da = hrrr_data.get("total_precip")

        if temp_da is not None and "valid_time" in temp_da.dims:
            temp_ts = nearest_point(temp_da)
            if temp_ts is not None:
                times = [str(t) for t in temp_ts.valid_time.values]
                temps_f = kelvin_to_fahrenheit(temp_ts).values.tolist()

        if u_da is not None and v_da is not None:
            u_ts = nearest_point(u_da)
            v_ts = nearest_point(v_da)
            if u_ts is not None and v_ts is not None:
                speed = np.sqrt(u_ts.values ** 2 + v_ts.values ** 2)
                winds_mph = ms_to_mph(
                    __import__("xarray").DataArray(speed)
                ).values.tolist()

        if precip_da is not None:
            p_ts = nearest_point(precip_da)
            if p_ts is not None:
                # Convert kg/m² to inches (1 kg/m² ≈ 0.03937 inches)
                precip = (p_ts.values * 0.03937).tolist()

        if not times:
            return html.Div(
                "Could not extract point data from HRRR",
                className="meteogram-empty",
            )

        # Build figure with subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.4, 0.3, 0.3],
            subplot_titles=("Temperature (°F)", "Wind Speed (mph)", "Precip (in)"),
        )

        # Temperature
        fig.add_trace(
            go.Scatter(
                x=times,
                y=temps_f,
                mode="lines+markers",
                line=dict(color="#ef4444", width=2),
                marker=dict(size=4),
                name="Temp",
                hovertemplate="%{y:.0f}°F<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Wind
        if winds_mph:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=winds_mph,
                    mode="lines+markers",
                    line=dict(color="#3b82f6", width=2),
                    marker=dict(size=4),
                    name="Wind",
                    hovertemplate="%{y:.0f} mph<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # Precipitation
        if precip:
            fig.add_trace(
                go.Bar(
                    x=times,
                    y=precip,
                    marker_color="#22c55e",
                    name="Precip",
                    hovertemplate="%{y:.2f} in<extra></extra>",
                ),
                row=3,
                col=1,
            )

        fig.update_layout(
            height=350,
            margin=dict(l=40, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,35,50,0.8)",
            font=dict(color="#8899aa", size=10, family="JetBrains Mono"),
            showlegend=False,
        )

        fig.update_xaxes(
            gridcolor="rgba(42,53,72,0.5)",
            tickformat="%H:%M",
        )
        fig.update_yaxes(gridcolor="rgba(42,53,72,0.5)")

        # Style subplot titles
        for ann in fig.layout.annotations:
            ann.font.size = 10
            ann.font.color = "#8899aa"

        return dcc.Graph(
            figure=fig,
            config={"displayModeBar": False},
            className="meteogram-graph",
        )

    except Exception as e:
        return html.Div(
            f"Meteogram error: {e}",
            className="meteogram-empty",
        )
