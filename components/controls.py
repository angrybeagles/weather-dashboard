"""
Dashboard control components — layer toggles, time slider, status bar.
"""

from dash import dcc, html


def build_layer_controls() -> html.Div:
    """Build the layer toggle panel."""
    return html.Div(
        className="layer-controls",
        children=[
            html.H4("Layers", className="controls-title"),
            dcc.Checklist(
                id="layer-toggles",
                options=[
                    {"label": " Temperature", "value": "temperature"},
                    {"label": " Wind", "value": "wind"},
                    {"label": " Reflectivity", "value": "reflectivity"},
                    {"label": " NEXRAD Radar", "value": "nexrad"},
                    {"label": " Observations", "value": "observations"},
                    {"label": " Alerts", "value": "alerts"},
                    {"label": " Satellite IR", "value": "satellite_ir"},
                ],
                value=["observations", "alerts", "nexrad"],
                className="layer-checklist",
                inputClassName="layer-checkbox",
                labelClassName="layer-label",
            ),
        ],
    )


def build_time_slider() -> html.Div:
    """Build the forecast hour time slider."""
    return html.Div(
        className="time-slider-container",
        children=[
            html.Label("Forecast Hour", className="slider-label"),
            dcc.Slider(
                id="forecast-hour-slider",
                min=0,
                max=18,
                step=1,
                value=0,
                marks={
                    0: {"label": "Now", "style": {"color": "#00ffcc"}},
                    3: {"label": "+3h"},
                    6: {"label": "+6h"},
                    9: {"label": "+9h"},
                    12: {"label": "+12h"},
                    15: {"label": "+15h"},
                    18: {"label": "+18h"},
                },
                className="forecast-slider",
            ),
        ],
    )


def build_satellite_selector() -> html.Div:
    """Build satellite channel selector."""
    return html.Div(
        className="satellite-selector",
        children=[
            html.Label("Satellite Channel", className="selector-label"),
            dcc.RadioItems(
                id="satellite-channel",
                options=[
                    {"label": " Infrared", "value": "ir"},
                    {"label": " Visible", "value": "visible"},
                    {"label": " Water Vapor", "value": "water_vapor"},
                ],
                value="ir",
                className="satellite-radio",
                inputClassName="satellite-radio-input",
                labelClassName="satellite-radio-label",
            ),
        ],
    )


def build_status_bar(last_updated: dict) -> html.Div:
    """Build the data freshness status bar."""
    items = []
    # Pick the most recent GOES channel timestamp (any of ir/visible/water_vapor).
    goes_ts = max(
        (v for k, v in last_updated.items() if k.startswith("goes_")),
        default=None,
    )
    entries = [
        ("HRRR", last_updated.get("hrrr")),
        ("GOES", goes_ts),
        ("NEXRAD", last_updated.get("nexrad")),
        ("Alerts", last_updated.get("alerts")),
        ("Obs", last_updated.get("observations")),
    ]
    for label, ts in entries:
        if ts:
            age_min = (
                __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                )
                - ts
            ).total_seconds() / 60
            if age_min < 5:
                status = "fresh"
            elif age_min < 30:
                status = "aging"
            else:
                status = "stale"
            time_str = ts.strftime("%H:%M UTC")
        else:
            status = "missing"
            time_str = "—"

        items.append(
            html.Span(
                className=f"status-item status-{status}",
                children=[
                    html.Span("●", className="status-dot"),
                    html.Span(f" {label}: {time_str}"),
                ],
            )
        )

    return html.Div(className="status-bar", children=items)
