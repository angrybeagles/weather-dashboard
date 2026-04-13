"""Tests for map layer rendering — alerts opacity and satellite layer."""

import re

import plotly.graph_objects as go

from components.map_layer import (
    _hex_to_rgba,
    add_alerts_layer,
    add_satellite_layer,
    create_base_map,
)


def test_hex_to_rgba_basic():
    assert _hex_to_rgba("#FF0000", 0.2) == "rgba(255,0,0,0.2)"
    assert _hex_to_rgba("#00cc66", 1.0) == "rgba(0,204,102,1.0)"


def test_hex_to_rgba_passthrough_rgba():
    assert _hex_to_rgba("rgba(1,2,3,0.5)", 0.2) == "rgba(1,2,3,0.5)"


def test_alert_polygon_uses_translucent_fill():
    """Regression: alert polygons used to render as fully opaque blobs because
    fillcolor was set to the raw hex color. They should now use rgba with low
    alpha so the underlying map remains visible."""
    fig = create_base_map()
    alert = {
        "color": "#FF0000",
        "headline": "Severe Thunderstorm Warning",
        "event": "Severe Thunderstorm",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-100, 40], [-99, 40], [-99, 41], [-100, 41], [-100, 40]]],
        },
    }
    fig = add_alerts_layer(fig, [alert])

    # last trace is the polygon
    polygon = fig.data[-1]
    fillcolor = polygon.fillcolor
    assert fillcolor.startswith("rgba("), f"Expected rgba fill, got {fillcolor!r}"

    # alpha should be < 1 — the whole point of the fix
    alpha = float(re.findall(r"[\d.]+", fillcolor)[-1])
    assert 0 < alpha < 1, f"Alert fill alpha must be translucent, got {alpha}"


def test_add_satellite_layer_attaches_image_overlay():
    """Satellite imagery used to never be rendered. add_satellite_layer should
    attach an image layer to the mapbox layout and add a Satellite trace."""
    fig = create_base_map()
    img = {
        "source": "data:image/png;base64,AAAA",
        "coordinates": [[-125, 50], [-66, 50], [-66, 24], [-125, 24]],
    }
    fig = add_satellite_layer(fig, img)

    layers = fig.layout.mapbox.layers
    assert layers and len(layers) == 1
    assert layers[0].sourcetype == "image"
    assert layers[0].source == img["source"]

    names = [t.name for t in fig.data]
    assert "Satellite" in names


def test_add_satellite_layer_handles_empty_dict():
    fig = create_base_map()
    fig2 = add_satellite_layer(fig, {})
    # nothing added
    assert len(fig2.data) == 0
