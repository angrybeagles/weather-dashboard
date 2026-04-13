"""Compact memory-stats sidebar panel with per-cache Clear buttons."""

from __future__ import annotations

from dash import html


def build_memory_panel() -> html.Div:
    """Container populated by the refresh callback."""
    return html.Div(
        className="memory-panel",
        children=[
            html.H4("Memory", className="controls-title"),
            html.Div(id="memory-stats-content", className="memory-rows"),
        ],
    )


def _fmt_mb(b: int) -> str:
    mb = b / (1024 * 1024)
    if mb < 0.05:
        return "—"
    if mb < 10:
        return f"{mb:.2f} MB"
    return f"{mb:.1f} MB"


def render_rows(breakdown: dict[str, int], rss_mb: float, disk_mb: float) -> list:
    """Build the row children given pre-computed sizes."""
    rows = []
    for name in ("hrrr", "goes", "nexrad", "alerts", "observations"):
        bytes_ = breakdown.get(name, 0)
        rows.append(
            html.Div(
                className="memory-row",
                children=[
                    html.Span(name.upper(), className="memory-row-label"),
                    html.Span(_fmt_mb(bytes_), className="memory-row-size"),
                    html.Button(
                        "Clear",
                        id={"type": "mem-clear", "cache": name},
                        n_clicks=0,
                        className="memory-clear-btn",
                    ),
                ],
            )
        )
    rows.append(html.Hr(className="memory-divider"))
    rows.append(
        html.Div(
            className="memory-totals",
            children=[
                html.Div(
                    [html.Span("RSS", className="memory-total-label"),
                     html.Span(f"{rss_mb:.0f} MB", className="memory-total-value")],
                    className="memory-total-row",
                ),
                html.Div(
                    [html.Span("DISK", className="memory-total-label"),
                     html.Span(f"{disk_mb:.0f} MB", className="memory-total-value")],
                    className="memory-total-row",
                ),
            ],
        )
    )
    return rows
