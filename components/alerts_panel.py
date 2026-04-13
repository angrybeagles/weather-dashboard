"""
Alerts panel component for the sidebar.
"""

from datetime import datetime

from dash import html


def format_time(iso_str: str) -> str:
    """Format ISO timestamp to readable local time."""
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d %I:%M %p %Z")
    except (ValueError, AttributeError):
        return iso_str


def build_alert_card(alert: dict) -> html.Div:
    """Build a single alert card."""
    severity_classes = {
        "Extreme": "alert-extreme",
        "Severe": "alert-severe",
        "Moderate": "alert-moderate",
        "Minor": "alert-minor",
    }
    css_class = severity_classes.get(alert["severity"], "alert-minor")

    return html.Div(
        className=f"alert-card {css_class}",
        children=[
            html.Div(
                className="alert-header",
                children=[
                    html.Span(alert["severity"].upper(), className="alert-badge"),
                    html.Span(alert["event"], className="alert-event"),
                ],
            ),
            html.P(alert["headline"], className="alert-headline"),
            html.Div(
                className="alert-meta",
                children=[
                    html.Span(f"Areas: {alert['areas'][:80]}...") if len(alert.get("areas", "")) > 80
                    else html.Span(f"Areas: {alert.get('areas', '—')}"),
                    html.Br(),
                    html.Span(f"Until: {format_time(alert.get('expires', ''))}"),
                ],
            ),
        ],
    )


def build_alerts_panel(alerts: list[dict]) -> html.Div:
    """Build the full alerts sidebar panel."""
    if not alerts:
        return html.Div(
            className="alerts-panel",
            children=[
                html.H3("Active Alerts", className="panel-title"),
                html.P(
                    "No active alerts",
                    className="no-alerts",
                ),
            ],
        )

    # Group by severity
    extreme = [a for a in alerts if a["severity"] == "Extreme"]
    severe = [a for a in alerts if a["severity"] == "Severe"]
    moderate = [a for a in alerts if a["severity"] == "Moderate"]
    minor = [a for a in alerts if a["severity"] == "Minor"]

    total = len(alerts)
    summary = f"{total} active alert{'s' if total != 1 else ''}"
    if extreme:
        summary += f" • {len(extreme)} EXTREME"
    if severe:
        summary += f" • {len(severe)} Severe"

    cards = [build_alert_card(a) for a in alerts[:25]]  # Cap display

    return html.Div(
        className="alerts-panel",
        children=[
            html.H3("Active Alerts", className="panel-title"),
            html.P(summary, className="alerts-summary"),
            html.Div(className="alerts-list", children=cards),
        ],
    )
