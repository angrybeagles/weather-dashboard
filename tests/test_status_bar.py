"""Tests for the status bar — GOES freshness should track any channel."""

from datetime import datetime, timedelta, timezone

from components.controls import build_status_bar


def _flatten_text(node) -> str:
    """Recursively pull all string content out of a Dash component tree."""
    if isinstance(node, str):
        return node
    children = getattr(node, "children", None)
    if children is None:
        return ""
    if isinstance(children, (list, tuple)):
        return " ".join(_flatten_text(c) for c in children)
    return _flatten_text(children)


def test_status_bar_lists_all_data_sources():
    bar = build_status_bar({})
    text = _flatten_text(bar)
    for label in ("HRRR", "GOES", "NEXRAD", "Alerts", "Obs"):
        assert label in text, f"{label} missing from status bar"


def test_status_bar_picks_up_goes_visible_when_ir_missing():
    """Regression: status bar used to hardcode 'goes_ir'. Now it should accept
    any goes_<channel> key (e.g. visible, water_vapor)."""
    now = datetime.now(timezone.utc)
    bar = build_status_bar({"goes_visible": now - timedelta(minutes=2)})
    text = _flatten_text(bar)
    # status spans get class names like 'status-fresh' — we surface that via children
    # easier check: the rendered time string is present, not the dash placeholder
    assert "GOES" in text
    # if goes still showed missing, GOES segment would read 'GOES: —'
    assert "GOES: —" not in text


def test_status_bar_marks_missing_when_no_timestamps():
    bar = build_status_bar({})
    text = _flatten_text(bar)
    assert "GOES: —" in text
    assert "HRRR: —" in text
