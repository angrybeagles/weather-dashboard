"""
Background scheduler for periodic data refresh.

Uses APScheduler to run data fetching pipelines at configurable intervals
without blocking the Dash UI thread.
"""

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from config import (
    ALERTS_REFRESH_INTERVAL,
    GOES_REFRESH_INTERVAL,
    HRRR_REFRESH_INTERVAL,
    OBS_REFRESH_INTERVAL,
)

logger = logging.getLogger(__name__)


class DataStore:
    """
    Thread-safe-ish in-memory store for the latest fetched data.
    Dash callbacks read from here; the scheduler writes to here.
    """

    def __init__(self):
        self.hrrr_data: dict = {}
        self.goes_data: dict = {}
        self.nexrad_data: dict | None = None
        self.alerts: list = []
        self.observations: list = []
        self.last_updated: dict[str, datetime] = {}

    def update_hrrr(self, data: dict) -> None:
        self.hrrr_data = data
        self.last_updated["hrrr"] = datetime.now(timezone.utc)
        logger.info("HRRR data updated (%d variables)", len(data))

    def update_goes(self, channel: str, data) -> None:
        self.goes_data[channel] = data
        self.last_updated[f"goes_{channel}"] = datetime.now(timezone.utc)
        logger.info("GOES %s updated", channel)

    def update_nexrad(self, data) -> None:
        self.nexrad_data = data
        self.last_updated["nexrad"] = datetime.now(timezone.utc)
        logger.info("NEXRAD radar updated")

    def update_alerts(self, alerts: list) -> None:
        self.alerts = alerts
        self.last_updated["alerts"] = datetime.now(timezone.utc)
        logger.info("Alerts updated (%d active)", len(alerts))

    def update_observations(self, obs: list) -> None:
        self.observations = obs
        self.last_updated["observations"] = datetime.now(timezone.utc)
        logger.info("Observations updated (%d stations)", len(obs))


# Global data store
store = DataStore()


def _refresh_hrrr():
    """Fetch latest HRRR cycle."""
    try:
        from pipeline.hrrr import fetch_hrrr, clean_cache

        data = fetch_hrrr()
        store.update_hrrr(data)
        clean_cache(max_age_hours=24)
    except Exception as e:
        logger.error("HRRR refresh failed: %s", e)


def _refresh_goes():
    """Fetch latest GOES imagery for key channels."""
    try:
        from pipeline.goes import fetch_goes_channel

        for channel in ["ir", "visible", "water_vapor"]:
            data = fetch_goes_channel(channel_name=channel)
            if data is not None:
                store.update_goes(channel, data)
    except Exception as e:
        logger.error("GOES refresh failed: %s", e)


def _refresh_nexrad():
    """Fetch latest MRMS CONUS radar composite."""
    try:
        from pipeline.nexrad import fetch_mrms_composite, radar_to_plotly_data

        raw = fetch_mrms_composite()
        if raw is not None:
            plotly_data = radar_to_plotly_data(raw, subsample=6)
            store.update_nexrad(plotly_data)
    except Exception as e:
        logger.error("NEXRAD refresh failed: %s", e)


def _refresh_alerts():
    """Fetch active NWS alerts."""
    try:
        from pipeline.nws import fetch_active_alerts

        alerts = fetch_active_alerts()
        store.update_alerts(alerts)
    except Exception as e:
        logger.error("Alerts refresh failed: %s", e)


def _refresh_observations():
    """Fetch surface observations for major CONUS stations."""
    try:
        from pipeline.nws import fetch_observations_bulk

        # Key stations across CONUS for a good coverage map
        major_stations = [
            "KSEA", "KPDX", "KSFO", "KLAX", "KLAS", "KPHX", "KDEN",
            "KSLC", "KABQ", "KDFW", "KIAH", "KSAT", "KORD", "KDTW",
            "KMSP", "KSTL", "KATL", "KMIA", "KTPA", "KCLT", "KBOS",
            "KJFK", "KPHL", "KIAD", "KBWI", "KRDU", "KBNA", "KMEM",
            "KMCI", "KOKC", "KOMA", "KBOI", "KGEG", "KFAT", "KSMF",
            "KAUS", "KMSN", "KCLE", "KPIT", "KCMH", "KIND", "KMKE",
            "KJAN", "KBHM", "KLIT", "KTUL", "KICT", "KFSD", "KBIL",
            "KMSO", "KRAP", "KFAR",
        ]
        obs = fetch_observations_bulk(major_stations)
        store.update_observations(obs)
    except Exception as e:
        logger.error("Observations refresh failed: %s", e)


def start_scheduler() -> BackgroundScheduler:
    """
    Start the background data refresh scheduler.

    On startup, does an immediate fetch of alerts and observations (fast).
    HRRR and GOES are heavier and run on their intervals.
    """
    scheduler = BackgroundScheduler(daemon=True)

    # Alerts — fast, refresh often
    scheduler.add_job(
        _refresh_alerts,
        "interval",
        seconds=ALERTS_REFRESH_INTERVAL,
        id="alerts",
        next_run_time=datetime.now(timezone.utc),  # run immediately
    )

    # Observations — moderate frequency
    scheduler.add_job(
        _refresh_observations,
        "interval",
        seconds=OBS_REFRESH_INTERVAL,
        id="observations",
        next_run_time=datetime.now(timezone.utc),
    )

    # GOES satellite — every 5 min
    scheduler.add_job(
        _refresh_goes,
        "interval",
        seconds=GOES_REFRESH_INTERVAL,
        id="goes",
        next_run_time=datetime.now(timezone.utc),  # run immediately
    )

    # NEXRAD radar — every 2 min (MRMS updates every ~2 min)
    scheduler.add_job(
        _refresh_nexrad,
        "interval",
        seconds=ALERTS_REFRESH_INTERVAL,  # reuse 2-min interval
        id="nexrad",
        next_run_time=datetime.now(timezone.utc),  # run immediately
    )

    # HRRR model — every hour
    scheduler.add_job(
        _refresh_hrrr,
        "interval",
        seconds=HRRR_REFRESH_INTERVAL,
        id="hrrr",
        next_run_time=datetime.now(timezone.utc),  # run immediately
    )

    scheduler.start()
    logger.info("Background scheduler started")
    return scheduler
