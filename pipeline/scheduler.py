"""
Background scheduler for periodic data refresh.

Uses APScheduler to run data fetching pipelines at configurable intervals
without blocking the Dash UI thread.
"""

import gc
import logging
from datetime import datetime, timezone

import dask
from apscheduler.schedulers.background import BackgroundScheduler

# Dash callbacks run single-threaded; the threaded scheduler only adds
# GIL contention and cross-thread allocation churn without parallelism
# benefit on our I/O-bound paths.
dask.config.set(scheduler="synchronous")

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
        self.hrrr_data: dict = {}          # full cycle — still set after _refresh_hrrr
        self.hrrr_window: dict = {}        # active slice for rendering
        self.hrrr_window_spec: tuple | None = None  # (cycle_str, center_fhr, radius, frozenset(vars))
        self.goes_data: dict = {}
        self.nexrad_data: dict | None = None
        self.alerts: list = []
        self.observations: list = []
        self.last_updated: dict[str, datetime] = {}

    def update_hrrr(self, data: dict) -> None:
        old = self.hrrr_data
        self.hrrr_data = data
        del old
        gc.collect()
        self.last_updated["hrrr"] = datetime.now(timezone.utc)
        logger.info("HRRR data updated (%d variables)", len(data))

    def update_goes(self, channel: str, data) -> None:
        # No explicit gc.collect(): per-channel swaps are small (~14 MB);
        # refcounting handles the drop in microseconds.
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

    # --- Eviction API (called from app callbacks / Clear buttons) ---

    def evict_hrrr(self) -> None:
        if self.hrrr_data or self.hrrr_window:
            self.hrrr_data = {}
            self.hrrr_window = {}
            self.hrrr_window_spec = None
            gc.collect()
            logger.info("HRRR cache evicted")

    def evict_goes(self, channel: str | None = None) -> None:
        """Evict one channel or all. Also drops its rendered PNG."""
        from pipeline.goes import clear_png_cache

        if channel is None:
            self.goes_data.clear()
            clear_png_cache()
            gc.collect()
            logger.info("GOES cache evicted (all channels)")
        elif channel in self.goes_data:
            del self.goes_data[channel]
            clear_png_cache(channel)
            gc.collect()
            logger.info("GOES cache evicted (%s)", channel)

    def evict_nexrad(self) -> None:
        if self.nexrad_data is not None:
            self.nexrad_data = None
            gc.collect()
            logger.info("NEXRAD cache evicted")

    def ensure_nexrad(self) -> None:
        """Load latest MRMS composite synchronously if not resident."""
        if self.nexrad_data is not None:
            return
        try:
            from pipeline.nexrad import fetch_mrms_composite, radar_to_plotly_data

            raw = fetch_mrms_composite()
            if raw is not None:
                self.nexrad_data = radar_to_plotly_data(raw, subsample=6)
                self.last_updated["nexrad"] = datetime.now(timezone.utc)
                logger.info("NEXRAD loaded on demand")
        except Exception as e:
            logger.error("NEXRAD on-demand load failed: %s", e)

    def evict_alerts(self) -> None:
        self.alerts = []

    def evict_observations(self) -> None:
        self.observations = []

    # --- On-demand loaders (called by view callbacks) ---

    def ensure_hrrr_window(
        self,
        variables: list[str],
        center_fhr: int,
        radius: int = 2,
    ) -> None:
        """Load only the needed variables within ±radius fhrs of center_fhr
        from the on-disk NetCDF cache. Drops the prior window if its spec
        differs (different cycle / center / vars)."""
        from pipeline.hrrr import latest_cached_cycle, load_hrrr_window

        cycle = latest_cached_cycle()
        if cycle is None:
            return  # no disk cache yet; scheduler will fill it
        cycle_str = cycle.strftime("%Y%m%d_%H")
        spec = (cycle_str, center_fhr, radius, frozenset(variables))
        if spec == self.hrrr_window_spec and self.hrrr_window:
            return

        # Drop prior window before loading the new one to minimize peak.
        self.hrrr_window = {}
        gc.collect()
        self.hrrr_window = load_hrrr_window(
            cycle=cycle, center_fhr=center_fhr, radius=radius, variables=variables
        )
        self.hrrr_window_spec = spec

    def ensure_goes(self, channel: str) -> None:
        """Load the given channel if absent; evict siblings."""
        from pipeline.goes import clear_png_cache, fetch_goes_channel

        stale = [c for c in self.goes_data if c != channel]
        for c in stale:
            del self.goes_data[c]
            clear_png_cache(c)
        if stale:
            gc.collect()
            logger.info("GOES evicted unused channels: %s", stale)

        if channel not in self.goes_data:
            data = fetch_goes_channel(channel_name=channel)
            if data is not None:
                self.goes_data[channel] = data
                self.last_updated[f"goes_{channel}"] = datetime.now(timezone.utc)
                logger.info("GOES %s loaded on demand", channel)


# Global data store
store = DataStore()


def _refresh_hrrr():
    """Ensure the on-disk HRRR cache is up to date with the latest cycle.

    The in-memory store no longer holds the full concatenated cycle —
    the view callback loads a small fhr window from disk on demand.
    This job's job is purely to keep the per-fhr NetCDF files fresh.
    """
    try:
        from pipeline.hrrr import clean_cache, fetch_hrrr

        clean_cache(max_age_hours=6)
        fetch_hrrr()  # writes the per-fhr NetCDF files
        clean_cache(max_age_hours=6)
        store.last_updated["hrrr"] = datetime.now(timezone.utc)
        # Invalidate active window so the next view call rebuilds against
        # the new cycle.
        store.hrrr_window_spec = None
    except Exception as e:
        logger.error("HRRR refresh failed: %s", e)


def _refresh_goes():
    """Refresh only the channels currently resident in the store.

    The initial active channel is seeded lazily by app callbacks via
    store.ensure_goes(); subsequent refreshes update just those.
    """
    try:
        from pipeline.goes import fetch_goes_channel

        channels = list(store.goes_data.keys())
        if not channels:
            return  # nothing resident; the first view toggle will load on demand
        for channel in channels:
            data = fetch_goes_channel(channel_name=channel)
            if data is not None:
                store.update_goes(channel, data)
    except Exception as e:
        logger.error("GOES refresh failed: %s", e)


def _refresh_nexrad():
    """Fetch latest MRMS CONUS radar composite — skip if layer is off."""
    if store.nexrad_data is None:
        # Layer has never been activated (or was evicted); skip the 100 MB fetch.
        return
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
