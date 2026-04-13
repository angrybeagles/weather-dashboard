"""Verify the background scheduler kicks off heavy fetchers immediately."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from pipeline import scheduler as sched_mod


def test_all_jobs_have_immediate_next_run_time():
    """HRRR + GOES used to wait a full interval before first run, leaving the
    map empty at startup. Every job should have next_run_time set to now-ish."""
    with patch.object(sched_mod, "BackgroundScheduler") as MockSched:
        instance = MockSched.return_value
        sched_mod.start_scheduler()

    job_kwargs = {
        call.kwargs["id"]: call.kwargs for call in instance.add_job.call_args_list
    }

    expected_ids = {"hrrr", "goes", "nexrad", "alerts", "observations"}
    assert expected_ids.issubset(job_kwargs.keys()), (
        f"Missing jobs: {expected_ids - set(job_kwargs)}"
    )

    now = datetime.now(timezone.utc)
    for job_id in expected_ids:
        next_run = job_kwargs[job_id].get("next_run_time")
        assert next_run is not None, f"{job_id} has no next_run_time"
        # should be effectively immediate (within a few seconds)
        assert abs((now - next_run).total_seconds()) < 30, (
            f"{job_id} next_run_time too far from now: {next_run}"
        )
