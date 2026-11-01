from qtrain.run import summarize


def test_summary_reports_reliability():
    rows = [{"episode": 1, "steps": 40, "play_seconds": 3.0, "achieved_hz": 15.0, "reset_seconds": 8.0},
            {"episode": 2, "steps": 30, "play_seconds": 2.0, "achieved_hz": 15.0, "reset_seconds": 12.0},
            {"episode": 3, "error": "TimeoutError: could not get into a run", "seconds": 90.0},
            {"episode": 4, "steps": 50, "play_seconds": 3.5, "achieved_hz": 14.0, "reset_seconds": 7.0}]
    s = summarize(rows, {"tap:red_x": 2, "app_restarts": 1, "episodes": 3}, wall_seconds=360)
    assert s["episodes_attempted"] == 4 and s["episodes_completed"] == 3 and s["failures"] == 1
    assert s["success_rate_pct"] == 75.0
    assert s["reset_seconds_p50"] == 8.0 and s["reset_seconds_max"] == 12.0
    assert s["recoveries"]["tap:red_x"] == 2 and s["recoveries_per_hour"] == 30.0
    assert s["steps_per_hour"] == 1200 and s["failure_reasons"][0].startswith("TimeoutError")
