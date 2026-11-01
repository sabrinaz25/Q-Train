from qtrain.bench import summarize, to_markdown


def test_summarize():
    s = summarize([0.010] * 19 + [0.100])
    assert s["n"] == 20
    assert s["p50_ms"] == 10.0
    assert s["p95_ms"] == 10.0  # one outlier in 20 sits above the 95th percentile
    assert s["per_sec"] > 0


def test_markdown_handles_errors_and_skips():
    md = to_markdown({
        "capture": {"mss": {"p50_ms": 5, "p95_ms": 7, "per_sec": 180, "frame_shape": [960, 540, 3]},
                    "dxcam": {"error": "ImportError: no dxcam"}},
        "input": {"spawn": {"round_trip": {"p50_ms": 150, "p95_ms": 220}},
                  "cmd": {"skipped": "needs Android 10+"}},
    })
    assert "960x540" in md and "ImportError" in md and "needs Android 10+" in md


def test_combine_runs_takes_medians_and_keeps_ranges():
    from qtrain.bench import combine_runs
    runs = [{"p50_ms": v, "frame_shape": [2, 2, 3]} for v in (10.0, 30.0, 12.0)] + [{"error": "boom"}]
    c = combine_runs(runs)
    assert c["p50_ms"] == 12.0 and c["p50_ms_range"] == [10.0, 30.0] and c["runs"] == 3
    assert combine_runs([{"error": "boom"}]) == {"error": "boom"}
