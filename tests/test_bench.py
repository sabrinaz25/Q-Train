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
