"""End-to-end input latency: from "touch sent" to "touch visible on screen".

Round-trip timings only show how long the host waited, and they can be fooled:
a command that fails instantly looks very fast (that's exactly what happened
with `cmd input`). This test can't be fooled. It turns on Android's
"pointer location" developer overlay, presses at a known spot, and watches the
captured frames until the overlay reacts. The overlay's crosshair is only 1
device pixel wide (too faint once the window is captured and scaled), so we
watch its info bar across the top of the screen instead: it prints the
touch's coordinates, and since presses alternate between two spots, its
digits change on every press.

Run it on a static screen (the pause screen is ideal) so nothing else moves.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .adb import run_adb
from .vision import CANON_W, GameArea, canonicalize


def column_change(before: np.ndarray, after: np.ndarray, x: int, band: int = 3,
                  min_diff: int = 60) -> float:
    """Share of rows whose pixels in a thin vertical strip at column x changed a lot.

    The overlay's crosshair is a line down the whole screen, so most rows of
    the strip change at once; ordinary animation only touches a few rows.
    """
    x0, x1 = max(x - band, 0), min(x + band + 1, before.shape[1])
    a = before[:, x0:x1].astype(np.int16)
    b = after[:, x0:x1].astype(np.int16)
    per_row = np.abs(a - b).reshape(a.shape[0], -1).max(axis=1)
    return float((per_row > min_diff).mean())


def bar_change(before: np.ndarray, after: np.ndarray, rows: int = 10, min_diff: int = 40) -> int:
    """Number of pixels that changed clearly in the overlay's info bar (top `rows` rows)."""
    a = before[:rows].astype(np.int16)
    b = after[:rows].astype(np.int16)
    return int((np.abs(a - b).max(axis=2) > min_diff).sum())


def set_pointer_location(serial: str, on: bool) -> None:
    run_adb("shell", "settings", "put", "system", "pointer_location", "1" if on else "0", serial=serial)


def measure(device, capture, area: GameArea | None, serial: str, trials: int = 12,
            hold: float = 0.4, timeout: float = 3.0, threshold: int = 20,
            debug_dir: str | None = "data/debug") -> list[float | None]:
    """Seconds from sending each press until it's visible; None if never seen."""
    results: list[float | None] = []
    set_pointer_location(serial, True)
    time.sleep(1.0)
    try:
        for i in range(trials):
            fx = 0.3 if i % 2 == 0 else 0.7            # alternate columns so every press is a new change
            fy = 0.5
            cx = int(fx * CANON_W)
            dx, dy = int(fx * device.width), int(fy * device.height)
            time.sleep(0.6)                             # let the previous press fade
            baseline = canonicalize(capture.grab(), area).copy()

            t0 = time.perf_counter()
            worker = threading.Thread(target=device.press, args=(dx, dy, hold), daemon=True)
            worker.start()
            seen, best, best_frame, frames = None, 0, baseline, 0
            while time.perf_counter() - t0 < timeout:
                frame = canonicalize(capture.grab(), area)
                frames += 1
                score = bar_change(baseline, frame)
                if score > best:
                    best, best_frame = score, frame.copy()
                if score >= threshold:
                    seen = time.perf_counter() - t0
                    break
            worker.join(timeout=timeout + 5)
            results.append(seen)
            if seen:
                print(f"  press {i + 1}/{trials}: {seen * 1000:.0f} ms", flush=True)
            else:
                print(f"  press {i + 1}/{trials}: not seen (best: {best} changed pixels in the overlay "
                      f"bar, need {threshold}; {frames} frames checked)", flush=True)
                if debug_dir and not any(r is None for r in results[:-1]):  # save the first miss only
                    d = Path(debug_dir)
                    d.mkdir(parents=True, exist_ok=True)
                    tag = getattr(device, "backend", "input")
                    marked = best_frame.copy()
                    cv2.line(marked, (cx, 0), (cx, marked.shape[0] - 1), (0, 255, 255), 1)
                    cv2.imwrite(str(d / f"e2e-{tag}-baseline.png"), baseline)
                    cv2.imwrite(str(d / f"e2e-{tag}-best.png"), marked)
    finally:
        set_pointer_location(serial, False)
    return results

