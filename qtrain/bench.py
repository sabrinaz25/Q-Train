"""Latency benchmark for every capture and input backend.

Usage (PowerShell, with BlueStacks running Subway Surfers on its main menu):

    python -m qtrain.bench --list-windows
    python -m qtrain.bench --serial 127.0.0.1:5555 --window "BlueStacks App Player"

Prints a markdown table (paste it into the README) and saves the raw numbers
to results/bench-<timestamp>.json so you can track improvements over time.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from .adb import BACKENDS as INPUT_BACKENDS
from .adb import AdbDevice, list_devices, run_adb
from .capture import CAPTURE_BACKENDS, list_windows, make_capture


def summarize(seconds: list[float]) -> dict:
    ms = sorted(s * 1000 for s in seconds)
    p95 = ms[min(len(ms) - 1, int(round(0.95 * (len(ms) - 1))))]
    mean = statistics.fmean(ms)
    return {
        "n": len(ms),
        "p50_ms": round(statistics.median(ms), 2),
        "p95_ms": round(p95, 2),
        "mean_ms": round(mean, 2),
        "per_sec": round(1000 / mean, 1) if mean > 0 else float("inf"),
    }


def bench_capture(backend: str, args) -> dict:
    cap = make_capture(backend, window=args.window, serial=args.serial)
    try:
        frame = cap.grab()  # warm-up; also checks it works at all
        times = []
        for _ in range(args.n):
            t0 = time.perf_counter()
            frame = cap.grab()
            times.append(time.perf_counter() - t0)
        res = summarize(times) | {"frame_shape": list(frame.shape)}
        if hasattr(cap, "fresh"):
            res["fresh_frames"] = cap.fresh
        if args.save_frames:
            _save_frame(frame, Path("data") / f"sample-{backend}.png")
        return res
    finally:
        cap.close()


def supports_cmd_input(serial: str) -> bool:
    try:
        return "input" in run_adb("shell", "cmd", "-l", serial=serial).split()
    except RuntimeError:
        return False


def bench_input(backend: str, args) -> dict:
    if backend == "cmd" and not supports_cmd_input(args.serial):
        return {"skipped": "`cmd input` unavailable (needs Android 10+)"}
    moves = ["left", "right"]
    with AdbDevice(args.serial, backend=backend) as dev:
        for i in range(3):
            dev.act(moves[i % 2], wait=True)
        n = max(10, args.n // 4)  # inputs are slow; fewer samples keep it short
        rt = []
        for i in range(n):
            t0 = time.perf_counter()
            dev.act(moves[i % 2], wait=True)
            rt.append(time.perf_counter() - t0)
        res = {"round_trip": summarize(rt)}
        if backend != "spawn":
            host = [dev.act(moves[i % 2], wait=False) for i in range(n)]
            res["host_only"] = summarize(host)
            time.sleep(1.0)  # let queued swipes finish before closing
        return res


def _save_frame(frame, path: Path) -> None:
    try:
        import cv2
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.ascontiguousarray(frame))


def to_markdown(results: dict) -> str:
    lines = ["**Frame capture** (ms per frame)", "",
             "| backend | p50 | p95 | frames/s | frame |", "|---|---|---|---|---|"]
    for name, r in results["capture"].items():
        if "error" in r:
            lines.append(f"| {name} | – | – | – | {r['error']} |")
        else:
            shape = "x".join(map(str, r["frame_shape"][:2]))
            lines.append(f"| {name} | {r['p50_ms']} | {r['p95_ms']} | {r['per_sec']} | {shape} |")
    lines += ["", "**Input** (ms per swipe)", "",
              "| backend | round-trip p50 | round-trip p95 | host-side p50 |", "|---|---|---|---|"]
    for name, r in results["input"].items():
        if "round_trip" not in r:
            lines.append(f"| {name} | – | – | {r.get('skipped') or r.get('error')} |")
        else:
            rt, host = r["round_trip"], r.get("host_only")
            lines.append(f"| {name} | {rt['p50_ms']} | {rt['p95_ms']} | "
                         f"{host['p50_ms'] if host else rt['p50_ms']} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Benchmark capture and input latency.")
    p.add_argument("--serial", default="127.0.0.1:5555", help="ADB serial (BlueStacks default 127.0.0.1:5555)")
    p.add_argument("--window", default="BlueStacks", help="substring of the emulator window title")
    p.add_argument("--capture", nargs="*", default=list(CAPTURE_BACKENDS), choices=CAPTURE_BACKENDS)
    p.add_argument("--input", nargs="*", default=list(INPUT_BACKENDS), choices=INPUT_BACKENDS)
    p.add_argument("-n", type=int, default=100, help="samples per capture backend")
    p.add_argument("--save-frames", action="store_true", help="write one sample frame per backend to data/")
    p.add_argument("--list-windows", action="store_true")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args(argv)

    if args.list_windows:
        for hwnd, title in list_windows():
            print(f"{hwnd:>10}  {title}")
        return
    if args.list_devices:
        print("\n".join(list_devices()) or "(no devices; is ADB enabled in BlueStacks settings?)")
        return

    results = {"capture": {}, "input": {},
               "machine": {"platform": platform.platform(), "processor": platform.processor()}}
    for b in args.capture:
        print(f"capture: {b} ...", flush=True)
        try:
            results["capture"][b] = bench_capture(b, args)
        except Exception as e:  # report and keep going; one broken backend shouldn't hide the rest
            results["capture"][b] = {"error": f"{type(e).__name__}: {e}"}
    for b in args.input:
        print(f"input: {b} ...", flush=True)
        try:
            results["input"][b] = bench_input(b, args)
        except Exception as e:
            results["input"][b] = {"error": f"{type(e).__name__}: {e}"}

    out = Path("results") / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("\n" + to_markdown(results))
    print(f"\nraw numbers: {out}")


if __name__ == "__main__":
    main()
