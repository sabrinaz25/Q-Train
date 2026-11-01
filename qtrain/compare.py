"""Explorer vs. Android's monkey on the same app, measured the same way.

    python -m qtrain.compare --serial 127.0.0.1:5555 --list-packages
    python -m qtrain.compare --serial 127.0.0.1:5555 --package org.example.app --minutes 10

Each driver gets the same budget and the same start (app force-stopped, then
launched), and the same CoverageObserver watches both: unique screens (pixel
fingerprints that held still for one sample), unique activities (from Android),
and crashes/ANRs (from the device's event log). Results go to results/.

monkey runs with its default event mix; --ignore-* flags keep it going after a
crash so it uses its whole budget, like the explorer. It sends system events
(volume, notification shade, ...), so use an emulator instance you don't mind
being poked at.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

from .adb import adb_path, run_adb

DRIVERS = ("explorer", "monkey")


def third_party_packages(serial: str) -> list[str]:
    out = run_adb("shell", "pm", "list", "packages", "-3", serial=serial)
    return sorted(line.split(":", 1)[1].strip() for line in out.splitlines() if line.startswith("package:"))


def monkey_command(serial: str, package: str, seed: int, throttle_ms: int) -> list[str]:
    return [adb_path(), "-s", serial, "shell", "monkey", "-p", package, "-s", str(seed),
            "--throttle", str(throttle_ms), "--ignore-crashes", "--ignore-timeouts",
            "--ignore-security-exceptions", "-v", "100000000"]


def stop_monkey(serial: str) -> None:
    for cmd in ("pkill -f com.android.commands.monkey",
                "kill $(pidof com.android.commands.monkey) 2>/dev/null"):
        try:
            run_adb("shell", cmd, serial=serial, timeout=10)
        except Exception:  # noqa: BLE001 - try the next way
            pass


def run_monkey(serial: str, package: str, seconds: float, seed: int, throttle_ms: int) -> dict:
    proc = subprocess.Popen(monkey_command(serial, package, seed, throttle_ms), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")
    counts = {"events": 0}

    def read():
        for line in proc.stdout:
            if line.startswith(":Sending"):
                counts["events"] += 1

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    time.sleep(seconds)
    stop_monkey(serial)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    reader.join(timeout=5)
    return {"events_sent": counts["events"], "throttle_ms": throttle_ms}


def comparison_table(results: dict[str, dict]) -> str:
    rows = [("Unique screens", "unique_screens"), ("Screens per hour", "screens_per_hour"),
            ("Unique activities", "unique_activities"), ("Crashes (unique)", None),
            ("ANRs", "anrs"), ("Time in app (%)", "time_in_app_pct")]
    names = list(results)
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, key in rows:
        cells = []
        for n in names:
            c = results[n]["coverage"]
            cells.append(f"{c['crashes']} ({c['unique_crashes']})" if key is None else str(c[key]))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Compare the pixel explorer with Android's monkey on one app.")
    p.add_argument("--serial", default="127.0.0.1:5555")
    p.add_argument("--package", help="app under test, e.g. org.example.app")
    p.add_argument("--list-packages", action="store_true", help="print installed third-party apps and exit")
    p.add_argument("--driver", default="both", choices=[*DRIVERS, "both"])
    p.add_argument("--minutes", type=float, default=10.0, help="budget per driver")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--monkey-throttle", type=int, default=200, help="ms between monkey events")
    p.add_argument("--max-size", type=int, default=480, help="video size for frames and touches")
    p.add_argument("--keep-data", action="store_true",
                   help="don't wipe the app's data before each driver (default: wipe, so both start fresh)")
    p.add_argument("--targets", default="hybrid", choices=["hybrid", "a11y", "pixels"],
                   help="explorer: where tappable elements come from")
    args = p.parse_args(argv)

    if args.list_packages:
        print("\n".join(third_party_packages(args.serial)))
        return
    if not args.package:
        p.error("--package is required (see --list-packages)")

    from .android import AndroidApp, CrashWatcher
    from .coverage import CoverageObserver
    from .explorer import Explorer, ExplorerConfig
    from .scrcpy_client import ScrcpyCapture, ScrcpyDevice, ScrcpySession

    app = AndroidApp(args.serial, args.package)
    drivers = DRIVERS if args.driver == "both" else (args.driver,)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    Path("results").mkdir(exist_ok=True)
    results: dict[str, dict] = {}
    for name in drivers:
        print(f"\n== {name}: {args.minutes} min on {args.package}")
        app.force_stop()
        if not args.keep_data:
            app.clear_data()      # same fresh-install state for every driver
        granted = app.grant_permissions()   # skip the system permission dialogs (see android.py)
        print(f"granted: {', '.join(g.rsplit('.', 1)[-1] for g in granted) or 'nothing'}")
        app.launch()
        time.sleep(3)
        session = ScrcpySession(args.serial, decode=True, max_size=args.max_size, max_fps=30,
                                bit_rate=4_000_000)
        capture = ScrcpyCapture(session=session)
        crashes = CrashWatcher(args.serial, args.package).start(time.perf_counter)
        observer = CoverageObserver(capture, app, crashes=crashes).start()
        try:
            seconds = args.minutes * 60
            if name == "explorer":
                device = ScrcpyDevice(session=session)
                cfg = ExplorerConfig(seed=args.seed, targets=args.targets)
                driver_stats = Explorer(device, capture, app, cfg).run(seconds)
                device.close()
            else:
                driver_stats = run_monkey(args.serial, args.package, seconds, args.seed, args.monkey_throttle)
        finally:
            observer.stop()
            crashes.stop()
            session.close()
        cov = observer.summary()
        results[name] = {"driver": driver_stats, "coverage": cov}
        print(f"{name}: {cov['unique_screens']} screens, {cov['unique_activities']} activities, "
              f"{cov['crashes']} crashes ({cov['unique_crashes']} unique), {cov['anrs']} ANRs, "
              f"{cov['time_in_app_pct']}% of samples in the app")
        if cov.get("outside_packages"):
            print(f"  in front instead of the app: {cov['outside_packages']}")

    out = Path("results") / f"compare-{args.package}-{stamp}.json"
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print("\n" + comparison_table(results))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
