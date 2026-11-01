"""Measuring how much of an app a driver reaches, identically for every driver.

The observer runs on its own thread, independent of whatever is driving the
app (the explorer, Android's monkey, a person), and samples:

  - the screen, every 1/sample_hz s: a fingerprint counts as a screen once it
    has stayed up for two samples in a row, so frames caught mid-transition or
    mid-animation never count;
  - the foreground activity, every `activity_every` s: unique activities is the
    coarse coverage measure Android testing papers report;
  - crashes and ANRs of the package, from the device log (CrashWatcher).

Frames are only counted while the app under test is in front, so the launcher,
crash dialogs and other apps add nothing.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .fingerprint import Crop, ScreenSet, distance, fingerprint


@dataclass
class CoverageObserver:
    capture: object               # .grab() -> BGR frame of the device screen
    app: object                   # .package, .current_activity() -> "pkg/Activity" or None
    sample_hz: float = 2.0
    activity_every: float = 1.0
    crop: Crop = field(default_factory=Crop)
    clock: object = time.perf_counter
    crashes: object | None = None  # CrashWatcher (or anything with .events)
    screens: ScreenSet = field(default_factory=ScreenSet)
    activities: dict[str, int] = field(default_factory=dict)
    timeline: list[dict] = field(default_factory=list)
    in_app: bool = True
    samples: int = 0
    out_of_app_samples: int = 0

    def __post_init__(self):
        self._pending: np.ndarray | None = None
        self._t0 = self.clock()
        self._next_activity = self._t0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------- sampling
    def sample(self) -> None:
        now = self.clock()
        if now >= self._next_activity:
            self._next_activity = now + self.activity_every
            act = self.app.current_activity()
            self.in_app = bool(act) and act.split("/")[0] == self.app.package
            if self.in_app:
                self.activities[act] = self.activities.get(act, 0) + 1
        self.samples += 1
        if not self.in_app:
            self.out_of_app_samples += 1
            self._pending = None
        else:
            fp = fingerprint(self.capture.grab(), self.crop)
            if self._pending is not None and distance(fp, self._pending) <= self.screens.threshold:
                self.screens.add(fp)          # held still for a full sample interval
            self._pending = fp
        if not self.timeline or now - self.timeline[-1]["t"] >= 10:
            self.timeline.append(self.point(now))

    def point(self, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        return {"t": round(now - self._t0, 1), "screens": len(self.screens),
                "activities": len(self.activities), "crashes": len(self.crash_events())}

    def crash_events(self) -> list:
        return list(self.crashes.events) if self.crashes is not None else []

    # ---------------------------------------------------------- background
    def start(self) -> CoverageObserver:
        self._t0 = self._next_activity = self.clock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        period = 1.0 / self.sample_hz
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                self.sample()
            except Exception as e:  # noqa: BLE001 - a bad sample must not end the measurement
                self.timeline.append({"t": round(self.clock() - self._t0, 1), "error": str(e)[:200]})
            self._stop.wait(max(period - (time.perf_counter() - t), 0.0))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    # -------------------------------------------------------------- result
    def summary(self) -> dict:
        wall = self.clock() - self._t0
        hours = wall / 3600 if wall else 0
        crashes = [e for e in self.crash_events() if e[1] == "crash"]
        anrs = [e for e in self.crash_events() if e[1] == "anr"]
        self.timeline.append(self.point())
        return {
            "wall_minutes": round(wall / 60, 2),
            "unique_screens": len(self.screens),
            "unique_activities": len(self.activities),
            "screens_per_hour": round(len(self.screens) / hours, 1) if hours else None,
            "crashes": len(crashes),
            "unique_crashes": len({sig for _, _, sig in crashes}),
            "anrs": len(anrs),
            "time_in_app_pct": round(100 * (1 - self.out_of_app_samples / self.samples), 1)
            if self.samples else None,
            "activities": sorted(self.activities),
            "crash_signatures": sorted({sig for _, _, sig in crashes}),
            "timeline": self.timeline,
        }
