"""The game loop: turns "screen + swipes" into reset()/step() for an RL agent.

reset()  navigates whatever is on screen (menus, crash screens, popups,
         reward notifications) until a run is in progress, then returns the
         first observation.
step(a)  sends one swipe, waits for the next control tick (fixed rate, so the
         agent sees evenly spaced frames), captures, and decides whether the
         run is still going.

Kept free of Gymnasium and of real devices: anything with the same methods
can stand in for the emulator, which is how the tests drive it.

Known limitation: the game shows its "Save me?" prompt roughly a second after
the actual collision, so crash detection (and the penalty) lags the mistake.
Stage 5 can tighten this by also watching for the character/HUD freezing.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .screens import Calibration, Detection, ScreenClassifier
from .vision import CANON_H, CANON_W, canonicalize, find_red_xs, locate_game, to_obs

ACTIONS = ("noop", "left", "right", "up", "down")   # up = jump, down = roll


class Device(Protocol):
    width: int
    height: int

    def act(self, action: str) -> object: ...
    def tap(self, x: int, y: int) -> None: ...
    def back(self) -> None: ...


class FrameSource(Protocol):
    def grab(self) -> np.ndarray: ...


@dataclass
class ControllerConfig:
    hz: float = 15.0                 # agent decisions per second
    frame_stack: int = 4
    obs_size: int = 84
    alive_reward: float = 0.1
    crash_penalty: float = -1.0
    lost_after: float = 6.0          # s of unrecognised screens mid-run before giving up on the episode
                                     # (crash animations hide the HUD for up to ~3 s before "Continue?")
    max_steps: int | None = None
    reset_timeout: float = 90.0
    settle: float = 0.8              # s to wait after tapping through a screen
    poll: float = 0.25               # s between checks while waiting
    back_after_unknown: int = 12     # press Android Back after this many unknown screens in a row
    restart_after: float = 45.0      # s into a reset that hasn't reached a run: restart the app (once)
    stuck_taps: int = 3              # same screen after this many taps...
    stuck_seconds: float = 4.0       # ...and this long: the tap isn't working (not just a slow screen)
    debug_dir: str = "data/debug"


class GameController:
    def __init__(self, device: Device, capture: FrameSource, calibration: Calibration,
                 config: ControllerConfig | None = None, clock=time.perf_counter, sleep=time.sleep,
                 restart_app=None):
        self.dev = device
        self.restart_app = restart_app   # callable that relaunches the app, or None
        self.cap = capture
        self.area = calibration.game_area
        self.classifier = ScreenClassifier(calibration.templates)
        self.cfg = config or ControllerConfig()
        self.clock, self.sleep = clock, sleep
        self.frames: deque[np.ndarray] = deque(maxlen=self.cfg.frame_stack)
        self.stats: Counter = Counter()
        self.last_canon: np.ndarray | None = None
        self.last_detection: Detection | None = None
        self._steps = 0
        self._tick = 0.0
        self._lost_since: float | None = None
        self.crash_delays: list[float] = []   # s from the HUD vanishing to the crash screen
        self.relocate()

    # ------------------------------------------------------------- helpers
    def relocate(self) -> bool:
        """Re-find the game inside the window (e.g. after an ad panel shifts the layout).

        Returns True if the area changed noticeably. Frame sources that deliver the
        device's own screen (scrcpy capture) have nothing to locate.
        """
        if getattr(self.cap, "full_screen", False):
            self.area = None
            return False
        w, h = getattr(self.dev, "width", 0), getattr(self.dev, "height", 0)
        aspect = (w / h, h / w) if w and h else None   # device shape first, then rotated
        found = locate_game(self.cap.grab(), **({"aspect": aspect} if aspect else {}))
        if found is None:
            self.stats["relocate_failed"] += 1
            return False
        area = found[0]
        old = self.area
        moved = old is None or max(abs(area.x - old.x), abs(area.y - old.y),
                                   abs(area.w - old.w), abs(area.h - old.h)) > 0.01
        self.area = area
        if moved:
            self.stats["relocated"] += 1
        return moved

    def to_device(self, point: tuple[int, int]) -> tuple[int, int]:
        x, y = point
        return int(x / CANON_W * self.dev.width), int(y / CANON_H * self.dev.height)

    def look(self) -> Detection:
        self.last_canon = canonicalize(self.cap.grab(), self.area)
        self.last_detection = self.classifier.classify(self.last_canon)
        return self.last_detection

    def _obs(self) -> np.ndarray:
        return np.stack(self.frames)

    def _save_debug(self, tag: str) -> Path | None:
        if self.last_canon is None:
            return None
        path = Path(self.cfg.debug_dir) / f"{tag}-{time.strftime('%Y%m%d-%H%M%S')}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.last_canon)
        return path

    def _act_on(self, det: Detection) -> None:
        """Do what a recognized screen asks: press Back or tap its point."""
        if det.key == "back":
            self.dev.back()
        elif det.tap is not None:
            self.dev.tap(*self.to_device(det.tap))

    def _unstick(self, screen: str) -> None:
        """Taps on `screen` stopped working: close a popup if one is visible, else press Back."""
        self.stats["stuck"] += 1
        if self.stats["stuck"] <= 5:
            self._save_debug(f"stuck-{screen}")
        xs = find_red_xs(self.last_canon) if self.last_canon is not None else []
        if xs:
            self.dev.tap(*self.to_device(xs[0].center))
            self.stats["stuck:red_x"] += 1
        else:
            self.dev.back()
            self.stats["stuck:back"] += 1

    # --------------------------------------------------------------- reset
    def reset(self) -> tuple[np.ndarray, dict]:
        """Get from whatever is on screen into a run.

        Recovery escalates: wait through transitions -> tap through known screens and
        popups -> re-find the game (layout changed) -> Android Back -> restart the app
        (once, after `restart_after` s) -> give up with a saved frame.
        """
        cfg = self.cfg
        t0 = self.clock()
        unknown = 0
        restarted = False
        last_tapped, same_taps, unsticks, first_tap_at = None, 0, 0, 0.0
        path: list[str] = []
        while True:
            # restart after `restart_after` s, or sooner once cheaper unsticking has failed twice
            if (not restarted and self.restart_app is not None
                    and (self.clock() - t0 > cfg.restart_after or unsticks >= 2)):
                self._save_debug("before-restart")
                self.restart_app()
                self.stats["app_restarts"] += 1
                restarted = True
                path.append("RESTART")
                self.sleep(cfg.settle * 5)          # app launch takes a few seconds
                continue
            if self.clock() - t0 > cfg.reset_timeout:
                dbg = self._save_debug("reset-timeout")
                raise TimeoutError(f"could not get into a run within {cfg.reset_timeout:.0f}s "
                                   f"(screens seen: {' > '.join(path[-12:])}); last frame saved to {dbg}")
            det = self.look()
            if not path or path[-1] != det.name:
                path.append(det.name)
            if det.role == "playing":
                break
            if det.role == "unknown":
                unknown += 1
                self.stats["unknown"] += 1
                if unknown % cfg.back_after_unknown == 0:
                    # a layout change can make every screen look unknown: re-find the game first
                    if not self.relocate():
                        self.dev.back()
                        self.stats["back"] += 1
                self.sleep(cfg.poll)
                continue
            unknown = 0
            if det.key == "back":
                self.dev.back()
                self.stats[f"back:{det.name}"] += 1
                self.sleep(cfg.settle)
            elif det.tap is not None:
                if det.name != last_tapped:
                    last_tapped, same_taps, first_tap_at = det.name, 0, self.clock()
                same_taps += 1
                # Both conditions: a slow screen also survives a few taps, and recovering
                # from it is harmful (Back during a run that did start pauses the game).
                if same_taps > cfg.stuck_taps and self.clock() - first_tap_at >= cfg.stuck_seconds:
                    # Still on the same screen after several taps: something is covering the
                    # button (e.g. a popup over a screen whose template still matches).
                    self._unstick(det.name)
                    unsticks += 1
                    last_tapped = None
                    self.sleep(cfg.settle)
                    continue
                self.dev.tap(*self.to_device(det.tap))
                self.stats[f"tap:{det.name}"] += 1
                self.sleep(cfg.settle)
            else:                       # recognised screen with nothing to tap: let it time out
                self.sleep(cfg.poll)

        frame = to_obs(self.last_canon, cfg.obs_size)
        self.frames.clear()
        self.frames.extend([frame] * cfg.frame_stack)
        self._steps = 0
        self._lost_since = None
        self._tick = self.clock()
        self.stats["episodes"] += 1
        return self._obs(), {"reset_seconds": round(self.clock() - t0, 2), "screens": path}

    # ---------------------------------------------------------------- step
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        cfg = self.cfg
        self.dev.act(ACTIONS[action])

        # fixed-rate control: wait for the next tick; if we're late, don't try to catch up
        self._tick += 1.0 / cfg.hz
        wait = self._tick - self.clock()
        if wait > 0:
            self.sleep(wait)
        else:
            self.stats["late_ticks"] += 1
            self._tick = self.clock()

        det = self.look()
        self.frames.append(to_obs(self.last_canon, cfg.obs_size))
        self._steps += 1

        terminated = det.role == "crash"
        truncated = False
        now = self.clock()
        info = {"screen": det.name, "match": det.score}
        if terminated and self._lost_since is not None:
            # how long the crash took to show up as a crash screen: the reward delay
            delay = round(now - self._lost_since, 3)
            self.crash_delays.append(delay)
            info["crash_delay_s"] = delay
        if det.role == "dismiss":
            # a popup mid-run (e.g. the game paused): clear it now rather than abandon the run
            self._act_on(det)
            self.stats["mid_run_dismiss"] += 1
        if det.role in ("playing", "crash"):
            self._lost_since = None
        else:
            # popup mid-run, transition frame, or something we don't know: tolerate briefly
            self._lost_since = self._lost_since or now
            if now - self._lost_since >= cfg.lost_after:
                truncated = True
                self.stats["lost_track"] += 1
                if self.stats["lost_track"] <= 5:      # keep evidence, but don't fill the disk
                    self._save_debug(f"lost-track-{self.stats['lost_track']}")
        if cfg.max_steps is not None and self._steps >= cfg.max_steps:
            truncated = True
        reward = cfg.crash_penalty if terminated else cfg.alive_reward
        return self._obs(), reward, terminated, truncated, info
