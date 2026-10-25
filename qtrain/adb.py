"""Sending game inputs to an Android device/emulator over ADB.

Three input backends, from slowest to fastest, so the benchmark can show
the effect of each optimization:

    "spawn"   - a fresh `adb shell input swipe ...` process per action
                (the naive approach most tutorials use)
    "shell"   - one persistent `adb shell` session; commands are written to
                its stdin, so no host-side process is spawned per action
    "cmd"     - persistent shell + `cmd input ...`, which talks to the input
                service directly instead of starting the `input` wrapper
                (needs Android 10+; in BlueStacks pick an Android 11 instance)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

ACTIONS = ("noop", "left", "right", "up", "down")  # up = jump, down = roll
BACKENDS = ("spawn", "shell", "cmd")


def adb_path() -> str:
    """Locate adb: $QTRAIN_ADB, then PATH."""
    path = os.environ.get("QTRAIN_ADB") or shutil.which("adb")
    if not path:
        raise FileNotFoundError(
            "adb not found. Install Android platform-tools and add it to PATH, "
            "or set QTRAIN_ADB to the full path of adb.exe."
        )
    return path


def run_adb(*args: str, serial: str | None = None, timeout: float = 15.0) -> str:
    cmd = [adb_path()] + (["-s", serial] if serial else []) + list(args)
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout


def connect(serial: str) -> None:
    """`adb connect` for network devices such as BlueStacks (127.0.0.1:5555)."""
    if ":" in serial:
        msg = run_adb("connect", serial)
        if "connected" not in msg:
            raise RuntimeError(f"adb connect {serial}: {msg.strip()}")


def list_devices() -> list[str]:
    lines = run_adb("devices").strip().splitlines()[1:]
    return [ln.split()[0] for ln in lines if ln.strip().endswith("device")]


# ---------------------------------------------------------------- geometry
def parse_display_size(dumpsys: str, wm_size: str = "") -> tuple[int, int]:
    """Current-orientation display size in pixels.

    Prefers `dumpsys window displays` (cur=WxH reflects rotation), falling back
    to `wm size` (physical size, may be unrotated).
    """
    m = re.search(r"cur=(\d+)x(\d+)", dumpsys)
    if m:
        return int(m.group(1)), int(m.group(2))
    sizes = re.findall(r"(\d+)x(\d+)", wm_size)
    if sizes:  # "Override size" line comes last when present
        w, h = sizes[-1]
        return int(w), int(h)
    raise ValueError("could not parse display size")


@dataclass(frozen=True)
class SwipeGeometry:
    width: int
    height: int
    frac: float = 0.25      # swipe length as a fraction of the screen
    duration_ms: int = 40

    def swipe(self, action: str) -> tuple[int, int, int, int] | None:
        cx, cy = self.width // 2, self.height // 2
        dx, dy = int(self.width * self.frac), int(self.height * self.frac)
        return {
            "noop": None,
            "left": (cx + dx // 2, cy, cx - dx // 2, cy),
            "right": (cx - dx // 2, cy, cx + dx // 2, cy),
            "up": (cx, cy + dy // 2, cx, cy - dy // 2),
            "down": (cx, cy - dy // 2, cx, cy + dy // 2),
        }[action]

    def command(self, action: str, prefix: str = "input") -> str | None:
        pts = self.swipe(action)
        if pts is None:
            return None
        return f"{prefix} swipe {pts[0]} {pts[1]} {pts[2]} {pts[3]} {self.duration_ms}"


# ------------------------------------------------------------------ device
class AdbDevice:
    def __init__(self, serial: str, backend: str = "shell"):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.serial = serial
        self.backend = backend
        connect(serial)
        disp = run_adb("shell", "dumpsys", "window", "displays", serial=serial)
        wm = run_adb("shell", "wm", "size", serial=serial)
        self.geom = SwipeGeometry(*parse_display_size(disp, wm))
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._marker = 0
        if backend != "spawn":
            self._open_shell()

    def _open_shell(self) -> None:
        self._proc = subprocess.Popen(
            [adb_path(), "-s", self.serial, "shell"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

    @property
    def _prefix(self) -> str:
        return "cmd input" if self.backend == "cmd" else "input"

    def act(self, action: str, wait: bool = False) -> float:
        """Send one game action. Returns seconds spent on the host side.

        With wait=True, blocks until the device has finished executing the
        command (used by the benchmark to measure true round-trip latency).
        """
        cmd = self.geom.command(action, self._prefix)
        if cmd is None:
            return 0.0
        t0 = time.perf_counter()
        if self.backend == "spawn":
            run_adb("shell", *cmd.split(), serial=self.serial)
        else:
            self._shell(cmd, wait=wait)
        return time.perf_counter() - t0

    def _shell(self, cmd: str, wait: bool) -> None:
        assert self._proc and self._proc.stdin and self._proc.stdout
        with self._lock:
            if not wait:
                # fire-and-forget; silence output so the unread pipe never fills
                self._proc.stdin.write(f"{cmd} >/dev/null 2>&1 &\n")
                self._proc.stdin.flush()
                return
            self._marker += 1
            mark = f"__QT{self._marker}__"
            self._proc.stdin.write(f"{cmd}; echo {mark}\n")
            self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("adb shell closed unexpectedly")
                if line.strip() == mark:
                    return

    def tap(self, x: int, y: int) -> None:
        cmd = f"{self._prefix} tap {x} {y}"
        if self.backend == "spawn":
            run_adb("shell", *cmd.split(), serial=self.serial)
        else:
            self._shell(cmd, wait=True)

    def close(self) -> None:
        if self._proc:
            try:
                self._proc.stdin.write("exit\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
