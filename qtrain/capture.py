"""Grabbing frames of the emulator screen.

Backends (benchmark them with `python -m qtrain.bench`):

    "adb"    - `adb exec-out screencap` raw pixels. Works anywhere, but slow:
               the device has to read back the framebuffer and ship it over ADB.
    "mss"    - screenshot of the emulator *window* on the desktop via GDI.
    "dxcam"  - Windows Desktop Duplication API; usually the fastest on Windows.

All backends return frames as uint8 NumPy arrays, shape (H, W, 3), BGR order
(the OpenCV convention).

Window coordinates: on a high-DPI laptop screen with display scaling (e.g.
3840x2400 at 250%), Windows lies to non-DPI-aware programs about window
positions. `set_dpi_aware()` must run before any window lookup, otherwise the
captured region is offset and too small.
"""
from __future__ import annotations

import struct
import subprocess
import sys
import time

import numpy as np

CAPTURE_BACKENDS = ("adb", "mss", "dxcam")


# ------------------------------------------------------------ window lookup
def match_window(windows: list[tuple[int, str]], query: str) -> int:
    """Pick the window handle whose title best matches `query` (case-insensitive).

    Exact title match wins; otherwise the shortest title containing `query`.
    """
    q = query.lower()
    exact = [h for h, t in windows if t.lower() == q]
    if exact:
        return exact[0]
    hits = sorted((len(t), h) for h, t in windows if q in t.lower())
    if not hits:
        titles = "\n  ".join(t for _, t in windows if t.strip()) or "(none)"
        raise LookupError(f"no window title contains {query!r}. Visible windows:\n  {titles}")
    return hits[0][1]


def _require_windows() -> None:
    if sys.platform != "win32":
        raise OSError("window capture is Windows-only; use the 'adb' capture backend elsewhere")


def set_dpi_aware() -> None:
    _require_windows()
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


def list_windows() -> list[tuple[int, str]]:
    """(hwnd, title) for every visible top-level window."""
    _require_windows()
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    out: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                out.append((int(hwnd), buf.value))
        return True

    user32.EnumWindows(_cb, 0)
    return out


def find_window(query: str) -> int:
    set_dpi_aware()
    return match_window(list_windows(), query)


def client_region(hwnd: int) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the window's client area in screen pixels."""
    _require_windows()
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise OSError("GetClientRect failed (window closed?)")
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, pt.x + rect.right, pt.y + rect.bottom


# ----------------------------------------------------------------- backends
class AdbScreencap:
    """Raw framebuffer over ADB (no PNG encoding on the device)."""

    def __init__(self, serial: str):
        from .adb import adb_path, connect
        connect(serial)
        self.cmd = [adb_path(), "-s", serial, "exec-out", "screencap"]

    @staticmethod
    def decode(raw: bytes) -> np.ndarray:
        # header: width, height, pixel format (+ colorspace on Android 9+)
        w, h, _fmt = struct.unpack_from("<III", raw)
        n = w * h * 4
        header = len(raw) - n
        if header not in (12, 16):
            raise ValueError(f"unexpected screencap size {len(raw)} for {w}x{h}")
        rgba = np.frombuffer(raw, np.uint8, n, header).reshape(h, w, 4)
        return rgba[:, :, 2::-1].copy()  # RGBA -> BGR

    def grab(self) -> np.ndarray:
        raw = subprocess.run(self.cmd, capture_output=True, check=True).stdout
        return self.decode(raw)

    def close(self) -> None:
        pass


class MssCapture:
    def __init__(self, hwnd: int):
        import mss
        self.hwnd = hwnd
        self.sct = mss.mss()

    def grab(self) -> np.ndarray:
        left, top, right, bottom = client_region(self.hwnd)  # tracks window moves
        shot = self.sct.grab({"left": left, "top": top, "width": right - left, "height": bottom - top})
        return np.asarray(shot)[:, :, :3]  # BGRA -> BGR (view, no copy)

    def close(self) -> None:
        self.sct.close()


class DxcamCapture:
    """Desktop Duplication. Only returns a *new* frame when the screen changed;
    otherwise we hand back the previous one, which is what a game loop wants."""

    def __init__(self, hwnd: int):
        import dxcam
        self.hwnd = hwnd
        self.cam = dxcam.create(output_color="BGR")
        self.last: np.ndarray | None = None
        self.fresh = 0      # how many grabs returned a genuinely new frame

    def grab(self) -> np.ndarray:
        frame = self.cam.grab(region=client_region(self.hwnd))
        if frame is not None:
            self.last = frame
            self.fresh += 1
        elif self.last is None:  # very first grab on a static screen
            while frame is None:
                frame = self.cam.grab(region=client_region(self.hwnd))
            self.last = frame
        return self.last

    def grab_fresh(self, timeout: float = 1.0) -> np.ndarray:
        """Block until the screen region actually changes (or time out)."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            frame = self.cam.grab(region=client_region(self.hwnd))
            if frame is not None:
                self.last = frame
                self.fresh += 1
                return frame
        raise TimeoutError("no new frame; is the game animating and the window visible?")

    def close(self) -> None:
        del self.cam


def make_capture(backend: str, window: str | None = None, serial: str | None = None):
    if backend == "adb":
        if not serial:
            raise ValueError("adb capture needs --serial")
        return AdbScreencap(serial)
    if backend in ("mss", "dxcam"):
        if not window:
            raise ValueError(f"{backend} capture needs --window")
        hwnd = find_window(window)
        return MssCapture(hwnd) if backend == "mss" else DxcamCapture(hwnd)
    raise ValueError(f"backend must be one of {CAPTURE_BACKENDS}")
