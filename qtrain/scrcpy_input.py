"""Fast touch input through scrcpy's on-device server.

Why: Android's `input` command starts a fresh runtime for every swipe
(~600 ms on BlueStacks), and the `cmd input` shortcut isn't implemented on
this Android 11 build. scrcpy instead runs ONE long-lived program on the
device (pushed over ADB, started with app_process) that injects touch events
it receives over a socket, so each touch is a 32-byte write.

Wire format (scrcpy app/src/control_msg.c, INJECT_TOUCH_EVENT, big-endian):
    u8 type=2 | u8 action | u64 pointer_id | i32 x | i32 y | u16 w | u16 h
    | u16 pressure (fixed point, 1.0 = 0xffff) | i32 action_button | i32 buttons
Positions are in the coordinate space of the server's video stream, so we let
it stream a tiny low-fps video (which we read and discard) and use its size.

The server version must match the installed scrcpy exactly; we push the
`scrcpy-server` file that ships with the scrcpy on this machine.
"""
from __future__ import annotations

import glob
import os
import queue
import random
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

from .adb import SwipeGeometry, adb_path, connect, run_adb

SERVER_DEVICE_PATH = "/data/local/tmp/qtrain-scrcpy-server.jar"

TYPE_INJECT_KEYCODE = 0
TYPE_INJECT_TOUCH = 2
ACTION_DOWN, ACTION_UP, ACTION_MOVE = 0, 1, 2
POINTER_FINGER = -2          # scrcpy's SC_POINTER_ID_GENERIC_FINGER
KEYCODE_BACK = 4


# --------------------------------------------------------------- encoding
def encode_touch(action: int, x: int, y: int, width: int, height: int, pressure: float = 1.0,
                 pointer_id: int = POINTER_FINGER) -> bytes:
    p = 0xFFFF if pressure >= 1.0 else max(0, int(pressure * 65536))
    return struct.pack(">BBqiiHHHii", TYPE_INJECT_TOUCH, action, pointer_id, int(x), int(y),
                       width, height, p, 0, 0)


def encode_key(action: int, keycode: int) -> bytes:
    return struct.pack(">BBiii", TYPE_INJECT_KEYCODE, action, keycode, 0, 0)


def swipe_messages(x1: int, y1: int, x2: int, y2: int, width: int, height: int,
                   steps: int = 4) -> list[bytes]:
    """down, `steps` moves along the line, up."""
    msgs = [encode_touch(ACTION_DOWN, x1, y1, width, height)]
    for i in range(1, steps + 1):
        t = i / steps
        msgs.append(encode_touch(ACTION_MOVE, round(x1 + (x2 - x1) * t), round(y1 + (y2 - y1) * t),
                                 width, height))
    msgs.append(encode_touch(ACTION_UP, x2, y2, width, height, pressure=0.0))
    return msgs


# ------------------------------------------------------------ locating it
def scrcpy_version() -> str:
    """Version of the scrcpy installed on this machine, e.g. '5.0.1'."""
    if os.environ.get("QTRAIN_SCRCPY_VERSION"):
        return os.environ["QTRAIN_SCRCPY_VERSION"]
    exe = shutil.which("scrcpy")
    if not exe:
        raise FileNotFoundError("scrcpy not found on PATH (install: winget install --id Genymobile.scrcpy)")
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=15).stdout
    m = re.search(r"scrcpy\s+v?(\d+(?:\.\d+)+)", out)
    if not m:
        raise RuntimeError(f"couldn't read the scrcpy version from: {out[:200]!r}")
    return m.group(1)


def find_server() -> Path:
    """The `scrcpy-server` file that came with the installed scrcpy."""
    if os.environ.get("QTRAIN_SCRCPY_SERVER"):
        return Path(os.environ["QTRAIN_SCRCPY_SERVER"])
    candidates = []
    exe = shutil.which("scrcpy")
    if exe:
        candidates.append(Path(os.path.realpath(exe)).parent / "scrcpy-server")
        candidates.append(Path(exe).parent / "scrcpy-server")
    local = os.environ.get("LOCALAPPDATA", "")
    if local:  # winget installs under ...\WinGet\Packages\Genymobile.scrcpy_<id>\scrcpy-win64-vX\
        pattern = os.path.join(local, "Microsoft", "WinGet", "Packages", "Genymobile.scrcpy*", "**",
                               "scrcpy-server")
        candidates += [Path(p) for p in glob.glob(pattern, recursive=True)]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("couldn't find the scrcpy-server file; "
                            "set QTRAIN_SCRCPY_SERVER to its full path "
                            "(it sits next to scrcpy.exe)")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("scrcpy server closed the connection")
        buf += chunk
    return buf


CODEC_TAGS = (b"h264", b"h265", b"av01")


def read_stream_header(sock: socket.socket, limit: int = 1024) -> tuple[int, int]:
    """Video size from the stream header: a codec tag ("h264"...), then the size.

    Rather than assume exactly where the header starts (it has moved between
    scrcpy versions as metadata was added), read until the codec tag appears.
    Newer servers (seen with 5.0.1) then send a "session" packet: a u32 with the
    top bit set, followed by u32 width and u32 height; older ones send the size
    directly.
    """
    buf = b""
    while len(buf) < limit:
        buf += _recv_exact(sock, 1)
        for tag in CODEC_TAGS:
            if buf.endswith(tag):
                (first,) = struct.unpack(">I", _recv_exact(sock, 4))
                if first & 0x80000000:                     # session packet marker
                    width, height = struct.unpack(">II", _recv_exact(sock, 8))
                else:
                    width, (height,) = first, struct.unpack(">I", _recv_exact(sock, 4))
                if not (16 <= width <= 8192 and 16 <= height <= 8192):
                    raise RuntimeError(f"implausible scrcpy video size {width}x{height}")
                return width, height
    raise RuntimeError(f"no codec tag in the first {limit} bytes from scrcpy: {buf[:64]!r}")


# ------------------------------------------------------------------ device
class ScrcpyDevice:
    """Same interface as AdbDevice (width, height, act, tap, back, close)."""

    backend = "scrcpy"

    def __init__(self, serial: str, max_size: int = 480, swipe_ms: int = 40, start_timeout: float = 15.0):
        self.serial = serial
        self.swipe_ms = swipe_ms
        self._proc: subprocess.Popen | None = None
        self._port: int | None = None
        connect(serial)
        server, version = find_server(), scrcpy_version()
        scid = random.randint(1, 0x7FFFFFFF)
        run_adb("push", str(server), SERVER_DEVICE_PATH, serial=serial, timeout=60)
        self._port = _free_port()
        run_adb("forward", f"tcp:{self._port}", f"localabstract:scrcpy_{scid:08x}", serial=serial)
        cmd = [adb_path(), "-s", serial, "shell", f"CLASSPATH={SERVER_DEVICE_PATH}", "app_process", "/",
               "com.genymobile.scrcpy.Server", version, f"scid={scid:08x}", "log_level=warn",
               "tunnel_forward=true", "audio=false", "control=true", "send_device_meta=false",
               f"max_size={max_size}", "max_fps=5", "video_bit_rate=250000", "cleanup=true"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            video = self._connect_first(start_timeout)
            control = socket.create_connection(("127.0.0.1", self._port), timeout=5)
            self._attach(video, control)
        except Exception:
            self.close()
            raise

    def _connect_first(self, timeout: float) -> socket.socket:
        """Connect the first (video) socket; the server greets it with one dummy byte.

        Through `adb forward` the connect succeeds even before the server is
        listening, so a connection that closes without the byte means "retry".
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc and self._proc.poll() is not None:
                out = self._proc.stdout.read() if self._proc.stdout else ""
                raise RuntimeError(f"scrcpy server exited: {out.strip()[-500:]}")
            try:
                s = socket.create_connection(("127.0.0.1", self._port), timeout=2)
                s.settimeout(2)
                if s.recv(1) == b"\x00":
                    s.settimeout(None)
                    return s
                s.close()
            except OSError:
                pass
            time.sleep(0.1)
        raise TimeoutError("scrcpy server didn't start in time")

    def _attach(self, video: socket.socket, control: socket.socket) -> None:
        """Read the video stream header for the coordinate space, then start the workers."""
        self._video, self._control = video, control
        self.width, self.height = read_stream_header(video)
        self.geom = SwipeGeometry(self.width, self.height)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._gestures: queue.Queue = queue.Queue()
        threading.Thread(target=self._drain_video, daemon=True).start()
        threading.Thread(target=self._gesture_worker, daemon=True).start()

    # ------------------------------------------------------------ plumbing
    def _drain_video(self) -> None:
        try:
            while not self._closed.is_set() and self._video.recv(65536):
                pass
        except OSError:
            pass

    def _send(self, data: bytes) -> None:
        with self._lock:
            self._control.sendall(data)

    def _swipe_now(self, pts: tuple[int, int, int, int]) -> None:
        msgs = swipe_messages(*pts, self.width, self.height)
        gap = self.swipe_ms / 1000 / (len(msgs) - 1)
        for i, m in enumerate(msgs):
            if i:
                time.sleep(gap)
            self._send(m)

    def _gesture_worker(self) -> None:
        while not self._closed.is_set():
            try:
                pts = self._gestures.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._swipe_now(pts)
            except OSError:
                return

    # ----------------------------------------------------------- interface
    def act(self, action: str, wait: bool = False) -> float:
        """Swipe for a game action. Non-blocking by default (a swipe takes
        `swipe_ms` to perform); returns host-side seconds spent."""
        pts = self.geom.swipe(action)
        if pts is None:
            return 0.0
        t0 = time.perf_counter()
        if wait:
            self._swipe_now(pts)
        else:
            self._gestures.put(pts)
        return time.perf_counter() - t0

    def tap(self, x: int, y: int, hold: float = 0.03) -> None:
        self._send(encode_touch(ACTION_DOWN, x, y, self.width, self.height))
        time.sleep(hold)
        self._send(encode_touch(ACTION_UP, x, y, self.width, self.height, pressure=0.0))

    def press(self, x: int, y: int, seconds: float) -> None:
        """Hold a finger down at (x, y); used by the end-to-end latency test."""
        self.tap(x, y, hold=seconds)

    def back(self) -> None:
        self._send(encode_key(ACTION_DOWN, KEYCODE_BACK) + encode_key(ACTION_UP, KEYCODE_BACK))

    def close(self) -> None:
        closed = getattr(self, "_closed", None)
        if closed:
            closed.set()
        for s in (getattr(self, "_control", None), getattr(self, "_video", None)):
            if s:
                try:
                    s.close()
                except OSError:
                    pass
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._port:
            try:
                run_adb("forward", "--remove", f"tcp:{self._port}", serial=self.serial)
            except (RuntimeError, OSError, subprocess.SubprocessError):
                pass
            self._port = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_device(serial: str, backend: str = "auto"):
    """The fastest input device that works: scrcpy, else a persistent adb shell."""
    from .adb import AdbDevice, best_backend
    if backend in ("auto", "scrcpy"):
        try:
            return ScrcpyDevice(serial)
        except Exception as e:  # noqa: BLE001 - any failure here means "fall back"
            if backend == "scrcpy":
                raise
            print(f"[qtrain] scrcpy input unavailable ({type(e).__name__}: {e}); using adb shell input")
            backend = "auto"
    return AdbDevice(serial, backend=best_backend(serial) if backend == "auto" else backend)
