"""scrcpy's on-device server as Q-Train's input channel and (optionally) its frame source.

One long-lived server per device (pushed over ADB, started with app_process)
streams the device's screen as video and accepts input over a second socket.

Input. Android's `input` command starts a fresh runtime for every touch
(~200 ms to show on screen on BlueStacks); here each touch is a 32-byte
message (scrcpy app/src/control_msg.c, INJECT_TOUCH_EVENT, big-endian):
    u8 type=2 | u8 action | u64 pointer_id | i32 x | i32 y | u16 w | u16 h
    | u16 pressure (1.0 = 0xffff) | i32 action_button | i32 buttons
Touch positions are in the coordinate space of the video stream, so the
current video size is tracked from the stream itself.

Frames. Decoding the stream gives the device's own pixels: no desktop
screenshot, so no locating the screen inside an emulator window, no need for
the window to be visible, and the same path for any app or emulator.

Stream format (scrcpy app/src/demuxer.c): after a 4-byte codec tag ("h264"),
a sequence of 12-byte headers.
    session packet  first bit set; bytes 4-7 width, 8-11 height. Sent at the
                    start and again whenever the screen rotates or resizes.
    media packet    u64 pts_and_flags (bit 62 = codec config, bit 61 = key
                    frame), u32 length, then `length` bytes of encoded video.
Config packets (H.264 SPS/PPS) are prepended to the next frame before
decoding, as scrcpy's own client does.

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

import numpy as np

from .adb import SwipeGeometry, adb_path, connect, run_adb

SERVER_DEVICE_PATH = "/data/local/tmp/qtrain-scrcpy-server.jar"

TYPE_INJECT_KEYCODE = 0
TYPE_INJECT_TOUCH = 2
ACTION_DOWN, ACTION_UP, ACTION_MOVE = 0, 1, 2
POINTER_FINGER = -2          # scrcpy's SC_POINTER_ID_GENERIC_FINGER
KEYCODE_BACK = 4

CODEC_TAGS = {b"h264": "h264", b"h265": "hevc", b"av01": "av1"}
FLAG_CONFIG = 1 << 62
FLAG_KEY_FRAME = 1 << 61


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
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("scrcpy server closed the connection")
        buf += chunk
    return bytes(buf)


def read_stream_header(sock: socket.socket, limit: int = 1024) -> tuple[str, int, int]:
    """(codec, width, height) from the start of the video stream.

    Reads until a codec tag appears (rather than assuming its exact offset),
    then the initial size: a session packet on current servers, or plain
    width/height on older ones.
    """
    buf = b""
    while len(buf) < limit:
        buf += _recv_exact(sock, 1)
        for tag, codec in CODEC_TAGS.items():
            if buf.endswith(tag):
                first = _recv_exact(sock, 4)
                if first[0] & 0x80:                                  # session packet
                    width, height = struct.unpack(">II", _recv_exact(sock, 8))
                else:                                                # older servers: plain width, height
                    width = struct.unpack(">I", first)[0]
                    (height,) = struct.unpack(">I", _recv_exact(sock, 4))
                if not (16 <= width <= 8192 and 16 <= height <= 8192):
                    raise RuntimeError(f"implausible scrcpy video size {width}x{height}")
                return codec, width, height
    raise RuntimeError(f"no codec tag in the first {limit} bytes from scrcpy: {buf[:64]!r}")


# ---------------------------------------------------------------- decoding
class AvDecoder:
    """H.264/H.265/AV1 to BGR frames with PyAV (FFmpeg), tuned for latency."""

    def __init__(self, codec: str):
        try:
            import av
        except ImportError as e:
            raise ImportError("frame capture from scrcpy needs PyAV: pip install av") from e
        self._av = av
        self.ctx = av.CodecContext.create(codec, "r")
        # Frame threading buffers several frames before output; slice threading doesn't.
        for attr, value in (("thread_type", "SLICE"), ("options", {"flags": "low_delay"})):
            try:
                setattr(self.ctx, attr, value)
            except (AttributeError, TypeError, ValueError):
                pass

    def decode(self, data: bytes) -> list:
        """Decoded frames, still in the decoder's own format (YUV). Converting to BGR
        is left to whoever actually reads a frame (see ScrcpySession.latest_frame)."""
        return list(self.ctx.decode(self._av.Packet(data)))


# ----------------------------------------------------------------- session
class ScrcpySession:
    """One scrcpy server on one device: video stream (parsed, optionally decoded) and control."""

    def __init__(self, serial: str, *, control: bool = True, decode: bool = False, max_size: int = 480,
                 max_fps: int = 5, bit_rate: int = 250_000, start_timeout: float = 15.0):
        self.serial = serial
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
               "tunnel_forward=true", "audio=false", f"control={'true' if control else 'false'}",
               "send_device_meta=false", f"max_size={max_size}", f"max_fps={max_fps}",
               f"video_bit_rate={bit_rate}", "cleanup=true"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            video = self._connect_first(start_timeout)
            ctrl = socket.create_connection(("127.0.0.1", self._port), timeout=5) if control else None
            self.attach(video, ctrl, decode=decode)
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

    def attach(self, video: socket.socket, control: socket.socket | None, decode: bool = False,
               decoder_factory=AvDecoder) -> None:
        """Read the stream header, then parse (and optionally decode) the stream in the background."""
        self._video, self._control = video, control
        codec, width, height = read_stream_header(video)
        self._size = (width, height)
        self._decoder = decoder_factory(codec) if decode else None
        self._send_lock = threading.Lock()
        self._frame_cond = threading.Condition()
        self._frame = None               # newest decoded frame, not yet converted
        self._converted: tuple[int, np.ndarray] | None = None   # (frame id, BGR) cache
        self._frame_id = 0
        self._frame_time = 0.0
        self.stats = {"packets": 0, "frames": 0, "resizes": 0, "bytes": 0}
        self._closed = threading.Event()
        threading.Thread(target=self._video_loop, daemon=True).start()

    # ------------------------------------------------------------ stream
    def _video_loop(self) -> None:
        pending_config = b""
        try:
            while not self._closed.is_set():
                header = _recv_exact(self._video, 12)
                if header[0] & 0x80:                                  # session packet: new size
                    size = struct.unpack(">II", header[4:12])
                    if size != self._size:
                        self._size = size
                        self.stats["resizes"] += 1
                    continue
                pts_flags, length = struct.unpack(">QI", header)
                payload = _recv_exact(self._video, length)
                self.stats["packets"] += 1
                self.stats["bytes"] += length
                if self._decoder is None:
                    continue
                if pts_flags & FLAG_CONFIG:                            # SPS/PPS: keep for the next frame
                    pending_config += payload
                    continue
                data, pending_config = pending_config + payload, b""
                for frame in self._decoder.decode(data):
                    self._publish(frame)
        except (OSError, ConnectionError):
            pass
        finally:
            with self._frame_cond:
                self._frame_cond.notify_all()

    def _publish(self, frame) -> None:
        with self._frame_cond:
            self._frame = frame
            self._frame_id += 1
            self._frame_time = time.perf_counter()
            self.stats["frames"] += 1
            self._frame_cond.notify_all()

    @property
    def size(self) -> tuple[int, int]:
        """Current video size: the coordinate space for touches."""
        return self._size

    def latest_frame(self, after_id: int = 0, timeout: float = 1.0) -> tuple[np.ndarray, int]:
        """(frame, id) of the newest decoded frame with id > after_id, waiting up to `timeout`."""
        deadline = time.perf_counter() + timeout
        with self._frame_cond:
            while self._frame_id <= after_id:
                left = deadline - time.perf_counter()
                if left <= 0 or self._closed.is_set():
                    raise TimeoutError("no new frame from scrcpy (static screen, or the stream stopped)")
                self._frame_cond.wait(left)
            raw, fid = self._frame, self._frame_id
        # Convert outside the lock, and only frames someone actually reads: the stream
        # delivers up to 60 fps but the control loop reads ~15, so most frames are
        # decoded (H.264 needs every one) but never converted.
        if self._converted is None or self._converted[0] != fid:
            bgr = raw.to_ndarray(format="bgr24") if hasattr(raw, "to_ndarray") else raw
            self._converted = (fid, bgr)
        return self._converted[1], fid

    def send(self, data: bytes) -> None:
        if self._control is None:
            raise RuntimeError("this scrcpy session was started without control")
        with self._send_lock:
            self._control.sendall(data)

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


# ------------------------------------------------------------------ device
class ScrcpyDevice:
    """Touch input on a scrcpy session. Same interface as AdbDevice."""

    backend = "scrcpy"

    def __init__(self, serial: str | None = None, *, session: ScrcpySession | None = None,
                 swipe_ms: int = 40):
        self.session = session or ScrcpySession(serial)
        self._owns_session = session is None
        self.swipe_ms = swipe_ms
        self._gestures: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        threading.Thread(target=self._gesture_worker, daemon=True).start()

    @property
    def width(self) -> int:
        return self.session.size[0]

    @property
    def height(self) -> int:
        return self.session.size[1]

    @property
    def geom(self) -> SwipeGeometry:
        return SwipeGeometry(self.width, self.height)    # follows rotations and resizes

    def _swipe_now(self, pts: tuple[int, int, int, int]) -> None:
        msgs = swipe_messages(*pts, self.width, self.height)
        gap = self.swipe_ms / 1000 / (len(msgs) - 1)
        for i, m in enumerate(msgs):
            if i:
                time.sleep(gap)
            self.session.send(m)

    def _gesture_worker(self) -> None:
        while not self._stop.is_set():
            try:
                pts = self._gestures.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._swipe_now(pts)
            except OSError:
                return

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
        w, h = self.width, self.height
        self.session.send(encode_touch(ACTION_DOWN, x, y, w, h))
        time.sleep(hold)
        self.session.send(encode_touch(ACTION_UP, x, y, w, h, pressure=0.0))

    def press(self, x: int, y: int, seconds: float) -> None:
        """Hold a finger down at (x, y); used by the end-to-end latency test."""
        self.tap(x, y, hold=seconds)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        """A drag from (x1, y1) to (x2, y2), e.g. to scroll a list. Blocks until done."""
        msgs = swipe_messages(x1, y1, x2, y2, self.width, self.height, steps=10)
        gap = duration_ms / 1000 / (len(msgs) - 1)
        for i, m in enumerate(msgs):
            if i:
                time.sleep(gap)
            self.session.send(m)

    def back(self) -> None:
        self.session.send(encode_key(ACTION_DOWN, KEYCODE_BACK) + encode_key(ACTION_UP, KEYCODE_BACK))

    def close(self) -> None:
        self._stop.set()
        if self._owns_session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ----------------------------------------------------------------- capture
class ScrcpyCapture:
    """Frames decoded from a scrcpy session. Same interface as the desktop capture backends.

    Frames are the device's own screen, so `full_screen` tells callers there is
    no emulator window to crop.
    """

    full_screen = True

    def __init__(self, serial: str | None = None, *, session: ScrcpySession | None = None,
                 max_size: int = 360, max_fps: int = 60, bit_rate: int = 8_000_000):
        self.session = session or ScrcpySession(serial, control=False, decode=True, max_size=max_size,
                                                max_fps=max_fps, bit_rate=bit_rate)
        self._owns_session = session is None
        self._last_id = 0
        self.fresh = 0

    def grab(self) -> np.ndarray:
        """Newest frame; repeats the previous one if the screen hasn't changed (like dxcam)."""
        frame, fid = self.session.latest_frame(after_id=0, timeout=5.0)
        if fid != self._last_id:
            self._last_id, self.fresh = fid, self.fresh + 1
        return frame

    def grab_fresh(self, timeout: float = 1.0) -> np.ndarray:
        """Block until a frame newer than the last one returned arrives."""
        frame, self._last_id = self.session.latest_frame(after_id=self._last_id, timeout=timeout)
        self.fresh += 1
        return frame

    def close(self) -> None:
        if self._owns_session:
            self.session.close()


def open_device(serial: str, backend: str = "auto", session: ScrcpySession | None = None):
    """The fastest input device that works: scrcpy, else a persistent adb shell."""
    from .adb import AdbDevice, best_backend
    if backend in ("auto", "scrcpy"):
        try:
            return ScrcpyDevice(serial, session=session)
        except Exception as e:  # noqa: BLE001 - any failure here means "fall back"
            if backend == "scrcpy":
                raise
            print(f"[qtrain] scrcpy input unavailable ({type(e).__name__}: {e}); using adb shell input")
    return AdbDevice(serial, backend=best_backend(serial) if backend == "auto" else backend)
