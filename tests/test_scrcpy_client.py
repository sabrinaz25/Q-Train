"""scrcpy client: wire formats, stream parsing, input and capture, against a fake server (no device)."""
import socket
import struct
import time

import numpy as np
import pytest

from qtrain.e2e import bar_change, column_change
from qtrain.scrcpy_client import (
    ACTION_DOWN,
    ACTION_MOVE,
    ACTION_UP,
    FLAG_CONFIG,
    FLAG_KEY_FRAME,
    ScrcpyCapture,
    ScrcpyDevice,
    ScrcpySession,
    encode_key,
    encode_touch,
    read_stream_header,
    swipe_messages,
)

TOUCH = ">BBqiiHHHii"


# ----------------------------------------------------------- wire formats
def test_touch_message_layout_matches_scrcpy():
    msg = encode_touch(ACTION_DOWN, 100, 200, 270, 480)
    assert len(msg) == 32
    typ, action, pointer, x, y, w, h, pressure, action_button, buttons = struct.unpack(TOUCH, msg)
    assert (typ, action, pointer) == (2, 0, -2)          # INJECT_TOUCH_EVENT, DOWN, generic finger
    assert (x, y, w, h) == (100, 200, 270, 480)
    assert pressure == 0xFFFF and action_button == 0 and buttons == 0
    assert msg[2:10] == b"\xff" * 7 + b"\xfe"            # pointer id -2 as big-endian u64
    assert struct.unpack(TOUCH, encode_touch(ACTION_UP, 1, 1, 9, 9, pressure=0.0))[7] == 0


def test_key_message_layout():
    msg = encode_key(0, 4)
    assert len(msg) == 14 and struct.unpack(">BBiii", msg) == (0, 0, 4, 0, 0)


def test_swipe_is_down_moves_up_along_a_line():
    msgs = [struct.unpack(TOUCH, m) for m in swipe_messages(100, 300, 200, 300, 270, 480, steps=4)]
    assert [m[1] for m in msgs] == [ACTION_DOWN] + [ACTION_MOVE] * 4 + [ACTION_UP]
    assert [m[3] for m in msgs] == [100, 125, 150, 175, 200, 200]
    assert {m[4] for m in msgs} == {300}


# --------------------------------------------------------- stream format
def session_packet(width, height):
    return bytes([0x80, 0, 0, 0]) + struct.pack(">II", width, height)


def media_packet(payload, pts=0, config=False, key=False):
    flags = (FLAG_CONFIG if config else 0) | (FLAG_KEY_FRAME if key else 0)
    return struct.pack(">QI", pts | flags, len(payload)) + payload


def test_stream_header_with_session_packet():
    a, b = socket.socketpair()
    try:
        a.sendall(b"h264" + session_packet(270, 480))
        assert read_stream_header(b) == ("h264", 270, 480)
    finally:
        a.close()
        b.close()


def test_stream_header_found_after_extra_metadata():
    a, b = socket.socketpair()
    try:
        a.sendall(b"BlueStacks".ljust(64, b"\0") + b"h264" + session_packet(264, 480))
        assert read_stream_header(b) == ("h264", 264, 480)
    finally:
        a.close()
        b.close()


class FakeDecoder:
    """Stands in for PyAV: records what it was fed, returns one 'frame' per packet."""

    instances: list = []

    def __init__(self, codec):
        self.codec, self.fed = codec, []
        FakeDecoder.instances.append(self)

    def decode(self, data):
        self.fed.append(data)
        return [np.full((4, 4, 3), len(self.fed), np.uint8)]


def fake_session(decode=False, width=270, height=480):
    """A ScrcpySession wired to socket pairs instead of a real server."""
    v_dev, v_cli = socket.socketpair()
    c_dev, c_cli = socket.socketpair()
    v_dev.sendall(b"h264" + session_packet(width, height))
    s = ScrcpySession.__new__(ScrcpySession)
    s.serial, s._proc, s._port = "fake", None, None
    s.attach(v_cli, c_cli, decode=decode, decoder_factory=FakeDecoder)
    return s, v_dev, c_dev


def wait_for(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_config_packet_is_prepended_to_the_next_frame():
    s, v_dev, _ = fake_session(decode=True)
    try:
        v_dev.sendall(media_packet(b"SPSPPS", config=True) + media_packet(b"IDR", pts=1, key=True)
                      + media_packet(b"P1", pts=2))
        assert wait_for(lambda: s.stats["frames"] == 2)
        assert FakeDecoder.instances[-1].fed == [b"SPSPPSIDR", b"P1"]
    finally:
        s.close()


def test_rotation_updates_touch_coordinates():
    s, v_dev, c_dev = fake_session(width=270, height=480)
    dev = ScrcpyDevice(session=s)
    try:
        assert (dev.width, dev.height) == (270, 480)
        v_dev.sendall(session_packet(480, 270))                # the app rotated to landscape
        assert wait_for(lambda: s.size == (480, 270))
        assert (dev.width, dev.height) == (480, 270) and s.stats["resizes"] == 1
        dev.tap(10, 20, hold=0.0)
        down = struct.unpack(TOUCH, read_messages(c_dev, 1)[0])
        assert down[5:7] == (480, 270)                         # touch carries the new screen size
    finally:
        dev.close()
        s.close()


def test_capture_returns_new_frames_and_repeats_on_static_screen():
    s, v_dev, _ = fake_session(decode=True)
    cap = ScrcpyCapture(session=s)
    try:
        assert cap.full_screen
        v_dev.sendall(media_packet(b"SPS", config=True) + media_packet(b"F1", key=True))
        first = cap.grab_fresh(timeout=2)
        assert first[0, 0, 0] == 1
        assert cap.grab()[0, 0, 0] == 1                        # nothing new: same frame again
        with pytest.raises(TimeoutError):
            cap.grab_fresh(timeout=0.1)                        # ...but no *new* frame
        v_dev.sendall(media_packet(b"F2", pts=1))
        assert cap.grab_fresh(timeout=2)[0, 0, 0] == 2
    finally:
        s.close()


def test_input_only_session_skips_decoding():
    s, v_dev, _ = fake_session(decode=False)
    try:
        v_dev.sendall(media_packet(b"x" * 1000, key=True))
        assert wait_for(lambda: s.stats["packets"] == 1)
        assert s.stats["frames"] == 0 and s.stats["bytes"] == 1000
    finally:
        s.close()


# ------------------------------------------------------------------ input
def read_messages(sock, n, size=32, timeout=2.0):
    sock.settimeout(timeout)
    buf = b""
    while len(buf) < n * size:
        buf += sock.recv(4096)
    return [buf[i:i + size] for i in range(0, n * size, size)]


def test_act_sends_a_swipe_without_blocking():
    s, _, c_dev = fake_session(width=270, height=480)
    dev = ScrcpyDevice(session=s, swipe_ms=20)
    try:
        t0 = time.perf_counter()
        dev.act("left")
        assert time.perf_counter() - t0 < 0.01            # queued, not performed inline
        msgs = [struct.unpack(TOUCH, m) for m in read_messages(c_dev, 6)]
        assert msgs[0][1] == ACTION_DOWN and msgs[-1][1] == ACTION_UP
        assert msgs[-1][3] < msgs[0][3]                    # moved left
        assert all(m[5:7] == (270, 480) for m in msgs)
    finally:
        dev.close()
        s.close()


def test_tap_and_back():
    s, _, c_dev = fake_session()
    dev = ScrcpyDevice(session=s)
    try:
        dev.tap(10, 20, hold=0.0)
        down, up = (struct.unpack(TOUCH, m) for m in read_messages(c_dev, 2))
        assert (down[1], up[1], down[3], down[4]) == (ACTION_DOWN, ACTION_UP, 10, 20)
        dev.back()
        keys = [struct.unpack(">BBiii", m) for m in read_messages(c_dev, 2, size=14)]
        assert [(k[1], k[2]) for k in keys] == [(0, 4), (1, 4)]
    finally:
        dev.close()
        s.close()


# ------------------------------------------------------ e2e test helpers
def test_column_change_spots_a_full_height_line_not_a_small_blob():
    before = np.full((640, 360, 3), 90, np.uint8)
    line = before.copy()
    line[:, 108] = 255                                     # crosshair down column 108
    blob = before.copy()
    blob[300:340, 100:120] = 255                           # something small animating nearby
    assert column_change(before, line, 108) > 0.9
    assert column_change(before, blob, 108) < 0.1
    assert column_change(before, line, 250) == 0.0         # wrong column: no change


def test_bar_change_counts_changed_overlay_digits():
    before = np.full((640, 360, 3), 200, np.uint8)
    after = before.copy()
    after[2:8, 40:80] = 30                                 # new digits in the top bar
    elsewhere = before.copy()
    elsewhere[300:340, 100:200] = 30                       # change far below the bar
    assert bar_change(before, after) == 6 * 40
    assert bar_change(before, elsewhere) == 0


class FakeYuvFrame:
    """Stands in for a PyAV frame: counts conversions to BGR."""
    conversions = 0

    def __init__(self, value):
        self.value = value

    def to_ndarray(self, format):
        FakeYuvFrame.conversions += 1
        return np.full((4, 4, 3), self.value, np.uint8)


def test_only_frames_that_are_read_get_converted():
    s, _, _ = fake_session()
    FakeYuvFrame.conversions = 0
    for i in range(1, 11):                                     # 10 frames decoded...
        s._publish(FakeYuvFrame(i))
    frame, fid = s.latest_frame()                              # ...one read
    again, _ = s.latest_frame()                                # read again: cached, no new conversion
    assert fid == 10 and frame[0, 0, 0] == 10 and again is frame
    assert FakeYuvFrame.conversions == 1
    s.close()
