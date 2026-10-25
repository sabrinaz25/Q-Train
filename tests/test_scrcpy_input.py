"""scrcpy input: wire format and handshake, against a fake server (no device needed)."""
import socket
import struct
import time

import numpy as np

from qtrain.e2e import bar_change, column_change
from qtrain.scrcpy_input import (
    ACTION_DOWN,
    ACTION_MOVE,
    ACTION_UP,
    ScrcpyDevice,
    encode_key,
    encode_touch,
    read_stream_header,
    swipe_messages,
)

TOUCH = ">BBqiiHHHii"


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


def fake_server(width=270, height=480):
    """Socket pairs standing in for scrcpy's video and control sockets."""
    v_dev, v_cli = socket.socketpair()
    c_dev, c_cli = socket.socketpair()
    v_dev.sendall(struct.pack(">III", 0x68323634, width, height) + b"\x00" * 500)  # header + some "video"
    dev = ScrcpyDevice.__new__(ScrcpyDevice)
    dev.serial, dev.swipe_ms, dev._proc, dev._port = "fake", 20, None, None
    dev._attach(v_cli, c_cli)
    return dev, v_dev, c_dev


def read_messages(sock, n, size=32, timeout=2.0):
    sock.settimeout(timeout)
    buf = b""
    while len(buf) < n * size:
        buf += sock.recv(4096)
    return [buf[i:i + size] for i in range(0, n * size, size)]


def test_handshake_reads_video_size_for_coordinates():
    dev, v_dev, c_dev = fake_server(270, 480)
    try:
        assert (dev.width, dev.height) == (270, 480)
    finally:
        dev.close()


def test_stream_header_found_after_extra_metadata():
    # newer scrcpy versions can put metadata (e.g. a 64-byte device name) before the codec tag
    a, b = socket.socketpair()
    try:
        a.sendall(b"BlueStacks".ljust(64, b"\0") + b"h264" + struct.pack(">II", 264, 480))
        assert read_stream_header(b) == (264, 480)
    finally:
        a.close()
        b.close()


def test_stream_header_with_session_packet():
    # scrcpy 5.0.1 sends a session marker (top bit set) between the codec tag and the size
    a, b = socket.socketpair()
    try:
        a.sendall(b"h264" + struct.pack(">III", 0x80000000, 270, 480))
        assert read_stream_header(b) == (270, 480)
    finally:
        a.close()
        b.close()


def test_act_sends_a_swipe_without_blocking():
    dev, v_dev, c_dev = fake_server(270, 480)
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


def test_tap_and_back():
    dev, v_dev, c_dev = fake_server()
    try:
        dev.tap(10, 20, hold=0.0)
        down, up = (struct.unpack(TOUCH, m) for m in read_messages(c_dev, 2))
        assert (down[1], up[1], down[3], down[4]) == (ACTION_DOWN, ACTION_UP, 10, 20)
        dev.back()
        keys = [struct.unpack(">BBiii", m) for m in read_messages(c_dev, 2, size=14)]
        assert [(k[1], k[2]) for k in keys] == [(0, 4), (1, 4)]
    finally:
        dev.close()


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
