import struct

import numpy as np
import pytest

from qtrain.capture import AdbScreencap, match_window

WINDOWS = [(1, "Visual Studio Code"), (2, "BlueStacks App Player 1"), (3, "BlueStacks App Player"),
           (4, "BlueStacks Multi Instance Manager")]


def test_exact_title_wins():
    assert match_window(WINDOWS, "bluestacks app player") == 3


def test_shortest_substring_match():
    assert match_window(WINDOWS, "BlueStacks") == 3
    assert match_window(WINDOWS, "player 1") == 2


def test_no_match_lists_titles():
    with pytest.raises(LookupError, match="Visual Studio Code"):
        match_window(WINDOWS, "emulator")


@pytest.mark.parametrize("header_len", [12, 16])
def test_screencap_decode(header_len):
    w, h = 3, 2
    rgba = np.arange(w * h * 4, dtype=np.uint8).reshape(h, w, 4)
    header = struct.pack("<III", w, h, 1) + b"\0" * (header_len - 12)
    bgr = AdbScreencap.decode(header + rgba.tobytes())
    assert bgr.shape == (h, w, 3)
    np.testing.assert_array_equal(bgr[..., 0], rgba[..., 2])  # B
    np.testing.assert_array_equal(bgr[..., 2], rgba[..., 0])  # R


def test_screencap_decode_rejects_garbage():
    with pytest.raises(ValueError):
        AdbScreencap.decode(struct.pack("<III", 10, 10, 1) + b"\0" * 7)
