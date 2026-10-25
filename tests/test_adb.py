import pytest

from qtrain.adb import SwipeGeometry, parse_display_size


def test_display_size_prefers_current_rotation():
    dumpsys = "Display: mDisplayId=0\n  init=1080x1920 420dpi cur=1920x1080 app=1920x1032"
    assert parse_display_size(dumpsys, "Physical size: 1080x1920") == (1920, 1080)


def test_display_size_falls_back_to_wm_size_override():
    wm = "Physical size: 1080x1920\nOverride size: 720x1280"
    assert parse_display_size("", wm) == (720, 1280)


def test_display_size_unparseable():
    with pytest.raises(ValueError):
        parse_display_size("nothing", "here")


@pytest.mark.parametrize("action,check", [
    ("left", lambda x1, y1, x2, y2: x2 < x1 and y1 == y2),
    ("right", lambda x1, y1, x2, y2: x2 > x1 and y1 == y2),
    ("up", lambda x1, y1, x2, y2: y2 < y1 and x1 == x2),     # jump
    ("down", lambda x1, y1, x2, y2: y2 > y1 and x1 == x2),   # roll
])
def test_swipe_directions(action, check):
    g = SwipeGeometry(1080, 1920)
    x1, y1, x2, y2 = g.swipe(action)
    assert check(x1, y1, x2, y2)
    assert all(0 <= v <= 1920 for v in (x1, y1, x2, y2))


def test_noop_sends_nothing():
    assert SwipeGeometry(1080, 1920).command("noop") is None


def test_command_prefix():
    g = SwipeGeometry(1000, 2000, frac=0.2, duration_ms=30)
    assert g.command("right") == "input swipe 400 1600 600 1600 30"
    assert g.command("right", "cmd input").startswith("cmd input swipe ")


def test_swipes_stay_below_the_continue_dialog():
    # the "Continue?" dialog (and its key button) spans roughly 29%-71% of the screen height
    g = SwipeGeometry(270, 480)
    for action in ("left", "right", "up", "down"):
        x1, y1, x2, y2 = g.swipe(action)
        assert min(y1, y2) > 0.71 * 480 and max(y1, y2) < 480
