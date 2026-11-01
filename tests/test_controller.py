"""Drives GameController against a scripted fake game, no emulator needed."""
import tempfile

import cv2
import numpy as np
import pytest

from qtrain.controller import ControllerConfig, GameController
from qtrain.screens import Calibration, ScreenTemplate
from qtrain.vision import CANON_H, CANON_W, GameArea

DEV_W, DEV_H = 1080, 1920


def background():
    rng = np.random.default_rng(0)
    return cv2.GaussianBlur(rng.integers(40, 160, (CANON_H, CANON_W, 3), dtype=np.uint8), (5, 5), 0)


def label(img, text, org):
    img = img.copy()
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    return img


def add_red_x(img, cx=320, cy=60):
    img = img.copy()
    cv2.circle(img, (cx, cy), 16, (40, 40, 220), -1)
    cv2.line(img, (cx - 7, cy - 7), (cx + 7, cy + 7), (255, 255, 255), 3)
    cv2.line(img, (cx - 7, cy + 7), (cx + 7, cy - 7), (255, 255, 255), 3)
    return img


BG = background()
SCREENS = {
    "menu": label(BG, "MENU", (100, 580)),
    "playing": label(BG, "||", (15, 45)),
    "save_me": label(BG, "SAVE", (110, 200)),
    "game_over": label(BG, "OVER", (110, 300)),
    "popup": add_red_x(label(BG, "GIFT!", (100, 400))),
    "mystery": label(BG, "???", (150, 450)),
}
SPECS = [("menu", "start", (90, 545, 140, 50), (180, 320)),
         ("playing", "playing", (5, 10, 60, 45), None),
         ("save_me", "crash", (100, 165, 140, 50), (60, 600)),
         ("game_over", "crash", (100, 265, 140, 50), (180, 500))]


def calibration():
    templates = []
    for name, role, (x, y, w, h), tap in SPECS:
        gray = cv2.cvtColor(SCREENS[name], cv2.COLOR_BGR2GRAY)
        templates.append(ScreenTemplate(name, role, (x, y, w, h), tap, image=gray[y:y + h, x:x + w].copy()))
    return Calibration(GameArea(0, 0, 1, 1), templates)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += max(dt, 0.0)


class FakeGame:
    """Device + frame source in one: a tiny state machine of the real game's screens."""
    width, height = DEV_W, DEV_H

    def __init__(self, start="menu", crash_after=20):
        self.screen = start
        self.crash_after = crash_after
        self.run_steps = 0
        self.actions, self.taps, self.backs = [], [], 0

    def grab(self):
        return SCREENS[self.screen]

    def act(self, action):
        self.actions.append(action)
        if self.screen == "playing":
            self.run_steps += 1
            if self.run_steps >= self.crash_after:
                self.screen = "save_me"

    def tap(self, x, y):
        cx, cy = x * CANON_W / DEV_W, y * CANON_H / DEV_H     # back to canonical coords
        self.taps.append((self.screen, round(cx), round(cy)))
        if self.screen == "menu":
            self.screen, self.run_steps = "playing", 0
        elif self.screen == "popup" and abs(cx - 320) < 12 and abs(cy - 60) < 12:
            self.screen = "menu"           # only a tap ON the X closes the popup
        elif self.screen == "save_me":
            self.screen = "game_over"
        elif self.screen == "game_over":
            self.screen = "menu"

    def back(self):
        self.backs += 1
        if self.screen == "mystery":
            self.screen = "menu"


def make(start="menu", crash_after=20, **cfg):
    cfg.setdefault("debug_dir", tempfile.mkdtemp())     # never write into the project's data/
    game, clock = FakeGame(start, crash_after), FakeClock()
    ctrl = GameController(game, game, calibration(), ControllerConfig(**cfg), clock=clock, sleep=clock.sleep)
    return ctrl, game, clock


def test_reset_from_menu_starts_a_run():
    ctrl, game, _ = make()
    obs, info = ctrl.reset()
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
    assert info["screens"] == ["menu", "playing"]
    assert game.screen == "playing" and game.taps == [("menu", 180, 320)]


def test_episode_ends_on_crash_screen():
    ctrl, game, _ = make(crash_after=10)
    ctrl.reset()
    rewards, done, steps = [], False, 0
    while not done:
        _, r, term, trunc, info = ctrl.step(1)
        rewards.append(r)
        done, steps = term or trunc, steps + 1
    assert term and not trunc and info["screen"] == "save_me"
    assert steps == 10
    assert rewards[-1] == ctrl.cfg.crash_penalty and all(r == ctrl.cfg.alive_reward for r in rewards[:-1])


def test_reset_after_crash_navigates_back_into_a_run():
    ctrl, game, _ = make(start="save_me")
    _, info = ctrl.reset()
    assert info["screens"] == ["save_me", "game_over", "menu", "playing"]


def test_popup_dismissed_with_red_x():
    ctrl, game, _ = make(start="popup")
    _, info = ctrl.reset()
    assert info["screens"][:2] == ["red_x", "menu"]
    assert ctrl.stats["tap:red_x"] == 1


def test_unknown_screen_presses_back():
    ctrl, game, _ = make(start="mystery", back_after_unknown=5)
    _, info = ctrl.reset()
    assert game.backs == 1 and info["screens"] == ["unknown", "menu", "playing"]


def test_reset_gives_up_with_a_useful_error(tmp_path):
    ctrl, game, _ = make(start="mystery", back_after_unknown=10**9, reset_timeout=5,
                         debug_dir=str(tmp_path))
    with pytest.raises(TimeoutError, match="unknown"):
        ctrl.reset()
    assert list(tmp_path.glob("reset-timeout-*.png"))


def test_lost_track_mid_run_truncates():
    ctrl, game, clock = make(lost_after=1.0, hz=10)
    ctrl.reset()
    game.screen = "mystery"
    truncated, steps = False, 0
    while not truncated:
        *_, term, truncated, _ = ctrl.step(0)
        steps += 1
        assert not term
    assert 10 <= steps <= 12      # ~1 s at 10 Hz


def test_steps_are_paced_at_fixed_rate():
    ctrl, game, clock = make(hz=15, crash_after=10**9)
    ctrl.reset()
    t0 = clock.t
    for _ in range(30):
        ctrl.step(0)
    assert clock.t - t0 == pytest.approx(2.0, abs=1e-6)
    assert ctrl.stats["late_ticks"] == 0


class FakeWindowGame(FakeGame):
    """FakeGame drawn inside an emulator-style window whose layout can shift (e.g. an ad panel appears)."""

    def __init__(self, *a, left=20, **kw):
        super().__init__(*a, **kw)
        self.left = left

    def grab(self):
        win = np.zeros((760, 760, 3), np.uint8)
        win[:] = (66, 38, 35)
        win[82:82 + CANON_H, self.left:self.left + CANON_W] = SCREENS[self.screen]
        return win


def test_layout_shift_is_recovered_by_relocating_not_back():
    game, clock = FakeWindowGame("menu", left=20), FakeClock()
    ctrl = GameController(game, game, calibration(), ControllerConfig(back_after_unknown=4),
                          clock=clock, sleep=clock.sleep)
    assert round(ctrl.area.x * 760) == 20               # found the game at startup
    before = ctrl.stats["relocated"]
    game.left = 380                                      # an ad panel pushes the game right
    _, info = ctrl.reset()
    assert info["screens"] == ["unknown", "menu", "playing"]
    assert ctrl.stats["relocated"] == before + 1 and game.backs == 0
    assert round(ctrl.area.x * 760) == 380


class FullScreenGame(FakeGame):
    """Frames straight from the device (like scrcpy capture): nothing to locate."""
    full_screen = True


def test_device_frames_skip_screen_location():
    game, clock = FullScreenGame("menu"), FakeClock()
    ctrl = GameController(game, game, calibration(), ControllerConfig(), clock=clock, sleep=clock.sleep)
    assert ctrl.area is None and ctrl.relocate() is False
    assert ctrl.stats["relocate_failed"] == 0
    _, info = ctrl.reset()
    assert info["screens"] == ["menu", "playing"]


def test_stuck_reset_restarts_the_app_once():
    game, clock = FakeGame("mystery"), FakeClock()     # a screen that Back can't leave...
    game.back = lambda: setattr(game, "backs", game.backs + 1)
    restarts = []

    def restart():                                      # ...but relaunching the app can
        restarts.append(clock.t)
        game.screen = "menu"

    ctrl = GameController(game, game, calibration(),
                          ControllerConfig(restart_after=20, reset_timeout=90, back_after_unknown=8,
                                           debug_dir=tempfile.mkdtemp()),
                          clock=clock, sleep=clock.sleep, restart_app=restart)
    _, info = ctrl.reset()
    assert len(restarts) == 1 and restarts[0] >= 20
    assert info["screens"] == ["unknown", "RESTART", "menu", "playing"]
    assert ctrl.stats["app_restarts"] == 1 and game.backs >= 1      # Back was tried first


def test_reset_gives_up_after_one_restart(tmp_path):
    game, clock = FakeGame("mystery"), FakeClock()
    game.back = lambda: None
    calls = []
    ctrl = GameController(game, game, calibration(),
                          ControllerConfig(restart_after=10, reset_timeout=30, debug_dir=str(tmp_path)),
                          clock=clock, sleep=clock.sleep, restart_app=lambda: calls.append(1))
    with pytest.raises(TimeoutError):
        ctrl.reset()
    assert len(calls) == 1                                          # never restarts in a loop


def test_screen_with_back_key_is_left_with_back_not_a_tap():
    cal = calibration()
    for t in cal.templates:
        if t.name == "save_me":
            t.tap, t.key = None, "back"
    game, clock = FakeGame("save_me"), FakeClock()

    def back():
        game.backs += 1
        if game.screen == "save_me":          # like the real game: Back skips "Continue?" for free
            game.screen = "game_over"
    game.back = back
    ctrl = GameController(game, game, cal, ControllerConfig(), clock=clock, sleep=clock.sleep)
    _, info = ctrl.reset()
    assert info["screens"] == ["save_me", "game_over", "menu", "playing"]
    assert ctrl.stats["back:save_me"] == 1 and not any(s == "save_me" for s, *_ in game.taps)


SCREENS["over_popup"] = label(SCREENS["game_over"], "AD", (40, 520))   # popup drawn over the score screen


class PopupOverScoreGame(FakeGame):
    """A popup sits on top of the score screen: its template still matches, but taps hit the popup."""

    def tap(self, x, y):
        if self.screen == "over_popup":
            self.taps.append((self.screen, x, y))            # swallowed by the popup
            return
        super().tap(x, y)

    def back(self):
        self.backs += 1
        if self.screen == "over_popup":
            self.screen = "game_over"


def test_taps_that_stop_working_escalate_to_back():
    game, clock = PopupOverScoreGame("over_popup"), FakeClock()
    ctrl = GameController(game, game, calibration(), ControllerConfig(debug_dir=tempfile.mkdtemp()),
                          clock=clock, sleep=clock.sleep)
    _, info = ctrl.reset()
    assert ctrl.stats["stuck"] == 1 and ctrl.stats["stuck:back"] == 1
    assert sum(1 for s, *_ in game.taps if s == "over_popup") >= ctrl.cfg.stuck_taps
    assert game.screen == "playing" and clock.t < 15        # recovered in seconds, not a 45 s restart


def test_slow_crash_animation_still_counts_as_a_crash():
    ctrl, game, clock = make(crash_after=10**9, hz=10)
    ctrl.reset()
    game.screen = "mystery"                                  # HUD gone: crash animation playing
    for _ in range(30):                                      # 3 s at 10 Hz, under lost_after (6 s)
        *_, term, trunc, _ = ctrl.step(0)
        assert not term and not trunc
    game.screen = "save_me"
    *_, term, trunc, info = ctrl.step(0)
    assert term and not trunc
    assert info["crash_delay_s"] == pytest.approx(3.0, abs=0.15)
    assert ctrl.crash_delays == [info["crash_delay_s"]]


def test_popup_back_cannot_close_triggers_early_restart():
    game, clock = PopupOverScoreGame("over_popup"), FakeClock()
    game.back = lambda: setattr(game, "backs", game.backs + 1)      # Back does nothing to this popup

    def restart():
        game.screen = "menu"
    ctrl = GameController(game, game, calibration(), ControllerConfig(debug_dir=tempfile.mkdtemp()),
                          clock=clock, sleep=clock.sleep, restart_app=restart)
    _, info = ctrl.reset()
    assert ctrl.stats["stuck"] == 2 and ctrl.stats["app_restarts"] == 1
    assert clock.t < ctrl.cfg.restart_after                          # didn't wait the full 45 s
    assert "RESTART" in info["screens"] and game.screen == "playing"


def test_slow_screen_is_not_mistaken_for_stuck():
    """A screen that takes a few taps' worth of time to change must not trigger recovery."""
    game, clock = FakeGame("game_over"), FakeClock()
    real_tap = FakeGame.tap

    def slow_tap(x, y):
        if game.screen != "game_over":
            return real_tap(game, x, y)
        game.taps.append((game.screen, x, y))
        if len(game.taps) >= 4:                                      # responds on the 4th tap (~2.4 s)
            game.taps.pop()
            real_tap(game, x, y)
    game.tap = slow_tap
    ctrl = GameController(game, game, calibration(), ControllerConfig(debug_dir=tempfile.mkdtemp()),
                          clock=clock, sleep=clock.sleep)
    ctrl.reset()
    assert ctrl.stats["stuck"] == 0 and game.backs == 0


def test_popup_mid_run_is_dismissed_immediately():
    ctrl, game, clock = make(crash_after=10**9, hz=10)
    ctrl.reset()
    game.screen = "popup"                                     # red-X popup appears during the run
    *_, term, trunc, info = ctrl.step(0)
    assert info["screen"] == "red_x" and not term and not trunc
    assert ctrl.stats["mid_run_dismiss"] == 1 and game.screen == "menu"   # tapped its X
