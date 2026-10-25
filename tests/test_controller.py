"""Drives GameController against a scripted fake game, no emulator needed."""
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
