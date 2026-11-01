from fake_app import FakeApp

from qtrain.compare import comparison_table, monkey_command
from qtrain.coverage import CoverageObserver


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Crashes:
    events = [(5.0, "crash", "java.lang.NullPointerException"),
              (9.0, "crash", "java.lang.NullPointerException"),
              (12.0, "anr", "Input dispatching timed out")]


def observe(app, script, crashes=None):
    """Sample twice per state in `script` (a screen name, or None for 'left the app')."""
    clock = Clock()
    obs = CoverageObserver(app, app, activity_every=0.0, clock=clock, crashes=crashes)
    for state in script:
        if state is None:
            app.foreground = "launcher"
        else:
            app.foreground, app.screen = app.package, state
        for _ in range(2):
            obs.sample()
            clock.t += 0.5
    return obs


def test_screens_count_once_they_hold_still():
    app = FakeApp()
    obs = observe(app, ["home", "list", "home", "detail"])
    assert len(obs.screens) == 3 and len(obs.activities) == 3


def test_a_screen_seen_for_one_sample_does_not_count():
    app = FakeApp()
    clock = Clock()
    obs = CoverageObserver(app, app, activity_every=0.0, clock=clock)
    for screen in ["home", "list", "form", "about", "about"]:   # only 'about' holds still
        app.screen = screen
        obs.sample()
        clock.t += 0.5
    assert len(obs.screens) == 1


def test_time_outside_the_app_adds_nothing():
    app = FakeApp()
    obs = observe(app, ["home", None, None, "home"], crashes=Crashes())
    s = obs.summary()
    assert s["unique_screens"] == 1 and s["time_in_app_pct"] == 50.0
    assert s["crashes"] == 2 and s["unique_crashes"] == 1 and s["anrs"] == 1


def test_comparison_table_and_monkey_command(monkeypatch):
    monkeypatch.setenv("QTRAIN_ADB", "adb")
    cov = {"unique_screens": 30, "screens_per_hour": 180.0, "unique_activities": 6, "crashes": 2,
           "unique_crashes": 1, "anrs": 0, "time_in_app_pct": 95.0}
    monkey = {**cov, "unique_screens": 12}
    table = comparison_table({"explorer": {"coverage": cov}, "monkey": {"coverage": monkey}})
    assert "| Unique screens | 30 | 12 |" in table and "| Crashes (unique) | 2 (1) | 2 (1) |" in table
    cmd = monkey_command("127.0.0.1:5555", "org.example", seed=3, throttle_ms=200)
    assert cmd[cmd.index("-p") + 1] == "org.example" and cmd[cmd.index("--throttle") + 1] == "200"
