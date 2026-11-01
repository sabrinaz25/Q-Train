from fake_app import SCREENS, FakeApp, draw

from qtrain.explorer import Explorer, ExplorerConfig, find_targets


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def explorer(app, **cfg):
    c = Clock()
    return Explorer(app, app, app, ExplorerConfig(**cfg), clock=c, sleep=c.sleep), c


def test_targets_cover_every_button():
    for name, (_, _, _, buttons) in SCREENS.items():
        if name == "crash":
            continue
        targets = find_targets(draw(name))
        for (bx, by, bw, bh), _ in buttons:
            cx, cy = bx + bw // 2, by + bh // 2
            assert any(t.x <= cx < t.x + t.w and t.y <= cy < t.y + t.h for t in targets), (name, cx, cy)


def test_reaches_every_screen_and_survives_a_crash():
    app = FakeApp()
    ex, clock = explorer(app)
    result = ex.run(seconds=120)
    # all 6 app screens fingerprinted, plus the launcher it crashed out to
    assert len(ex.screens) == len(SCREENS) - 1 + 1
    assert app.crashes >= 1 and ex.stats["relaunches"] >= 1        # crashed, came back
    assert result["screens_fully_explored"] >= 5


def test_dead_targets_are_not_retried():
    app = FakeApp()
    ex, _ = explorer(app)
    for _ in range(60):
        ex.step()
    # each (screen, target) is tried once; repeats only happen as deliberate navigation
    taps_without_navigation = ex.stats["taps"] + ex.stats["scrolls"] - ex.stats["navigate"]
    total_actions = sum(len(n.actions) for n in ex.nodes.values())
    assert taps_without_navigation <= total_actions


def test_leaving_the_app_presses_back_then_relaunches():
    app = FakeApp()
    app.foreground = "com.android.launcher"
    ex, _ = explorer(app)
    ex.step()
    ex.step()
    assert ex.stats["back:outside"] == 1 and ex.stats["relaunches"] == 1
    assert app.foreground == app.package
