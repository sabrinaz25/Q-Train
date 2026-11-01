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


class TreeApp(FakeApp):
    """FakeApp that also answers `ui_dump` like Android: its real buttons, in display pixels."""

    def __init__(self, *a, scale=4, empty=False, **kw):
        super().__init__(*a, **kw)
        self.scale, self.empty, self.dumps = scale, empty, 0

    def ui_dump(self):
        self.dumps += 1
        s, H_, W_ = self.scale, self.height * self.scale, self.width * self.scale
        nodes = [] if self.empty or self.foreground != self.package else [
            f'<node package="{self.package}" clickable="true" enabled="true" '
            f'bounds="[{bx * s},{by * s}][{(bx + bw) * s},{(by + bh) * s}]"/>'
            for (bx, by, bw, bh), _ in SCREENS[self.screen][3]]
        return (f'<?xml version="1.0"?><hierarchy><node package="{self.package}" '
                f'bounds="[0,0][{W_},{H_}]">' + "".join(nodes) + "</node></hierarchy>")


def test_tree_targets_hit_real_buttons():
    app = TreeApp()
    ex, _ = explorer(app, scroll=False)
    dead_sources = []
    for _ in range(300):
        r = ex.step()
        if r.get("changed") is False:
            dead_sources.append(ex.nodes[r["screen"]].source)
    assert ex.stats["screens:a11y"] >= 5
    assert app.dumps == len(ex.nodes)                       # the tree is read once per new screen
    # taps from the tree always hit a real button; only pixel guesses (on the one screen
    # with no buttons at all, where the tree is empty) can be dead
    assert "a11y" not in dead_sources


def test_empty_tree_falls_back_to_pixels():
    app = TreeApp(empty=True)
    ex, _ = explorer(app)
    ex.step()
    assert ex.nodes[ex.root].source == "pixels" and len(ex.nodes[ex.root].actions) > 1


def test_never_presses_back_on_the_starting_screen():
    app = TreeApp(empty=True)
    ex, _ = explorer(app, targets="a11y", scroll=False)     # no targets anywhere: always exhausted
    for _ in range(30):
        ex.step()
        assert app.foreground == app.package                 # never exits the app
    assert ex.stats["random_taps"] >= 1 and ex.stats["relaunches"] == 0


def test_learns_where_back_exits_the_app():
    app = TreeApp()
    ex, _ = explorer(app)
    sid = ex.current()
    ex._backed_from = sid
    app.foreground = "launcher"                              # that Back left the app
    ex.step()
    assert sid in ex.back_exits and ex.stats["learned_back_exits"] == 1


def test_screen_wrongly_cached_as_outside_is_corrected():
    app = TreeApp()
    app.foreground = "launcher"                  # app still starting: Android reports the launcher
    ex, _ = explorer(app)
    app.foreground = app.package                 # ...but frames already show the app's screen
    sid = ex.current()
    ex.nodes[sid].outside = True                 # what the old cache would have stuck with
    ex.step()
    assert ex.stats["outside_corrected"] == 1 and ex.stats["back:outside"] == 0
    assert ex.stats["relaunches"] == 0
