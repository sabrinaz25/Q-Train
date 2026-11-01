from qtrain.waits import wait_for_change, wait_until_stable


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def reader(values_by_time, clock):
    """read() returning the value scheduled for the current fake time."""
    def read():
        return [v for t, v in values_by_time if t <= clock.t + 1e-9][-1]
    return read


def test_change_returns_as_soon_as_it_happens():
    c = Clock()
    res = wait_for_change(reader([(0, "a"), (0.2, "b")], c), lambda v: v != "a",
                          timeout=0.8, poll=0.05, clock=c, sleep=c.sleep)
    assert res.done and res.value == "b" and abs(res.seconds - 0.2) < 1e-6


def test_change_times_out_at_the_old_fixed_wait():
    c = Clock()
    res = wait_for_change(lambda: "a", lambda v: v != "a", timeout=0.8, poll=0.05, clock=c, sleep=c.sleep)
    assert not res.done and 0.8 <= res.seconds < 0.86


def test_stable_waits_out_an_animation():
    c = Clock()
    frames = [(0, 1), (0.1, 2), (0.2, 3), (0.3, 4)]          # changing until 0.3 s
    res = wait_until_stable(reader(frames, c), lambda a, b: a == b, stable_for=0.2, timeout=2,
                            poll=0.05, clock=c, sleep=c.sleep)
    assert res.done and res.value == 4 and 0.5 <= res.seconds < 0.56


def test_stable_gives_up_on_a_screen_that_never_settles():
    c = Clock()
    res = wait_until_stable(lambda: c.t, lambda a, b: a == b, stable_for=0.2, timeout=1,
                            poll=0.05, clock=c, sleep=c.sleep)
    assert not res.done and res.seconds >= 1
