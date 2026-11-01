import itertools

import cv2
import numpy as np
from fake_app import SCREENS, draw

from qtrain.fingerprint import ScreenSet, distance, fingerprint


def test_clock_and_cursor_do_not_make_a_new_screen():
    a = fingerprint(draw("form", clock="12:00"))
    b = fingerprint(draw("form", clock="09:47", cursor=True))
    assert distance(a, b) < 0.03


def test_every_layout_is_its_own_screen():
    names = [n for n in SCREENS if n != "crash"]
    prints = {n: fingerprint(draw(n)) for n in names}
    for x, y in itertools.combinations(names, 2):
        assert distance(prints[x], prints[y]) > 0.15, (x, y, distance(prints[x], prints[y]))


def test_small_noise_and_scaling_keep_the_same_screen():
    img = draw("list")
    noisy = np.clip(img.astype(np.int16) + np.random.default_rng(0).normal(0, 6, img.shape), 0, 255)
    smaller = cv2.resize(img, (270, 480), interpolation=cv2.INTER_AREA)   # a different stream size
    base = fingerprint(img)
    assert distance(base, fingerprint(noisy.astype(np.uint8))) < 0.03
    assert distance(base, fingerprint(smaller)) < 0.03


def test_screen_set_counts_distinct_screens():
    s = ScreenSet()
    for name in ["home", "list", "home", "detail", "list", "home"]:
        s.add(fingerprint(draw(name, clock=f"12:{len(s):02d}")))
    assert len(s) == 3 and sorted(s.visits) == [1, 2, 3]
