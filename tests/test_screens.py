import cv2
import numpy as np
import pytest

from qtrain.screens import Calibration, ScreenClassifier, ScreenTemplate
from qtrain.vision import CANON_H, CANON_W, GameArea


def background(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 160, (CANON_H, CANON_W, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (5, 5), 0)


def with_label(text, org):
    img = background()
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return img


def template_from(canon, name, role, roi, tap=None):
    x, y, w, h = roi
    gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
    return ScreenTemplate(name, role, roi, tap, image=gray[y:y + h, x:x + w].copy())


MENU = with_label("MENU", (100, 580))
OVER = with_label("OVER", (110, 300))


def make_templates():
    return [template_from(MENU, "menu", "start", (90, 545, 140, 50), (180, 320)),
            template_from(OVER, "game_over", "crash", (100, 265, 140, 50), (180, 500))]


def test_recognises_each_screen():
    clf = ScreenClassifier(make_templates())
    assert clf.classify(MENU).name == "menu"
    assert clf.classify(OVER).name == "game_over"
    assert clf.classify(OVER).tap == (180, 500)


def test_tolerates_small_shifts():
    shifted = np.roll(MENU, (6, -8), axis=(0, 1))
    assert ScreenClassifier(make_templates()).classify(shifted).name == "menu"


def test_unknown_and_red_x_fallback():
    clf = ScreenClassifier(make_templates())
    plain = background()
    assert clf.classify(plain).role == "unknown"
    cv2.circle(plain, (320, 60), 16, (40, 40, 220), -1)
    cv2.line(plain, (313, 53), (327, 67), (255, 255, 255), 3)
    cv2.line(plain, (313, 67), (327, 53), (255, 255, 255), 3)
    det = clf.classify(plain)
    assert det.name == "red_x" and det.role == "dismiss"
    assert abs(det.tap[0] - 320) <= 3 and abs(det.tap[1] - 60) <= 3


def test_flat_template_never_matches():
    flat = np.full((CANON_H, CANON_W, 3), 128, np.uint8)
    t = template_from(flat, "flat", "dismiss", (10, 10, 40, 40))
    assert ScreenClassifier([t], detect_red_x=False).score(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY), t) == 0.0


def test_calibration_round_trip(tmp_path):
    cal = Calibration(GameArea(0.0, 0.07, 0.88, 0.93), make_templates())
    cal.save(tmp_path)
    loaded = Calibration.load(tmp_path)
    assert loaded.game_area == cal.game_area
    assert [(t.name, t.role, t.roi, t.tap) for t in loaded.templates] == \
           [(t.name, t.role, t.roi, t.tap) for t in cal.templates]
    np.testing.assert_array_equal(loaded.templates[0].image, cal.templates[0].image)


def test_rejects_bad_role():
    with pytest.raises(ValueError):
        ScreenTemplate("x", "celebrate", (0, 0, 1, 1))
