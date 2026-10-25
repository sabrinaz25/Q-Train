import cv2
import numpy as np
import pytest

from qtrain.vision import CANON_H, CANON_W, GameArea, canonicalize, find_red_xs, to_obs, x_structure

RED = (35, 35, 205)      # BGR
WHITE = (255, 255, 255)


def blank():
    return np.full((CANON_H, CANON_W, 3), (120, 90, 70), np.uint8)  # dull blue-grey backdrop


def draw_close_button(img, cx, cy, r=16, outline=True, ribbon=False):
    """A close button drawn like the game's: red disc, white rim, and a thick
    white X that sits a little above centre with its lower arms spread wider."""
    if ribbon:  # red banner running into the button, as on the Daily Login popup
        top_left, bottom_right = (cx - 4 * r, cy - int(r * 0.6)), (cx + r // 2, cy + int(r * 0.6))
        cv2.rectangle(img, top_left, bottom_right, (30, 30, 200), -1)
    cv2.circle(img, (cx, cy), r, RED, -1, cv2.LINE_AA)
    if outline:
        cv2.circle(img, (cx, cy), r, WHITE, max(2, r // 8), cv2.LINE_AA)
    t, xc, yc = max(2, int(r * 0.2)), cx, cy - int(r * 0.12)
    cv2.line(img, (xc - int(r * .36), yc - int(r * .40)), (xc + int(r * .50), yc + int(r * .45)), WHITE, t,
             cv2.LINE_AA)
    cv2.line(img, (xc + int(r * .36), yc - int(r * .40)), (xc - int(r * .50), yc + int(r * .45)), WHITE, t,
             cv2.LINE_AA)


def draw_badge(img, cx, cy, text, r=13, scale=0.55, thick=2):
    cv2.circle(img, (cx, cy), r, RED, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, WHITE, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, thick,
                cv2.LINE_AA)


@pytest.mark.parametrize("r", [10, 13, 16, 20, 26, 32])
@pytest.mark.parametrize("style", [{}, {"outline": False}, {"ribbon": True}])
def test_detects_close_button(r, style):
    img = blank()
    draw_close_button(img, 250, 200, r=r, **style)
    xs = find_red_xs(img)
    assert len(xs) == 1
    assert abs(xs[0].cx - 250) <= 3 and abs(xs[0].cy - 200) <= 3


@pytest.mark.parametrize("digit", list("0123456789"))
@pytest.mark.parametrize("size", [(11, 0.45, 1), (13, 0.55, 2), (13, 0.5, 1), (16, 0.7, 2), (18, 0.8, 2),
                                  (22, 1.0, 3)])
def test_ignores_notification_badges(digit, size):
    r, scale, thick = size
    img = blank()
    draw_badge(img, 100, 300, digit, r=r, scale=scale, thick=thick)
    assert find_red_xs(img) == []


def test_finds_x_among_badges():
    img = blank()
    for i, d in enumerate("1238"):
        draw_badge(img, 40 + 70 * i, 500, d)
    draw_close_button(img, 320, 60)
    xs = find_red_xs(img)
    assert len(xs) == 1 and abs(xs[0].cx - 320) <= 3 and abs(xs[0].cy - 60) <= 3


def _white_in_button(img, cx=50, cy=50, r=14):
    hsv = cv2.cvtColor(img[cy - r:cy + r, cx - r:cx + r], cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, 185), (180, 80, 255))


def test_x_structure_separates_x_from_digit():
    x_img, one_img = blank(), blank()
    draw_close_button(x_img, 50, 50)
    draw_badge(one_img, 50, 50, "1", r=16, scale=0.7)
    center, _, gap = x_structure(_white_in_button(x_img))
    _, arm, _ = x_structure(_white_in_button(one_img))
    assert center > 0.7 and gap < 0.25    # X: white centre, empty gaps
    assert arm < 0.25                     # "1": no diagonal arms


def test_ignores_large_red_banner_and_plain_disc():
    img = blank()
    cv2.rectangle(img, (20, 20), (340, 140), RED, -1)           # big red panel
    cv2.circle(img, (180, 400), 16, RED, -1)                     # disc with no glyph
    assert find_red_xs(img) == []


def test_game_area_crop_and_canonical_size():
    frame = np.zeros((1165, 693, 3), np.uint8)
    frame[82:1165, 0:610] = 200                                  # "game" region
    area = GameArea.from_pixels((0, 82, 610, 1083), frame.shape)
    canon = canonicalize(frame, area)
    assert canon.shape == (CANON_H, CANON_W, 3)
    assert canon.min() == 200                                    # nothing outside the game leaked in
    assert to_obs(canon).shape == (84, 84)
