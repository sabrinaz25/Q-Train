"""Turning raw emulator frames into something the rest of the system can reason about.

1. `GameArea` crops the BlueStacks window down to just the game (dropping the
   title bar and side toolbar) and resizes it to a fixed *canonical* size.
   Everything downstream (templates, tap points, popup detection) works in
   canonical coordinates, so resizing or moving the emulator window doesn't
   break calibration.

2. `find_red_xs` finds red close buttons ("X") on popups without any stored
   game artwork: it looks for round red buttons and then checks the white mark
   inside has an X's structure. That second check matters because the main
   menu is full of red notification badges with white digits ("1", "2", "3")
   that a colour-only detector would happily tap.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CANON_W, CANON_H = 360, 640  # portrait, same 9:16 aspect as the game


@dataclass(frozen=True)
class GameArea:
    """Game region inside the captured window, as fractions of the frame size."""
    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_pixels(cls, rect: tuple[int, int, int, int], frame_shape) -> GameArea:
        fh, fw = frame_shape[:2]
        x, y, w, h = rect
        return cls(x / fw, y / fh, w / fw, h / fh)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        fh, fw = frame.shape[:2]
        x0, y0 = int(round(self.x * fw)), int(round(self.y * fh))
        x1, y1 = int(round((self.x + self.w) * fw)), int(round((self.y + self.h) * fh))
        return frame[y0:y1, x0:x1]


def canonicalize(frame: np.ndarray, area: GameArea | None) -> np.ndarray:
    """Crop to the game and resize to CANON_W x CANON_H (BGR)."""
    game = area.crop(frame) if area else frame
    return cv2.resize(np.ascontiguousarray(game), (CANON_W, CANON_H), interpolation=cv2.INTER_AREA)


def to_obs(canon: np.ndarray, size: int = 84) -> np.ndarray:
    """Agent observation: small grayscale image, uint8."""
    gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------ red X buttons
@dataclass(frozen=True)
class RedX:
    cx: int
    cy: int
    w: int
    h: int
    score: float  # how X-like the button is, 0..1

    @property
    def center(self) -> tuple[int, int]:
        return self.cx, self.cy


def _white_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, 185), (180, 80, 255))


def _red_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, (0, 110, 100), (10, 255, 255))
    hi = cv2.inRange(hsv, (168, 110, 100), (180, 255, 255))
    return lo | hi


# Where the parts of the game's close-button X sit, as fractions of the red
# button's bounding box (row0, row1, col0, col1). Measured from real popups:
# the X sits slightly above centre and its lower arms spread wider.
_CENTER = (0.34, 0.54, 0.38, 0.62)
_ARMS = [(0.18, 0.36, 0.22, 0.42), (0.18, 0.36, 0.58, 0.78),    # upper-left, upper-right
         (0.52, 0.72, 0.16, 0.40), (0.52, 0.72, 0.60, 0.84)]    # lower-left, lower-right
_GAPS = [(0.10, 0.28, 0.42, 0.58), (0.62, 0.80, 0.42, 0.58),    # top, bottom
         (0.36, 0.52, 0.08, 0.26), (0.36, 0.52, 0.74, 0.92)]    # left, right


def _fill(mask: np.ndarray, region: tuple[float, float, float, float]) -> float:
    h, w = mask.shape
    r0, r1, c0, c1 = region
    sub = mask[int(r0 * h):max(int(r1 * h), int(r0 * h) + 1), int(c0 * w):max(int(c1 * w), int(c0 * w) + 1)]
    return float((sub > 0).mean())


def x_structure(white: np.ndarray) -> tuple[float, float, float]:
    """(centre, weakest arm, fullest gap) white fractions inside a button, each 0..1.

    An X is white in the middle and along all four arms, and *red* in the four
    gaps between the arms (above, below, left and right of centre). Digits on
    notification badges fail: they leave arms empty (1, 7) and/or put ink in a
    gap (the bars and loops of 0, 2, 3, 5, 6, 8, 9).
    """
    return (_fill(white, _CENTER), min(_fill(white, a) for a in _ARMS),
            max(_fill(white, g) for g in _GAPS))


def find_red_xs(canon: np.ndarray, min_size: int = 14, max_size: int = 90, min_center: float = 0.7,
                min_arm: float = 0.25, max_gap: float = 0.25) -> list[RedX]:
    """Red close buttons with a white X, in canonical coordinates, best first."""
    red = cv2.morphologyEx(_red_mask(canon), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    white = _white_mask(canon)

    found: list[RedX] = []
    # red alone: the button's white outline keeps it separate from red backgrounds
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (min_size <= w <= max_size and min_size <= h <= max_size and 0.75 <= w / h <= 1.33):
            continue
        # a filled disc or rounded square (the X's holes don't count against it)
        if cv2.contourArea(cv2.convexHull(c)) / (np.pi * w * h / 4) < 0.75:
            continue
        center, arm, gap = x_structure(white[y:y + h, x:x + w])
        if center >= min_center and arm >= min_arm and gap <= max_gap:
            score = round((center + min(arm / 0.5, 1.0) + (1 - gap)) / 3, 3)
            found.append(RedX(x + w // 2, y + h // 2, w, h, score))
    return sorted(found, key=lambda r: -r.score)
