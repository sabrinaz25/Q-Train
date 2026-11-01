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


GAME_ASPECT = 9 / 16  # width / height of the game's portrait viewport


def _edge_candidates(score: np.ndarray, k: int, min_score: float) -> list[int]:
    """Positions of the k strongest local maxima of an edge-strength profile."""
    order = np.argsort(score)[::-1]
    picked: list[int] = []
    for i in order:
        if score[i] < min_score or len(picked) >= k:
            break
        if all(abs(int(i) - j) > 2 for j in picked):
            picked.append(int(i))
    return picked


def locate_game(frame: np.ndarray, aspect: float = GAME_ASPECT, min_height_frac: float = 0.5,
                k: int = 12, jump: int = 25, min_side: float = 0.5) -> tuple[GameArea, float] | None:
    """Find the game's rectangle inside the emulator window, with no calibration.

    The game is drawn as a sharp-edged 9:16 rectangle inside the emulator's
    frame, while toolbars, title bars and ad panels move around it. Steps:
      1. Edge maps: where the colour jumps sharply between neighbouring pixels.
      2. Candidate lines: columns/rows that are a clear edge along much of
         their length (top, bottom, left, right borders are among them).
      3. For each top/bottom pair and each candidate vertical edge, try it as
         the left side and as the right side; the opposite side is *computed*
         from the 9:16 shape (width = height x 9/16), not searched for.
      4. Keep rectangles where at least three sides are clear edges, and
         return the one with the strongest sides overall.
    Computing the fourth side matters twice over: the game's own border fades
    wherever the game is as dark as the frame (so requiring all four sides
    fails), and a look-alike line a few pixels away, like an ad panel's
    divider, produces a rectangle whose computed opposite side lands on
    nothing and scores low. Returns (area, score), score = mean fraction of
    each side that is a clear edge; None if nothing qualifies.
    """
    img = frame.astype(np.int16)
    h, w = img.shape[:2]
    # boundary b sits between pixel b-1 and b
    v_edge = np.abs(np.diff(img, axis=1)).max(axis=2) > jump    # (h, w-1)
    h_edge = np.abs(np.diff(img, axis=0)).max(axis=2) > jump    # (h-1, w)
    v_cum = np.vstack([np.zeros((1, w - 1)), np.cumsum(v_edge, axis=0)])   # running count down each column
    h_cum = np.hstack([np.zeros((h - 1, 1)), np.cumsum(h_edge, axis=1)])   # running count along each row

    def v_side(b: int, top: int, bottom: int) -> float:
        if not 1 <= b <= w - 1:
            return 0.0
        return (v_cum[bottom, b - 1] - v_cum[top, b - 1]) / (bottom - top)

    def h_side(b: int, left: int, right: int) -> float:
        if not 1 <= b <= h - 1:
            return 0.0
        return (h_cum[b - 1, right] - h_cum[b - 1, left]) / (right - left)

    cols = [c + 1 for c in _edge_candidates(v_edge.mean(axis=0), k, 0.15)]
    rows = [r + 1 for r in _edge_candidates(h_edge.mean(axis=1), k, 0.2)]
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for top in rows:
        for bottom in rows:
            height = bottom - top
            if height < min_height_frac * h:
                continue
            width = round(height * aspect)
            for c in cols:
                for left in (c, c - width):
                    right = left + width
                    if left < 0 or right > w:
                        continue
                    # the computed side may be off by a pixel of rounding: take its best position
                    vl = v_side(left, top, bottom) if left == c else max(
                        v_side(left + d, top, bottom) for d in (-1, 0, 1))
                    vr = v_side(right, top, bottom) if right == c else max(
                        v_side(right + d, top, bottom) for d in (-1, 0, 1))
                    sides = sorted((vl, vr, h_side(top, left, right), h_side(bottom, left, right)))
                    if sides[1] < min_side:          # the three strongest sides must be clear edges
                        continue
                    score = float(np.mean(sides))
                    if best is None or score > best[0]:
                        best = (score, (left, top, width, height))
    if best is None:
        return None
    score, rect = best
    return GameArea.from_pixels(rect, frame.shape), round(score, 3)


def resolve_game_area(frame: np.ndarray, fallback: GameArea | None) -> tuple[GameArea | None, str]:
    """Auto-detected game area if found, else the calibrated one. Returns (area, source)."""
    found = locate_game(frame)
    if found is not None:
        return found[0], "auto"
    return fallback, "calibration" if fallback is not None else "whole window"


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
