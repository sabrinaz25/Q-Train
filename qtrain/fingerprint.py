"""Telling screens apart from pixels alone, for any app.

A screen's *fingerprint* is a 512-bit map of where its layout has edges: blur the
frame, shrink it to a 17x17 grid of grey cells, and mark every pair of
neighbouring cells (left-right and up-down) whose brightness differs. Blurring and
shrinking keep the big shapes (toolbars, list rows, buttons, images) and drop what
changes on an unchanged screen: the clock, a blinking cursor, small animations,
video noise. The status bar is cropped off first because its clock and icons
change on their own.

Two fingerprints within `threshold` (fraction of differing bits) are the same
screen. `ScreenSet` keeps one representative per screen seen, which is what
"unique screens reached" counts, for the explorer and the monkey baseline alike.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

GRID = 16
BITS = 2 * GRID * GRID


@dataclass(frozen=True)
class Crop:
    """Fractions of the frame to ignore: Android's status bar on top, a nav bar at the bottom."""
    top: float = 0.04
    bottom: float = 0.0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        h = frame.shape[0]
        return frame[int(h * self.top): h - int(h * self.bottom)]


STATUS_BAR = Crop()


def fingerprint(frame: np.ndarray, crop: Crop = STATUS_BAR) -> np.ndarray:
    """512-bit edge-layout map of a BGR (or grey) frame, packed into 64 bytes."""
    img = crop.apply(frame)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.GaussianBlur(cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA), (5, 5), 0)
    c = cv2.resize(small, (GRID + 1, GRID + 1), interpolation=cv2.INTER_AREA).astype(np.int16)
    # a small margin keeps flat areas (equal neighbours) from flipping on noise
    horiz = np.abs(c[:GRID, 1:] - c[:GRID, :-1]) > 2
    vert = np.abs(c[1:, :GRID] - c[:-1, :GRID]) > 2
    return np.packbits(np.concatenate([horiz.ravel(), vert.ravel()]))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of differing bits, 0 (identical) to 1."""
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum()) / BITS


@dataclass
class ScreenSet:
    """Distinct screens seen so far; a new fingerprint joins the nearest one within `threshold`."""
    threshold: float = 0.10
    prints: list[np.ndarray] = field(default_factory=list)
    visits: list[int] = field(default_factory=list)

    def match(self, fp: np.ndarray) -> int | None:
        """Id of the closest known screen within the threshold, or None."""
        if not self.prints:
            return None
        d = np.unpackbits(np.bitwise_xor(np.stack(self.prints), fp), axis=1).sum(axis=1) / BITS
        i = int(d.argmin())
        return i if d[i] <= self.threshold else None

    def add(self, fp: np.ndarray) -> tuple[int, bool]:
        """(screen id, True if it was new)."""
        i = self.match(fp)
        if i is not None:
            self.visits[i] += 1
            return i, False
        self.prints.append(fp)
        self.visits.append(1)
        return len(self.prints) - 1, True

    def __len__(self) -> int:
        return len(self.prints)
