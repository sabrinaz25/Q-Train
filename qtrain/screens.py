"""Recognising which screen the game is on.

Each known screen is described by a small *template*: a patch of the
canonical game image that only appears on that screen (the "Tap to Play"
label, the in-run pause button, ...), captured from YOUR emulator by
`python -m qtrain.calibrate`. Templates live in assets/ (gitignored), so no
game artwork is ever committed.

Every template has a role, which tells the controller what to do:
    start    - main menu: tap to begin a run
    playing  - a run is in progress (the agent is in control)
    crash    - the run ended ("Save me?" prompt, score screen): episode over
    dismiss  - a popup/notification to tap away

Anything that matches no template is checked for a red close button (X);
failing that it's "unknown" and the controller waits / presses Back.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .vision import GameArea, find_red_xs

ROLES = ("start", "playing", "crash", "dismiss")
CALIBRATION_FILE = "calibration.json"


@dataclass
class ScreenTemplate:
    name: str
    role: str
    roi: tuple[int, int, int, int]          # x, y, w, h in canonical coordinates
    tap: tuple[int, int] | None = None      # where to tap to leave this screen (canonical)
    threshold: float = 0.8
    image: np.ndarray | None = field(default=None, repr=False)  # grayscale patch

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")
        self.roi = tuple(int(v) for v in self.roi)
        self.tap = tuple(int(v) for v in self.tap) if self.tap is not None else None


@dataclass
class Calibration:
    game_area: GameArea
    templates: list[ScreenTemplate]

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        (d / "templates").mkdir(parents=True, exist_ok=True)
        meta = {"game_area": asdict(self.game_area), "templates": []}
        for t in self.templates:
            cv2.imwrite(str(d / "templates" / f"{t.name}.png"), t.image)
            meta["templates"].append({"name": t.name, "role": t.role, "roi": list(t.roi),
                                      "tap": list(t.tap) if t.tap else None, "threshold": t.threshold})
        path = d / CALIBRATION_FILE
        path.write_text(json.dumps(meta, indent=2))
        return path

    @classmethod
    def load(cls, directory: str | Path) -> Calibration:
        d = Path(directory)
        path = d / CALIBRATION_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `python -m qtrain.calibrate` first")
        meta = json.loads(path.read_text())
        templates = []
        for m in meta["templates"]:
            img = cv2.imread(str(d / "templates" / f"{m['name']}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"template image for {m['name']!r} is missing")
            templates.append(ScreenTemplate(m["name"], m["role"], tuple(m["roi"]),
                                            tuple(m["tap"]) if m["tap"] else None,
                                            m.get("threshold", 0.8), img))
        return cls(GameArea(**meta["game_area"]), templates)


def is_flat(patch: np.ndarray, min_std: float = 4.0) -> bool:
    """True if a patch has too little detail to be recognised reliably."""
    return patch is None or patch.size == 0 or float(patch.std()) < min_std


@dataclass(frozen=True)
class Detection:
    name: str             # template name, "red_x" or "unknown"
    role: str             # one of ROLES, or "unknown"
    score: float
    tap: tuple[int, int] | None = None


UNKNOWN = Detection("unknown", "unknown", 0.0)


class ScreenClassifier:
    """Matches each template inside a search window around where it was calibrated.

    Searching only near the expected spot (instead of the whole frame) keeps
    each check to well under a millisecond and avoids matching look-alike
    patches elsewhere on screen.
    """

    def __init__(self, templates: list[ScreenTemplate], margin: int = 24, detect_red_x: bool = True):
        self.templates = templates
        self.margin = margin
        self.detect_red_x = detect_red_x

    def score(self, gray: np.ndarray, t: ScreenTemplate) -> float:
        if is_flat(t.image):
            return 0.0  # a featureless patch "matches" anything; never trust it
        x, y, w, h = t.roi
        m = self.margin
        H, W = gray.shape
        x0, y0 = max(x - m, 0), max(y - m, 0)
        x1, y1 = min(x + w + m, W), min(y + h + m, H)
        window = gray[y0:y1, x0:x1]
        th, tw = t.image.shape
        if window.shape[0] < th or window.shape[1] < tw:
            return 0.0
        res = cv2.matchTemplate(window, t.image, cv2.TM_CCOEFF_NORMED)
        # a perfectly flat patch makes the normalised score undefined; treat as no match
        return float(np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0).max())

    def scores(self, canon: np.ndarray) -> dict[str, float]:
        gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
        return {t.name: self.score(gray, t) for t in self.templates}

    def classify(self, canon: np.ndarray) -> Detection:
        gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
        best: Detection | None = None
        for t in self.templates:
            s = self.score(gray, t)
            if s >= t.threshold and (best is None or s > best.score):
                best = Detection(t.name, t.role, round(s, 3), t.tap)
        if best is not None:
            return best
        if self.detect_red_x:
            xs = find_red_xs(canon)
            if xs:
                return Detection("red_x", "dismiss", xs[0].score, xs[0].center)
        return UNKNOWN
