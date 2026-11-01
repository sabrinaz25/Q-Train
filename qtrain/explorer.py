"""Systematic app exploration from screen pixels: the second application on Q-Train's core.

Works on any Android app with no app-specific setup: no templates, no source
code. Each step:

  1. Wait for the screen to settle, then fingerprint it (qtrain.fingerprint).
     A new fingerprint is a new node in the app's screen graph.
  2. On a new screen, find what's tappable. Primary source: Android's
     accessibility tree (`uiautomator dump`), which lists every element the app
     marks clickable, with exact bounds. Fallback when the tree is empty (games,
     web views, custom-drawn UI): find_targets, from pixels. Plus one "scroll".
     The tree takes ~1-2 s to read, so it is read once per *new* screen only.
     (Pixel-only targets measured 7 of 8 taps dead on a real app.)
  3. Act on an untried target. If the screen didn't change, the target is dead.
     If it did, record the edge (screen, target) -> new screen.
  4. When a screen has nothing left to try, walk the recorded edges to the
     nearest screen that still has untried targets (breadth-first), or press
     Back if none is reachable.
  5. If Android says another app is in front (we tapped out, or the app
     crashed), press Back once, then relaunch the app.

Compared with Android's `monkey` (random events), the explorer never repeats an
action it has learned is useless and goes back to unexplored screens on
purpose. `python -m qtrain.compare` measures both with the same observer.
"""
from __future__ import annotations

import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .fingerprint import STATUS_BAR, Crop, ScreenSet, distance, fingerprint
from .waits import wait_for_change, wait_until_stable

SCROLL = "scroll"


@dataclass(frozen=True)
class Target:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


def _iou(a: Target, b: Target) -> float:
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / float(a.w * a.h + b.w * b.h - inter) if inter else 0.0


def _contains(outer: Target, inner: Target) -> bool:
    return (outer.x <= inner.x and outer.y <= inner.y and outer.x + outer.w >= inner.x + inner.w
            and outer.y + outer.h >= inner.y + inner.h and outer != inner)


def find_targets(frame: np.ndarray, crop: Crop = STATUS_BAR, min_frac: float = 0.05,
                 max_area_frac: float = 0.35, max_targets: int = 40) -> list[Target]:
    """Boxes that look like tappable elements (buttons, rows, icons, text blocks), from pixels.

    Edges are closed into solid blocks so a button's outline, label and icon form
    one region. Boxes bigger than `max_area_frac` of the screen are page containers,
    and boxes that contain two or more others are groups (a card around buttons):
    tapping those hits empty space, so they're dropped in favour of their contents.
    """
    H, W = frame.shape[:2]
    top, bottom = int(H * crop.top), H - int(H * crop.bottom)
    gray = cv2.cvtColor(frame[top:bottom], cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame[top:bottom]
    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, W // 45), max(3, H // 110)))
    blocks = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(blocks, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_side = W * min_frac
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if max(w, h) < min_side or min(w, h) < min_side / 3:
            continue
        if w * h > max_area_frac * W * (bottom - top):
            continue
        boxes.append(Target(x, y + top, w, h))
    boxes = [b for b in boxes if sum(_contains(b, o) for o in boxes) < 2]
    kept: list[Target] = []
    for b in sorted(boxes, key=lambda t: t.w * t.h, reverse=True):
        if all(_iou(b, k) < 0.5 and not _contains(k, b) for k in kept):
            kept.append(b)
    return sorted(kept, key=lambda t: (t.y, t.x))[:max_targets]


@dataclass
class Node:
    actions: list                     # Target or SCROLL
    tried: set = field(default_factory=set)
    outside: bool | None = None       # None: not checked yet
    source: str = ""                  # where its targets came from: "a11y", "pixels" or "none"

    def untried(self) -> list:
        return [a for i, a in enumerate(self.actions) if i not in self.tried]


@dataclass
class ExplorerConfig:
    change_timeout: float = 0.6       # s for a tap to change the screen before it counts as dead
    stable_for: float = 0.25          # screen must hold still this long before it's fingerprinted
    settle_timeout: float = 2.0       # ...or we take it as it is (live content, video)
    poll: float = 0.05
    threshold: float = 0.10           # fingerprint distance that makes a different screen
    scroll: bool = True
    seed: int = 0
    targets: str = "hybrid"           # "hybrid" (tree, else pixels), "a11y" or "pixels"
    launch_wait: float = 2.5          # s to let a relaunched app come to the front


class Explorer:
    def __init__(self, device, capture, app, config: ExplorerConfig | None = None,
                 crop: Crop = STATUS_BAR, clock=time.perf_counter, sleep=time.sleep):
        self.dev, self.cap, self.app = device, capture, app
        self.cfg = config or ExplorerConfig()
        self.crop, self.clock, self.sleep = crop, clock, sleep
        self.screens = ScreenSet(threshold=self.cfg.threshold)
        self.nodes: dict[int, Node] = {}
        self.edges: dict[int, dict[int, int]] = {}      # screen -> {action index: next screen}
        self.stats: Counter = Counter()
        self.rng = random.Random(self.cfg.seed)
        self._frame: np.ndarray | None = None
        self._outside_streak = 0
        self.root: int | None = None      # the app's starting screen: Back there exits the app
        self.back_exits: set[int] = set()  # screens where Back was seen to leave the app
        self._backed_from: int | None = None

    # ------------------------------------------------------------- seeing
    def _fp(self) -> np.ndarray:
        self._frame = self.cap.grab()
        return fingerprint(self._frame, self.crop)

    def settle(self) -> np.ndarray:
        res = wait_until_stable(self._fp, lambda a, b: distance(a, b) <= 0.02,
                                stable_for=self.cfg.stable_for, timeout=self.cfg.settle_timeout,
                                poll=self.cfg.poll, clock=self.clock, sleep=self.sleep)
        if not res.done:
            self.stats["unsettled"] += 1
        return res.value

    def current(self) -> int:
        fp = self.settle()
        sid, new = self.screens.add(fp)
        if new:
            targets, source = self._find_targets()
            self.rng.shuffle(targets)
            self.nodes[sid] = Node(targets + ([SCROLL] if self.cfg.scroll else []), source=source)
        node = self.nodes[sid]
        if node.outside is None:                 # one Android query per screen, not per step
            node.outside = not self._in_app()
        if self.root is None and not node.outside:
            self.root = sid
        return sid

    def _in_app(self) -> bool:
        pkg = self.app.current_package()
        if pkg != self.app.package:
            self.stats[f"outside:{pkg or 'unknown'}"] += 1   # which app/dialog was in front
        return pkg == self.app.package

    def _find_targets(self) -> tuple[list[Target], str]:
        if self.cfg.targets in ("hybrid", "a11y"):
            targets = self._tree_targets()
            if targets or self.cfg.targets == "a11y":
                source = "a11y" if targets else "none"
                self.stats[f"screens:{source}"] += 1
                return targets, source
        self.stats["screens:pixels"] += 1
        return find_targets(self._frame, self.crop), "pixels"

    def _tree_targets(self) -> list[Target]:
        """Clickable elements from Android's accessibility tree, in frame coordinates."""
        dump = getattr(self.app, "ui_dump", None)
        if dump is None:
            return []
        t0 = self.clock()
        xml = dump()
        ms = int((self.clock() - t0) * 1000)
        self.stats["a11y_ms"] += ms
        self.stats["a11y_reads"] += 1
        self.stats["a11y_max_ms"] = max(self.stats["a11y_max_ms"], ms)
        if not xml:
            self.stats["a11y_failed"] += 1
            return []
        from .android import parse_clickables
        boxes, display = parse_clickables(xml, self.app.package)
        if not boxes or display is None:
            return []
        fh, fw = self._frame.shape[:2]
        sx, sy = fw / display[0], fh / display[1]
        return [Target(int(x1 * sx), int(y1 * sy), max(1, int((x2 - x1) * sx)), max(1, int((y2 - y1) * sy)))
                for x1, y1, x2, y2 in boxes]

    # ------------------------------------------------------------- acting
    def _to_device(self, x: int, y: int) -> tuple[int, int]:
        fh, fw = self._frame.shape[:2]
        return int(x * self.dev.width / fw), int(y * self.dev.height / fh)

    def _do(self, action) -> None:
        if action == SCROLL:
            fh, fw = self._frame.shape[:2]
            x0, y0 = self._to_device(fw // 2, int(fh * 0.75))
            x1, y1 = self._to_device(fw // 2, int(fh * 0.30))
            self.dev.swipe(x0, y0, x1, y1, 300)
            self.stats["scrolls"] += 1
        else:
            self.dev.tap(*self._to_device(*action.center))
            self.stats["taps"] += 1

    def _path_to_untried(self, start: int) -> list[int] | None:
        """Action indices leading from `start` to the nearest screen with untried actions."""
        seen, queue = {start}, deque([(start, [])])
        while queue:
            sid, path = queue.popleft()
            if path and self.nodes[sid].untried() and not self.nodes[sid].outside:
                return path
            for a, nxt in self.edges.get(sid, {}).items():
                if nxt not in seen and nxt in self.nodes:
                    seen.add(nxt)
                    queue.append((nxt, path + [a]))
        return None

    def step(self) -> dict:
        sid = self.current()
        node = self.nodes[sid]
        if node.outside and self._backed_from is not None:
            self.back_exits.add(self._backed_from)    # learned: Back there leaves the app
            self.stats["learned_back_exits"] += 1
        self._backed_from = None
        if node.outside and self._in_app():
            # The cached answer came from a transition (Android briefly reports the launcher
            # while an app starts): this screen is the app's. Caching "outside" here made
            # the splash screen trigger Back -> exit -> relaunch -> splash, in a loop.
            node.outside = False
            self.stats["outside_corrected"] += 1
        if node.outside:
            self._outside_streak += 1
            if self._outside_streak == 1:
                self.dev.back()
                self.stats["back:outside"] += 1
            else:
                self.app.launch()
                self.stats["relaunches"] += 1
                self._outside_streak = 0
                self.sleep(self.cfg.launch_wait)
            self.sleep(0.5)
            return {"screen": sid, "action": "recover"}
        self._outside_streak = 0

        untried = [i for i in range(len(node.actions)) if i not in node.tried]
        if untried:
            idx = untried[0]
            node.tried.add(idx)
        else:
            path = self._path_to_untried(sid)
            if path is None:
                if sid == self.root or sid in self.back_exits:
                    # Back on the starting screen exits the app (then a relaunch lands right
                    # back here): try a random spot instead, which can find what we missed.
                    fh, fw = self._frame.shape[:2]
                    x = self.rng.randint(int(fw * 0.05), int(fw * 0.95))
                    y = self.rng.randint(int(fh * 0.10), int(fh * 0.95))
                    self.dev.tap(*self._to_device(x, y))
                    self.stats["random_taps"] += 1
                    self.sleep(self.cfg.change_timeout)
                    return {"screen": sid, "action": "random"}
                self.dev.back()
                self.stats["back:exhausted"] += 1
                self._backed_from = sid
                return {"screen": sid, "action": "back"}
            idx = path[0]                      # replay a known edge towards unexplored screens
            self.stats["navigate"] += 1

        before = self.screens.prints[sid]
        self._do(node.actions[idx])
        res = wait_for_change(self._fp, lambda fp: distance(fp, before) > self.cfg.threshold,
                              timeout=self.cfg.change_timeout, poll=self.cfg.poll,
                              clock=self.clock, sleep=self.sleep)
        if not res.done:
            self.stats["dead"] += 1
            return {"screen": sid, "action": idx, "changed": False}
        nxt = self.current()
        self.edges.setdefault(sid, {})[idx] = nxt
        return {"screen": sid, "action": idx, "changed": True, "next": nxt}

    def run(self, seconds: float) -> dict:
        t0 = self.clock()
        steps = 0
        while self.clock() - t0 < seconds:
            self.step()
            steps += 1
        explored = sum(1 for n in self.nodes.values() if not n.untried())
        return {"steps": steps, "screens_seen": len(self.screens),
                "screens_fully_explored": explored, **dict(self.stats)}
