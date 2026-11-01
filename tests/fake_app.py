"""A tiny fake Android app drawn with OpenCV: screens, buttons, Back, a status-bar clock.

Shared by the fingerprint, explorer and monkey tests so they run without a device.
Each screen has a different layout; buttons are dark rounded boxes that lead to
another screen. Screen "crash" makes the app crash (it relaunches on "home").
"""
from __future__ import annotations

import cv2
import numpy as np

W, H = 360, 640
BAR = 26   # status bar height

# screen -> (title, accent colour BGR, layout, [(button box, target screen)])
SCREENS = {
    "home": ("Home", (180, 120, 40), "grid",
             [((30, 120, 140, 90), "list"), ((190, 120, 140, 90), "form"),
              ((30, 260, 140, 90), "detail"), ((190, 260, 140, 90), "about")]),
    "list": ("Inbox", (60, 140, 60), "rows", [((20, 100, 320, 60), "detail"), ((20, 180, 320, 60), "detail2"),
                                              ((20, 260, 320, 60), "crash")]),
    "form": ("Sign up", (40, 40, 160), "form", [((100, 520, 160, 56), "home")]),
    "detail": ("Message", (120, 60, 140), "text", [((250, 560, 90, 50), "list")]),
    "detail2": ("Thread", (140, 100, 30), "image", []),
    "about": ("About", (90, 90, 90), "text2", [((30, 500, 300, 50), "detail2")]),
    "crash": ("", (0, 0, 0), "blank", []),
}
PARENT = {"list": "home", "form": "home", "detail": "list", "detail2": "list", "about": "home"}


def draw(screen: str, clock: str = "12:00", cursor: bool = False) -> np.ndarray:
    title, accent, layout, buttons = SCREENS[screen]
    img = np.full((H, W, 3), 245, np.uint8)
    img[:BAR] = (30, 30, 30)
    cv2.putText(img, clock, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    img[BAR:BAR + 56] = accent
    cv2.putText(img, title, (16, BAR + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if layout == "rows":
        for y in range(330, 620, 70):
            cv2.line(img, (20, y), (340, y), (200, 200, 200), 1)
    elif layout == "form":
        for i, y in enumerate((130, 230, 330, 430)):
            cv2.rectangle(img, (40, y), (320, y + 50), (120, 120, 120), 2)
            if cursor and i == 0:
                cv2.line(img, (50, y + 10), (50, y + 40), (0, 0, 0), 2)
    elif layout == "text":
        for y in range(120, 500, 28):
            cv2.line(img, (24, y), (24 + (y * 37) % 280 + 40, y), (90, 90, 90), 6)
    elif layout == "text2":
        for y in range(140, 460, 40):
            cv2.line(img, (60, y), (300, y), (60, 60, 60), 10)
    elif layout == "image":
        cv2.circle(img, (180, 330), 120, (40, 150, 220), -1)
        cv2.rectangle(img, (40, 480), (320, 600), (60, 160, 60), -1)
    elif layout == "grid":
        cv2.putText(img, "Welcome", (100, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2)
    for (x, y, w, h), _ in buttons:
        cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), -1)
        cv2.rectangle(img, (x + 12, y + h // 2 - 4), (x + w // 2, y + h // 2 + 4), (230, 230, 230), -1)
    return img


class FakeApp:
    """Device + frame source + Android state for one app."""

    def __init__(self, package: str = "com.example.app"):
        self.package = package
        self.screen = "home"
        self.foreground = package
        self.crashes = 0
        self.taps: list[tuple[int, int]] = []
        self.backs = 0
        self.swipes = 0
        self.launches = 0
        self.t = 0
        self.width, self.height = W, H

    # frames
    def grab(self) -> np.ndarray:
        self.t += 1
        if self.foreground != self.package:
            return np.full((H, W, 3), 20, np.uint8)               # launcher / another app
        return draw(self.screen, clock=f"12:{self.t % 60:02d}")

    # input
    def tap(self, x: int, y: int) -> None:
        self.taps.append((x, y))
        if self.foreground != self.package:
            return
        for (bx, by, bw, bh), target in SCREENS[self.screen][3]:
            if bx <= x < bx + bw and by <= y < by + bh:
                if target == "crash":
                    self.crashes += 1
                    self.foreground = "launcher"
                else:
                    self.screen = target
                return

    def back(self) -> None:
        self.backs += 1
        if self.foreground != self.package:
            return
        if self.screen in PARENT:
            self.screen = PARENT[self.screen]
        else:
            self.foreground = "launcher"                         # Back on home leaves the app

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None:
        self.swipes += 1

    # android state
    def current_package(self) -> str:
        return self.foreground

    def current_activity(self) -> str:
        return f"{self.foreground}/.{self.screen.capitalize()}Activity"

    def launch(self) -> None:
        self.launches += 1
        self.foreground, self.screen = self.package, "home"
