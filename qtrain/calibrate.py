"""Interactive calibration: teach Q-Train what each game screen looks like.

    python -m qtrain.calibrate --window "BlueStacks App Player 1"

Put this terminal and the BlueStacks window side by side. For each screen you
get the game to that screen, press Enter, then in the picture window:
  1. drag a box around something that ONLY appears on that screen, press Enter
  2. click where the bot should tap to leave that screen, press Enter
     (Esc = no tap)

Everything is saved to assets/ (gitignored): the templates are crops of the
game's artwork and aren't ours to publish.

Re-do one screen:          --only save_me
Teach it a popup:          --add-popup daily_reward
Re-select the game area:   --redo-area
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .capture import make_capture
from .screens import CALIBRATION_FILE, Calibration, ScreenClassifier, ScreenTemplate, is_flat
from .vision import GameArea, canonicalize


@dataclass(frozen=True)
class ScreenSpec:
    name: str
    role: str
    when: str           # how to get the game to this screen
    box_hint: str       # what to drag a box around
    tap_hint: str | None  # where to click (None = this screen gets no tap)


STANDARD = [
    ScreenSpec("menu", "start",
               "the MAIN MENU (the one that says 'Tap to Play')",
               "the Missions / Me / Shop buttons along the bottom (they don't animate)",
               "an empty spot in the middle of the screen; tapping there starts a run"),
    ScreenSpec("playing", "playing",
               "a RUN IN PROGRESS: start a run, and capture while the character is running "
               "(type a delay like 3 to give yourself time)",
               "the pause button in the top-left corner (not the score; it changes)",
               None),
    ScreenSpec("save_me", "crash",
               "the 'SAVE ME' prompt that appears right after you crash (it only lasts a few seconds; "
               "use a delay)",
               "the 'Save me' heading or the key button",
               "an EMPTY area away from the Save Me button, so no keys get spent "
               "(or Esc to just let it time out)"),
    ScreenSpec("game_over", "crash",
               "the SCORE screen that appears after a run ends",
               "a heading or button that only this screen has",
               "the button that goes back to the menu or straight into a new run"),
]


# ------------------------------------------------------------ UI helpers
def select_box(img: np.ndarray, title: str) -> tuple[int, int, int, int] | None:
    print("   (drag a box, then Enter/Space to confirm; c to cancel)")
    x, y, w, h = cv2.selectROI(title, img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    return None if w == 0 or h == 0 else (int(x), int(y), int(w), int(h))


def pick_point(img: np.ndarray, title: str) -> tuple[int, int] | None:
    print("   (click a point, then Enter to confirm; Esc for no tap)")
    state: dict = {"pt": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pt"] = (x, y)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        view = img.copy()
        if state["pt"]:
            cv2.circle(view, state["pt"], 9, (0, 255, 0), 2)
            cv2.drawMarker(view, state["pt"], (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
        cv2.imshow(title, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32) and state["pt"]:
            break
        if key == 27:
            state["pt"] = None
            break
    cv2.destroyWindow(title)
    return state["pt"]


def wait_for_user(prompt: str) -> bool:
    """Enter = capture now, a number = capture after that many seconds, s = skip."""
    ans = input(f"{prompt}\n   Enter = capture now, a number = delay in seconds, s = skip: ").strip().lower()
    if ans == "s":
        return False
    if ans:
        try:
            delay = float(ans)
        except ValueError:
            delay = 0
        for left in range(int(delay), 0, -1):
            print(f"   capturing in {left}...", flush=True)
            time.sleep(1)
        time.sleep(delay - int(delay))
    return True


# ----------------------------------------------------------- cross-check
def cross_check(cal: Calibration, snapshots: dict[str, np.ndarray]) -> list[str]:
    """Score every template on every saved screen; flag templates that also
    match a screen they shouldn't (they would make the bot misread screens)."""
    clf = ScreenClassifier(cal.templates, detect_red_x=False)
    names = [t.name for t in cal.templates]
    warnings = []
    print("\nCross-check (rows = screen captured, columns = template score; want high only on the diagonal):")
    print(" " * 14 + "".join(f"{n[:10]:>11}" for n in names))
    for shot_name, canon in snapshots.items():
        scores = clf.scores(canon)
        row = "".join(f"{scores[n]:>11.2f}" for n in names)
        print(f"{shot_name[:13]:<14}{row}")
        for t in cal.templates:
            if t.name != shot_name and scores[t.name] >= t.threshold:
                warnings.append(f"template '{t.name}' also matches the '{shot_name}' screen "
                                f"({scores[t.name]:.2f}); pick a more distinctive box with --only {t.name}")
            if t.name == shot_name and scores[t.name] < t.threshold:
                warnings.append(f"template '{t.name}' doesn't match its own screen; "
                                f"redo with --only {t.name}")
    return warnings


# ------------------------------------------------------------------ main
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Calibrate screen recognition for Q-Train.")
    p.add_argument("--window", default="BlueStacks", help="substring of the emulator window title")
    p.add_argument("--dir", default="assets", help="where calibration is stored (gitignored)")
    p.add_argument("--only", nargs="*", help="re-calibrate just these screens")
    p.add_argument("--add-popup", nargs="*", default=[], help="teach popups to dismiss, by name")
    p.add_argument("--redo-area", action="store_true", help="re-select the game area")
    args = p.parse_args(argv)

    root = Path(args.dir)
    existing = Calibration.load(root) if (root / CALIBRATION_FILE).exists() else None
    cap = make_capture("mss", window=args.window)
    try:
        if existing and not args.redo_area:
            area = existing.game_area
            print("Using the saved game area (--redo-area to change it).")
        else:
            print("\nStep 1: game area. Drag a box around JUST the game: leave out BlueStacks' "
                  "title bar and the toolbar on the side.")
            frame = cap.grab()
            box = select_box(np.ascontiguousarray(frame), "Q-Train: select the GAME area")
            if box is None:
                raise SystemExit("No game area selected; nothing saved.")
            area = GameArea.from_pixels(box, frame.shape)

        templates = {t.name: t for t in existing.templates} if existing else {}
        if args.only:
            specs = [s for s in STANDARD if s.name in args.only]
        elif args.add_popup:
            specs = []      # just adding popups: keep the standard screens as they are
        else:
            specs = list(STANDARD)
        if args.only and not args.add_popup:
            unknown = set(args.only) - {s.name for s in STANDARD} - set(templates)
            if unknown:
                raise SystemExit(f"unknown screen(s): {', '.join(sorted(unknown))}")
        specs += [ScreenSpec(n, "dismiss", f"the '{n}' popup",
                             "something that only this popup has (its title, an icon)",
                             "its close / OK / claim button") for n in args.add_popup]
        if args.only:  # re-doing a previously added popup
            specs += [ScreenSpec(n, templates[n].role, f"the '{n}' screen", "something unique to it",
                                 "where to tap to leave it") for n in args.only
                      if n in templates and n not in {s.name for s in specs}]

        snap_dir = root / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        for i, spec in enumerate(specs, 1):
            print(f"\nScreen {i}/{len(specs)}: '{spec.name}'")
            if not wait_for_user(f" Get the game to {spec.when}."):
                print("   skipped")
                continue
            canon = canonicalize(cap.grab(), area)
            print(f" Box: {spec.box_hint}")
            box = select_box(canon, f"Q-Train: {spec.name} - select a distinctive region")
            if box is None:
                print("   no box selected; skipped")
                continue
            tap = None
            if spec.tap_hint:
                print(f" Tap: {spec.tap_hint}")
                tap = pick_point(canon, f"Q-Train: {spec.name} - click where to tap")
            x, y, w, h = box
            gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
            if is_flat(gray[y:y + h, x:x + w]):
                print("   that region is nearly blank, so it can't identify the screen; "
                      f"skipped (redo with --only {spec.name})")
                continue
            templates[spec.name] = ScreenTemplate(spec.name, spec.role, box, tap,
                                                  image=gray[y:y + h, x:x + w].copy())
            cv2.imwrite(str(snap_dir / f"{spec.name}.png"), canon)
            print(f"   saved '{spec.name}' ({spec.role}" + (f", tap at {tap})" if tap else ", no tap)"))

        cal = Calibration(area, list(templates.values()))
        path = cal.save(root)
        print(f"\nSaved {len(cal.templates)} screen(s) to {path}")

        snapshots = {}
        for t in cal.templates:
            img = cv2.imread(str(snap_dir / f"{t.name}.png"))
            if img is not None:
                snapshots[t.name] = img
        warnings = cross_check(cal, snapshots)
        if warnings:
            print("\nWarnings:")
            for w_ in warnings:
                print(f"  ! {w_}")
        else:
            print("\nAll templates are distinct. Next: python -m qtrain.watch")
    finally:
        cap.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
