"""Live view of what the bot "sees": detected screen, template scores, red X's.

    python -m qtrain.watch --window "BlueStacks App Player 1"

Keys: s = save the current frame to data/screens/ (handy for tuning)
      q / Esc = quit
Works before calibration too (shows red X detection on the raw window).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .capture import make_capture
from .screens import CALIBRATION_FILE, Calibration, ScreenClassifier
from .vision import canonicalize, find_red_xs

COLORS = {"start": (255, 200, 0), "playing": (0, 220, 0), "crash": (0, 0, 255),
          "dismiss": (255, 0, 255), "unknown": (160, 160, 160)}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Live debug view of screen recognition.")
    p.add_argument("--window", default="BlueStacks")
    p.add_argument("--dir", default="assets")
    p.add_argument("--capture", default="mss", choices=["mss", "dxcam"])
    args = p.parse_args(argv)

    cal = Calibration.load(args.dir) if (Path(args.dir) / CALIBRATION_FILE).exists() else None
    if cal is None:
        print("No calibration yet: showing red X detection only.")
    clf = ScreenClassifier(cal.templates) if cal else None
    cap = make_capture(args.capture, window=args.window)
    t_prev, fps = time.perf_counter(), 0.0
    try:
        while True:
            canon = canonicalize(cap.grab(), cal.game_area if cal else None)
            t0 = time.perf_counter()
            det = clf.classify(canon) if clf else None
            xs = find_red_xs(canon)
            scores = clf.scores(canon) if clf else {}
            ms = (time.perf_counter() - t0) * 1000

            view = canon.copy()
            for x in xs:
                cv2.rectangle(view, (x.cx - x.w // 2, x.cy - x.h // 2), (x.cx + x.w // 2, x.cy + x.h // 2),
                              (255, 0, 255), 2)
            if cal and det and det.name in scores:
                t = next(t for t in cal.templates if t.name == det.name)
                x, y, w, h = t.roi
                cv2.rectangle(view, (x, y), (x + w, y + h), COLORS[det.role], 2)
            if det and det.tap:
                cv2.drawMarker(view, det.tap, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            lines = [f"{det.name if det else '-'} ({det.role if det else ''}) {det.score if det else 0:.2f}",
                     f"{fps:.0f} fps, classify {ms:.1f} ms, red X: {len(xs)}"]
            lines += [f"{n}: {s:.2f}" for n, s in sorted(scores.items(), key=lambda kv: -kv[1])[:4]]
            for i, line in enumerate(lines):
                y = 22 + 20 * i
                cv2.putText(view, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                color = COLORS.get(det.role, (255, 255, 255)) if det and i == 0 else (255, 255, 255)
                cv2.putText(view, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.imshow("Q-Train watch", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = Path("data/screens") / f"{time.strftime('%Y%m%d-%H%M%S')}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), canon)
                print(f"saved {out}")
    finally:
        cap.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
