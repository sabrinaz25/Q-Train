"""Play whole episodes end to end: the stage 2 acceptance test.

    python -m qtrain.run --serial 127.0.0.1:5556 --window "BlueStacks App Player 1" --episodes 5

Uses a random (or do-nothing) policy, so expect short runs: the point is that
reset -> play -> crash -> reset works hands-free, and to measure how much
time the loop spends playing versus navigating menus.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .controller import ControllerConfig
from .env import SubwaySurfersEnv


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run episodes with a baseline policy.")
    p.add_argument("--serial", default="127.0.0.1:5556")
    p.add_argument("--window", default="BlueStacks")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--policy", default="random", choices=["random", "noop"])
    p.add_argument("--hz", type=float, default=15.0)
    p.add_argument("--capture", default="dxcam", choices=["dxcam", "mss"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input", default="auto", choices=["auto", "scrcpy", "shell", "spawn"])
    args = p.parse_args(argv)

    env = SubwaySurfersEnv(args.serial, args.window, capture=args.capture,
                           config=ControllerConfig(hz=args.hz), input_backend=args.input)
    env.action_space.seed(args.seed)
    rows = []
    t_start = time.perf_counter()
    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset(seed=args.seed + ep)
            t0 = time.perf_counter()
            steps, ret, done, last = 0, 0.0, False, {}
            while not done:
                a = env.action_space.sample() if args.policy == "random" else 0
                obs, r, term, trunc, last = env.step(a)
                steps, ret, done = steps + 1, ret + r, term or trunc
            dur = time.perf_counter() - t0
            row = {"episode": ep, "steps": steps, "return": round(ret, 2), "play_seconds": round(dur, 2),
                   "achieved_hz": round(steps / dur, 1) if dur else 0.0,
                   "reset_seconds": info["reset_seconds"], "ended_on": last.get("screen"),
                   "truncated": bool(trunc)}
            rows.append(row)
            print(f"ep {ep}: {steps} steps ({dur:.1f}s at {row['achieved_hz']} Hz), return {ret:.1f}, "
                  f"reset took {info['reset_seconds']}s via {' > '.join(info['screens'])}, "
                  f"ended on '{row['ended_on']}'")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        env.close()

    if not rows:
        return
    total = time.perf_counter() - t_start
    play = sum(r["play_seconds"] for r in rows)
    summary = {
        "episodes": len(rows),
        "mean_steps": round(float(np.mean([r["steps"] for r in rows])), 1),
        "mean_reset_seconds": round(float(np.mean([r["reset_seconds"] for r in rows])), 2),
        "mean_achieved_hz": round(float(np.mean([r["achieved_hz"] for r in rows])), 1),
        "time_playing_pct": round(100 * play / total, 1),
        "steps_per_hour": round(sum(r["steps"] for r in rows) / total * 3600),
        "controller_stats": dict(env.ctrl.stats),
    }
    print("\n" + json.dumps(summary, indent=2))
    out = Path("results") / f"run-{args.policy}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "summary": summary, "episodes": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
