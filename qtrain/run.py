"""Play whole episodes end to end, unattended, and measure reliability.

    python -m qtrain.run --serial 127.0.0.1:5555 --window "BlueStacks App Player 1" --episodes 5
    python -m qtrain.run --serial 127.0.0.1:5555 --window "BlueStacks App Player 1" --episodes 50  (soak)

Uses a random (or do-nothing) policy, so runs are short: the point is that
reset -> play -> crash -> reset works with nobody watching, and to measure how
often it needs to recover and how much time goes to playing versus menus.

A failed episode (e.g. a reset that gives up) is recorded and the run carries
on; every episode is appended to a .jsonl file as soon as it finishes, so a
crash or Ctrl+C never loses the results so far.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .controller import ControllerConfig

RECOVERY_KEYS = ("tap:red_x", "relocated", "back", "stuck", "mid_run_dismiss", "app_restarts",
                 "lost_track")


def percentile(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 2) if values else None


def summarize(rows: list[dict], stats: dict, wall_seconds: float,
              crash_delays: list[float] | None = None) -> dict:
    ok = [r for r in rows if "error" not in r]
    failed = [r for r in rows if "error" in r]
    resets = [r["reset_seconds"] for r in ok]
    play = sum(r["play_seconds"] for r in ok)
    hours = wall_seconds / 3600
    recoveries = {k: stats.get(k, 0) for k in RECOVERY_KEYS}
    return {
        "episodes_attempted": len(rows),
        "episodes_completed": len(ok),
        "failures": len(failed),
        "success_rate_pct": round(100 * len(ok) / len(rows), 1) if rows else None,
        "wall_minutes": round(wall_seconds / 60, 1),
        "reset_seconds_p50": percentile(resets, 50),
        "reset_seconds_p95": percentile(resets, 95),
        "reset_seconds_max": max(resets) if resets else None,
        "mean_steps": round(float(np.mean([r["steps"] for r in ok])), 1) if ok else None,
        "mean_achieved_hz": round(float(np.mean([r["achieved_hz"] for r in ok])), 1) if ok else None,
        "time_playing_pct": round(100 * play / wall_seconds, 1) if wall_seconds else None,
        "steps_per_hour": round(sum(r["steps"] for r in ok) / hours) if hours else None,
        "crash_detection_delay_s_p50": percentile(crash_delays or [], 50),
        "crash_detection_delay_s_p95": percentile(crash_delays or [], 95),
        "recoveries": recoveries,
        "recoveries_per_hour": round(sum(recoveries.values()) / hours, 1) if hours else None,
        "failure_reasons": [r["error"][:160] for r in failed],
        "controller_stats": dict(stats),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run episodes unattended with a baseline policy.")
    p.add_argument("--serial", default="127.0.0.1:5555")
    p.add_argument("--window", default="BlueStacks")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--minutes", type=float, default=None,
                   help="stop after this long, even if episodes remain")
    p.add_argument("--policy", default="random", choices=["random", "noop"])
    p.add_argument("--hz", type=float, default=15.0)
    p.add_argument("--capture", default="dxcam", choices=["dxcam", "mss", "scrcpy"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input", default="auto", choices=["auto", "scrcpy", "shell", "spawn"])
    p.add_argument("--app-package", default="com.kiloo.subwaysurf",
                   help="restarted as a last-resort recovery; '' to disable")
    args = p.parse_args(argv)

    from .env import SubwaySurfersEnv  # heavy (Gymnasium, devices): only when actually running

    env = SubwaySurfersEnv(args.serial, args.window, capture=args.capture,
                           config=ControllerConfig(hz=args.hz), input_backend=args.input,
                           app_package=args.app_package or None)
    env.action_space.seed(args.seed)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    Path("results").mkdir(exist_ok=True)
    log = Path("results") / f"run-{args.policy}-{stamp}.jsonl"
    rows: list[dict] = []
    t_start = time.perf_counter()
    deadline = t_start + args.minutes * 60 if args.minutes else None
    try:
        for ep in range(1, args.episodes + 1):
            if deadline and time.perf_counter() > deadline:
                print("time limit reached")
                break
            t_ep = time.perf_counter()
            try:
                obs, info = env.reset(seed=args.seed + ep)
                t0 = time.perf_counter()
                steps, ret, done, last, trunc = 0, 0.0, False, {}, False
                while not done:
                    a = env.action_space.sample() if args.policy == "random" else 0
                    obs, r, term, trunc, last = env.step(a)
                    steps, ret, done = steps + 1, ret + r, term or trunc
                dur = time.perf_counter() - t0
                row = {"episode": ep, "steps": steps, "return": round(ret, 2), "play_seconds": round(dur, 2),
                       "achieved_hz": round(steps / dur, 1) if dur else 0.0,
                       "reset_seconds": info["reset_seconds"], "screens": info["screens"],
                       "ended_on": last.get("screen"), "truncated": bool(trunc),
                       "crash_delay_s": last.get("crash_delay_s")}
                print(f"ep {ep}: {steps} steps ({dur:.1f}s at {row['achieved_hz']} Hz), "
                      f"reset {info['reset_seconds']}s via {' > '.join(info['screens'])}, "
                      f"ended on '{row['ended_on']}'")
            except KeyboardInterrupt:
                raise
            except Exception as e:  # record it and keep going: that's what a soak test measures
                row = {"episode": ep, "error": f"{type(e).__name__}: {e}",
                       "seconds": round(time.perf_counter() - t_ep, 1)}
                print(f"ep {ep}: FAILED: {row['error'][:200]}")
            rows.append(row)
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        env.close()

    if not rows:
        return
    summary = summarize(rows, env.ctrl.stats, time.perf_counter() - t_start, env.ctrl.crash_delays)
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "controller_stats"}, indent=2))
    out = log.with_suffix(".json")
    out.write_text(json.dumps({"args": vars(args), "summary": summary, "episodes": rows}, indent=2))
    print(f"saved {out} (per-episode log: {log})")


if __name__ == "__main__":
    main()
