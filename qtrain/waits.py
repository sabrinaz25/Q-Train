"""Waiting on the screen instead of sleeping a fixed time.

After a tap the next screen takes anywhere from a few ms to a second to appear.
A fixed sleep has to cover the slowest case, so every fast transition wastes the
difference. These helpers poll something cheap (a screen label, a fingerprint)
and return as soon as it changes or settles, with the old sleep as the timeout.

They take any `read` function and any comparison. The app explorer uses them to
wait until a screen's fingerprint changes, then stops changing. (The game
controller keeps its fixed settle sleeps: a change-based wait there returned on
fade-out frames and caused repeated Back presses.)
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class WaitResult(Generic[T]):
    done: bool          # condition met (False: timed out)
    seconds: float
    value: T            # last value read


def wait_for_change(read: Callable[[], T], changed: Callable[[T], bool], *, timeout: float,
                    poll: float = 0.05, clock=time.perf_counter, sleep=time.sleep) -> WaitResult[T]:
    """Poll `read()` until `changed(value)` is true, or `timeout` seconds pass."""
    t0 = clock()
    while True:
        value = read()
        elapsed = clock() - t0
        if changed(value):
            return WaitResult(True, elapsed, value)
        if elapsed >= timeout:
            return WaitResult(False, elapsed, value)
        sleep(poll)


def wait_until_stable(read: Callable[[], T], same: Callable[[T, T], bool], *, stable_for: float,
                      timeout: float, poll: float = 0.05, clock=time.perf_counter,
                      sleep=time.sleep) -> WaitResult[T]:
    """Poll `read()` until it has stayed the same for `stable_for` seconds (animations finished)."""
    t0 = clock()
    last = read()
    since = clock()
    while True:
        now = clock()
        if now - since >= stable_for:
            return WaitResult(True, now - t0, last)
        if now - t0 >= timeout:
            return WaitResult(False, now - t0, last)
        sleep(poll)
        value = read()
        if not same(last, value):
            last, since = value, clock()
