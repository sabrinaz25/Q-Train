"""What Android itself knows about an app: which one is in front, launching it, its crashes.

Used to measure coverage the same way for any app and any driver (the explorer or
Android's monkey), and to notice when the driver has left the app under test.
"""
from __future__ import annotations

import re
import subprocess
import threading
from dataclasses import dataclass, field

from .adb import adb_path, connect, run_adb

# "mResumedActivity: ActivityRecord{1a2b u0 com.app/.MainActivity t12}" (older Androids),
# "topResumedActivity=ActivityRecord{... u0 com.app/.Main t12}" (Android 10+),
# "mCurrentFocus=Window{... u0 com.app/com.app.Main}" (window manager)
_COMPONENT = re.compile(r"(?:ResumedActivity|mCurrentFocus)[:=]\s*\S+\{[^}]*?\s([\w.]+)/([\w.$]+)")


def parse_resumed(dumpsys: str) -> tuple[str, str] | None:
    """(package, activity) of the foreground activity, activity fully qualified."""
    m = _COMPONENT.search(dumpsys)
    if not m:
        return None
    pkg, act = m.group(1), m.group(2)
    return pkg, pkg + act if act.startswith(".") else act


def parse_event(line: str, package: str) -> tuple[str, str] | None:
    """A crash or ANR of `package` from an `events` log line: (kind, signature) or None.

    am_crash: [..., process, flags, exception, message, ...]   signature: exception class
    am_anr:   [..., package, flags, reason]                    signature: reason
    The message is free text that can contain commas, so only fields before it are trusted.
    """
    m = re.search(r"\bam_(crash|anr)\b.*?:\s*\[(.*)\]\s*$", line)
    if not m:
        return None
    kind, fields = m.group(1), [f.strip() for f in m.group(2).split(",")]
    i = next((k for k, f in enumerate(fields) if f == package or f.startswith(package + ":")), None)
    if i is None:
        return None
    detail = fields[i + 2] if len(fields) > i + 2 else ""
    if kind == "crash":
        return "crash", detail or "?"
    return "anr", ",".join(fields[i + 2:])[:120] or "anr"


class AndroidApp:
    """The app under test on one device."""

    def __init__(self, serial: str, package: str):
        self.serial, self.package = serial, package
        connect(serial)

    def foreground(self) -> tuple[str, str] | None:
        out = run_adb("shell", "dumpsys activity activities | grep -E 'ResumedActivity' | head -n 2",
                      serial=self.serial)
        found = parse_resumed(out)
        if found is None:       # some builds only show it in the window manager
            out = run_adb("shell", "dumpsys window | grep -E 'mCurrentFocus' | head -n 1", serial=self.serial)
            found = parse_resumed(out)
        return found

    def current_package(self) -> str | None:
        fg = self.foreground()
        return fg[0] if fg else None

    def current_activity(self) -> str | None:
        fg = self.foreground()
        return f"{fg[0]}/{fg[1]}" if fg else None

    def launch(self) -> None:
        run_adb("shell", "monkey", "-p", self.package, "-c", "android.intent.category.LAUNCHER", "1",
                serial=self.serial)

    def force_stop(self) -> None:
        run_adb("shell", "am", "force-stop", self.package, serial=self.serial)


@dataclass
class CrashWatcher:
    """Follows the device's `events` log and records crashes and ANRs of one package."""
    serial: str
    package: str
    events: list[tuple[float, str, str]] = field(default_factory=list)   # (t, kind, signature)

    def start(self, clock) -> CrashWatcher:
        self._clock = clock
        run_adb("logcat", "-b", "events", "-c", serial=self.serial)          # only count new ones
        cmd = [adb_path(), "-s", self.serial, "logcat", "-b", "events", "-v", "brief"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                      errors="replace")
        threading.Thread(target=self._read, daemon=True).start()
        return self

    def _read(self) -> None:
        for line in self._proc.stdout:
            found = parse_event(line, self.package)
            if found:
                self.events.append((self._clock(), *found))

    def stop(self) -> None:
        if getattr(self, "_proc", None) and self._proc.poll() is None:
            self._proc.terminate()
