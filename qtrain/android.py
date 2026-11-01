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


_GRANTED = re.compile(r"^\s*([\w.]+): granted=(true|false)", re.M)


def parse_ungranted(dumpsys_package: str) -> list[str]:
    """Permissions the app requests but doesn't hold, from `dumpsys package <pkg>`.

    Starts from the "requested permissions:" list, because older Androids only print a
    runtime permission's state once the app has asked for it (a freshly wiped app shows
    none), then drops those reported granted=true. Install-time permissions are in the
    list too; `pm grant` refuses those, and grant_permissions() skips them.
    """
    requested, lines = [], dumpsys_package.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "requested permissions:":
            indent = len(line) - len(line.lstrip())
            for nxt in lines[i + 1:]:
                if not nxt.strip() or len(nxt) - len(nxt.lstrip()) <= indent:
                    break
                requested.append(nxt.strip().split(":")[0])
            break
    state = {}
    for perm, val in _GRANTED.findall(dumpsys_package):
        state[perm] = state.get(perm, False) or val == "true"
    candidates = set(requested) | set(state)
    return sorted(p for p in candidates if not state.get(p, False))


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


_BOUNDS = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


def parse_clickables(xml: str, package: str, max_area_frac: float = 0.6
                     ) -> tuple[list[tuple[int, int, int, int]], tuple[int, int] | None]:
    """Clickable elements of `package` from a `uiautomator dump`, and the display size.

    Returns boxes as (x1, y1, x2, y2) in display pixels. Only enabled elements the
    app itself marks as clickable, long-clickable or checkable count; system UI
    (status bar, navigation) belongs to other packages and is skipped. Boxes covering
    most of the screen are page containers, so tapping their centre is meaningless.
    """
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return [], None
    nodes, dw, dh = [], 0, 0
    for n in root.iter("node"):
        m = _BOUNDS.match(n.get("bounds", ""))
        if not m:
            continue
        x1, y1, x2, y2 = map(int, m.groups())
        dw, dh = max(dw, x2), max(dh, y2)
        nodes.append((n, (x1, y1, x2, y2)))
    boxes: list[tuple[int, int, int, int]] = []
    for n, (x1, y1, x2, y2) in nodes:
        if n.get("package") != package or n.get("enabled") == "false":
            continue
        if not any(n.get(a) == "true" for a in ("clickable", "long-clickable", "checkable")):
            continue
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        if dw and dh and (x2 - x1) * (y2 - y1) > max_area_frac * dw * dh:
            continue
        if (x1, y1, x2, y2) not in boxes:
            boxes.append((x1, y1, x2, y2))
    return boxes, ((dw, dh) if dw and dh else None)


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

    def clear_data(self) -> None:
        """Wipe the app's data so it starts as freshly installed (fair, repeatable starts)."""
        run_adb("shell", "pm", "clear", self.package, serial=self.serial)

    def grant_permissions(self) -> list[str]:
        """Grant every runtime permission the app asks for and doesn't have yet.

        Android 6+ asks for "dangerous" permissions (storage, camera, ...) at run time, in a
        dialog owned by the system's package installer, not the app. A tester sitting in that
        dialog is outside the app, and Back there means "deny", so apps like AnkiDroid ask
        again: a loop. Test setups normally grant up front (`adb install -g` does the same).
        Returns the permissions granted; ones Android refuses to grant are skipped.
        """
        out = run_adb("shell", "dumpsys", "package", self.package, serial=self.serial)
        granted = []
        for perm in parse_ungranted(out):
            try:
                out = run_adb("shell", "pm", "grant", self.package, perm, serial=self.serial)
            except RuntimeError:     # not a runtime permission on this Android version
                continue
            if "exception" not in out.lower() and "error" not in out.lower():   # older pm exits 0 on failure
                granted.append(perm)
        return granted

    def ui_dump(self) -> str | None:
        """The current screen's accessibility tree as XML, or None if Android can't produce
        one right now (e.g. "could not get idle state" while something animates)."""
        try:
            out = run_adb("shell", "uiautomator dump /sdcard/qtrain_ui.xml >/dev/null 2>&1 && "
                          "cat /sdcard/qtrain_ui.xml", serial=self.serial, timeout=6)
        except (RuntimeError, subprocess.SubprocessError):
            return None
        start = out.find("<?xml")
        return out[start:] if start >= 0 else None


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
