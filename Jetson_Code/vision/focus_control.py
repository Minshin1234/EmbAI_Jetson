"""Manual focus control for the Arducam lens."""

from __future__ import annotations

from pathlib import Path
import subprocess
import threading

from vision.config import (
    DEPTH_FOCUS_BUS,
    DEPTH_FOCUS_DEFAULT,
    DEPTH_FOCUS_ENABLED,
    DEPTH_FOCUS_MAX,
    DEPTH_FOCUS_MIN,
)


_lock = threading.RLock()
_current_focus = DEPTH_FOCUS_DEFAULT
_last_error = None
_active_bus = None
_PROBE_BUSES = [DEPTH_FOCUS_BUS, 10, 9, 7]


def _clamp(value: int) -> int:
    return max(DEPTH_FOCUS_MIN, min(DEPTH_FOCUS_MAX, int(value)))


def _device_exists(bus: int) -> bool:
    return Path(f"/dev/i2c-{bus}").exists()


def _run_i2cset(bus: int, register: int, value: int) -> tuple[bool, str | None]:
    cmd = ["i2cset", "-y", str(bus), "0x0c", f"0x{register:02x}", f"0x{value:02x}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True, None
    error = (result.stderr or result.stdout or "i2cset failed").strip()
    return False, error


def _write_focus_raw(bus: int, focus_value: int) -> tuple[bool, str | None]:
    scaled = int(_clamp(focus_value) / 1000.0 * 4095.0)
    shifted = scaled << 4
    high = (shifted >> 8) & 0xFF
    low = shifted & 0xFF

    ok, error = _run_i2cset(bus, 0x02, 0x00)
    if not ok:
        return False, error
    ok, error = _run_i2cset(bus, 0x00, high)
    if not ok:
        return False, error
    ok, error = _run_i2cset(bus, 0x01, low)
    if not ok:
        return False, error
    return True, None


def _candidate_buses() -> list[int]:
    seen = set()
    buses = []
    for bus in _PROBE_BUSES:
        if bus in seen:
            continue
        seen.add(bus)
        buses.append(bus)
    return buses


def _ensure_focus_bus():
    global _active_bus, _last_error

    if not DEPTH_FOCUS_ENABLED:
        _last_error = "Focus control disabled in config."
        return None

    if _active_bus is not None and _device_exists(_active_bus):
        return _active_bus

    last_error = None
    for bus in _candidate_buses():
        if not _device_exists(bus):
            continue
        ok, error = _write_focus_raw(bus, _current_focus)
        if ok:
            _active_bus = bus
            _last_error = None
            return _active_bus
        last_error = f"i2c-{bus}: {error}"

    _active_bus = None
    _last_error = last_error or "No usable /dev/i2c-* focus bus found."
    return None


def get_focus_state():
    """Return the current focus state for the UI."""
    with _lock:
        active_bus = _ensure_focus_bus()
        return {
            "enabled": bool(DEPTH_FOCUS_ENABLED),
            "available": active_bus is not None,
            "value": int(_current_focus),
            "min": int(DEPTH_FOCUS_MIN),
            "max": int(DEPTH_FOCUS_MAX),
            "bus": None if active_bus is None else int(active_bus),
            "error": _last_error,
        }


def set_focus_value(value: int):
    """Set the Arducam manual focus value."""
    global _current_focus, _last_error, _active_bus

    with _lock:
        clamped = _clamp(value)
        bus = _ensure_focus_bus()
        if bus is None:
            return {"ok": False, "focus": get_focus_state()}

        ok, error = _write_focus_raw(bus, clamped)
        if not ok:
            _last_error = error
            _active_bus = None
            return {"ok": False, "focus": get_focus_state()}

        _current_focus = clamped
        _last_error = None
        return {"ok": True, "focus": get_focus_state()}
