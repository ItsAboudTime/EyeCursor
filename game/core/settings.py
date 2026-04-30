from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULTS: Dict[str, Any] = {
    "photo_trigger": "left_click",
    "countdown_duration": 1.0,
    "cart_speed": "normal",
}

VALID_TRIGGERS = ("left_click", "spacebar", "right_click")
VALID_DURATIONS = (0.5, 1.0, 1.5, 2.0)
VALID_SPEEDS = ("slow", "normal", "fast")

SPEED_VALUES = {"slow": 1.5, "normal": 3.0, "fast": 6.0}

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def _validate(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(DEFAULTS)
    if d.get("photo_trigger") in VALID_TRIGGERS:
        out["photo_trigger"] = d["photo_trigger"]
    if d.get("countdown_duration") in VALID_DURATIONS:
        out["countdown_duration"] = d["countdown_duration"]
    if d.get("cart_speed") in VALID_SPEEDS:
        out["cart_speed"] = d["cart_speed"]
    return out


def load() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return dict(DEFAULTS)
    try:
        raw = json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULTS)
    if not isinstance(raw, dict):
        return dict(DEFAULTS)
    return _validate(raw)


def save(data: Dict[str, Any]) -> None:
    merged = _validate(data)
    try:
        CONFIG_PATH.write_text(json.dumps(merged, indent=2))
    except OSError:
        pass


def cycle(values: tuple, current: Any) -> Any:
    try:
        i = values.index(current)
    except ValueError:
        return values[0]
    return values[(i + 1) % len(values)]


def cycle_back(values: tuple, current: Any) -> Any:
    try:
        i = values.index(current)
    except ValueError:
        return values[-1]
    return values[(i - 1) % len(values)]
