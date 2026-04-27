from __future__ import annotations

import struct
import tempfile
import wave
from math import pi, sin
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QSoundEffect

_SAMPLE_RATE = 44100
_cache_dir: Path | None = None
_effects: dict[str, QSoundEffect] = {}


def _get_cache_dir() -> Path:
    global _cache_dir
    if _cache_dir is None:
        _cache_dir = Path(tempfile.mkdtemp(prefix="eyecursor_sfx_"))
    return _cache_dir


def _generate_wav(path: Path, tones: list[tuple[float, float, float]]) -> None:
    """Generate a WAV file from a list of (frequency_hz, duration_s, amplitude 0-1) segments."""
    samples: list[int] = []
    for freq, duration, amplitude in tones:
        n = int(_SAMPLE_RATE * duration)
        for i in range(n):
            fade = 1.0
            tail = int(0.01 * _SAMPLE_RATE)
            if i >= n - tail:
                fade = (n - i) / tail
            if i < tail:
                fade = min(fade, i / tail)
            t = i / _SAMPLE_RATE
            value = amplitude * fade * sin(2 * pi * freq * t)
            samples.append(int(value * 32767))
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _ensure_effects() -> None:
    if _effects:
        return
    cache = _get_cache_dir()

    ding_path = cache / "ding.wav"
    _generate_wav(ding_path, [(880, 0.08, 0.35), (1320, 0.10, 0.30)])
    _effects["ding"] = QSoundEffect()
    _effects["ding"].setSource(QUrl.fromLocalFile(str(ding_path)))
    _effects["ding"].setVolume(0.5)

    success_path = cache / "success.wav"
    _generate_wav(success_path, [(660, 0.08, 0.30), (880, 0.08, 0.30), (1100, 0.12, 0.35)])
    _effects["success"] = QSoundEffect()
    _effects["success"].setSource(QUrl.fromLocalFile(str(success_path)))
    _effects["success"].setVolume(0.5)

    error_path = cache / "error.wav"
    _generate_wav(error_path, [(220, 0.15, 0.40), (165, 0.20, 0.35)])
    _effects["error"] = QSoundEffect()
    _effects["error"].setSource(QUrl.fromLocalFile(str(error_path)))
    _effects["error"].setVolume(0.5)

    tick_path = cache / "tick.wav"
    _generate_wav(tick_path, [(1000, 0.05, 0.25)])
    _effects["tick"] = QSoundEffect()
    _effects["tick"].setSource(QUrl.fromLocalFile(str(tick_path)))
    _effects["tick"].setVolume(0.4)

    warning_path = cache / "warning.wav"
    _generate_wav(warning_path, [(600, 0.10, 0.30), (600, 0.10, 0.30)])
    _effects["warning"] = QSoundEffect()
    _effects["warning"].setSource(QUrl.fromLocalFile(str(warning_path)))
    _effects["warning"].setVolume(0.5)


def play(name: str) -> None:
    _ensure_effects()
    effect = _effects.get(name)
    if effect:
        effect.play()
