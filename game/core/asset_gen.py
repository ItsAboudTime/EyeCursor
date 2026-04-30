from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np


ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
TEXTURES_DIR = ASSETS_DIR / "textures"
FONTS_DIR = ASSETS_DIR / "fonts"

GRASS_PATH = TEXTURES_DIR / "grass.png"
SKY_PATH = TEXTURES_DIR / "sky.png"
FONT_PATH = FONTS_DIR / "pixel.ttf"

FONT_URL = "https://github.com/google/fonts/raw/main/ofl/pressstart2p/PressStart2P-Regular.ttf"


def _make_grass(path: Path) -> None:
    rng = np.random.default_rng(42)
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    base[..., 1] = 110
    base[..., 0] = 30
    base[..., 2] = 40
    jitter = rng.integers(-25, 25, size=(64, 64, 3), dtype=np.int16)
    out = np.clip(base.astype(np.int16) + jitter, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


def _make_sky(path: Path) -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for y in range(64):
        t = y / 63.0
        r = int(80 + (180 - 80) * t)
        g = int(140 + (220 - 140) * t)
        b = int(200 + (240 - 200) * t)
        img[y, :, 0] = b
        img[y, :, 1] = g
        img[y, :, 2] = r
    cv2.imwrite(str(path), img)


def _try_download_font(path: Path) -> None:
    try:
        with urllib.request.urlopen(FONT_URL, timeout=5) as resp:
            data = resp.read()
        if data and len(data) > 1024:
            path.write_bytes(data)
    except Exception:
        pass


def ensure_assets() -> None:
    TEXTURES_DIR.mkdir(parents=True, exist_ok=True)
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    if not GRASS_PATH.exists():
        try:
            _make_grass(GRASS_PATH)
        except Exception:
            pass
    if not SKY_PATH.exists():
        try:
            _make_sky(SKY_PATH)
        except Exception:
            pass
    if not FONT_PATH.exists():
        _try_download_font(FONT_PATH)


def font_path() -> Path | None:
    return FONT_PATH if FONT_PATH.exists() else None


def grass_path() -> Path | None:
    return GRASS_PATH if GRASS_PATH.exists() else None


def sky_path() -> Path | None:
    return SKY_PATH if SKY_PATH.exists() else None
