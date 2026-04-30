from __future__ import annotations

import math
import random
from typing import List, Tuple

from panda3d.core import NodePath, Vec3, Vec4


HORSE_COLORS: List[Tuple[str, Vec4]] = [
    ("red", Vec4(0.85, 0.18, 0.18, 1.0)),
    ("blue", Vec4(0.20, 0.35, 0.85, 1.0)),
    ("yellow", Vec4(0.95, 0.85, 0.15, 1.0)),
    ("green", Vec4(0.25, 0.70, 0.30, 1.0)),
    ("brown", Vec4(0.50, 0.32, 0.18, 1.0)),
    ("white", Vec4(0.92, 0.92, 0.90, 1.0)),
    ("black", Vec4(0.10, 0.10, 0.12, 1.0)),
]


def _box(loader, parent: NodePath, scale: Vec3, pos: Vec3) -> NodePath:
    np_ = loader.loadModel("models/box")
    np_.reparentTo(parent)
    np_.setScale(scale)
    np_.setPos(pos.x - scale.x / 2.0, pos.y - scale.y / 2.0, pos.z)
    return np_


def build_horse(loader, color: Vec4) -> NodePath:
    root = NodePath("horse")
    body = _box(loader, root, Vec3(2.4, 1.0, 1.1), Vec3(0, 0, 1.4))
    _box(loader, root, Vec3(1.0, 0.9, 1.4), Vec3(1.4, 0, 1.6))
    _box(loader, root, Vec3(0.6, 0.6, 0.8), Vec3(2.0, 0, 2.6))
    _box(loader, root, Vec3(0.4, 0.4, 1.4), Vec3(-0.9, -0.4, 0.0))
    _box(loader, root, Vec3(0.4, 0.4, 1.4), Vec3(-0.9, 0.4, 0.0))
    _box(loader, root, Vec3(0.4, 0.4, 1.4), Vec3(0.9, -0.4, 0.0))
    _box(loader, root, Vec3(0.4, 0.4, 1.4), Vec3(0.9, 0.4, 0.0))
    _box(loader, root, Vec3(0.3, 0.3, 0.9), Vec3(-1.4, 0, 1.5))
    root.setColor(color)
    body.setColor(color)
    return root


def spawn_horses(
    loader,
    parent: NodePath,
    track_a: float,
    track_b: float,
    count: int = 10,
    seed: int = 7,
) -> List[NodePath]:
    rng = random.Random(seed)
    out: List[NodePath] = []
    for i in range(count):
        angle = (i / count) * 2.0 * math.pi + rng.uniform(-0.15, 0.15)
        outset = rng.uniform(8.0, 22.0)
        rx = (track_a + outset) * math.cos(angle)
        ry = (track_b + outset) * math.sin(angle)
        color_name, color = rng.choice(HORSE_COLORS)
        h = build_horse(loader, color)
        h.reparentTo(parent)
        h.setPos(rx, ry, 0.0)
        h.setH(rng.uniform(0, 360))
        h.setTag("horse_color", color_name)
        out.append(h)
    return out
