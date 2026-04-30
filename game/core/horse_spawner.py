from __future__ import annotations

import math
import random
from typing import List

from panda3d.core import NodePath, Vec3

from game.core.horse_roster import HorseSpecies, pick_species


def _box(loader, parent: NodePath, scale: Vec3, pos: Vec3) -> NodePath:
    np_ = loader.loadModel("models/box")
    np_.reparentTo(parent)
    np_.setScale(scale)
    np_.setPos(pos.x - scale.x / 2.0, pos.y - scale.y / 2.0, pos.z)
    return np_


def build_horse(loader, species: HorseSpecies) -> NodePath:
    root = NodePath("horse")
    body = _box(loader, root, Vec3(2.4, 1.0, 1.1), Vec3(0, 0, 1.4))
    head = _box(loader, root, Vec3(1.0, 0.9, 1.4), Vec3(1.4, 0, 1.6))
    snout = _box(loader, root, Vec3(0.6, 0.6, 0.8), Vec3(2.0, 0, 2.6))
    leg1 = _box(loader, root, Vec3(0.4, 0.4, 1.4), Vec3(-0.9, -0.4, 0.0))
    leg2 = _box(loader, root, Vec3(0.4, 0.4, 1.4), Vec3(-0.9, 0.4, 0.0))
    leg3 = _box(loader, root, Vec3(0.4, 0.4, 1.4), Vec3(0.9, -0.4, 0.0))
    leg4 = _box(loader, root, Vec3(0.4, 0.4, 1.4), Vec3(0.9, 0.4, 0.0))
    mane = _box(loader, root, Vec3(0.3, 0.3, 0.9), Vec3(-1.4, 0, 1.5))
    body.setColor(species.body)
    head.setColor(species.head)
    snout.setColor(species.head)
    for leg in (leg1, leg2, leg3, leg4):
        leg.setColor(species.legs)
    mane.setColor(species.mane)
    return root


def spawn_horses(
    loader,
    parent: NodePath,
    track,
    count: int = 10,
    map_id: str = "meadow",
    seed: int = 7,
) -> List[NodePath]:
    rng = random.Random(seed)
    out: List[NodePath] = []
    for i in range(count):
        jitter = rng.uniform(-0.5, 0.5) / count
        t = ((i / count) + jitter) % 1.0
        pos, _ = track.evaluate(t)
        outset = rng.uniform(8.0, 22.0)
        r = math.hypot(pos.x, pos.y) or 1.0
        ux, uy = pos.x / r, pos.y / r
        hx = pos.x + ux * outset
        hy = pos.y + uy * outset
        species = pick_species(rng, map_id)
        h = build_horse(loader, species)
        h.reparentTo(parent)
        h.setPos(hx, hy, 0.0)
        h.setH(rng.uniform(0, 360))
        h.setTag("species_id", species.id)
        out.append(h)
    return out
