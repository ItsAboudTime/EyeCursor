from __future__ import annotations

import math

from panda3d.core import NodePath, Vec3, Vec4

from game.core.track import OvalTrack


RAIL_COLOR = Vec4(0.62, 0.64, 0.68, 1.0)
SLEEPER_COLOR = Vec4(0.42, 0.27, 0.15, 1.0)

RAIL_GAUGE = 1.10
RAIL_WIDTH = 0.10
RAIL_HEIGHT = 0.12
RAIL_BASE_Z = 0.10

SLEEPER_LENGTH = 1.50
SLEEPER_WIDTH = 0.30
SLEEPER_HEIGHT = 0.08
SLEEPER_BASE_Z = 0.00

NUM_SAMPLES = 240
SLEEPERS_EVERY = 4


def _oriented_box(
    loader,
    parent: NodePath,
    sx: float,
    sy: float,
    sz: float,
    pos: Vec3,
    h_deg: float,
    color: Vec4,
) -> NodePath:
    holder = parent.attachNewNode("seg")
    np_ = loader.loadModel("models/box")
    np_.reparentTo(holder)
    np_.setScale(sx, sy, sz)
    np_.setPos(-sx / 2.0, -sy / 2.0, -sz / 2.0)
    holder.setPos(pos)
    holder.setH(h_deg)
    holder.setColor(color)
    return holder


def build_rails(loader, parent: NodePath, track: OvalTrack) -> NodePath:
    root = parent.attachNewNode("rails")

    samples = []
    for i in range(NUM_SAMPLES):
        t = i / NUM_SAMPLES
        pos, _ = track.evaluate(t)
        samples.append(pos)

    for i in range(NUM_SAMPLES):
        p0 = samples[i]
        p1 = samples[(i + 1) % NUM_SAMPLES]
        seg = p1 - p0
        seg_len = max(0.001, seg.length())
        h_deg = math.degrees(math.atan2(-seg.x, seg.y))
        mid_x = (p0.x + p1.x) / 2.0
        mid_y = (p0.y + p1.y) / 2.0

        # Right-perpendicular unit vector in the xy plane.
        nx = seg.y / seg_len
        ny = -seg.x / seg_len

        rail_z = RAIL_BASE_Z + RAIL_HEIGHT / 2.0
        right_pos = Vec3(
            mid_x + nx * RAIL_GAUGE / 2.0,
            mid_y + ny * RAIL_GAUGE / 2.0,
            rail_z,
        )
        left_pos = Vec3(
            mid_x - nx * RAIL_GAUGE / 2.0,
            mid_y - ny * RAIL_GAUGE / 2.0,
            rail_z,
        )
        # Overshoot length slightly so adjacent segments overlap and don't gap on curves.
        seg_y = seg_len + 0.02
        _oriented_box(loader, root, RAIL_WIDTH, seg_y, RAIL_HEIGHT, right_pos, h_deg, RAIL_COLOR)
        _oriented_box(loader, root, RAIL_WIDTH, seg_y, RAIL_HEIGHT, left_pos, h_deg, RAIL_COLOR)

        if i % SLEEPERS_EVERY == 0:
            sleeper_z = SLEEPER_BASE_Z + SLEEPER_HEIGHT / 2.0
            _oriented_box(
                loader,
                root,
                SLEEPER_LENGTH,
                SLEEPER_WIDTH,
                SLEEPER_HEIGHT,
                Vec3(mid_x, mid_y, sleeper_z),
                h_deg,
                SLEEPER_COLOR,
            )

    root.flattenStrong()
    return root
