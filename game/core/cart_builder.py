from __future__ import annotations

from panda3d.core import NodePath, Vec4


CART_BODY_COLOR = Vec4(0.55, 0.32, 0.26, 1.0)
CART_TRIM_COLOR = Vec4(0.30, 0.18, 0.14, 1.0)
WHEEL_COLOR = Vec4(0.10, 0.10, 0.12, 1.0)


def _box(loader, parent, sx, sy, sz, x, y, z, color):
    np_ = loader.loadModel("models/box")
    np_.reparentTo(parent)
    np_.setScale(sx, sy, sz)
    np_.setPos(x - sx / 2.0, y - sy / 2.0, z - sz / 2.0)
    np_.setColor(color)
    return np_


def build_cart(loader, parent: NodePath) -> NodePath:
    """Minecart visual parented to the cart node. Local axes: +Y forward, +X right,
    +Z up. Camera sits at parent origin, so the cabin walls top out just below z=0."""
    root = parent.attachNewNode("cart_visual")

    floor_z = -1.05
    _box(loader, root, 1.30, 1.80, 0.10, 0.0, 0.0, floor_z, CART_TRIM_COLOR)

    wall_top = -0.40
    wall_bot = -1.00
    wall_h = wall_top - wall_bot
    wall_cz = (wall_top + wall_bot) / 2.0
    wall_thick = 0.10

    _box(loader, root, wall_thick, 1.80, wall_h, -0.65, 0.0, wall_cz, CART_BODY_COLOR)
    _box(loader, root, wall_thick, 1.80, wall_h, +0.65, 0.0, wall_cz, CART_BODY_COLOR)
    _box(loader, root, 1.40, wall_thick, wall_h, 0.0, +0.90, wall_cz, CART_BODY_COLOR)
    _box(loader, root, 1.40, wall_thick, wall_h, 0.0, -0.90, wall_cz, CART_BODY_COLOR)

    rim_thick = 0.06
    rim_z = wall_top + 0.03
    _box(loader, root, 1.50, 0.10, rim_thick, 0.0, +0.95, rim_z, CART_TRIM_COLOR)
    _box(loader, root, 1.50, 0.10, rim_thick, 0.0, -0.95, rim_z, CART_TRIM_COLOR)
    _box(loader, root, 0.10, 1.80, rim_thick, -0.70, 0.0, rim_z, CART_TRIM_COLOR)
    _box(loader, root, 0.10, 1.80, rim_thick, +0.70, 0.0, rim_z, CART_TRIM_COLOR)

    wheel_z = -1.30
    for x in (-0.55, +0.55):
        for y in (-0.70, +0.70):
            _box(loader, root, 0.30, 0.30, 0.30, x, y, wheel_z, WHEEL_COLOR)

    root.flattenStrong()
    return root
