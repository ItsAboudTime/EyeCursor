from __future__ import annotations

import math
from typing import Tuple

from panda3d.core import Vec3


class OvalTrack:
    def __init__(self, a: float = 30.0, b: float = 18.0, height: float = 1.5) -> None:
        self.a = a
        self.b = b
        self.height = height
        h = ((a - b) ** 2) / ((a + b) ** 2)
        self.circumference = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))

    def evaluate(self, t: float) -> Tuple[Vec3, Vec3]:
        theta = t * 2.0 * math.pi
        x = self.a * math.cos(theta)
        y = self.b * math.sin(theta)
        pos = Vec3(x, y, self.height)
        dx = -self.a * math.sin(theta)
        dy = self.b * math.cos(theta)
        length = math.sqrt(dx * dx + dy * dy) or 1.0
        tangent = Vec3(dx / length, dy / length, 0.0)
        return pos, tangent

    def advance(self, t: float, speed: float, dt: float) -> float:
        if self.circumference <= 0:
            return 0.0
        t += (speed * dt) / self.circumference
        return t % 1.0
