from __future__ import annotations

import math
from typing import Tuple

from panda3d.core import Vec3


# Closed polar curve r(theta) = BASE + sum(amp_k * cos(k*theta + phase_k)).
# Tuned for a non-oval scenic loop noticeably longer than the meadow oval.
# All r > 0 -> no self-crossings.
BASE_R = 32.0
HARMONICS = (
    (3, 10.0, 0.0),
    (5, 4.0, 0.5),
)
HEIGHT = 1.5


class WindingTrack:
    def __init__(self) -> None:
        self.height = HEIGHT
        self.circumference = self._compute_length(samples=2048)

    def _r(self, theta: float) -> float:
        r = BASE_R
        for k, amp, phase in HARMONICS:
            r += amp * math.cos(k * theta + phase)
        return r

    def _r_dot(self, theta: float) -> float:
        d = 0.0
        for k, amp, phase in HARMONICS:
            d += -amp * k * math.sin(k * theta + phase)
        return d

    def _point(self, theta: float) -> Vec3:
        r = self._r(theta)
        return Vec3(r * math.cos(theta), r * math.sin(theta), self.height)

    def _compute_length(self, samples: int) -> float:
        n = max(8, samples)
        prev = self._point(0.0)
        total = 0.0
        for i in range(1, n + 1):
            cur = self._point(2.0 * math.pi * i / n)
            total += (cur - prev).length()
            prev = cur
        return total

    def evaluate(self, t: float) -> Tuple[Vec3, Vec3]:
        theta = t * 2.0 * math.pi
        pos = self._point(theta)
        r = self._r(theta)
        rd = self._r_dot(theta)
        tx = rd * math.cos(theta) - r * math.sin(theta)
        ty = rd * math.sin(theta) + r * math.cos(theta)
        length = math.sqrt(tx * tx + ty * ty) or 1.0
        return pos, Vec3(tx / length, ty / length, 0.0)

    def advance(self, t: float, speed: float, dt: float) -> float:
        if self.circumference <= 0:
            return 0.0
        t += (speed * dt) / self.circumference
        return t % 1.0
