import math
import time
from abc import ABC, abstractmethod
from typing import Tuple

from cursor.config import DEFAULT_SPEED_PX_PER_SEC, DEFAULT_FRAME_RATE


class Cursor(ABC):
    """
    Abstract cursor interface + shared animation logic.

    Subclasses must implement:
      - get_pos
      - set_pos
      - get_virtual_bounds
    """

    def __init__(
        self,
        speed_px_per_sec: float = DEFAULT_SPEED_PX_PER_SEC,
        frame_rate: int = DEFAULT_FRAME_RATE,
    ) -> None:
        self.speed_px_per_sec = float(speed_px_per_sec)
        self.frame_rate = int(frame_rate)

    # ---- primitives that OS-specific subclasses must implement ----

    @abstractmethod
    def get_pos(self) -> Tuple[int, int]:
        """Return the current cursor position as (x, y)."""
        raise NotImplementedError

    @abstractmethod
    def set_pos(self, x: int, y: int) -> None:
        """Set the cursor position to absolute coordinates (x, y)."""
        raise NotImplementedError

    @abstractmethod
    def get_virtual_bounds(self) -> Tuple[int, int, int, int]:
        """
        Return virtual desktop bounds as (minx, miny, maxx, maxy).

        Should account for multi-monitor setups when possible.
        """
        raise NotImplementedError

    # ---- shared helpers implemented in the base class ----

    def clamp_target(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp (x, y) to the virtual desktop bounds."""
        minx, miny, maxx, maxy = self.get_virtual_bounds()
        x = max(minx, min(x, maxx))
        y = max(miny, min(y, maxy))
        return x, y

    def move_to(self, target_x: int, target_y: int) -> None:
        """
        Smoothly move the cursor to (target_x, target_y) using the configured
        speed and frame rate.
        """
        cx, cy = self.get_pos()
        target_x, target_y = self.clamp_target(int(target_x), int(target_y))

        dx = target_x - cx
        dy = target_y - cy
        dist = math.hypot(dx, dy)

        if dist < 1:
            self.set_pos(target_x, target_y)
            return

        duration = dist / max(1e-6, self.speed_px_per_sec)
        steps = max(1, int(self.frame_rate * duration))

        start_time = time.perf_counter()
        for i in range(1, steps + 1):
            t = i / steps
            nx = round(cx + dx * t)
            ny = round(cy + dy * t)
            self.set_pos(nx, ny)

            # keep approximate timing
            target_elapsed = t * duration
            now = time.perf_counter()
            sleep_time = (start_time + target_elapsed) - now
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.set_pos(target_x, target_y)
