import math
import time
from abc import ABC, abstractmethod
from typing import Tuple

from cursor.config import DEFAULT_SPEED_PX_PER_SEC, DEFAULT_FRAME_RATE, DEFAULT_SCROLL_SPEED_PX_PER_SEC


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
        scroll_speed_sec: float = DEFAULT_SCROLL_SPEED_PX_PER_SEC,
    ) -> None:
        self.speed_px_per_sec = float(speed_px_per_sec)
        self.frame_rate = int(frame_rate)
        self.scroll_speed_sec = float(scroll_speed_sec)

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

    @abstractmethod
    def left_click(self) -> None:
        """Perform a left mouse click."""
        raise NotImplementedError

    @abstractmethod
    def right_click(self) -> None:
        """Perform a right mouse click."""
        raise NotImplementedError

    @abstractmethod
    def scroll(self, delta: int) -> None:
        """Scroll the mouse wheel. Positive delta scrolls up, negative scrolls down."""
        raise NotImplementedError

    def clamp_target(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp (x, y) to the virtual desktop bounds."""
        minx, miny, maxx, maxy = self.get_virtual_bounds()
        x = max(minx, min(x, maxx))
        y = max(miny, min(y, maxy))
        return x, y

    def move_to_with_speed(self, target_x: int, target_y: int) -> None:
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
            target_elapsed = t * duration
            now = time.perf_counter()
            sleep_time = (start_time + target_elapsed) - now
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.set_pos(target_x, target_y)

    def scroll_with_speed(self, delta: int) -> None:
        """
        Scroll the mouse wheel with the configured scroll speed in pixels per second.
        Positive delta scrolls up, negative scrolls down.
        """
        steps = max(1, int(self.frame_rate * abs(delta) / self.scroll_speed_sec))
        step_delta = 1 if delta > 0 else -1
        total_duration = abs(delta) / self.scroll_speed_sec

        start_time = time.perf_counter()
        for i in range(1, steps + 1):
            t = i / steps
            self.scroll(step_delta)
            target_elapsed = t * total_duration
            now = time.perf_counter()
            sleep_time = (start_time + target_elapsed) - now
            if sleep_time > 0:
                time.sleep(sleep_time)
