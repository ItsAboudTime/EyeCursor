import math
import time
from abc import ABC, abstractmethod
from typing import Tuple

from cursor.constants import DEFAULT_MOVE_PX_PER_SEC, DEFAULT_FRAME_RATE, DEFAULT_SCROLL_UNITS_PER_SEC


class Cursor(ABC):
    """
    Abstract cursor interface + shared animation logic.

    Subclasses must implement:
      - get_pos
      - set_pos
      - get_virtual_bounds
      - left_click
      - right_click
      - scroll
    """

    def __init__(
        self,
        move_px_per_sec: float = DEFAULT_MOVE_PX_PER_SEC,
        frame_rate: int = DEFAULT_FRAME_RATE,
        scroll_units_per_sec: float = DEFAULT_SCROLL_UNITS_PER_SEC,
    ) -> None:
        self.move_px_per_sec = float(move_px_per_sec)
        self.frame_rate = int(frame_rate)
        self.scroll_units_per_sec = float(scroll_units_per_sec)

    def update_config(
        self,
        move_px_per_sec: float,
        frame_rate: int,
        scroll_units_per_sec: float,
    ) -> None:
        """Update cursor configuration."""
        self.move_px_per_sec = float(move_px_per_sec)
        self.frame_rate = int(frame_rate)
        self.scroll_units_per_sec = float(scroll_units_per_sec)

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
        move_px_per_sec and frame rate.
        """
        cx, cy = self.get_pos()
        target_x, target_y = self.clamp_target(int(target_x), int(target_y))

        dx = target_x - cx
        dy = target_y - cy
        dist = math.hypot(dx, dy)

        if dist < 1:
            self.set_pos(target_x, target_y)
            return

        duration = dist / max(1e-6, self.move_px_per_sec)
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
    
    def step_towards(self, target_x: int, target_y: int) -> None:
        """
        Non-blocking: Moves the cursor one 'step' towards the target.
        Automatically calculates the time delta (dt) since the last call.
        """
        import time
        import math
        
        now = time.perf_counter()
        
        if not hasattr(self, "_last_step_time") or self._last_step_time is None:
            self._last_step_time = now
            return 

        dt = now - self._last_step_time

        min_interval = 1.0 / max(1, self.frame_rate)
        if dt < min_interval:
            return

        self._last_step_time = now

        cx, cy = self.get_pos()
        
        dx = target_x - cx
        dy = target_y - cy
        dist = math.hypot(dx, dy)

        step_size = self.move_px_per_sec * dt

        if dist <= step_size:
            self.set_pos(int(target_x), int(target_y))
        else:
            ratio = step_size / dist
            nx = cx + (dx * ratio)
            ny = cy + (dy * ratio)
            self.set_pos(int(nx), int(ny))
        
    def scroll_with_speed(self, delta: int) -> None:
        """
        Scroll the mouse wheel with the configured scroll speed.
        Uses an accumulator to handle sub-unit scrolling smoothly.
        """
        if delta == 0:
            return

        total_duration = abs(delta) / max(1e-6, self.scroll_units_per_sec)
        steps = max(1, int(self.frame_rate * total_duration))
        per_step_scroll = delta / steps
        accumulator = 0.0
        start_time = time.perf_counter()

        for i in range(1, steps + 1):
            # Add the fractional amount to our "bucket"
            accumulator += per_step_scroll
            
            # Check if we have enough in the bucket to actually scroll (>= 1 or <= -1)
            scroll_amount = int(accumulator)
            
            if scroll_amount != 0:
                self.scroll(scroll_amount)
                accumulator -= scroll_amount

            target_elapsed = (i / steps) * total_duration
            now = time.perf_counter()
            sleep_time = (start_time + target_elapsed) - now
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Handles cases where rounding errors left 1 tick behind
        remaining = int(accumulator + 0.5) if delta > 0 else int(accumulator - 0.5)
        if remaining != 0:
            self.scroll(remaining)
