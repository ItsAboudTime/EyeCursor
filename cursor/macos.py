import time
from typing import Tuple
from cursor.base import Cursor

from Quartz import (
    CGEventCreateMouseEvent,
    CGEventPost,
    CGEventSetIntegerValueField,
    kCGEventMouseMoved,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGEventRightMouseDown,
    kCGEventRightMouseUp,
    kCGMouseEventClickState,
    CGDisplayBounds,
    CGMainDisplayID,
    CGEventCreate,
    CGEventGetLocation,
    CGEventCreateScrollWheelEvent,
)


class MacOSCursor(Cursor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._double_click_max_interval = 0.45
        self._left_click_count = 0
        self._right_click_count = 0
        self._left_last_up_time = 0.0
        self._right_last_up_time = 0.0
        self._left_down_click_count = 1
        self._right_down_click_count = 1

    def _next_click_count(self, last_up_time: float, current_click_count: int) -> int:
        now = time.perf_counter()
        if now - last_up_time <= self._double_click_max_interval:
            return min(3, current_click_count + 1)
        return 1

    def get_pos(self) -> Tuple[int, int]:
        event = CGEventCreate(None)
        location = CGEventGetLocation(event)
        return int(location.x), int(location.y)

    def set_pos(self, x: int, y: int) -> None:
        event = CGEventCreateMouseEvent(
            None, kCGEventMouseMoved, (int(x), int(y)), 0
        )
        CGEventPost(0, event)

    def get_virtual_bounds(self) -> Tuple[int, int, int, int]:
        bounds = CGDisplayBounds(CGMainDisplayID())
        minx = int(bounds.origin.x)
        miny = int(bounds.origin.y)
        maxx = int(bounds.origin.x + bounds.size.width - 1)
        maxy = int(bounds.origin.y + bounds.size.height - 1)
        return minx, miny, maxx, maxy

    def left_click(self) -> None:
        self.left_down()
        self.left_up()

    def left_down(self) -> None:
        self._left_down_click_count = self._next_click_count(
            self._left_last_up_time,
            self._left_click_count,
        )
        event_down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, self.get_pos(), 0)
        CGEventSetIntegerValueField(event_down, kCGMouseEventClickState, self._left_down_click_count)
        CGEventPost(0, event_down)

    def left_up(self) -> None:
        event_up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, self.get_pos(), 0)
        CGEventSetIntegerValueField(event_up, kCGMouseEventClickState, self._left_down_click_count)
        CGEventPost(0, event_up)
        self._left_click_count = self._left_down_click_count
        self._left_last_up_time = time.perf_counter()

    def right_click(self) -> None:
        self.right_down()
        self.right_up()

    def right_down(self) -> None:
        self._right_down_click_count = self._next_click_count(
            self._right_last_up_time,
            self._right_click_count,
        )
        event_down = CGEventCreateMouseEvent(None, kCGEventRightMouseDown, self.get_pos(), 0)
        CGEventSetIntegerValueField(event_down, kCGMouseEventClickState, self._right_down_click_count)
        CGEventPost(0, event_down)

    def right_up(self) -> None:
        event_up = CGEventCreateMouseEvent(None, kCGEventRightMouseUp, self.get_pos(), 0)
        CGEventSetIntegerValueField(event_up, kCGMouseEventClickState, self._right_down_click_count)
        CGEventPost(0, event_up)
        self._right_click_count = self._right_down_click_count
        self._right_last_up_time = time.perf_counter()

    def scroll(self, delta: int) -> None:
        event = CGEventCreateScrollWheelEvent(None, 0, 1, delta)
        CGEventPost(0, event)
