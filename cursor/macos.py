from typing import Tuple
from cursor.base import Cursor

from Quartz import CGEventCreateMouseEvent, CGEventPost, kCGEventMouseMoved
from Quartz import CGDisplayBounds, CGMainDisplayID, CGEventCreate, CGEventGetLocation


# Currently only supports single display setups.
class MacOSCursor(Cursor):
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
