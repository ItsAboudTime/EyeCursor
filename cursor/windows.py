import ctypes
from typing import Tuple

from cursor.base import Cursor

user32 = ctypes.windll.user32

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class WindowsCursor(Cursor):
    def get_pos(self) -> Tuple[int, int]:
        pt = POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def set_pos(self, x: int, y: int) -> None:
        user32.SetCursorPos(int(x), int(y))

    def get_virtual_bounds(self) -> Tuple[int, int, int, int]:
        minx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        miny = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        w = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        h = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
        maxx = minx + w - 1
        maxy = miny + h - 1
        return minx, miny, maxx, maxy

    def left_click(self) -> None:
        user32.mouse_event(0x0002, 0, 0, 0, 0) 
        user32.mouse_event(0x0004, 0, 0, 0, 0)  

    def right_click(self) -> None:
        user32.mouse_event(0x0008, 0, 0, 0, 0) 
        user32.mouse_event(0x0010, 0, 0, 0, 0)

    def scroll(self, delta: int) -> None:
        user32.mouse_event(0x0800, 0, 0, delta, 0)
