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
