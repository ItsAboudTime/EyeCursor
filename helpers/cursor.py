# Helpers and constants for cursor movement (Windows)

import ctypes
import time
import math

user32 = ctypes.windll.user32

# Virtual desktop metrics (works with multi-monitor)
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

# Tweakable speed settings
SPEED_PX_PER_SEC = 1000.0   # pixels per second (increase for faster, decrease for slower)
FRAME_RATE = 120            # animation updates per second (higher = smoother)

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

def get_cursor_pos():
    pt = POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

def set_cursor_pos(x, y):
    user32.SetCursorPos(int(x), int(y))

def get_virtual_bounds():
    minx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    miny = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    w = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    h = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    return minx, miny, minx + w - 1, miny + h - 1

def clamp_target(x, y):
    minx, miny, maxx, maxy = get_virtual_bounds()
    x = max(minx, min(x, maxx))
    y = max(miny, min(y, maxy))
    return x, y

def move_to(target_x, target_y):
    cx, cy = get_cursor_pos()
    target_x, target_y = clamp_target(int(target_x), int(target_y))

    dx = target_x - cx
    dy = target_y - cy
    dist = math.hypot(dx, dy)

    if dist < 1:
        set_cursor_pos(target_x, target_y)
        return

    duration = dist / max(1e-6, SPEED_PX_PER_SEC)
    steps = max(1, int(FRAME_RATE * duration))

    start_time = time.perf_counter()
    for i in range(1, steps + 1):
        t = i / steps
        nx = round(cx + dx * t)
        ny = round(cy + dy * t)
        set_cursor_pos(nx, ny)

        # keep approximate timing
        target_elapsed = t * duration
        now = time.perf_counter()
        sleep_time = (start_time + target_elapsed) - now
        if sleep_time > 0:
            time.sleep(sleep_time)

    set_cursor_pos(target_x, target_y)

def parse_coords(raw):
    s = raw.strip()
    if s.lower() == "q":
        return "quit"
    s = s.replace(",", " ")
    parts = [p for p in s.split() if p]
    if len(parts) != 2:
        raise ValueError("Please enter two numbers like: 800 400 or 800,400")
    x = int(float(parts[0]))
    y = int(float(parts[1]))
    return x, y
