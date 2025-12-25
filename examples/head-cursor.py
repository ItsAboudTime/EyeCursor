"""
Linux-only demo: control the mouse cursor with head pose.

Requires webcam, OpenCV, and MediaPipe. Press 'q' to quit, 'c' to calibrate.
"""

import sys
import os

# Ensure project root on sys.path for local imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cursor import create_cursor
import importlib.util
TRACKER_PATH = os.path.join(ROOT, "head-track", "tracker.py")
spec = importlib.util.spec_from_file_location("head_track_tracker", TRACKER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load tracker module from {TRACKER_PATH}")
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
HeadPoseTracker = _mod.HeadPoseTracker


def main() -> int:
    if not sys.platform.startswith("linux"):
        print("This demo currently supports Linux only.")
        return 1

    cur = create_cursor()
    minx, miny, maxx, maxy = cur.get_virtual_bounds()
    screen_w = maxx - minx + 1
    screen_h = maxy - miny + 1

    tracker = HeadPoseTracker(yaw_span=20.0, pitch_span=10.0, smooth_len=8)
    tracker.start()
    print("Head-Cursor demo running (Linux). Press 'q' to quit, 'c' to calibrate.")

    while True:
        pos, frame, angles = tracker.next_position(screen_w, screen_h)
        if pos is not None:
            sx, sy = pos
            sx = max(minx, min(maxx, sx + minx))
            sy = max(miny, min(maxy, sy + miny))
            cur.move_to_with_speed(sx, sy)

        import cv2
        cv2.putText(
            frame,
            "Press 'c' to calibrate center, 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Head Cursor (Linux)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord('c'):
            if angles is not None:
                yaw, pitch = angles
                tracker.calibrate_center(yaw, pitch)

    tracker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
