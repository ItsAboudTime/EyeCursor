"""
Linux-only demo: control the mouse cursor with head pose.

Requires webcam, OpenCV, and MediaPipe. Press 'q' to quit, 'c' to calibrate.
"""

import sys
import os
import threading
import queue
import time
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cursor import create_cursor
from ui.settings import SettingsWindow

TRACKER_PATH = os.path.join(ROOT, "head-track", "tracker.py")
spec = importlib.util.spec_from_file_location("head_track_tracker", TRACKER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load tracker module from {TRACKER_PATH}")
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
HeadPoseTracker = _mod.HeadPoseTracker


def run_tracking_loop(cur, tracker, stop_queue):
    import cv2
    
    minx, miny, maxx, maxy = cur.get_virtual_bounds()
    screen_w = maxx - minx + 1
    screen_h = maxy - miny + 1

    tracker.start()
    print("Head-Cursor demo running. Press 'q' to quit, 'c' to calibrate.")

    while True:
        if not stop_queue.empty():
            if stop_queue.get() == "QUIT":
                break

        pos, frame, angles = tracker.next_position(screen_w, screen_h)

        if pos is not None:
            raw_tx, raw_ty = pos
            target_x = max(minx, min(maxx, raw_tx + minx))
            target_y = max(miny, min(maxy, raw_ty + miny))

            cur.step_towards(target_x, target_y)

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
            stop_queue.put("QUIT")
            break
        if key == ord('c'):
            if angles is not None:
                yaw, pitch = angles
                tracker.calibrate_center(yaw, pitch)

    tracker.stop()
    cv2.destroyAllWindows()


def main():
    if not sys.platform.startswith("linux"):
        print("This demo currently supports Linux only.")
        return 1

    cur = create_cursor()
    tracker = HeadPoseTracker(yaw_span=20.0, pitch_span=10.0, smooth_len=8)
    msg_queue = queue.Queue()

    try:
        root = SettingsWindow.create_app(cursor=cur)
    except Exception as e:
        print(f"Fatal Error: Could not start Tkinter: {e}")
        return 1

    def check_queue():
        try:
            msg = msg_queue.get_nowait()
            if msg == "QUIT":
                root.quit()
                sys.exit(0)
        except queue.Empty:
            pass
        root.after(100, check_queue)

    t = threading.Thread(
        target=run_tracking_loop, 
        args=(cur, tracker, msg_queue), 
        daemon=True
    )
    t.start()

    check_queue()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
