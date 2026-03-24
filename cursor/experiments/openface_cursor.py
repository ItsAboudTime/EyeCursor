"""
Linux-only demo: control the mouse cursor with head pose via OpenFace 2.0.

Drop-in alternative to ``head_cursor.py`` that uses an OpenFace Docker
container instead of MediaPipe.

Prerequisites
-------------
1. Build the OpenFace ZMQ server image (once)::

       cd openface_server
       docker build -t openface-server .

2. Start the container::

       docker run --rm -p 5555:5555 openface-server

3. Install ``pyzmq`` on the host::

       pip install pyzmq

4. Run this program::

       python examples/openface_cursor.py

Press **c** to calibrate (centres the current head position),
**q** to quit.

Wink detection
--------------
OpenFace provides AU45_r (blink intensity).  A value above the threshold
triggers a left click.  A cooldown prevents repeated clicks from a single
blink.
"""

import sys
import threading
import queue
import time

from cursor import create_cursor
from ui.settings import SettingsWindow
from head_track import OpenFaceTracker


# AU45 blink-click parameters
AU45_CLICK_THRESHOLD = 2.0  # AU45_r intensity above which we register a click
CLICK_COOLDOWN = 0.6  # seconds between successive clicks


def run_tracking_loop(cur, tracker, stop_queue):
    import cv2

    minx, miny, maxx, maxy = cur.get_virtual_bounds()
    screen_w = maxx - minx + 1
    screen_h = maxy - miny + 1

    tracker.start()
    print("OpenFace Cursor demo running. Press 'q' to quit, 'c' to calibrate.")

    last_click_time = 0.0

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

        # Wink / blink click detection via AU45_r
        now = time.time()
        if tracker.last_au45 > AU45_CLICK_THRESHOLD:
            if now - last_click_time > CLICK_COOLDOWN:
                cur.left_click()
                last_click_time = now

        # HUD overlay
        status = "No face" if pos is None else f"AU45={tracker.last_au45:.2f}"
        cv2.putText(
            frame,
            f"'c' calibrate | 'q' quit | {status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("OpenFace Cursor (Linux)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            stop_queue.put("QUIT")
            break
        if key == ord("c"):
            if angles is not None:
                yaw, pitch = angles
                tracker.calibrate_center(yaw, pitch)
                print("Calibrated.")

    tracker.stop()
    cv2.destroyAllWindows()


def main():
    if not sys.platform.startswith("linux"):
        print("This demo currently supports Linux only.")
        return 1

    cur = create_cursor()
    tracker = OpenFaceTracker(
        host="localhost",
        port=5555,
        yaw_span=20.0,
        pitch_span=10.0,
        smooth_len=8,
        debug=True,
    )
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
        daemon=True,
    )
    t.start()

    check_queue()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
