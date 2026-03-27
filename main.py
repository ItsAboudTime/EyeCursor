"""
Combined demo: control the mouse cursor with head pose + wink gestures.

Requires webcam, OpenCV, MediaPipe, and the project's `cursor` and `head_track` modules.
Cross-platform note: keyboard controls are handled by the Tkinter window.
Press 'q' (or Esc) to quit, 'c' to calibrate (centers current head pose).
"""

import sys
import threading
import queue
import time

import cv2

from cursor import create_cursor
from ui.settings import SettingsWindow
from head_track.face_analysis_pipeline import FaceAnalysisPipeline


def run_tracking_loop(cur, stop_queue, control_queue):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0).")
        stop_queue.put("QUIT")
        return

    face_analysis_pipeline = FaceAnalysisPipeline(yaw_span=40.0, pitch_span=20.0, ema_alpha=0.1)

    minx, miny, maxx, maxy = cur.get_virtual_bounds()
    screen_w = maxx - minx + 1
    screen_h = maxy - miny + 1

    # Repeated winks keep the corresponding mouse button held for dragging.
    HOLD_REFRESH_WINDOW = 0.35
    left_is_down = False
    right_is_down = False
    left_held_until = 0.0
    right_held_until = 0.0
    latest_angles = None

    print("Head+Wink Cursor demo running.")
    print("Focus the settings window and press 'c' to calibrate, 'q' or Esc to quit.")

    try:
        while True:
            should_quit = False
            while not control_queue.empty():
                cmd = control_queue.get()
                if cmd == "QUIT":
                    should_quit = True
                    break
                if cmd == "CALIBRATE" and latest_angles is not None:
                    face_analysis_pipeline.calibrate_to_center(*latest_angles)

            if should_quit:
                stop_queue.put("QUIT")
                break

            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_analysis = face_analysis_pipeline.analyze(
                rgb_frame=rgb,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                screen_width=screen_w,
                screen_height=screen_h,
            )

            if face_analysis is not None:
                if face_analysis.screen_position is not None:
                    raw_tx, raw_ty = face_analysis.screen_position
                    target_x = max(minx, min(maxx, raw_tx + minx))
                    target_y = max(miny, min(maxy, raw_ty + miny))
                    cur.step_towards(target_x, target_y)
                latest_angles = face_analysis.angles

                now = time.time()
                if face_analysis.wink_direction == "left":
                    right_held_until = now + HOLD_REFRESH_WINDOW
                    if not right_is_down:
                        cur.right_down()
                        right_is_down = True
                elif face_analysis.wink_direction == "right":
                    left_held_until = now + HOLD_REFRESH_WINDOW
                    if not left_is_down:
                        cur.left_down()
                        left_is_down = True

                if left_is_down and now > left_held_until:
                    cur.left_up()
                    left_is_down = False
                if right_is_down and now > right_held_until:
                    cur.right_up()
                    right_is_down = False

    finally:
        if left_is_down:
            cur.left_up()
        if right_is_down:
            cur.right_up()
        cap.release()
        face_analysis_pipeline.release()


def main():
    cur = create_cursor()
    msg_queue = queue.Queue()
    control_queue = queue.Queue()

    try:
        root = SettingsWindow.create_app(cursor=cur)
    except Exception as e:
        print(f"Fatal Error: Could not start Tkinter: {e}")
        return 1

    def send_quit(_event=None):
        control_queue.put("QUIT")

    def send_calibrate(_event=None):
        control_queue.put("CALIBRATE")

    root.bind("q", send_quit)
    root.bind("<Escape>", send_quit)
    root.bind("c", send_calibrate)
    root.protocol("WM_DELETE_WINDOW", send_quit)

    def check_queue():
        try:
            msg = msg_queue.get_nowait()
            if msg == "QUIT":
                root.quit()
                root.destroy()
                return
        except queue.Empty:
            pass
        root.after(100, check_queue)

    t = threading.Thread(
        target=run_tracking_loop,
        args=(cur, msg_queue, control_queue),
        daemon=True,
    )
    t.start()

    check_queue()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
