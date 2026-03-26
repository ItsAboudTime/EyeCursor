"""
Combined demo: control the mouse cursor with head pose + wink gestures.

Requires webcam, OpenCV, MediaPipe, and the project's `cursor` and `head_track` modules.
Press 'q' to quit, 'c' to calibrate (centers current head pose).
"""

import sys
import threading
import queue
import time

import cv2

from cursor import create_cursor
from ui.settings import SettingsWindow
from head_track.perception_pipeline import FaceAnalysisPipeline


def run_tracking_loop(cur, stop_queue):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0).")
        stop_queue.put("QUIT")
        return

    face_analysis_pipeline = FaceAnalysisPipeline(yaw_span=20.0, pitch_span=10.0, smooth_len=8)

    minx, miny, maxx, maxy = cur.get_virtual_bounds()
    screen_w = maxx - minx + 1
    screen_h = maxy - miny + 1

    # Simple cooldown to avoid repeated clicks
    last_left_click = 0.0
    last_right_click = 0.0
    CLICK_COOLDOWN = 0.6

    print("Head+Wink Cursor demo running. Press 'q' to quit, 'c' to calibrate.")

    try:
        while True:
            if not stop_queue.empty():
                if stop_queue.get() == "QUIT":
                    break

            ok, frame = cap.read()
            if not ok:
                continue

            angles = None
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
                angles = face_analysis.angles

                now = time.time()
                if face_analysis.wink_direction == "left":
                    if now - last_right_click > CLICK_COOLDOWN:
                        cur.right_click()
                        last_right_click = now
                elif face_analysis.wink_direction == "right":
                    if now - last_left_click > CLICK_COOLDOWN:
                        cur.left_click()
                        last_left_click = now

            cv2.putText(
                frame,
                "Press 'c' to calibrate center, 'q' to quit. Wink to click.",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Head+Wink Cursor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                stop_queue.put("QUIT")
                break
            if key == ord('c') and angles is not None:
                face_analysis_pipeline.calibrate_to_center(*angles)
    finally:
        cap.release()
        face_analysis_pipeline.release()
        cv2.destroyAllWindows()


def main():
    cur = create_cursor()
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
        args=(cur, msg_queue),
        daemon=True,
    )
    t.start()

    check_queue()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
