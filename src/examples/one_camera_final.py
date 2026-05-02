"""
Combined demo: control the mouse cursor with head pose + facial gestures.

Requires webcam, OpenCV, MediaPipe, and the project's `cursor` and `head_track` modules.
Cross-platform note: keyboard controls are handled by the Tkinter window.
Press 'q' (or Esc) to quit, 'c' to calibrate (centers current head pose).
"""

import sys
import threading
import queue
import time

import cv2

from src.cursor import create_cursor
from src.face_tracking.controllers.gesture import GestureController
from src.ui.settings import SettingsWindow
from src.face_tracking.pipelines.face_analysis import FaceAnalysisPipeline



def run_tracking_loop(cursor, stop_queue, control_queue):
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam (index 0).")
        stop_queue.put("QUIT")
        return

    analysis_pipeline = FaceAnalysisPipeline(
        yaw_span=40.0,
        pitch_span=20.0,
        ema_alpha=0.1,
    )

    minx, miny, maxx, maxy = cursor.get_virtual_bounds()
    screen_w = maxx - minx + 1
    screen_h = maxy - miny + 1

    gesture_controller = GestureController(cursor=cursor)
    latest_head_angles = None

    print("Head+Gesture Cursor demo running.")
    print("Focus the settings window and press 'c' to calibrate, 'q' or Esc to quit.")

    try:
        while True:
            should_quit = False
            while not control_queue.empty():
                cmd = control_queue.get()
                if cmd == "QUIT":
                    should_quit = True
                    break
                if cmd == "CALIBRATE" and latest_head_angles is not None:
                    analysis_pipeline.calibrate_to_center(*latest_head_angles)

            if should_quit:
                stop_queue.put("QUIT")
                break

            ok, frame = camera.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_analysis = analysis_pipeline.analyze(
                rgb_frame=rgb,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                screen_width=screen_w,
                screen_height=screen_h,
            )

            if face_analysis is not None:
                latest_head_angles = face_analysis.angles
                if latest_head_angles is not None:
                    yaw_deg, pitch_deg = latest_head_angles
                    print(f"\rHead pose -> yaw: {yaw_deg:7.2f} deg | pitch: {pitch_deg:7.2f} deg", end="", flush=True)
                gesture_controller.handle_face_analysis(face_analysis, now=time.time())

    finally:
        print()
        gesture_controller.shutdown()
        camera.release()
        analysis_pipeline.release()


def main():
    cursor = create_cursor()
    message_queue = queue.Queue()
    control_queue = queue.Queue()

    try:
        root = SettingsWindow.create_app(cursor=cursor)
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
            message = message_queue.get_nowait()
            if message == "QUIT":
                root.quit()
                root.destroy()
                return
        except queue.Empty:
            pass
        root.after(100, check_queue)

    tracking_thread = threading.Thread(
        target=run_tracking_loop,
        args=(cursor, message_queue, control_queue),
        daemon=True,
    )
    tracking_thread.start()

    check_queue()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
