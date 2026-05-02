"""
Stereo demo: control the mouse cursor with head pose + facial gestures using two cameras.

Pipeline:
left+right frames -> MediaPipe landmarks -> triangulation of important points ->
yaw/pitch/depth estimation -> cursor + gesture control.

Edit the calibration variables below with your stereo calibration output:
        K1, D1, K2, D2, R, T

Keyboard controls are handled by the Tkinter window:
    - q / Esc: quit
    - c: calibrate current head pose as center
"""

import sys
import threading
import queue
import time

import cv2
import numpy as np

from src.cursor import create_cursor
from src.face_tracking.controllers.gesture import GestureController
from src.face_tracking.pipelines.stereo_face_analysis import StereoCalibration, StereoFaceAnalysisPipeline
from src.ui.settings import SettingsWindow


LEFT_CAMERA_INDEX = 4
RIGHT_CAMERA_INDEX = 6
BASELINE_METERS = 0.0788

K1 = np.array([
    [542.975661, 0.000000, 347.621721],
    [0.000000, 542.580855, 266.597383],
    [0.000000, 0.000000, 1.000000],
], dtype=np.float64)

D1 = np.array([
    [0.139821, -0.092846, 0.013859, 0.018398, -1.438426],
], dtype=np.float64)

K2 = np.array([
    [550.591389, 0.000000, 354.946646],
    [0.000000, 547.426744, 257.201464],
    [0.000000, 0.000000, 1.000000],
], dtype=np.float64)

D2 = np.array([
    [0.061811, 0.096200, 0.009729, 0.016548, -0.325093],
], dtype=np.float64)

R = np.array([
    [0.999955, -0.009529, -0.000044],
    [0.009528, 0.999776, 0.018882],
    [-0.000136, -0.018881, 0.999822],
], dtype=np.float64)

T = np.array([
    [-0.078688],
    [-0.000478],
    [0.004312],
], dtype=np.float64)

EMA_ALPHA = 0.08


def run_tracking_loop(cursor, stop_queue, control_queue):
    left_camera = cv2.VideoCapture(LEFT_CAMERA_INDEX)
    right_camera = cv2.VideoCapture(RIGHT_CAMERA_INDEX)

    if not left_camera.isOpened() or not right_camera.isOpened():
        print(
            "Could not open stereo cameras "
            f"(left={LEFT_CAMERA_INDEX}, right={RIGHT_CAMERA_INDEX})."
        )
        stop_queue.put("QUIT")
        return

    try:
        stereo_calibration = StereoCalibration(
            k1=K1,
            d1=D1,
            k2=K2,
            d2=D2,
            r=R,
            t=T,
        )

        analysis_pipeline = StereoFaceAnalysisPipeline(
            stereo_calibration=stereo_calibration,
            yaw_span=40.0,
            pitch_span=20.0,
            ema_alpha=EMA_ALPHA,
        )
    except Exception as exc:
        print(f"Failed to initialize stereo analysis pipeline: {exc}")
        left_camera.release()
        right_camera.release()
        stop_queue.put("QUIT")
        return

    minx, miny, maxx, maxy = cursor.get_virtual_bounds()
    screen_w = maxx - minx + 1
    screen_h = maxy - miny + 1

    gesture_controller = GestureController(cursor=cursor)
    latest_head_angles = None

    print("Stereo Head+Gesture Cursor demo running.")
    print(
        "Focus the settings window and press 'c' to calibrate, 'q' or Esc to quit. "
        f"Using left={LEFT_CAMERA_INDEX}, right={RIGHT_CAMERA_INDEX}."
    )

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

            ok_left, left_frame = left_camera.read()
            ok_right, right_frame = right_camera.read()
            if not ok_left or not ok_right:
                continue

            left_rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)

            face_analysis = analysis_pipeline.analyze(
                left_rgb_frame=left_rgb,
                right_rgb_frame=right_rgb,
                left_frame_width=left_frame.shape[1],
                left_frame_height=left_frame.shape[0],
                right_frame_width=right_frame.shape[1],
                right_frame_height=right_frame.shape[0],
                screen_width=screen_w,
                screen_height=screen_h,
            )

            if face_analysis is not None:
                latest_head_angles = face_analysis.angles
                gesture_controller.handle_face_analysis(face_analysis, now=time.time())
    finally:
        gesture_controller.shutdown()
        left_camera.release()
        right_camera.release()
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
