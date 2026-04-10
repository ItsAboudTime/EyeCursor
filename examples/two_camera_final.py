"""
Stereo demo: control the mouse cursor with head pose + wink gestures using two cameras.

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

from cursor import create_cursor
from face_tracking.controllers.gesture import GestureController
from face_tracking.pipelines.stereo_face_analysis import StereoCalibration, StereoFaceAnalysisPipeline
from ui.settings import SettingsWindow


LEFT_CAMERA_INDEX = 0
RIGHT_CAMERA_INDEX = 1
BASELINE_METERS = 0.08

# Placeholder intrinsics/distortion.
K1 = np.array(
    [
        [700.0, 0.0, 640.0],
        [0.0, 700.0, 360.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
D1 = np.zeros((5, 1), dtype=np.float64)

K2 = np.array(
    [
        [700.0, 0.0, 640.0],
        [0.0, 700.0, 360.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
D2 = np.zeros((5, 1), dtype=np.float64)

R = np.eye(3, dtype=np.float64)
# Parallel stereo rig with right camera baseline on +X in world terms maps to -baseline here.
T = np.array([[-BASELINE_METERS], [0.0], [0.0]], dtype=np.float64)

BOTH_EYES_SQUINT_SCROLL_THRESHOLD = 0.3
BOTH_EYES_OPEN_SCROLL_THRESHOLD = 0.65
EYE_SCROLL_HOLD_SECONDS = 1.0
EYE_SCROLL_DELTA = 120


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
            ema_alpha=0.08,
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

    gesture_controller = GestureController(
        cursor=cursor,
        hold_trigger_seconds=1.0,
        release_missed_frames=5,
        both_eyes_open_threshold=BOTH_EYES_OPEN_SCROLL_THRESHOLD,
        both_eyes_squint_threshold=BOTH_EYES_SQUINT_SCROLL_THRESHOLD,
        scroll_trigger_seconds=EYE_SCROLL_HOLD_SECONDS,
        scroll_delta=EYE_SCROLL_DELTA,
    )
    latest_head_angles = None

    print("Stereo Head+Wink Cursor demo running.")
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
