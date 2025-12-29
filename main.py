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
import mediapipe as mp

from cursor import create_cursor
from ui.settings import SettingsWindow
from head_track import HeadPoseTracker


def detect_wink(landmarks, left_eye_indices, right_eye_indices):
    def eye_aspect_ratio(eye):
        vertical_1 = ((eye[1][0] - eye[5][0])**2 + (eye[1][1] - eye[5][1])**2)**0.5
        vertical_2 = ((eye[2][0] - eye[4][0])**2 + (eye[2][1] - eye[4][1])**2)**0.5
        horizontal = ((eye[0][0] - eye[3][0])**2 + (eye[0][1] - eye[3][1])**2)**0.5
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    left_eye = [landmarks[i] for i in left_eye_indices]
    right_eye = [landmarks[i] for i in right_eye_indices]

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    return left_ear, right_ear


def run_tracking_loop(cur, tracker, stop_queue):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    minx, miny, maxx, maxy = cur.get_virtual_bounds()
    screen_w = maxx - minx + 1
    screen_h = maxy - miny + 1

    # Wink indices (MediaPipe face mesh)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

    # Simple cooldown to avoid repeated clicks
    last_left_click = 0.0
    last_right_click = 0.0
    CLICK_COOLDOWN = 0.6

    tracker.start()
    print("Head+Wink Cursor demo running. Press 'q' to quit, 'c' to calibrate.")

    while True:
        if not stop_queue.empty():
            if stop_queue.get() == "QUIT":
                break

        pos, frame, angles = tracker.next_position(screen_w, screen_h)

        # Move cursor towards head-derived position
        if pos is not None:
            raw_tx, raw_ty = pos
            target_x = max(minx, min(maxx, raw_tx + minx))
            target_y = max(miny, min(maxy, raw_ty + miny))
            cur.step_towards(target_x, target_y)

        # Detect winks from the same frame (if available)
        if frame is not None and frame.size != 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                    left_ear, right_ear = detect_wink(landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES)

                    now = time.time()
                    if left_ear < 0.2 and right_ear > 0.3:
                        if now - last_right_click > CLICK_COOLDOWN:
                            cur.right_click()
                            last_right_click = now
                    elif right_ear < 0.2 and left_ear > 0.3:
                        if now - last_left_click > CLICK_COOLDOWN:
                            cur.left_click()
                            last_left_click = now

        # Overlay guidance text
        try:
            cv2.putText(
                frame,
                "Press 'c' to calibrate center, 'q' to quit. Wink to click.",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        except Exception:
            pass

        cv2.imshow("Head+Wink Cursor (Linux)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            stop_queue.put("QUIT")
            break
        if key == ord('c'):
            if angles is not None:
                yaw, pitch = angles
                tracker.calibrate_center(yaw, pitch)

    tracker.stop()
    face_mesh.close()
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
        daemon=True,
    )
    t.start()

    check_queue()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
