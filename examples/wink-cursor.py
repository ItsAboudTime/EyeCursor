"""
Linux-only demo: control the mouse cursor with winks.

Requires webcam, OpenCV, and MediaPipe. Press 'q' to quit.
"""

import sys
import os

# Ensure project root on sys.path for local imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cursor import create_cursor
import cv2
import mediapipe as mp

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

def main() -> int:
    if not sys.platform.startswith("linux"):
        print("This demo currently supports Linux only.")
        return 1

    cur = create_cursor()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    cap = cv2.VideoCapture(0)
    print("Wink-Cursor demo running (Linux). Press 'q' to quit.")

    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                left_ear, right_ear = detect_wink(landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES)

                if left_ear < 0.2 and right_ear > 0.3:  # Left wink detected
                    cur.right_click()
                elif right_ear < 0.2 and left_ear > 0.3:  # Right wink detected
                    cur.left_click()

        cv2.putText(
            frame,
            "Wink to click: Left=Right Click, Right=Left Click",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Wink Cursor (Linux)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())