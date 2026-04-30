"""
Demo: control the mouse cursor with smirk gestures.

Each smirk fires a single left or right click. The user must relax their
face before another click can fire (boolean armed flag with relax-diff
hysteresis).

Requires webcam, OpenCV, and MediaPipe. Press 'q' to quit.
"""

import sys

import cv2

from src.cursor import create_cursor
from src.face_tracking.controllers.blendshape_gesture_constants import (
    SMIRK_RELAX_DIFF,
    SMIRK_TRIGGER_DIFF,
)
from src.face_tracking.pipelines.face_analysis import FaceAnalysisPipeline
from src.face_tracking.signals.blendshapes import compute_smirk_activations


def main():
    cur = create_cursor()
    pipeline = FaceAnalysisPipeline(yaw_span=20.0, pitch_span=10.0, ema_alpha=0.25)

    cap = cv2.VideoCapture(0)
    print("Smirk-Cursor demo running. Press 'q' to quit.")

    click_armed = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pipeline.analyze(
            rgb_frame=rgb_frame,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            screen_width=frame.shape[1],
            screen_height=frame.shape[0],
        )
        if result is not None:
            blendshapes = result.blendshapes or {}
            left_act, right_act = compute_smirk_activations(blendshapes)
            diff = left_act - right_act

            if not click_armed:
                if abs(diff) < SMIRK_RELAX_DIFF:
                    click_armed = True
            else:
                if diff > SMIRK_TRIGGER_DIFF:
                    cur.left_click()
                    click_armed = False
                elif (-diff) > SMIRK_TRIGGER_DIFF:
                    cur.right_click()
                    click_armed = False

        cv2.putText(
            frame,
            "Smirk left = left click, smirk right = right click. Relax to re-arm.",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Smirk Cursor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    pipeline.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
