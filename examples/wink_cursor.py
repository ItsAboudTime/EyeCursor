"""
Demo: control the mouse cursor with winks.

Requires webcam, OpenCV, and MediaPipe. Press 'q' to quit.
"""

import sys

from cursor import create_cursor
import cv2
from face_tracking.pipelines.face_analysis import FaceAnalysisPipeline

def main():
    cur = create_cursor()
    face_analysis_pipeline = FaceAnalysisPipeline(yaw_span=20.0, pitch_span=10.0, ema_alpha=0.25)

    cap = cv2.VideoCapture(0)
    print("Wink-Cursor demo running. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_analysis = face_analysis_pipeline.analyze(
            rgb_frame=rgb_frame,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            screen_width=frame.shape[1],
            screen_height=frame.shape[0],
        )
        if face_analysis is not None:
            wink_direction = face_analysis.wink_direction

            if wink_direction == "left":  # Left wink detected
                cur.right_click()
            elif wink_direction == "right":  # Right wink detected
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

        cv2.imshow("Wink Cursor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    face_analysis_pipeline.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
