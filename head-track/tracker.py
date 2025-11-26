import sys
import math
from collections import deque
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class HeadPoseTracker:
    """
    Linux-only head pose tracker using MediaPipe FaceMesh.

    Provides a simple API to stream cursor positions mapped from head yaw/pitch.
    """

    def __init__(self, yaw_span: float = 20.0, pitch_span: float = 10.0, smooth_len: int = 8) -> None:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("HeadPoseTracker currently supports Linux only.")

        self.yaw_span = float(yaw_span)
        self.pitch_span = float(pitch_span)
        self.smooth_len = int(smooth_len)

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._cap: Optional[cv2.VideoCapture] = None
        self._ray_dirs: deque[np.ndarray] = deque(maxlen=self.smooth_len)

        self.calib_yaw: float = 0.0
        self.calib_pitch: float = 0.0

        # Landmark indices (consistent with prototype)
        self._LMK = {
            "left": 234,
            "right": 454,
            "top": 10,
            "bottom": 152,
            "front": 1,
        }

    def start(self) -> None:
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0)")

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()

    def calibrate_center(self, yaw: float, pitch: float) -> None:
        """Set calibration offsets so current yaw/pitch map to screen center."""
        cx = 180.0
        cy = 180.0
        self.calib_yaw = cx - yaw
        self.calib_pitch = cy - pitch

    def _compute_angles(self, avg_dir: np.ndarray) -> Tuple[float, float]:
        ref_fwd = np.array([0.0, 0.0, -1.0])
        xz = np.array([avg_dir[0], 0.0, avg_dir[2]])
        xz /= (np.linalg.norm(xz) + 1e-9)
        yaw = math.degrees(math.acos(np.clip(np.dot(ref_fwd, xz), -1.0, 1.0)))
        if avg_dir[0] < 0:
            yaw = -yaw

        yz = np.array([0.0, avg_dir[1], avg_dir[2]])
        yz /= (np.linalg.norm(yz) + 1e-9)
        pitch = math.degrees(math.acos(np.clip(np.dot(ref_fwd, yz), -1.0, 1.0)))
        if avg_dir[1] > 0:
            pitch = -pitch

        if yaw < 0:
            yaw = abs(yaw)
        elif yaw < 180:
            yaw = 360 - yaw
        if pitch < 0:
            pitch = 360 + pitch

        yaw += self.calib_yaw
        pitch += self.calib_pitch
        return yaw, pitch

    def next_position(self, screen_w: int, screen_h: int) -> Tuple[Optional[Tuple[int, int]], np.ndarray, Optional[Tuple[float, float]]]:
        """
        Read the next camera frame, estimate yaw/pitch, and map to screen coords.

        Returns a tuple `(pos, frame, angles)` where:
          - `pos` is `(x, y)` int or `None` if no face detected
          - `frame` is the BGR image for optional display
          - `angles` is `(yaw, pitch)` in degrees or `None`
        """
        if self._cap is None:
            raise RuntimeError("Tracker not started. Call start() first.")

        ok, frame = self._cap.read()
        if not ok:
            return None, np.zeros((1, 1, 3), dtype=np.uint8), None

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, frame, None

        lm = results.multi_face_landmarks[0].landmark

        def lmk(i: int) -> np.ndarray:
            p = lm[i]
            return np.array([p.x * w, p.y * h, p.z * w], dtype=float)

        left = lmk(self._LMK["left"])   
        right = lmk(self._LMK["right"]) 
        top = lmk(self._LMK["top"])     
        bottom = lmk(self._LMK["bottom"]) 
        front = lmk(self._LMK["front"])   

        right_axis = right - left
        right_axis /= (np.linalg.norm(right_axis) + 1e-9)

        up_axis = top - bottom
        up_axis /= (np.linalg.norm(up_axis) + 1e-9)

        fwd = np.cross(right_axis, up_axis)
        fwd /= (np.linalg.norm(fwd) + 1e-9)
        fwd = -fwd

        self._ray_dirs.append(fwd)
        avg_dir = np.mean(self._ray_dirs, axis=0)
        avg_dir /= (np.linalg.norm(avg_dir) + 1e-9)

        yaw, pitch = self._compute_angles(avg_dir)

        sx = int(((yaw - (180.0 - self.yaw_span)) / (2.0 * self.yaw_span)) * screen_w)
        sy = int(((180.0 + self.pitch_span - pitch) / (2.0 * self.pitch_span)) * screen_h)

        sx = max(0, min(screen_w - 1, sx))
        sy = max(0, min(screen_h - 1, sy))

        return (sx, sy), frame, (yaw, pitch)
