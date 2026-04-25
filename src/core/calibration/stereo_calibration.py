from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class StereoCalibrationSession:
    def __init__(
        self,
        left_camera_id: int,
        right_camera_id: int,
        board_rows: int = 7,
        board_cols: int = 9,
        square_size: float = 0.020,
        min_pairs: int = 15,
    ) -> None:
        self._left_camera_id = left_camera_id
        self._right_camera_id = right_camera_id
        self._board_size = (board_cols, board_rows)
        self._square_size = square_size
        self._min_pairs = min_pairs

        self._objp = np.zeros((board_cols * board_rows, 3), np.float32)
        self._objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
        self._objp *= square_size

        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self._obj_points: List[np.ndarray] = []
        self._img_points_left: List[np.ndarray] = []
        self._img_points_right: List[np.ndarray] = []
        self._image_size: Optional[Tuple[int, int]] = None

    def detect_corners(
        self, left_frame: np.ndarray, right_frame: np.ndarray
    ) -> Tuple[bool, bool, Optional[np.ndarray], Optional[np.ndarray]]:
        gray_l = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        if self._image_size is None:
            self._image_size = (gray_l.shape[1], gray_l.shape[0])

        found_l, corners_l = cv2.findChessboardCorners(gray_l, self._board_size, None)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, self._board_size, None)

        if found_l:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self._criteria)
        if found_r:
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self._criteria)

        return found_l, found_r, corners_l, corners_r

    def add_frame_pair(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
    ) -> Tuple[bool, bool]:
        found_l, found_r, corners_l, corners_r = self.detect_corners(left_frame, right_frame)
        if found_l and found_r:
            self._obj_points.append(self._objp.copy())
            self._img_points_left.append(corners_l)
            self._img_points_right.append(corners_r)
        return found_l, found_r

    def get_pair_count(self) -> int:
        return len(self._obj_points)

    def can_calibrate(self) -> bool:
        return len(self._obj_points) >= self._min_pairs

    def compute_calibration(self) -> Optional[Dict]:
        if not self.can_calibrate() or self._image_size is None:
            return None

        _, K1, D1, _, _ = cv2.calibrateCamera(
            self._obj_points, self._img_points_left, self._image_size, None, None
        )
        _, K2, D2, _, _ = cv2.calibrateCamera(
            self._obj_points, self._img_points_right, self._image_size, None, None
        )

        rms_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            self._obj_points,
            self._img_points_left,
            self._img_points_right,
            K1, D1, K2, D2,
            self._image_size,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )

        baseline = float(np.linalg.norm(T))

        if rms_stereo < 0.5:
            quality_label = "Excellent"
        elif rms_stereo < 0.8:
            quality_label = "Good"
        elif rms_stereo < 1.0:
            quality_label = "Acceptable"
        else:
            quality_label = "Poor"

        return {
            "left_camera_id": self._left_camera_id,
            "right_camera_id": self._right_camera_id,
            "baseline_meters": round(baseline, 6),
            "K1": K1.tolist(),
            "D1": D1.tolist(),
            "K2": K2.tolist(),
            "D2": D2.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
            "image_size": list(self._image_size),
            "checkerboard_size": list(self._board_size),
            "square_size": self._square_size,
            "reprojection_error": round(float(rms_stereo), 4),
            "quality_label": quality_label,
            "num_pairs": len(self._obj_points),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def reset(self) -> None:
        self._obj_points.clear()
        self._img_points_left.clear()
        self._img_points_right.clear()
        self._image_size = None

    def draw_corners(
        self, frame: np.ndarray, corners: Optional[np.ndarray], found: bool
    ) -> np.ndarray:
        if found and corners is not None:
            display = frame.copy()
            cv2.drawChessboardCorners(display, self._board_size, corners, found)
            return display
        return frame
