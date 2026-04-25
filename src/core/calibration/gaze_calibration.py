from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.eye_tracking.controllers.gaze_cursor_controller import GazeCursorController

CALIBRATION_POINTS = GazeCursorController.calibration_points()
NUM_CAPTURE_FRAMES = 24


class GazeCalibrationSession:
    def __init__(self) -> None:
        self._controller = GazeCursorController(cursor_enabled=False)
        self._gaze_samples: List[Tuple[float, float]] = []
        self._target_points: List[Tuple[float, float]] = []
        self._current_capture: List[Tuple[float, float]] = []

    def capture_gaze_sample(self, pitch_rad: float, yaw_rad: float) -> None:
        self._current_capture.append((yaw_rad, pitch_rad))

    def get_capture_count(self) -> int:
        return len(self._current_capture)

    def has_enough_samples(self) -> bool:
        return len(self._current_capture) >= NUM_CAPTURE_FRAMES

    def finalize_target(self, target_norm: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if len(self._current_capture) < 5:
            return None

        yaws = [s[0] for s in self._current_capture]
        pitches = [s[1] for s in self._current_capture]

        yaw_med = np.median(yaws)
        pitch_med = np.median(pitches)

        yaw_mad = np.median(np.abs(np.array(yaws) - yaw_med))
        pitch_mad = np.median(np.abs(np.array(pitches) - pitch_med))

        filtered = [
            (y, p) for y, p in self._current_capture
            if abs(y - yaw_med) <= 3 * max(yaw_mad, 0.01) and
               abs(p - pitch_med) <= 3 * max(pitch_mad, 0.01)
        ]

        if len(filtered) < 3:
            filtered = self._current_capture

        avg_yaw = float(np.mean([s[0] for s in filtered]))
        avg_pitch = float(np.mean([s[1] for s in filtered]))

        self._gaze_samples.append((avg_yaw, avg_pitch))
        self._target_points.append(target_norm)
        self._current_capture.clear()
        return (avg_yaw, avg_pitch)

    def compute_calibration(self) -> Optional[Dict]:
        if len(self._gaze_samples) < 5:
            return None

        gaze_arr = np.array(self._gaze_samples, dtype=np.float64)
        target_arr = np.array(self._target_points, dtype=np.float64)

        ok = self._controller.fit_calibration(gaze_arr, target_arr)
        if not ok:
            return None

        affine = self._controller.affine
        norm_bounds = self._controller.norm_bounds

        ones = np.ones((gaze_arr.shape[0], 1), dtype=np.float64)
        augmented = np.concatenate([gaze_arr, ones], axis=1)
        pred = np.dot(augmented, affine.T)
        errors = np.linalg.norm(pred - target_arr, axis=1)
        mean_err = float(np.mean(errors))

        if mean_err < 0.04:
            quality_label = "Excellent"
        elif mean_err < 0.08:
            quality_label = "Good"
        elif mean_err < 0.12:
            quality_label = "Acceptable"
        else:
            quality_label = "Poor"

        quality_score = max(0.0, min(1.0, 1.0 - mean_err * 5.0))

        return {
            "affine": affine.tolist(),
            "norm_bounds": list(norm_bounds) if norm_bounds else None,
            "calibration_yaw": self._controller.calibration_yaw,
            "calibration_pitch": self._controller.calibration_pitch,
            "mean_error": round(mean_err, 4),
            "num_points": len(self._gaze_samples),
            "quality_score": round(quality_score, 3),
            "quality_label": quality_label,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def reset(self) -> None:
        self._gaze_samples.clear()
        self._target_points.clear()
        self._current_capture.clear()
        self._controller.affine = None
        self._controller.norm_bounds = None
