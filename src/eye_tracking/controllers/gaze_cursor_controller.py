from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.cursor import create_cursor


class GazeCursorController:
    def __init__(
        self,
        cursor_enabled: bool = True,
        cursor_yaw_span: float = 0.6,
        cursor_pitch_span: float = 0.4,
        cursor_ema_alpha: float = 0.1,
    ) -> None:
        self.cursor_enabled = bool(cursor_enabled)
        self.cursor_yaw_span = float(cursor_yaw_span)
        self.cursor_pitch_span = float(cursor_pitch_span)
        self.cursor_ema_alpha = float(cursor_ema_alpha)

        if self.cursor_yaw_span <= 0.0:
            raise ValueError("cursor_yaw_span must be > 0")
        if self.cursor_pitch_span <= 0.0:
            raise ValueError("cursor_pitch_span must be > 0")
        if not (0.0 < self.cursor_ema_alpha <= 1.0):
            raise ValueError("cursor_ema_alpha must be in (0, 1]")

        self.cursor = None
        self.cursor_bounds: Optional[Tuple[int, int, int, int]] = None
        self.calibration_yaw = 0.0
        self.calibration_pitch = 0.0
        self.ema_yaw: Optional[float] = None
        self.ema_pitch: Optional[float] = None
        self.affine: Optional[np.ndarray] = None
        self.norm_bounds: Optional[Tuple[float, float, float, float]] = None

        self._init_cursor()

    def _init_cursor(self) -> None:
        if not self.cursor_enabled:
            return

        try:
            self.cursor = create_cursor()
            self.cursor_bounds = self.cursor.get_virtual_bounds()
            print(f"cursor control enabled; virtual bounds={self.cursor_bounds}")
        except Exception as exc:
            self.cursor = None
            self.cursor_bounds = None
            self.cursor_enabled = False
            print(f"warning: failed to initialize cursor control: {exc}")

    @staticmethod
    def calibration_points() -> List[Tuple[float, float]]:
        # Start from center to settle the user, then spread to all edges/corners.
        return [
            (0.50, 0.50),
            (0.08, 0.08),
            (0.50, 0.08),
            (0.92, 0.08),
            (0.08, 0.50),
            (0.92, 0.50),
            (0.08, 0.92),
            (0.50, 0.92),
            (0.92, 0.92),
        ]

    @staticmethod
    def target_abs_point(
        bounds: Tuple[int, int, int, int],
        target_norm: Tuple[float, float],
    ) -> Tuple[int, int]:
        minx, miny, maxx, maxy = bounds
        width = maxx - minx + 1
        height = maxy - miny + 1
        tx = minx + int(round(target_norm[0] * (width - 1)))
        ty = miny + int(round(target_norm[1] * (height - 1)))
        return tx, ty

    def fit_calibration(
        self,
        gaze_samples: np.ndarray,
        target_points: np.ndarray,
    ) -> bool:
        affine, inlier_mask = cv2.estimateAffine2D(
            gaze_samples,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=0.06,
            maxIters=3000,
            confidence=0.995,
        )
        if affine is None:
            return False

        ones = np.ones((gaze_samples.shape[0], 1), dtype=np.float64)
        augmented = np.concatenate([gaze_samples, ones], axis=1)
        pred = np.dot(augmented, affine.T)

        errors = np.linalg.norm(pred - target_points, axis=1)
        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))

        inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else gaze_samples.shape[0]
        min_inliers = max(6, int(round(gaze_samples.shape[0] * 0.67)))
        if inliers < min_inliers or mean_err > 0.12 or max_err > 0.24:
            return False

        min_x = float(np.min(pred[:, 0]))
        max_x = float(np.max(pred[:, 0]))
        min_y = float(np.min(pred[:, 1]))
        max_y = float(np.max(pred[:, 1]))
        if (max_x - min_x) < 1e-4 or (max_y - min_y) < 1e-4:
            return False

        self.affine = affine.astype(np.float64)
        self.norm_bounds = (min_x, max_x, min_y, max_y)
        self.ema_yaw = None
        self.ema_pitch = None
        print(
            f"cursor calibration complete: inliers={inliers}/{gaze_samples.shape[0]}, "
            f"mean_err={mean_err:.4f}, max_err={max_err:.4f}"
        )
        return True

    def calibrate_center(self, yaw_rad: float, pitch_rad: float) -> None:
        self.calibration_yaw = -yaw_rad
        self.calibration_pitch = -pitch_rad
        self.ema_yaw = None
        self.ema_pitch = None
        print(
            f"cursor calibrated: yaw_offset={self.calibration_yaw:.4f}, "
            f"pitch_offset={self.calibration_pitch:.4f}"
        )

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def target_from_gaze(self, yaw_rad: float, pitch_rad: float) -> Optional[Tuple[int, int]]:
        if not self.cursor_enabled or self.cursor is None or self.cursor_bounds is None:
            return None

        yaw_adj = yaw_rad
        pitch_adj = pitch_rad
        yaw_adj += self.calibration_yaw
        pitch_adj += self.calibration_pitch

        if self.ema_yaw is None:
            self.ema_yaw = yaw_adj
            self.ema_pitch = pitch_adj
        else:
            if self.ema_pitch is None:
                self.ema_pitch = pitch_adj
            alpha = self.cursor_ema_alpha
            ema_yaw = float(self.ema_yaw)
            ema_pitch = float(self.ema_pitch)
            self.ema_yaw = alpha * yaw_adj + (1.0 - alpha) * ema_yaw
            self.ema_pitch = alpha * pitch_adj + (1.0 - alpha) * ema_pitch

        if self.affine is not None:
            point = np.array([self.ema_yaw, self.ema_pitch, 1.0], dtype=np.float64)
            mapped = np.dot(self.affine, point)
            norm_x = float(mapped[0])
            norm_y = float(mapped[1])

            if self.norm_bounds is not None:
                min_x, max_x, min_y, max_y = self.norm_bounds
                norm_x = (norm_x - min_x) / (max_x - min_x)
                norm_y = (norm_y - min_y) / (max_y - min_y)
        else:
            norm_x = (self.ema_yaw + self.cursor_yaw_span) / (2.0 * self.cursor_yaw_span)
            norm_y = (self.ema_pitch + self.cursor_pitch_span) / (2.0 * self.cursor_pitch_span)

        norm_x = self._clip01(norm_x)
        norm_y = self._clip01(norm_y)

        minx, miny, maxx, maxy = self.cursor_bounds
        width = maxx - minx + 1
        height = maxy - miny + 1

        target_x = minx + int(round(norm_x * (width - 1)))
        target_y = miny + int(round(norm_y * (height - 1)))
        return target_x, target_y

    def update_cursor(self, yaw_rad: float, pitch_rad: float) -> None:
        if self.cursor is None:
            return

        target = self.target_from_gaze(yaw_rad=yaw_rad, pitch_rad=pitch_rad)
        if target is None:
            return

        self.cursor.step_towards(*target)
