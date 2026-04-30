from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.face_tracking.providers.face_landmarks import FaceLandmarksProvider
from src.face_tracking.signals.wink import get_eye_aspect_ratios

NUM_CAPTURE_FRAMES = 30


class EyeGestureCalibrationSession:
    def __init__(self) -> None:
        self._provider = FaceLandmarksProvider()
        self._open_samples: List[Tuple[float, float]] = []
        self._left_wink_samples: List[Tuple[float, float]] = []
        self._right_wink_samples: List[Tuple[float, float]] = []
        self._squint_samples: List[Tuple[float, float]] = []
        self._wide_open_samples: List[Tuple[float, float]] = []

    def capture_open_eyes(self, rgb_frame) -> Optional[Tuple[float, float]]:
        ratios = self._get_ratios(rgb_frame)
        if ratios is None:
            return None
        self._open_samples.append(ratios)
        return ratios

    def capture_left_wink(self, rgb_frame) -> Optional[Tuple[float, float]]:
        ratios = self._get_ratios(rgb_frame)
        if ratios is None:
            return None
        self._left_wink_samples.append(ratios)
        return ratios

    def capture_right_wink(self, rgb_frame) -> Optional[Tuple[float, float]]:
        ratios = self._get_ratios(rgb_frame)
        if ratios is None:
            return None
        self._right_wink_samples.append(ratios)
        return ratios

    def capture_squint(self, rgb_frame) -> Optional[Tuple[float, float]]:
        ratios = self._get_ratios(rgb_frame)
        if ratios is None:
            return None
        self._squint_samples.append(ratios)
        return ratios

    def capture_wide_open(self, rgb_frame) -> Optional[Tuple[float, float]]:
        ratios = self._get_ratios(rgb_frame)
        if ratios is None:
            return None
        self._wide_open_samples.append(ratios)
        return ratios

    def get_sample_count(self, step: str) -> int:
        mapping = {
            "open": self._open_samples,
            "left_wink": self._left_wink_samples,
            "right_wink": self._right_wink_samples,
            "squint": self._squint_samples,
            "wide_open": self._wide_open_samples,
        }
        return len(mapping.get(step, []))

    def has_enough_samples(self, step: str) -> bool:
        return self.get_sample_count(step) >= NUM_CAPTURE_FRAMES

    def compute_calibration(self) -> Optional[Dict]:
        if not all([
            self._open_samples,
            self._left_wink_samples,
            self._right_wink_samples,
            self._squint_samples,
            self._wide_open_samples,
        ]):
            return None

        open_left_avg = float(np.median([s[0] for s in self._open_samples]))
        open_right_avg = float(np.median([s[1] for s in self._open_samples]))

        left_wink_left_ratio = float(np.median([s[0] for s in self._left_wink_samples]))
        right_wink_right_ratio = float(np.median([s[1] for s in self._right_wink_samples]))

        squint_left_avg = float(np.median([s[0] for s in self._squint_samples]))
        squint_right_avg = float(np.median([s[1] for s in self._squint_samples]))
        wide_open_left_avg = float(np.median([s[0] for s in self._wide_open_samples]))
        wide_open_right_avg = float(np.median([s[1] for s in self._wide_open_samples]))

        both_eyes_open_threshold = (wide_open_left_avg + wide_open_right_avg) / 2.0 * 0.9
        both_eyes_squint_threshold = (squint_left_avg + squint_right_avg) / 2.0 * 1.1

        if both_eyes_squint_threshold >= both_eyes_open_threshold:
            both_eyes_open_threshold = (open_left_avg + open_right_avg) / 2.0
            both_eyes_squint_threshold = (squint_left_avg + squint_right_avg) / 2.0

        wink_eye_closed_threshold = (left_wink_left_ratio + right_wink_right_ratio) / 2.0 * 1.3
        wink_eye_open_threshold = min(open_left_avg, open_right_avg) * 0.7

        open_closed_separation = min(
            open_left_avg - left_wink_left_ratio,
            open_right_avg - right_wink_right_ratio,
        )
        quality_score = max(0.0, min(1.0, open_closed_separation * 3.0))
        quality_label = self._quality_label(quality_score)

        return {
            "left_eye_open_avg": round(open_left_avg, 4),
            "right_eye_open_avg": round(open_right_avg, 4),
            "left_eye_wide_open_avg": round(wide_open_left_avg, 4),
            "right_eye_wide_open_avg": round(wide_open_right_avg, 4),
            "both_eyes_open_threshold": round(both_eyes_open_threshold, 4),
            "both_eyes_squint_threshold": round(both_eyes_squint_threshold, 4),
            "wink_eye_closed_threshold": round(wink_eye_closed_threshold, 4),
            "wink_eye_open_threshold": round(wink_eye_open_threshold, 4),
            "hold_duration_click": 1.0,
            "scroll_delta": 120,
            "quality_score": round(quality_score, 3),
            "quality_label": quality_label,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def release(self) -> None:
        self._provider.release()

    def reset(self) -> None:
        self._open_samples.clear()
        self._left_wink_samples.clear()
        self._right_wink_samples.clear()
        self._squint_samples.clear()
        self._wide_open_samples.clear()

    def _get_ratios(self, rgb_frame) -> Optional[Tuple[float, float]]:
        observation = self._provider.get_primary_face_observation(rgb_frame)
        if observation is None:
            return None
        left, right = get_eye_aspect_ratios(observation.landmarks)
        if left is None or right is None:
            return None
        return (float(left), float(right))

    @staticmethod
    def _quality_label(score: float) -> str:
        if score >= 0.9:
            return "Excellent"
        if score >= 0.7:
            return "Good"
        if score >= 0.5:
            return "Acceptable"
        if score >= 0.3:
            return "Poor"
        return "Failed"
