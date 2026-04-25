from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.face_tracking.pipelines.face_analysis import FaceAnalysisPipeline

CALIBRATION_POINTS = [
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

NUM_CAPTURE_FRAMES = 20


class HeadPoseCalibrationSession:
    def __init__(self) -> None:
        self._pipeline = FaceAnalysisPipeline(
            yaw_span=60.0,
            pitch_span=40.0,
            ema_alpha=0.3,
        )
        self._center_samples: List[Tuple[float, float]] = []
        self._target_samples: Dict[int, List[Tuple[float, float]]] = {}
        self._screen_w = 1920
        self._screen_h = 1080

    def set_screen_size(self, width: int, height: int) -> None:
        self._screen_w = width
        self._screen_h = height

    def capture_sample(self, rgb_frame, target_index: int) -> Optional[Tuple[float, float]]:
        h, w = rgb_frame.shape[:2]
        result = self._pipeline.analyze(
            rgb_frame=rgb_frame,
            frame_width=w,
            frame_height=h,
            screen_width=self._screen_w,
            screen_height=self._screen_h,
        )
        if result is None or result.angles is None:
            return None

        yaw, pitch = result.angles
        if target_index == 0:
            self._center_samples.append((yaw, pitch))
        if target_index not in self._target_samples:
            self._target_samples[target_index] = []
        self._target_samples[target_index].append((yaw, pitch))
        return (yaw, pitch)

    def get_capture_count(self, target_index: int) -> int:
        if target_index == 0 and not self._target_samples.get(0):
            return len(self._center_samples)
        return len(self._target_samples.get(target_index, []))

    def has_enough_samples(self, target_index: int) -> bool:
        return self.get_capture_count(target_index) >= NUM_CAPTURE_FRAMES

    def compute_calibration(self) -> Optional[Dict]:
        if not self._center_samples:
            return None
        if len(self._target_samples) < 5:
            return None

        center_yaws = [s[0] for s in self._center_samples]
        center_pitches = [s[1] for s in self._center_samples]
        center_yaw = float(np.median(center_yaws))
        center_pitch = float(np.median(center_pitches))

        all_yaw_deltas = []
        all_pitch_deltas = []
        for idx, samples in self._target_samples.items():
            if idx == 0:
                continue
            median_yaw = float(np.median([s[0] for s in samples]))
            median_pitch = float(np.median([s[1] for s in samples]))
            all_yaw_deltas.append(abs(median_yaw - center_yaw))
            all_pitch_deltas.append(abs(median_pitch - center_pitch))

        if not all_yaw_deltas or not all_pitch_deltas:
            return None

        yaw_span = float(np.max(all_yaw_deltas)) * 1.2
        pitch_span = float(np.max(all_pitch_deltas)) * 1.2
        yaw_span = max(yaw_span, 5.0)
        pitch_span = max(pitch_span, 3.0)

        center_jitter = float(np.std(center_yaws)) + float(np.std(center_pitches))
        range_coverage = min(len(all_yaw_deltas) / 8.0, 1.0)
        quality_score = max(0.0, min(1.0, range_coverage - center_jitter * 0.1))
        quality_label = self._quality_label(quality_score)

        return {
            "center_yaw": center_yaw,
            "center_pitch": center_pitch,
            "yaw_span": yaw_span,
            "pitch_span": pitch_span,
            "ema_alpha": 0.1,
            "quality_score": round(quality_score, 3),
            "quality_label": quality_label,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def release(self) -> None:
        self._pipeline.release()

    def reset(self) -> None:
        self._center_samples.clear()
        self._target_samples.clear()

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
