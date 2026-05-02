from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.face_tracking.controllers.blendshape_gesture_constants import SCROLL_INTENT_DELAY_SEC
from src.face_tracking.providers.face_landmarks import FaceLandmarksProvider
from src.face_tracking.signals.blendshapes import (
    compute_smirk_activations,
    extract_blendshapes,
    pucker_value,
    tuck_value,
)


NUM_CAPTURE_FRAMES = 30
SAFETY_FLOOR_PUCKER_MAX = 0.4       # never let calibrated pucker max drop below this
SAFETY_FLOOR_TUCK_MAX = 0.4         # same floor for the tuck-in signal
SMIRK_TRIGGER_FLOOR = 0.25
SMIRK_TRIGGER_CEIL = 0.50
CALIBRATION_SCHEMA_VERSION = 5


class FacialGestureCalibrationSession:
    """Captures smirk and lip-gesture samples and derives per-user thresholds.

    Each sample captures (left_activation, right_activation, pucker, tuck) where:
      - left/right activations come from mouthSmileLeft / mouthSmileRight
      - pucker comes from MediaPipe's mouthPucker blendshape only (cheekPuff
        in MediaPipe's default model is unreliable, so it is not used)
      - tuck comes from max(mouthRollUpper, mouthRollLower,
        mouthPressLeft, mouthPressRight)

    Output thresholds are computed in **baseline-adjusted units** so the
    runtime controller subtracts the same baseline before applying them.
    Asymmetric faces are handled correctly because the relax step captures
    each side's natural at-rest activation independently.
    """

    def __init__(self) -> None:
        self._provider = FaceLandmarksProvider()
        self._relax_samples: List[Tuple[float, float, float, float]] = []
        self._left_smirk_samples: List[Tuple[float, float, float, float]] = []
        self._right_smirk_samples: List[Tuple[float, float, float, float]] = []
        self._pucker_max_samples: List[Tuple[float, float, float, float]] = []
        self._tuck_in_max_samples: List[Tuple[float, float, float, float]] = []

    def capture_relax(self, rgb_frame) -> Optional[Tuple[float, float, float, float]]:
        sample = self._sample(rgb_frame)
        if sample is None:
            return None
        self._relax_samples.append(sample)
        return sample

    def capture_left_smirk(self, rgb_frame) -> Optional[Tuple[float, float, float, float]]:
        sample = self._sample(rgb_frame)
        if sample is None:
            return None
        self._left_smirk_samples.append(sample)
        return sample

    def capture_right_smirk(self, rgb_frame) -> Optional[Tuple[float, float, float, float]]:
        sample = self._sample(rgb_frame)
        if sample is None:
            return None
        self._right_smirk_samples.append(sample)
        return sample

    def capture_pucker_max(self, rgb_frame) -> Optional[Tuple[float, float, float, float]]:
        sample = self._sample(rgb_frame)
        if sample is None:
            return None
        self._pucker_max_samples.append(sample)
        return sample

    def capture_tuck_in_max(self, rgb_frame) -> Optional[Tuple[float, float, float, float]]:
        sample = self._sample(rgb_frame)
        if sample is None:
            return None
        self._tuck_in_max_samples.append(sample)
        return sample

    def get_sample_count(self, step: str) -> int:
        return len(self._samples_for(step))

    def has_enough_samples(self, step: str) -> bool:
        return self.get_sample_count(step) >= NUM_CAPTURE_FRAMES

    def compute_calibration(self) -> Optional[Dict]:
        if not all([
            self._relax_samples,
            self._left_smirk_samples,
            self._right_smirk_samples,
            self._pucker_max_samples,
            self._tuck_in_max_samples,
        ]):
            return None

        relax_left = float(np.median([s[0] for s in self._relax_samples]))
        relax_right = float(np.median([s[1] for s in self._relax_samples]))
        pucker_baseline = float(np.median([s[2] for s in self._relax_samples]))
        tuck_baseline = float(np.median([s[3] for s in self._relax_samples]))

        # Baseline-adjusted smirk activations during the left/right smirk steps.
        def adj(value, baseline):
            v = value - baseline
            return v if v > 0.0 else 0.0

        left_smirk_l = float(np.median([adj(s[0], relax_left) for s in self._left_smirk_samples]))
        left_smirk_r = float(np.median([adj(s[1], relax_right) for s in self._left_smirk_samples]))
        right_smirk_l = float(np.median([adj(s[0], relax_left) for s in self._right_smirk_samples]))
        right_smirk_r = float(np.median([adj(s[1], relax_right) for s in self._right_smirk_samples]))

        smirk_max_left_diff = max(0.0, left_smirk_l - left_smirk_r)
        smirk_max_right_diff = max(0.0, right_smirk_r - right_smirk_l)

        typical_max = (smirk_max_left_diff + smirk_max_right_diff) / 2.0
        smirk_trigger_diff = max(
            SMIRK_TRIGGER_FLOOR,
            min(SMIRK_TRIGGER_CEIL, typical_max * 0.6),
        )
        smirk_relax_diff = smirk_trigger_diff * 0.4

        pucker_max_raw = float(np.median([s[2] for s in self._pucker_max_samples]))
        pucker_max = max(SAFETY_FLOOR_PUCKER_MAX, pucker_max_raw)

        pucker_release = pucker_max * 0.20
        pucker_trigger_low = pucker_max * 0.33
        pucker_trigger_high = pucker_max * 0.78

        tuck_max_raw = float(np.median([s[3] for s in self._tuck_in_max_samples]))
        tuck_max = max(SAFETY_FLOOR_TUCK_MAX, tuck_max_raw - tuck_baseline)

        tuck_release = tuck_max * 0.20
        tuck_trigger_low = tuck_max * 0.33
        tuck_trigger_high = tuck_max * 0.78

        smirk_separation = min(smirk_max_left_diff, smirk_max_right_diff)
        pucker_range = max(0.0, pucker_max - pucker_baseline)
        smirk_score = min(1.0, smirk_separation / 0.4)
        pucker_score = min(1.0, pucker_range / 0.5)
        tuck_score = min(1.0, tuck_max / 0.5)
        quality_score = max(0.0, min(1.0, (smirk_score + pucker_score + tuck_score) / 3.0))
        quality_label = self._quality_label(quality_score)

        return {
            "version": CALIBRATION_SCHEMA_VERSION,
            "smirk_baseline_left": round(relax_left, 4),
            "smirk_baseline_right": round(relax_right, 4),
            "smirk_max_left_diff": round(smirk_max_left_diff, 4),
            "smirk_max_right_diff": round(smirk_max_right_diff, 4),
            "pucker_baseline": round(pucker_baseline, 4),
            "pucker_max": round(pucker_max, 4),
            "smirk_trigger_diff": round(smirk_trigger_diff, 4),
            "smirk_relax_diff": round(smirk_relax_diff, 4),
            "pucker_release": round(pucker_release, 4),
            "pucker_trigger_low": round(pucker_trigger_low, 4),
            "pucker_trigger_high": round(pucker_trigger_high, 4),
            "tuck_baseline": round(tuck_baseline, 4),
            "tuck_max": round(tuck_max, 4),
            "tuck_release": round(tuck_release, 4),
            "tuck_trigger_low": round(tuck_trigger_low, 4),
            "tuck_trigger_high": round(tuck_trigger_high, 4),
            "scroll_intent_delay_sec": SCROLL_INTENT_DELAY_SEC,
            "quality_score": round(quality_score, 3),
            "quality_label": quality_label,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def release(self) -> None:
        self._provider.release()

    def reset(self) -> None:
        self._relax_samples.clear()
        self._left_smirk_samples.clear()
        self._right_smirk_samples.clear()
        self._pucker_max_samples.clear()
        self._tuck_in_max_samples.clear()

    def _samples_for(self, step: str) -> List[Tuple[float, float, float, float]]:
        return {
            "relax": self._relax_samples,
            "left_smirk": self._left_smirk_samples,
            "right_smirk": self._right_smirk_samples,
            "pucker_max": self._pucker_max_samples,
            "tuck_in_max": self._tuck_in_max_samples,
        }.get(step, [])

    def _sample(self, rgb_frame) -> Optional[Tuple[float, float, float, float]]:
        observation = self._provider.get_primary_face_observation(rgb_frame)
        if observation is None:
            return None
        bs = extract_blendshapes(observation.blendshapes)
        left, right = compute_smirk_activations(bs)
        return (
            float(left),
            float(right),
            float(pucker_value(bs)),
            float(tuck_value(bs)),
        )

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
