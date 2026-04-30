"""Mode 3: Smooth Fade -- weighted blend that fades from gaze to head as you turn."""

from typing import Optional, Tuple

from src.core.modes.hybrid._base import _HybridGazeHeadModeBase
from src.core.modes.hybrid._constants import (
    FADE_PITCH_FAR_DEG,
    FADE_YAW_FAR_DEG,
)


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class HybridSmoothFadeMode(_HybridGazeHeadModeBase):
    id = "hybrid_smooth_fade"
    display_name = "Hybrid: Smooth Fade (experimental)"
    description = (
        "Smooth weighted blend; gaze when head is straight, head when you turn."
    )

    def _blend_targets(
        self,
        head_xy: Tuple[int, int],
        gaze_xy: Optional[Tuple[int, int]],
        yaw_deg: float,
        pitch_deg: float,
        screen_w: int,
        screen_h: int,
    ) -> Tuple[Tuple[float, float], dict]:
        yaw_norm = abs(yaw_deg) / FADE_YAW_FAR_DEG
        pitch_norm = abs(pitch_deg) / FADE_PITCH_FAR_DEG
        t = max(yaw_norm, pitch_norm)
        alpha = _smoothstep(t)

        if gaze_xy is None:
            return head_xy, {
                "active": "head_fallback",
                "alpha": 1.0,
                "alpha_raw": alpha,
                "t": t,
            }

        bx = (1.0 - alpha) * gaze_xy[0] + alpha * head_xy[0]
        by = (1.0 - alpha) * gaze_xy[1] + alpha * head_xy[1]
        return (bx, by), {
            "active": "blend",
            "alpha": alpha,
            "t": t,
            "yaw_norm": yaw_norm,
            "pitch_norm": pitch_norm,
        }
