"""Mode 4: Safe Zone Weighted Blend -- 80% gaze + 20% head inside the zone, head only outside."""

from typing import Optional, Tuple

from src.core.modes.hybrid._base import _HybridGazeHeadModeBase
from src.core.modes.hybrid._constants import (
    SAFE_ZONE_GAZE_WEIGHT,
    SAFE_ZONE_PITCH_DEG,
    SAFE_ZONE_YAW_DEG,
)


class HybridSafeZoneBlendMode(_HybridGazeHeadModeBase):
    id = "hybrid_safe_zone_blend"
    display_name = "Hybrid: Safe Zone Weighted Blend (experimental)"
    description = (
        "Inside the safe zone, blends 80% eye gaze with 20% head pose; "
        "outside the safe zone, head pose only."
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
        in_zone = self._in_safe_zone(
            yaw_deg, pitch_deg, SAFE_ZONE_YAW_DEG, SAFE_ZONE_PITCH_DEG
        )
        if not in_zone or gaze_xy is None:
            return head_xy, {"active": "head_only", "in_safe_zone": in_zone}

        w = SAFE_ZONE_GAZE_WEIGHT
        bx = w * gaze_xy[0] + (1.0 - w) * head_xy[0]
        by = w * gaze_xy[1] + (1.0 - w) * head_xy[1]
        return (bx, by), {
            "active": "blend",
            "in_safe_zone": True,
            "gaze_weight": w,
            "head_weight": 1.0 - w,
        }
