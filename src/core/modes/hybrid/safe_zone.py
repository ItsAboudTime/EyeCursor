"""Mode 1: Safe Zone Switch -- gaze when head is centered, head when turned."""

from typing import Optional, Tuple

from src.core.modes.hybrid._base import _HybridGazeHeadModeBase
from src.core.modes.hybrid._constants import (
    SAFE_ZONE_PITCH_DEG,
    SAFE_ZONE_YAW_DEG,
)


class HybridSafeZoneMode(_HybridGazeHeadModeBase):
    id = "hybrid_safe_zone"
    display_name = "Hybrid: Safe Zone Switch (experimental)"
    description = (
        "Eyes drive while head is centered; head takes over the moment you turn."
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
        if in_zone and gaze_xy is not None:
            return gaze_xy, {"active": "gaze", "in_safe_zone": True}
        return head_xy, {"active": "head", "in_safe_zone": in_zone}
