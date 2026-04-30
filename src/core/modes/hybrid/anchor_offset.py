"""Mode 2: Anchor + Eye Offset -- head anchors, eyes nudge inside the safe zone."""

from typing import Optional, Tuple

from src.core.modes.hybrid._base import _HybridGazeHeadModeBase
from src.core.modes.hybrid._constants import (
    ANCHOR_EYE_OFFSET_SCALE,
    SAFE_ZONE_PITCH_DEG,
    SAFE_ZONE_YAW_DEG,
)


class HybridAnchorOffsetMode(_HybridGazeHeadModeBase):
    id = "hybrid_anchor_offset"
    display_name = "Hybrid: Anchor + Eye Offset (experimental)"
    description = (
        "Head is the anchor; eyes nudge the cursor for fine targeting inside the safe zone."
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
        cx = screen_w * 0.5
        cy = screen_h * 0.5
        ox = (gaze_xy[0] - cx) * ANCHOR_EYE_OFFSET_SCALE
        oy = (gaze_xy[1] - cy) * ANCHOR_EYE_OFFSET_SCALE
        bx = max(0.0, min(float(screen_w - 1), head_xy[0] + ox))
        by = max(0.0, min(float(screen_h - 1), head_xy[1] + oy))
        return (bx, by), {
            "active": "head+offset",
            "in_safe_zone": True,
            "offset_px": (ox, oy),
        }
