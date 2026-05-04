"""Threshold constants for experimental hybrid head+gaze cursor modes."""

# Safe zone -- used by Modes 1 and 2.
SAFE_ZONE_YAW_DEG: float = 10.0     # |yaw|  <= this AND
SAFE_ZONE_PITCH_DEG: float = 5.0    # |pitch| <= this  -> in safe zone

# Mode 2 -- Anchor + Offset.
# Fraction of (gaze_target - screen_center) added to the head anchor.
# 0.2 -> at the screen corner, eyes shift the cursor by ~10% of screen size.
ANCHOR_EYE_OFFSET_SCALE: float = 0.2

# Mode 3 -- Smooth Fade.
FADE_YAW_FAR_DEG: float = 15.0      # |yaw|   >= this -> alpha = 1 (pure head)
FADE_PITCH_FAR_DEG: float = 8.0     # |pitch| >= this -> alpha = 1 (pure head)

# Mode 4 -- Safe Zone Weighted Blend.
# Inside the safe zone: blended = SAFE_ZONE_GAZE_WEIGHT * gaze + (1 - SAFE_ZONE_GAZE_WEIGHT) * head.
# Outside the safe zone: pure head.
SAFE_ZONE_GAZE_WEIGHT: float = 0.8
