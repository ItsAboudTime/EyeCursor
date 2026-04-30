"""Helpers for resolving stable camera identifiers in calibration data.

Calibration files (stereo, head pose, eye gaze, facial gestures) are stored
on disk and referenced across reboots. To make them resilient to /dev/videoN
re-shuffling, each calibration now records the stable identifier(s) of the
camera(s) it was built against, alongside the legacy numeric indices for
backwards compatibility.

This module centralises:

* the field names used in JSON ("camera_stable_id", "left_camera_stable_id",
  "right_camera_stable_id");
* the matching logic that decides whether a given stored calibration is
  compatible with the cameras the user has selected today;
* a best-effort migration helper that backfills stable IDs into legacy
  calibration files when the cameras are currently connected.

The matching never raises -- it returns a structured result so the UI/mode
can decide how to surface the situation to the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.devices.camera_manager import CameraManager
from src.core.devices.stable_camera_id import stable_id_for_index


# JSON keys used in calibration files.
KEY_LEFT_CAMERA_ID = "left_camera_id"
KEY_RIGHT_CAMERA_ID = "right_camera_id"
KEY_LEFT_CAMERA_STABLE_ID = "left_camera_stable_id"
KEY_RIGHT_CAMERA_STABLE_ID = "right_camera_stable_id"
KEY_CAMERA_STABLE_ID = "camera_stable_id"  # for single-camera calibrations
KEY_CAMERA_ID = "camera_id"


@dataclass
class CameraMatch:
    """Outcome of matching a stored calibration against the current cameras."""

    ok: bool
    # When ``ok`` is True these hold the resolved indices to use at runtime
    # (which may differ from what the user picked in the GUI if a stable-ID
    # remap took place).
    resolved_indices: List[int]
    # Human-readable reason when ok is False, suitable for QMessageBox.
    reason: str = ""


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_calibration_stable_ids(calibration: Optional[Dict]) -> Dict[str, Optional[str]]:
    """Extract the stable IDs stored in a calibration JSON, if present."""
    if not calibration:
        return {}
    result: Dict[str, Optional[str]] = {}
    if KEY_LEFT_CAMERA_STABLE_ID in calibration:
        result["left"] = calibration.get(KEY_LEFT_CAMERA_STABLE_ID)
    if KEY_RIGHT_CAMERA_STABLE_ID in calibration:
        result["right"] = calibration.get(KEY_RIGHT_CAMERA_STABLE_ID)
    if KEY_CAMERA_STABLE_ID in calibration:
        result["camera"] = calibration.get(KEY_CAMERA_STABLE_ID)
    return result


def match_stereo_cameras(
    stereo_calibration: Dict,
    selected_cameras: List[int],
    camera_manager: Optional[CameraManager] = None,
) -> CameraMatch:
    """Resolve `selected_cameras` against a stereo calibration's stable IDs.

    Returns a ``CameraMatch`` whose ``resolved_indices`` are the indices to
    actually open at runtime. They equal ``selected_cameras`` unless the user
    physically swapped cameras (in which case we transparently re-map) or one
    of the calibrated cameras moved to a different /dev/videoN.
    """
    if len(selected_cameras) < 2:
        return CameraMatch(ok=False, resolved_indices=[], reason="Two cameras are required.")

    stored_left_idx = _coerce_int(stereo_calibration.get(KEY_LEFT_CAMERA_ID))
    stored_right_idx = _coerce_int(stereo_calibration.get(KEY_RIGHT_CAMERA_ID))
    stored_left_sid = stereo_calibration.get(KEY_LEFT_CAMERA_STABLE_ID)
    stored_right_sid = stereo_calibration.get(KEY_RIGHT_CAMERA_STABLE_ID)

    selected_left, selected_right = selected_cameras[0], selected_cameras[1]
    selected_left_sid = _resolve_sid(selected_left, camera_manager)
    selected_right_sid = _resolve_sid(selected_right, camera_manager)

    # ----- New-style match: both calibration and current cameras have stable IDs.
    if stored_left_sid and stored_right_sid:
        # Special case: stable IDs are equal (e.g. two sub-streams of the
        # same physical camera) -- we can't disambiguate by stable ID
        # alone, so fall through to the legacy index comparison.
        if stored_left_sid == stored_right_sid:
            # Defer to legacy index logic below.
            pass
        else:
            available = [
                (selected_left_sid, selected_left),
                (selected_right_sid, selected_right),
            ]
            resolved_left = next(
                (idx for sid, idx in available if sid == stored_left_sid), None
            )
            resolved_right = next(
                (idx for sid, idx in available
                 if sid == stored_right_sid and idx != resolved_left),
                None,
            )
            if resolved_left is not None and resolved_right is not None:
                return CameraMatch(
                    ok=True,
                    resolved_indices=[resolved_left, resolved_right],
                )
            # One or both calibrated cameras are not present today.
            missing = []
            if resolved_left is None:
                missing.append("left")
            if resolved_right is None:
                missing.append("right")
            return CameraMatch(
                ok=False,
                resolved_indices=[],
                reason=(
                    f"Calibrated {' and '.join(missing)} camera not detected. "
                    "Reconnect the camera you used for calibration, or recalibrate."
                ),
            )

    # ----- Legacy match: calibration has only numeric indices.
    if stored_left_idx == selected_left and stored_right_idx == selected_right:
        return CameraMatch(ok=True, resolved_indices=[selected_left, selected_right])

    # If the user is doing the legacy "swap" workaround the indices are simply
    # reversed. Accept that too -- it matches the old behaviour exactly.
    if stored_left_idx == selected_right and stored_right_idx == selected_left:
        return CameraMatch(
            ok=True,
            resolved_indices=[selected_left, selected_right],
        )

    return CameraMatch(
        ok=False,
        resolved_indices=[],
        reason=(
            "Stereo calibration was created for different cameras. Please recalibrate, "
            "or use the Swap Left/Right button if the cables are crossed."
        ),
    )


def match_single_camera(
    calibration: Dict,
    selected_camera: int,
    *,
    label: str = "calibration",
    camera_manager: Optional[CameraManager] = None,
) -> CameraMatch:
    """Resolve a single-camera calibration against the current selection.

    Returns ``ok=True`` when the calibration either:
    - has no stable ID stored (legacy data, used to be index-based and
      blindly trusted; we keep that behaviour but add a warning hook for
      callers that want to surface it);
    - has a stable ID matching the currently-selected camera;
    - has a stable ID matching any *currently connected* camera (allowing
      transparent re-mapping when the camera is at a different index now).
    """
    stored_sid = calibration.get(KEY_CAMERA_STABLE_ID) if calibration else None
    if not stored_sid:
        # Legacy data without a stable ID -- assume it still matches.
        return CameraMatch(ok=True, resolved_indices=[selected_camera])

    selected_sid = _resolve_sid(selected_camera, camera_manager)
    if selected_sid and selected_sid == stored_sid:
        return CameraMatch(ok=True, resolved_indices=[selected_camera])

    # Try to find the calibrated camera at any currently-connected index.
    if camera_manager is not None:
        remap = camera_manager.index_for_stable_id(stored_sid)
        if remap is not None:
            return CameraMatch(ok=True, resolved_indices=[remap])

    return CameraMatch(
        ok=False,
        resolved_indices=[],
        reason=(
            f"This {label} was created on a different camera. "
            "Please recalibrate or reconnect the original camera."
        ),
    )


def warn_if_single_camera_mismatch(
    calibration: Optional[Dict],
    selected_camera: int,
    *,
    label: str = "calibration",
) -> Optional[str]:
    """Return a warning message (or None) when the recorded camera differs.

    Single-camera modes don't *block* on a mismatch -- the head pose / eye
    gaze / facial-gesture calibrations are largely tied to the user's face
    rather than the specific webcam, so they usually still work. But the
    user benefits from being told that the camera that was used during
    calibration is no longer the one they're about to track with.

    The returned string is suitable for printing to stdout and surfacing
    in any UI element that displays mode status.
    """
    if not calibration:
        return None
    stored_sid = calibration.get(KEY_CAMERA_STABLE_ID)
    if not stored_sid:
        return None
    selected_sid = _resolve_sid(selected_camera, None)
    if selected_sid == stored_sid:
        return None
    return (
        f"Note: this {label} was created on a different physical camera. "
        "It should still work but accuracy may be reduced; recalibrate "
        "if you notice issues."
    )


def annotate_stereo_calibration(
    calibration: Dict,
    left_index: int,
    right_index: int,
    camera_manager: Optional[CameraManager] = None,
) -> Dict:
    """Insert stable IDs into a stereo-calibration dict in place and return it."""
    left_sid = _resolve_sid(left_index, camera_manager)
    right_sid = _resolve_sid(right_index, camera_manager)
    if left_sid:
        calibration[KEY_LEFT_CAMERA_STABLE_ID] = left_sid
    if right_sid:
        calibration[KEY_RIGHT_CAMERA_STABLE_ID] = right_sid
    return calibration


def annotate_single_camera_calibration(
    calibration: Dict,
    camera_index: int,
    camera_manager: Optional[CameraManager] = None,
) -> Dict:
    """Insert ``camera_id`` and ``camera_stable_id`` into a single-camera calibration."""
    sid = _resolve_sid(camera_index, camera_manager)
    calibration.setdefault(KEY_CAMERA_ID, int(camera_index))
    if sid:
        calibration[KEY_CAMERA_STABLE_ID] = sid
    return calibration


def _resolve_sid(index: int, camera_manager: Optional[CameraManager]) -> Optional[str]:
    if camera_manager is not None:
        sid = camera_manager.stable_id_for_index(index)
        if sid:
            return sid
    return stable_id_for_index(index)
