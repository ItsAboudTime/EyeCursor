"""Best-effort backfill of stable camera IDs into legacy calibration data.

When the user upgrades to a build that records stable camera IDs in
calibration files, their existing calibration JSONs only have raw indices.
On the first launch after the upgrade, this module walks the active
profile's calibration directory and -- where the relevant camera is still
plugged in *and* the stable ID can be derived from its current index --
augments each file with the new ``camera_stable_id`` /
``left_camera_stable_id`` / ``right_camera_stable_id`` fields.

Migration is intentionally conservative:

* Only files lacking the stable ID are touched.
* Single-camera calibrations (head pose, eye gaze, eye gestures) are
  migrated when the user has an active per-mode camera selection on
  the profile that uniquely identifies the camera. The risk is low: if
  we end up recording a "wrong" stable ID the user just gets a recovery
  prompt the next time the camera changes.
* Stereo calibration migration is HIGHER risk: if /dev/videoN indices
  have already shifted between the time the user calibrated and the
  time of first launch on the new build, we'd record left/right stable
  IDs that don't match what the calibration matrices were computed
  against. That would silently swap left and right at runtime. We
  therefore only migrate stereo when the legacy indices currently
  resolve to two distinct stable IDs *and* the user has not had a
  chance to reshuffle since calibrating. Heuristically we treat
  ``preferred_cameras["two_camera_left"] == stored_left_camera_id`` and
  the analogous right check as the user's most recent confirmation that
  the indices still match. If those don't agree, we skip stereo
  migration; the legacy index check then surfaces a recoverable error
  on the next start.
* When a stable ID cannot be derived, the file is left untouched (no
  warning written -- the runtime mode will still load and use it via
  the legacy index path).
"""

from __future__ import annotations

from typing import Optional

from src.core.devices.camera_identity import (
    KEY_CAMERA_STABLE_ID,
    KEY_LEFT_CAMERA_ID,
    KEY_LEFT_CAMERA_STABLE_ID,
    KEY_RIGHT_CAMERA_ID,
    KEY_RIGHT_CAMERA_STABLE_ID,
)
from src.core.devices.camera_manager import CameraManager
from src.core.devices.stable_camera_id import stable_id_for_index
from src.core.profiles.profile_manager import ProfileManager
from src.core.profiles.profile_model import ProfileModel


_SINGLE_CAMERA_MODES = ("one_camera_head_pose", "eye_gaze", "eye_gestures")


def migrate_profile(
    profile: ProfileModel,
    profile_manager: ProfileManager,
    camera_manager: Optional[CameraManager] = None,
) -> None:
    """Backfill stable IDs into one profile's calibration files.

    Safe to call repeatedly; a no-op if every file is already migrated or
    the cameras can't be fingerprinted right now.
    """
    if profile is None or profile_manager is None:
        return

    # Stereo calibration -- backfill only when the user's current GUI
    # selection still agrees with the stored indices. Without that
    # agreement we have no way to verify left/right haven't already
    # shuffled since calibration time, and silently writing the wrong
    # stable IDs would swap left and right.
    stereo = profile_manager.load_stereo_calibration(profile.id)
    if stereo:
        cams = profile.preferred_cameras or {}
        left_pref = cams.get("two_camera_left")
        right_pref = cams.get("two_camera_right")
        stored_left = stereo.get("left_camera_id")
        stored_right = stereo.get("right_camera_id")
        if (
            left_pref is not None
            and right_pref is not None
            and left_pref == stored_left
            and right_pref == stored_right
        ):
            changed = _migrate_stereo(stereo)
            if changed:
                profile_manager.save_stereo_calibration(profile.id, stereo)

    # Single-camera calibrations -- backfill from the user's preferred
    # camera selections on this profile.
    cams = profile.preferred_cameras or {}
    one_cam_idx = cams.get("one_camera")
    eye_gaze_idx = cams.get("eye_gaze", one_cam_idx)
    one_cam_to_idx = {
        "one_camera_head_pose": one_cam_idx,
        "eye_gestures": one_cam_idx,
        "eye_gaze": eye_gaze_idx,
    }
    for mode_id in _SINGLE_CAMERA_MODES:
        cal = profile_manager.load_calibration(profile.id, mode_id)
        if not cal:
            continue
        idx = one_cam_to_idx.get(mode_id)
        if idx is None:
            continue
        changed = _migrate_single(cal, int(idx))
        if changed:
            profile_manager.save_calibration(profile.id, mode_id, cal)


def _migrate_stereo(stereo: dict) -> bool:
    """Return True iff stereo dict was mutated with new stable IDs."""
    if (
        KEY_LEFT_CAMERA_STABLE_ID in stereo
        and KEY_RIGHT_CAMERA_STABLE_ID in stereo
    ):
        return False
    left_idx = stereo.get(KEY_LEFT_CAMERA_ID)
    right_idx = stereo.get(KEY_RIGHT_CAMERA_ID)
    if left_idx is None or right_idx is None:
        return False
    try:
        left_sid = stable_id_for_index(int(left_idx))
        right_sid = stable_id_for_index(int(right_idx))
    except (TypeError, ValueError):
        return False
    # Only write when *both* sides resolve. Writing one without the other
    # would record an unverifiable mismatch on subsequent loads.
    if not left_sid or not right_sid:
        return False
    # Avoid pointing both stable IDs at the same physical camera -- that
    # would be a corrupted state that match_stereo_cameras should refuse.
    if left_sid == right_sid:
        return False
    stereo[KEY_LEFT_CAMERA_STABLE_ID] = left_sid
    stereo[KEY_RIGHT_CAMERA_STABLE_ID] = right_sid
    return True


def _migrate_single(cal: dict, camera_index: int) -> bool:
    if KEY_CAMERA_STABLE_ID in cal:
        return False
    sid = stable_id_for_index(camera_index)
    if not sid:
        return False
    cal[KEY_CAMERA_STABLE_ID] = sid
    return True


def migrate_all_profiles(
    profile_manager: ProfileManager,
    camera_manager: Optional[CameraManager] = None,
) -> None:
    """Migrate every profile in storage. Safe to call on every launch."""
    for profile in profile_manager.list_profiles():
        try:
            migrate_profile(profile, profile_manager, camera_manager)
        except Exception:
            # Migration must never crash the app; swallow per-profile errors.
            continue
