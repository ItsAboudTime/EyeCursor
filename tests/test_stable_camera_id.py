"""Unit tests for stable camera identification logic.

These tests don't touch real /sys hardware -- they exercise the pure-Python
helpers that compose stable IDs from raw metadata, and the matching /
migration logic in `camera_identity` / `calibration_migration`. They run
without any cameras attached.

Run with:

    python -m unittest discover tests
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Make ``src.*`` importable when this file is run directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.devices import camera_identity, stable_camera_id
from src.core.devices.calibration_migration import (
    _migrate_single,
    _migrate_stereo,
)
from src.core.devices.camera_identity import (
    KEY_CAMERA_STABLE_ID,
    KEY_LEFT_CAMERA_ID,
    KEY_LEFT_CAMERA_STABLE_ID,
    KEY_RIGHT_CAMERA_ID,
    KEY_RIGHT_CAMERA_STABLE_ID,
    annotate_single_camera_calibration,
    annotate_stereo_calibration,
    match_single_camera,
    match_stereo_cameras,
)


class BuildStableIdTests(unittest.TestCase):
    def test_unique_serial_wins(self):
        sid = stable_camera_id.build_stable_id(
            vendor_id="046d",
            product_id="0825",
            serial="ABC12345",
            usb_port="2-1.4",
            product_name="Logitech HD",
        )
        self.assertEqual(sid, "usb:046d:0825:serial:ABC12345")

    def test_falls_back_to_port_when_serial_missing(self):
        sid = stable_camera_id.build_stable_id(
            vendor_id="1bcf",
            product_id="d828",
            serial=None,
            usb_port="3-3",
        )
        self.assertEqual(sid, "usb:1bcf:d828:port:3-3")

    def test_falls_back_to_port_for_placeholder_serial(self):
        # WebCamera-style: serial equals product name -> not unique.
        sid = stable_camera_id.build_stable_id(
            vendor_id="1bcf",
            product_id="d828",
            serial="WebCamera",
            usb_port="3-3",
            product_name="WebCamera",
        )
        self.assertEqual(sid, "usb:1bcf:d828:port:3-3")

    def test_falls_back_to_port_for_firmware_revision_serial(self):
        # Lenovo built-in: "01.00.00" is not a per-unit serial.
        sid = stable_camera_id.build_stable_id(
            vendor_id="30c9",
            product_id="00c2",
            serial="01.00.00",
            usb_port="3-7",
            product_name="Integrated RGB Camera",
        )
        self.assertEqual(sid, "usb:30c9:00c2:port:3-7")

    def test_returns_none_without_vendor_or_product(self):
        self.assertIsNone(
            stable_camera_id.build_stable_id(
                vendor_id="",
                product_id="0825",
                serial="A",
                usb_port="1-1",
            )
        )
        self.assertIsNone(
            stable_camera_id.build_stable_id(
                vendor_id="046d",
                product_id="",
                serial="A",
                usb_port="1-1",
            )
        )

    def test_returns_none_when_no_serial_and_no_port(self):
        self.assertIsNone(
            stable_camera_id.build_stable_id(
                vendor_id="046d",
                product_id="0825",
                serial="",
                usb_port=None,
            )
        )


class MatchStereoCamerasTests(unittest.TestCase):
    """Exercise the runtime stereo-mode validation logic."""

    def _calibration(self, **kw):
        base = {
            KEY_LEFT_CAMERA_ID: 4,
            KEY_RIGHT_CAMERA_ID: 6,
            "K1": [], "D1": [], "K2": [], "D2": [], "R": [], "T": [],
        }
        base.update(kw)
        return base

    def test_legacy_indices_match_exactly(self):
        cal = self._calibration()
        match = match_stereo_cameras(cal, [4, 6])
        self.assertTrue(match.ok)
        self.assertEqual(match.resolved_indices, [4, 6])

    def test_legacy_indices_swapped_still_match(self):
        # User did the old "swap cables" workaround: indices reversed.
        cal = self._calibration()
        match = match_stereo_cameras(cal, [6, 4])
        self.assertTrue(match.ok)

    def test_legacy_indices_unrelated_fail_with_clear_message(self):
        cal = self._calibration()
        match = match_stereo_cameras(cal, [0, 2])
        self.assertFalse(match.ok)
        self.assertIn("calibration", match.reason.lower())

    def test_stable_ids_remap_when_indices_shift(self):
        cal = self._calibration(
            **{
                KEY_LEFT_CAMERA_STABLE_ID: "usb:1bcf:d828:port:3-3",
                KEY_RIGHT_CAMERA_STABLE_ID: "usb:1bcf:d828:port:3-2",
            }
        )

        class FakeCamMgr:
            def __init__(self, mapping):
                self._mapping = mapping
            def stable_id_for_index(self, idx):
                return self._mapping.get(idx)
            def index_for_stable_id(self, sid):
                for idx, s in self._mapping.items():
                    if s == sid:
                        return idx
                return None

        # User selected "left=8, right=2" -- the stable IDs say it's the
        # original left at index 8 and original right at 2. The match
        # should remap to [8, 2] explicitly (not swap them).
        cam_mgr = FakeCamMgr({
            8: "usb:1bcf:d828:port:3-3",  # original "left"
            2: "usb:1bcf:d828:port:3-2",  # original "right"
        })
        match = match_stereo_cameras(cal, [8, 2], camera_manager=cam_mgr)
        self.assertTrue(match.ok)
        self.assertEqual(match.resolved_indices, [8, 2])

    def test_stable_ids_remap_when_user_swapped_cables(self):
        # Calibration was created with port 3-3 = left, port 3-2 = right.
        # User physically swapped USB cables, so today index 8 is on port 3-2
        # and index 2 is on port 3-3. The user picked left=8, right=2 in the
        # GUI. Stable-ID matching should detect the swap and assign
        # left=2, right=8 so the calibration matrices apply to the right
        # physical cameras.
        cal = self._calibration(
            **{
                KEY_LEFT_CAMERA_STABLE_ID: "usb:1bcf:d828:port:3-3",
                KEY_RIGHT_CAMERA_STABLE_ID: "usb:1bcf:d828:port:3-2",
            }
        )

        class FakeCamMgr:
            def __init__(self, mapping):
                self._mapping = mapping
            def stable_id_for_index(self, idx):
                return self._mapping.get(idx)
            def index_for_stable_id(self, sid):
                for idx, s in self._mapping.items():
                    if s == sid:
                        return idx
                return None

        cam_mgr = FakeCamMgr({
            8: "usb:1bcf:d828:port:3-2",  # left is actually the *right* camera
            2: "usb:1bcf:d828:port:3-3",  # right is actually the *left* camera
        })
        match = match_stereo_cameras(cal, [8, 2], camera_manager=cam_mgr)
        self.assertTrue(match.ok)
        # Physical-left was at 3-3 originally and is now at index 2.
        # Physical-right was at 3-2 originally and is now at index 8.
        self.assertEqual(match.resolved_indices, [2, 8])

    def test_identical_stable_ids_falls_back_to_legacy_indices(self):
        # When both calibrated cameras share a stable ID (e.g. two
        # sub-streams of the same physical device), the matcher should
        # fall back to the legacy index check rather than collapse them.
        cal = self._calibration(
            **{
                KEY_LEFT_CAMERA_STABLE_ID: "usb:30c9:00c2:port:3-7",
                KEY_RIGHT_CAMERA_STABLE_ID: "usb:30c9:00c2:port:3-7",
                KEY_LEFT_CAMERA_ID: 0,
                KEY_RIGHT_CAMERA_ID: 2,
            }
        )

        class FakeCamMgr:
            def stable_id_for_index(self, idx):
                return "usb:30c9:00c2:port:3-7"
            def index_for_stable_id(self, sid):
                return 0

        # User picks the same indices as calibration -- legacy match wins.
        match = match_stereo_cameras(cal, [0, 2], camera_manager=FakeCamMgr())
        self.assertTrue(match.ok)
        self.assertEqual(match.resolved_indices, [0, 2])

    def test_stable_ids_missing_camera_emits_clear_error(self):
        cal = self._calibration(
            **{
                KEY_LEFT_CAMERA_STABLE_ID: "usb:1bcf:d828:port:3-3",
                KEY_RIGHT_CAMERA_STABLE_ID: "usb:046d:0825:serial:UNPLUGGED",
            }
        )

        class FakeCamMgr:
            def __init__(self, mapping):
                self._mapping = mapping
            def stable_id_for_index(self, idx):
                return self._mapping.get(idx)
            def index_for_stable_id(self, sid):
                for idx, s in self._mapping.items():
                    if s == sid:
                        return idx
                return None

        cam_mgr = FakeCamMgr({
            8: "usb:1bcf:d828:port:3-3",
            2: "usb:30c9:00c2:port:3-7",  # not the calibrated right camera
        })
        match = match_stereo_cameras(cal, [8, 2], camera_manager=cam_mgr)
        self.assertFalse(match.ok)
        # The new error must NOT use the misleading legacy message.
        self.assertNotIn("different cameras", match.reason.lower())
        self.assertIn("not detected", match.reason.lower())


class MatchSingleCameraTests(unittest.TestCase):
    def test_legacy_calibration_without_stable_id_passes(self):
        cal = {"foo": "bar"}
        match = match_single_camera(cal, 4, label="head pose")
        self.assertTrue(match.ok)
        self.assertEqual(match.resolved_indices, [4])

    def test_matching_stable_id_passes(self):
        cal = {KEY_CAMERA_STABLE_ID: "usb:1bcf:d828:port:3-3"}

        class FakeCamMgr:
            def stable_id_for_index(self, idx):
                return "usb:1bcf:d828:port:3-3"
            def index_for_stable_id(self, sid):
                return 4

        match = match_single_camera(cal, 4, label="x", camera_manager=FakeCamMgr())
        self.assertTrue(match.ok)

    def test_remap_when_camera_moved_to_different_index(self):
        cal = {KEY_CAMERA_STABLE_ID: "usb:1bcf:d828:port:3-3"}

        class FakeCamMgr:
            def stable_id_for_index(self, idx):
                # User has the camera at index 8 today.
                return "usb:30c9:00c2:port:3-7" if idx == 4 else "usb:1bcf:d828:port:3-3"
            def index_for_stable_id(self, sid):
                if sid == "usb:1bcf:d828:port:3-3":
                    return 8
                return None

        # User asked for index 4 but the camera the calibration was built
        # for is now at index 8.
        match = match_single_camera(cal, 4, label="x", camera_manager=FakeCamMgr())
        self.assertTrue(match.ok)
        self.assertEqual(match.resolved_indices, [8])

    def test_camera_not_present_returns_clear_error(self):
        cal = {KEY_CAMERA_STABLE_ID: "usb:abcd:1234:serial:GONE"}

        class FakeCamMgr:
            def stable_id_for_index(self, idx):
                return "usb:30c9:00c2:port:3-7"
            def index_for_stable_id(self, sid):
                return None

        match = match_single_camera(cal, 0, label="head pose", camera_manager=FakeCamMgr())
        self.assertFalse(match.ok)
        self.assertIn("head pose", match.reason)


class AnnotateCalibrationTests(unittest.TestCase):
    def test_annotate_stereo_writes_both_stable_ids(self):
        cal = {KEY_LEFT_CAMERA_ID: 4, KEY_RIGHT_CAMERA_ID: 6}

        class FakeCamMgr:
            def stable_id_for_index(self, idx):
                return f"usb:test:{idx}"

        out = annotate_stereo_calibration(cal, 4, 6, FakeCamMgr())
        self.assertEqual(out[KEY_LEFT_CAMERA_STABLE_ID], "usb:test:4")
        self.assertEqual(out[KEY_RIGHT_CAMERA_STABLE_ID], "usb:test:6")
        # Legacy indices remain.
        self.assertEqual(out[KEY_LEFT_CAMERA_ID], 4)
        self.assertEqual(out[KEY_RIGHT_CAMERA_ID], 6)

    def test_annotate_stereo_skips_when_no_stable_id(self):
        cal = {}

        class FakeCamMgr:
            def stable_id_for_index(self, idx):
                return None

        # Patch the function used as the module-level fallback inside
        # camera_identity so neither path resolves a stable ID.
        with patch.object(camera_identity, "stable_id_for_index", return_value=None):
            out = annotate_stereo_calibration(cal, 4, 6, FakeCamMgr())
        # No stable IDs should be written when nothing can be resolved.
        self.assertNotIn(KEY_LEFT_CAMERA_STABLE_ID, out)
        self.assertNotIn(KEY_RIGHT_CAMERA_STABLE_ID, out)

    def test_annotate_single_writes_camera_id_and_stable_id(self):
        cal = {}

        class FakeCamMgr:
            def stable_id_for_index(self, idx):
                return f"usb:test:{idx}"

        out = annotate_single_camera_calibration(cal, 4, FakeCamMgr())
        self.assertEqual(out["camera_id"], 4)
        self.assertEqual(out[KEY_CAMERA_STABLE_ID], "usb:test:4")


class MigrationTests(unittest.TestCase):
    def test_migrate_stereo_backfills_when_indices_resolve(self):
        cal = {KEY_LEFT_CAMERA_ID: 4, KEY_RIGHT_CAMERA_ID: 6, "K1": []}

        with patch(
            "src.core.devices.calibration_migration.stable_id_for_index",
            side_effect=lambda i: f"usb:test:port:{i}",
        ):
            changed = _migrate_stereo(cal)
        self.assertTrue(changed)
        self.assertEqual(cal[KEY_LEFT_CAMERA_STABLE_ID], "usb:test:port:4")
        self.assertEqual(cal[KEY_RIGHT_CAMERA_STABLE_ID], "usb:test:port:6")

    def test_migrate_stereo_skips_when_already_present(self):
        cal = {
            KEY_LEFT_CAMERA_ID: 4,
            KEY_RIGHT_CAMERA_ID: 6,
            KEY_LEFT_CAMERA_STABLE_ID: "x",
            KEY_RIGHT_CAMERA_STABLE_ID: "y",
        }
        with patch(
            "src.core.devices.calibration_migration.stable_id_for_index",
            side_effect=lambda i: f"usb:test:{i}",
        ):
            changed = _migrate_stereo(cal)
        self.assertFalse(changed)
        self.assertEqual(cal[KEY_LEFT_CAMERA_STABLE_ID], "x")  # unchanged

    def test_migrate_stereo_skips_when_only_one_resolves(self):
        cal = {KEY_LEFT_CAMERA_ID: 4, KEY_RIGHT_CAMERA_ID: 6}
        with patch(
            "src.core.devices.calibration_migration.stable_id_for_index",
            side_effect=lambda i: "usb:test:4" if i == 4 else None,
        ):
            changed = _migrate_stereo(cal)
        self.assertFalse(changed)
        self.assertNotIn(KEY_LEFT_CAMERA_STABLE_ID, cal)

    def test_migrate_stereo_skips_when_both_resolve_to_same(self):
        # Defensive: if the user's setup somehow returns the same SID for
        # both indices, do not record a corrupt migration.
        cal = {KEY_LEFT_CAMERA_ID: 4, KEY_RIGHT_CAMERA_ID: 6}
        with patch(
            "src.core.devices.calibration_migration.stable_id_for_index",
            return_value="usb:test:port:3-7",
        ):
            changed = _migrate_stereo(cal)
        self.assertFalse(changed)

    def test_migrate_single_backfills(self):
        cal = {"some": "data"}
        with patch(
            "src.core.devices.calibration_migration.stable_id_for_index",
            return_value="usb:test:4",
        ):
            changed = _migrate_single(cal, 4)
        self.assertTrue(changed)
        self.assertEqual(cal[KEY_CAMERA_STABLE_ID], "usb:test:4")

    def test_migrate_single_skips_when_already_present(self):
        cal = {KEY_CAMERA_STABLE_ID: "old"}
        with patch(
            "src.core.devices.calibration_migration.stable_id_for_index",
            return_value="usb:test:4",
        ):
            changed = _migrate_single(cal, 4)
        self.assertFalse(changed)
        self.assertEqual(cal[KEY_CAMERA_STABLE_ID], "old")


class StereoModeValidationTests(unittest.TestCase):
    """Confirm the stereo modes mutate selected_cameras to the resolved list."""

    def _stereo_calibration(self):
        return {
            KEY_LEFT_CAMERA_ID: 4,
            KEY_RIGHT_CAMERA_ID: 6,
            KEY_LEFT_CAMERA_STABLE_ID: "usb:1bcf:d828:port:3-3",
            KEY_RIGHT_CAMERA_STABLE_ID: "usb:1bcf:d828:port:3-2",
            "K1": [], "D1": [], "K2": [], "D2": [], "R": [], "T": [],
        }

    def test_two_camera_head_pose_remaps_in_place(self):
        from src.core.modes.two_camera_head_pose import TwoCameraHeadPoseMode

        cal = {
            "stereo": self._stereo_calibration(),
            "two_camera_head_pose": {"yaw_span": 30, "pitch_span": 20,
                                     "center_yaw": 0, "center_pitch": 0},
            "facial_gestures": {"version": 4},
        }
        # User asked for cameras [8, 2] but the calibration was built for
        # the cameras whose stable IDs are now at [2, 8] (cables swapped).
        # Patch the module-level resolver used inside camera_identity.
        with patch(
            "src.core.devices.camera_identity.stable_id_for_index",
            side_effect=lambda i: {
                8: "usb:1bcf:d828:port:3-2",
                2: "usb:1bcf:d828:port:3-3",
            }.get(i),
        ):
            mode = TwoCameraHeadPoseMode()
            selected = [8, 2]
            ok, reason = mode.validate_requirements(cal, selected)
            self.assertTrue(ok, msg=reason)
            # selected was rewritten to put physical-left first.
            self.assertEqual(selected, [2, 8])

    def test_eye_gaze_depth_sensitivity_clear_error_when_camera_missing(self):
        from src.core.modes.eye_gaze_depth_sensitivity import EyeGazeDepthSensitivityMode

        cal = {
            "stereo": self._stereo_calibration(),
            "eye_gaze": {"weights_path": ""},
            "facial_gestures": {},
        }
        # Only one of the two calibrated cameras is currently connected.
        with patch(
            "src.core.devices.camera_identity.stable_id_for_index",
            side_effect=lambda i: {
                4: "usb:1bcf:d828:port:3-3",
                6: "usb:30c9:00c2:port:3-7",  # not the calibrated right
            }.get(i),
        ):
            mode = EyeGazeDepthSensitivityMode()
            ok, reason = mode.validate_requirements(cal, [4, 6])
        self.assertFalse(ok)
        self.assertNotIn("different cameras", reason.lower())
        self.assertIn("not detected", reason.lower())


class EndToEndMigrationTests(unittest.TestCase):
    """Round-trip a profile through the migration helper using a tmp dir."""

    def test_migrate_profile_writes_stable_ids(self):
        from src.core.devices import calibration_migration
        from src.core.profiles.profile_manager import ProfileManager
        from src.core.profiles.profile_model import ProfileModel

        with tempfile.TemporaryDirectory() as tmp:
            pm = ProfileManager(Path(tmp))
            profile = pm.create_profile("Test")
            profile.preferred_cameras = {"one_camera": 4, "two_camera_left": 4,
                                         "two_camera_right": 6, "eye_gaze": 4}
            pm.save_profile(profile)

            # Pre-existing legacy calibrations without stable IDs.
            pm.save_stereo_calibration(profile.id, {
                KEY_LEFT_CAMERA_ID: 4,
                KEY_RIGHT_CAMERA_ID: 6,
                "K1": [], "D1": [], "K2": [], "D2": [], "R": [], "T": [],
            })
            pm.save_calibration(profile.id, "one_camera_head_pose",
                                {"yaw_span": 30})
            pm.save_calibration(profile.id, "eye_gaze",
                                {"weights_path": ""})
            pm.save_calibration(profile.id, "facial_gestures", {"version": 4})

            with patch(
                "src.core.devices.calibration_migration.stable_id_for_index",
                side_effect=lambda i: {
                    4: "usb:1bcf:d828:port:3-3",
                    6: "usb:1bcf:d828:port:3-2",
                }.get(i),
            ):
                calibration_migration.migrate_profile(profile, pm)

            stereo = pm.load_stereo_calibration(profile.id)
            self.assertEqual(stereo[KEY_LEFT_CAMERA_STABLE_ID],
                             "usb:1bcf:d828:port:3-3")
            self.assertEqual(stereo[KEY_RIGHT_CAMERA_STABLE_ID],
                             "usb:1bcf:d828:port:3-2")
            head = pm.load_calibration(profile.id, "one_camera_head_pose")
            self.assertEqual(head[KEY_CAMERA_STABLE_ID],
                             "usb:1bcf:d828:port:3-3")
            gaze = pm.load_calibration(profile.id, "eye_gaze")
            self.assertEqual(gaze[KEY_CAMERA_STABLE_ID],
                             "usb:1bcf:d828:port:3-3")
            gestures = pm.load_calibration(profile.id, "facial_gestures")
            self.assertEqual(gestures[KEY_CAMERA_STABLE_ID],
                             "usb:1bcf:d828:port:3-3")

    def test_stereo_migration_skipped_when_preferred_cams_disagree(self):
        """Don't migrate stereo when the user's GUI selection diverges from
        the stored indices -- we'd risk recording a swapped left/right."""
        from src.core.devices import calibration_migration
        from src.core.profiles.profile_manager import ProfileManager

        with tempfile.TemporaryDirectory() as tmp:
            pm = ProfileManager(Path(tmp))
            profile = pm.create_profile("Test")
            # GUI shows the OPPOSITE assignment compared with what was
            # stored at calibration time -- the user has already shuffled.
            profile.preferred_cameras = {"two_camera_left": 6,
                                         "two_camera_right": 4}
            pm.save_profile(profile)

            pm.save_stereo_calibration(profile.id, {
                KEY_LEFT_CAMERA_ID: 4,
                KEY_RIGHT_CAMERA_ID: 6,
                "K1": [], "D1": [], "K2": [], "D2": [], "R": [], "T": [],
            })

            with patch(
                "src.core.devices.calibration_migration.stable_id_for_index",
                side_effect=lambda i: {
                    4: "usb:1bcf:d828:port:3-3",
                    6: "usb:1bcf:d828:port:3-2",
                }.get(i),
            ):
                calibration_migration.migrate_profile(profile, pm)

            stereo = pm.load_stereo_calibration(profile.id)
            # No migration occurred -- legacy fields untouched.
            self.assertNotIn(KEY_LEFT_CAMERA_STABLE_ID, stereo)
            self.assertNotIn(KEY_RIGHT_CAMERA_STABLE_ID, stereo)


if __name__ == "__main__":
    unittest.main()
