"""Unit tests for ``IdleController``.

Pure Python -- no Qt, cameras, or sleeps that block the test runner.

Run with:

    python -m unittest tests.test_idle_controller -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.modes.idle import IdleController, apply_idle_settings


class IdleControllerStateTests(unittest.TestCase):
    def test_starts_active(self) -> None:
        idle = IdleController(idle_after_frames=3, idle_sleep_s=0.0)
        self.assertFalse(idle.is_idle)
        self.assertEqual(idle.streak_frames, 0)

    def test_streak_increments_only_on_no_face(self) -> None:
        idle = IdleController(idle_after_frames=10, idle_sleep_s=0.0)
        idle.observe(False)
        idle.observe(False)
        self.assertEqual(idle.streak_frames, 2)
        self.assertFalse(idle.is_idle)

    def test_streak_resets_on_face_detection(self) -> None:
        idle = IdleController(idle_after_frames=10, idle_sleep_s=0.0)
        idle.observe(False)
        idle.observe(False)
        idle.observe(True)
        self.assertEqual(idle.streak_frames, 0)

    def test_trips_into_idle_at_threshold(self) -> None:
        idle = IdleController(idle_after_frames=3, idle_sleep_s=0.0)
        # Frames 1, 2 -- not yet idle.
        self.assertFalse(idle.observe(False))
        self.assertFalse(idle.is_idle)
        self.assertFalse(idle.observe(False))
        self.assertFalse(idle.is_idle)
        # Frame 3 -- transition fires.
        self.assertTrue(idle.observe(False))
        self.assertTrue(idle.is_idle)

    def test_does_not_re_fire_while_already_idle(self) -> None:
        idle = IdleController(idle_after_frames=2, idle_sleep_s=0.0)
        idle.observe(False)
        idle.observe(False)  # transitions to idle
        # Subsequent no-face frames must not report another transition.
        for _ in range(5):
            self.assertFalse(idle.observe(False))
        self.assertTrue(idle.is_idle)

    def test_exits_idle_on_face_detection(self) -> None:
        idle = IdleController(idle_after_frames=2, idle_sleep_s=0.0)
        idle.observe(False)
        idle.observe(False)  # transitions to idle
        self.assertTrue(idle.is_idle)
        # First face frame fires the exit transition.
        self.assertTrue(idle.observe(True))
        self.assertFalse(idle.is_idle)
        self.assertEqual(idle.streak_frames, 0)
        # Subsequent face frames are no-op transitions.
        self.assertFalse(idle.observe(True))


class IdleControllerCallbackTests(unittest.TestCase):
    def test_callback_fires_on_each_transition(self) -> None:
        events: list[bool] = []
        idle = IdleController(idle_after_frames=2, idle_sleep_s=0.0)
        idle.set_on_change(events.append)

        idle.observe(False)
        idle.observe(False)  # → idle
        idle.observe(False)  # already idle, no event
        idle.observe(True)   # → active
        idle.observe(True)   # already active, no event
        idle.observe(False)
        idle.observe(False)  # → idle again

        self.assertEqual(events, [True, False, True])

    def test_callback_can_be_cleared(self) -> None:
        events: list[bool] = []
        idle = IdleController(idle_after_frames=1, idle_sleep_s=0.0)
        idle.set_on_change(events.append)
        idle.observe(False)  # → idle
        idle.set_on_change(None)
        idle.observe(True)   # → active, but no callback
        self.assertEqual(events, [True])

    def test_callback_exception_does_not_propagate(self) -> None:
        idle = IdleController(idle_after_frames=1, idle_sleep_s=0.0)

        def boom(_state: bool) -> None:
            raise RuntimeError("subscriber blew up")

        idle.set_on_change(boom)
        # Must not raise.
        self.assertTrue(idle.observe(False))
        self.assertTrue(idle.is_idle)


class IdleControllerSleepTests(unittest.TestCase):
    def test_sleep_is_noop_when_active(self) -> None:
        # Use a long sleep_s; if the gate fails it would be obvious in test runtime.
        idle = IdleController(idle_after_frames=10, idle_sleep_s=10.0)
        # Not idle -- maybe_sleep must return immediately.
        idle.maybe_sleep()  # If this actually sleeps 10s, the test takes 10s. It must not.

    def test_sleep_is_noop_when_sleep_s_zero(self) -> None:
        idle = IdleController(idle_after_frames=1, idle_sleep_s=0.0)
        idle.observe(False)
        self.assertTrue(idle.is_idle)
        idle.maybe_sleep()  # No-op even when idle.


class ApplyIdleSettingsTests(unittest.TestCase):
    def test_applies_known_keys(self) -> None:
        idle = IdleController(idle_after_frames=30, idle_sleep_s=0.15)
        apply_idle_settings(idle, {"idle_after_frames": 5, "idle_sleep_s": 0.5})
        self.assertEqual(idle.idle_after_frames, 5)
        self.assertEqual(idle.idle_sleep_s, 0.5)

    def test_ignores_missing_keys(self) -> None:
        idle = IdleController(idle_after_frames=30, idle_sleep_s=0.15)
        apply_idle_settings(idle, {"unrelated": 42})
        self.assertEqual(idle.idle_after_frames, 30)
        self.assertEqual(idle.idle_sleep_s, 0.15)

    def test_tolerates_bad_values(self) -> None:
        idle = IdleController(idle_after_frames=30, idle_sleep_s=0.15)
        apply_idle_settings(idle, {"idle_after_frames": "nope", "idle_sleep_s": "bad"})
        self.assertEqual(idle.idle_after_frames, 30)
        self.assertEqual(idle.idle_sleep_s, 0.15)

    def test_clamps_negatives(self) -> None:
        idle = IdleController(idle_after_frames=30, idle_sleep_s=0.15)
        apply_idle_settings(idle, {"idle_after_frames": -1, "idle_sleep_s": -1.0})
        self.assertEqual(idle.idle_after_frames, 1)
        self.assertEqual(idle.idle_sleep_s, 0.0)

    def test_handles_none_controller(self) -> None:
        # Must not raise if mode hasn't started yet (controller is None).
        apply_idle_settings(None, {"idle_after_frames": 10})


if __name__ == "__main__":
    unittest.main()
