"""Idle-state detection for tracking mode loops.

Counts consecutive no-face frames; trips into ``is_idle = True`` after a
configurable streak. Mode loops use this to back off from per-frame
inference work when the user has stepped away.

Designed to live entirely on the tracking thread (no locks; the mode loop
is the only caller). The Pi-future hook (``set_on_change``) is a single
callback slot meant for the capture supervisor to subscribe to so it can
throttle camera FPS over a control channel — left unwired today.
"""

import time
from typing import Callable, Optional


class IdleController:
    def __init__(
        self,
        idle_after_frames: int = 30,
        idle_sleep_s: float = 0.15,
    ) -> None:
        self.idle_after_frames = max(1, int(idle_after_frames))
        self.idle_sleep_s = max(0.0, float(idle_sleep_s))
        self._streak = 0
        self._is_idle = False
        self._on_change: Optional[Callable[[bool], None]] = None

    @property
    def is_idle(self) -> bool:
        return self._is_idle

    @property
    def streak_frames(self) -> int:
        return self._streak

    def observe(self, face_detected: bool) -> bool:
        """Record one loop iteration's detection result.

        Returns True iff the idle state flipped on this call (so the mode
        loop can force a one-shot visualization emission to flip the badge
        instantly).
        """
        if face_detected:
            self._streak = 0
            if self._is_idle:
                self._is_idle = False
                self._fire_change(False)
                return True
            return False
        self._streak += 1
        if not self._is_idle and self._streak >= self.idle_after_frames:
            self._is_idle = True
            self._fire_change(True)
            return True
        return False

    def maybe_sleep(self) -> None:
        if self._is_idle and self.idle_sleep_s > 0:
            time.sleep(self.idle_sleep_s)

    def set_on_change(self, callback: Optional[Callable[[bool], None]]) -> None:
        """Reserved Pi-future hook: a capture supervisor can subscribe here
        so it gets a True/False call on every active↔idle transition and
        relay that to the capture subprocess (control channel TBD) to drop
        camera FPS while idle. Today nothing besides the mode loop observes
        the idle state."""
        self._on_change = callback

    def _fire_change(self, is_idle: bool) -> None:
        if self._on_change is None:
            return
        try:
            self._on_change(is_idle)
        except Exception:
            # A faulty subscriber must never kill the tracking loop.
            pass


def apply_idle_settings(idle: Optional[IdleController], settings: dict) -> None:
    """Push live-mutable idle thresholds onto the controller. Tolerant of
    missing keys / bad input — mirrors ``_apply_cursor_settings`` etc."""
    if idle is None or not settings:
        return
    if "idle_after_frames" in settings:
        try:
            idle.idle_after_frames = max(1, int(settings["idle_after_frames"]))
        except (TypeError, ValueError) as exc:
            print(f"warning: bad idle_after_frames, ignored: {exc}")
    if "idle_sleep_s" in settings:
        try:
            idle.idle_sleep_s = max(0.0, float(settings["idle_sleep_s"]))
        except (TypeError, ValueError) as exc:
            print(f"warning: bad idle_sleep_s, ignored: {exc}")
