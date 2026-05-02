from typing import Dict, Optional, Tuple

from src.face_tracking.controllers.blendshape_gesture_constants import (
    CHEEK_PUFF_DOWN_HIGH,
    CHEEK_PUFF_DOWN_LOW,
    CHEEK_PUFF_RELEASE,
    CHEEK_PUFF_UP_HIGH,
    CLICK_HOLD_UNFREEZE_SEC,
    SCROLL_INTENT_DELAY_SEC,
    SCROLL_MIN_TICK_INTERVAL_SEC,
    SMIRK_RELAX_DIFF,
    SMIRK_TRIGGER_DIFF,
    TUCK_RELEASE,
    TUCK_TRIGGER_HIGH,
    TUCK_TRIGGER_LOW,
)
from src.face_tracking.signals.blendshapes import (
    compute_smirk_activations,
    puff_value,
    tuck_value,
)
class GestureController:
    """Maps face signals to cursor movement, click-and-hold actions, and smirk-driven scrolls.

    Gesture map:
      - Pucker lips (cheekPuff/mouthPucker proxy) -> press & hold LEFT mouse button
      - Tuck lips inward (mouthRoll/mouthPress proxy) -> press & hold RIGHT mouse button
      - Smirk LEFT (mouthSmileLeft > mouthSmileRight) -> scroll UP
      - Smirk RIGHT (mouthSmileRight > mouthSmileLeft) -> scroll DOWN

    Click-and-hold (drawing): a lip gesture (pucker or tuck) presses the
    corresponding mouse button DOWN immediately. The cursor is frozen for
    the first ``click_hold_unfreeze_sec`` of the held button so the press
    lands on a stable target. After that, the cursor resumes moving while
    the button stays held -- this enables drag/draw. Releasing the lip
    gesture (intensity drops below the calibrated release threshold)
    releases the button. A brief pucker/tuck produces a press+release,
    which apps treat as a single click. The cursor is NOT frozen during
    the ramp-up to a click trigger -- it keeps tracking the head pose
    smoothly until the press fires.

    Smirk scroll: the signed smirk differential drives continuous scroll
    with speed proportional to ``|diff|`` between ``smirk_relax_diff`` and
    ``smirk_trigger_diff``. Positive diff (left smirk) = scroll up;
    negative diff (right smirk) = scroll down. A 200 ms intent buffer
    filters accidental brief smirks. The cursor is NOT frozen during
    smirk scrolls -- the user is presumably looking at content.

    Asymmetric faces: ``smirk_baseline_left`` / ``smirk_baseline_right``
    and ``tuck_baseline`` / ``cheek_puff_baseline`` capture the user's
    natural at-rest activations from calibration. The runtime subtracts
    these so an asymmetric resting face reads as zero across all signals.
    """

    def __init__(
        self,
        cursor,
        smirk_trigger_diff: float = SMIRK_TRIGGER_DIFF,
        smirk_relax_diff: float = SMIRK_RELAX_DIFF,
        smirk_baseline_left: float = 0.0,
        smirk_baseline_right: float = 0.0,
        click_hold_unfreeze_sec: float = CLICK_HOLD_UNFREEZE_SEC,
        cheek_puff_release: float = CHEEK_PUFF_RELEASE,
        cheek_puff_down_low: float = CHEEK_PUFF_DOWN_LOW,
        cheek_puff_down_high: float = CHEEK_PUFF_DOWN_HIGH,
        cheek_puff_up_high: float = CHEEK_PUFF_UP_HIGH,
        cheek_puff_baseline: float = 0.0,
        tuck_release: float = TUCK_RELEASE,
        tuck_trigger_low: float = TUCK_TRIGGER_LOW,
        tuck_trigger_high: float = TUCK_TRIGGER_HIGH,
        tuck_baseline: float = 0.0,
        scroll_intent_delay_sec: float = SCROLL_INTENT_DELAY_SEC,
        scroll_min_tick_interval_sec: float = SCROLL_MIN_TICK_INTERVAL_SEC,
    ) -> None:
        if smirk_trigger_diff <= 0.0:
            raise ValueError("smirk_trigger_diff must be > 0")
        if smirk_relax_diff < 0.0 or smirk_relax_diff >= smirk_trigger_diff:
            raise ValueError("smirk_relax_diff must be in [0, smirk_trigger_diff)")
        if smirk_baseline_left < 0.0 or smirk_baseline_right < 0.0:
            raise ValueError("smirk baselines must be >= 0")
        if click_hold_unfreeze_sec < 0.0:
            raise ValueError("click_hold_unfreeze_sec must be >= 0")
        if not (0.0 <= cheek_puff_release < cheek_puff_down_low < cheek_puff_down_high <= cheek_puff_up_high):
            raise ValueError(
                "cheek_puff thresholds must satisfy: "
                "0 <= release < down_low < down_high <= up_high"
            )
        if cheek_puff_baseline < 0.0:
            raise ValueError("cheek_puff_baseline must be >= 0")
        if not (0.0 <= tuck_release < tuck_trigger_low < tuck_trigger_high):
            raise ValueError(
                "tuck thresholds must satisfy: 0 <= release < trigger_low < trigger_high"
            )
        if tuck_baseline < 0.0:
            raise ValueError("tuck_baseline must be >= 0")
        if scroll_intent_delay_sec < 0.0:
            raise ValueError("scroll_intent_delay_sec must be >= 0")
        if scroll_min_tick_interval_sec < 0.0:
            raise ValueError("scroll_min_tick_interval_sec must be >= 0")

        self.cursor = cursor
        minx, miny, maxx, maxy = self.cursor.get_virtual_bounds()
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy

        self.smirk_trigger_diff = float(smirk_trigger_diff)
        self.smirk_relax_diff = float(smirk_relax_diff)
        self.smirk_baseline_left = float(smirk_baseline_left)
        self.smirk_baseline_right = float(smirk_baseline_right)
        self.click_hold_unfreeze_sec = float(click_hold_unfreeze_sec)
        self.cheek_puff_release = float(cheek_puff_release)
        self.cheek_puff_down_low = float(cheek_puff_down_low)
        self.cheek_puff_down_high = float(cheek_puff_down_high)
        self.cheek_puff_up_high = float(cheek_puff_up_high)
        self.cheek_puff_baseline = float(cheek_puff_baseline)
        self.tuck_release = float(tuck_release)
        self.tuck_trigger_low = float(tuck_trigger_low)
        self.tuck_trigger_high = float(tuck_trigger_high)
        self.tuck_baseline = float(tuck_baseline)
        self.scroll_intent_delay_sec = float(scroll_intent_delay_sec)
        self.scroll_min_tick_interval_sec = float(scroll_min_tick_interval_sec)

        self.click_enabled = True
        self.scroll_enabled = True

        # Click / hold state
        self._click_armed: bool = True
        self._held_button: Optional[str] = None      # 'left' | 'right' | None
        self._held_started_at: float = 0.0
        self._last_click_side: Optional[str] = None  # one-frame visualizer flash
        self._last_click_consumed: bool = False

        # Scroll state
        self._smirk_scroll_intent_started_at: Optional[float] = None
        self._scroll_last_tick_at: float = 0.0
        self._scroll_accumulator: float = 0.0
        self.active_scroll_gesture: Optional[str] = None  # 'scroll_up' | 'scroll_down' | None

    # ---------------------------------------------------------------- helpers

    def _adjusted_smirk(self, blendshapes: Dict[str, float]) -> Tuple[float, float]:
        raw_left, raw_right = compute_smirk_activations(blendshapes)
        adj_left = raw_left - self.smirk_baseline_left
        adj_right = raw_right - self.smirk_baseline_right
        if adj_left < 0.0:
            adj_left = 0.0
        if adj_right < 0.0:
            adj_right = 0.0
        return adj_left, adj_right

    def _adjusted_puff(self, blendshapes: Dict[str, float]) -> float:
        raw = puff_value(blendshapes)
        v = raw - self.cheek_puff_baseline
        return v if v > 0.0 else 0.0

    def _adjusted_tuck(self, blendshapes: Dict[str, float]) -> float:
        raw = tuck_value(blendshapes)
        v = raw - self.tuck_baseline
        return v if v > 0.0 else 0.0

    def _press_held_button(self, side: str, now: float) -> None:
        if side == "left":
            self.cursor.left_down()
        elif side == "right":
            self.cursor.right_down()
        else:
            return
        self._held_button = side
        self._held_started_at = now
        self._last_click_side = side
        self._last_click_consumed = False

    def _release_held_button(self) -> None:
        if self._held_button == "left":
            self.cursor.left_up()
        elif self._held_button == "right":
            self.cursor.right_up()
        self._held_button = None
        self._held_started_at = 0.0

    # ---------------------------------------------------------------- click

    def _handle_lip_click_or_hold(self, puff: float, tuck: float, now: float) -> None:
        # Use the calibrated *_down_high / *_trigger_high as the click-trigger
        # point (~78% of user max) and *_release (~20% of user max) as the
        # button-release point. This gives natural hysteresis: a deliberate
        # gesture is required to fire, and a clear relaxation is required to
        # release the button.
        puff_active = puff >= self.cheek_puff_release
        tuck_active = tuck >= self.tuck_release
        puff_trigger = puff >= self.cheek_puff_down_high
        tuck_trigger = tuck >= self.tuck_trigger_high

        # Re-arm: no button held + previous click already fired + user has
        # released both gestures.
        if self._held_button is None and not self._click_armed:
            if not puff_active and not tuck_active:
                self._click_armed = True
            return

        if self._held_button is not None:
            # Holding -- check for release or side switch.
            if self._held_button == "left":
                # Pucker fired left. Release on pucker drop.
                if not puff_active:
                    self._release_held_button()
                    self._click_armed = True
                    return
                # Side switch: tuck strongly active while pucker lingers.
                if tuck_trigger:
                    self._release_held_button()
                    self._press_held_button("right", now)
            else:  # right
                if not tuck_active:
                    self._release_held_button()
                    self._click_armed = True
                    return
                if puff_trigger:
                    self._release_held_button()
                    self._press_held_button("left", now)
            return

        # No button held + click armed: check for fresh trigger. Pucker
        # checked first since the puff and tuck blendshapes are physically
        # incompatible -- only one can realistically be at trigger level.
        if puff_trigger:
            self._press_held_button("left", now)
            self._click_armed = False
        elif tuck_trigger:
            self._press_held_button("right", now)
            self._click_armed = False

    # ---------------------------------------------------------------- scroll

    def _emit_scroll_tick(self, direction: str, speed: float, now: float) -> None:
        if (now - self._scroll_last_tick_at) < self.scroll_min_tick_interval_sec:
            return
        if self._scroll_last_tick_at == 0.0:
            elapsed = self.scroll_min_tick_interval_sec
        else:
            elapsed = now - self._scroll_last_tick_at
        self._scroll_last_tick_at = now

        clamped_speed = speed
        speed_limit = getattr(self.cursor, "scroll_units_per_sec", None)
        if speed_limit is not None:
            try:
                limit_value = float(speed_limit)
            except (TypeError, ValueError):
                limit_value = None
            if limit_value is not None and limit_value > 0.0:
                clamped_speed = limit_value

        units = clamped_speed * elapsed
        if direction != "up":
            units = -units
        self._scroll_accumulator += units

        if self._scroll_accumulator >= 1.0 or self._scroll_accumulator <= -1.0:
            delta = int(self._scroll_accumulator)
            self._scroll_accumulator -= delta
            step = 1 if delta > 0 else -1
            for _ in range(abs(delta)):
                self.cursor.scroll(step)

    def _handle_smirk_scroll(self, smirk_diff: float, now: float) -> None:
        abs_diff = abs(smirk_diff)

        if abs_diff <= self.smirk_relax_diff:
            self._smirk_scroll_intent_started_at = None
            if self.active_scroll_gesture in ("scroll_up", "scroll_down"):
                self.active_scroll_gesture = None
            self._scroll_accumulator = 0.0
            self._scroll_last_tick_at = 0.0
            return

        # Past relax threshold: candidate scroll, gated by intent buffer.
        if self._smirk_scroll_intent_started_at is None:
            self._smirk_scroll_intent_started_at = now
            self._scroll_last_tick_at = 0.0
            return
        if (now - self._smirk_scroll_intent_started_at) < self.scroll_intent_delay_sec:
            return

        # Direction: smirk LEFT (positive diff) -> scroll UP; smirk RIGHT -> scroll DOWN.
        if smirk_diff > 0:
            direction = "up"
            self.active_scroll_gesture = "scroll_up"
        else:
            direction = "down"
            self.active_scroll_gesture = "scroll_down"

        speed_limit = getattr(self.cursor, "scroll_units_per_sec", None)
        try:
            speed = float(speed_limit)
        except (TypeError, ValueError):
            return
        self._emit_scroll_tick(direction, speed, now)

    # ---------------------------------------------------------------- public API

    def release_all(self) -> None:
        self._release_held_button()
        self._click_armed = True
        self._last_click_side = None
        self._last_click_consumed = False
        self._smirk_scroll_intent_started_at = None
        self._scroll_last_tick_at = 0.0
        self._scroll_accumulator = 0.0
        self.active_scroll_gesture = None

    def handle_face_analysis(self, face_analysis, now: float) -> None:
        blendshapes = face_analysis.blendshapes or {}

        # --- Click signals (lips) ---
        puff = self._adjusted_puff(blendshapes)
        tuck = self._adjusted_tuck(blendshapes)
        # Cursor freeze: only for the first click_hold_unfreeze_sec after the
        # button presses, so the press lands on a stable target before drag.
        # No ramp-up freeze -- the cursor keeps tracking smoothly while the
        # user is forming a gesture.
        cursor_frozen = (
            self._held_button is not None
            and (now - self._held_started_at) < self.click_hold_unfreeze_sec
        )

        if face_analysis.screen_position is not None and not cursor_frozen:
            raw_tx, raw_ty = face_analysis.screen_position
            target_x = max(self.minx, min(self.maxx, raw_tx + self.minx))
            target_y = max(self.miny, min(self.maxy, raw_ty + self.miny))
            self.cursor.step_towards(target_x, target_y)

        if self.click_enabled:
            self._handle_lip_click_or_hold(puff, tuck, now)
        elif self._held_button is not None:
            self._release_held_button()
            self._click_armed = True

        # One-frame visualizer flash for the click event.
        if self._last_click_side is not None and self._last_click_consumed:
            self._last_click_side = None
        elif self._last_click_side is not None:
            self._last_click_consumed = True

        # --- Scroll signal (smirk) ---
        if self.scroll_enabled:
            adj_left, adj_right = self._adjusted_smirk(blendshapes)
            self._handle_smirk_scroll(adj_left - adj_right, now)
        else:
            self._smirk_scroll_intent_started_at = None
            self.active_scroll_gesture = None

    def shutdown(self) -> None:
        self.release_all()
