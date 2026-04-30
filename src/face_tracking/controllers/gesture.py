from typing import Dict, Optional, Tuple

from src.face_tracking.controllers.blendshape_gesture_constants import (
    CHEEK_PUFF_DOWN_HIGH,
    CHEEK_PUFF_DOWN_LOW,
    CHEEK_PUFF_RELEASE,
    CHEEK_PUFF_UP_HIGH,
    CLICK_HOLD_UNFREEZE_SEC,
    SCROLL_INTENT_DELAY_SEC,
    SCROLL_MAX_UNITS_PER_SEC,
    SCROLL_MIN_TICK_INTERVAL_SEC,
    SCROLL_MIN_UNITS_PER_SEC,
    SCROLL_TICK_DELTA,
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


def _lerp(value: float, in_low: float, in_high: float, out_low: float, out_high: float) -> float:
    if in_high <= in_low:
        return out_low
    ratio = (value - in_low) / (in_high - in_low)
    if ratio < 0.0:
        ratio = 0.0
    elif ratio > 1.0:
        ratio = 1.0
    return out_low + ratio * (out_high - out_low)


class GestureController:
    """Maps face signals to cursor movement, click-and-hold actions, and lip-driven scrolls.

    Click-and-hold (drawing): a smirk presses the corresponding mouse button DOWN
    immediately. The cursor is frozen for the first ``click_hold_unfreeze_sec``
    of the held smirk so the press lands on a stable target. After that, the
    cursor resumes moving while the button stays held -- this lets the user
    drag/draw. Releasing the smirk (diff drops below ``smirk_relax_diff``)
    releases the button. A brief smirk produces a press+release sequence,
    which apps treat as a single click.

    Cursor freeze: still applied during the first ``click_hold_unfreeze_sec``
    of any smirk (smirks deform mouth landmarks slightly, leaking into the
    head-pose estimate). After that window the freeze lifts even while the
    button stays held.

    Scrolls -- two distinct lip gestures:
      - Cheek puff (mouthPucker proxy) -> scroll DOWN, speed lerps from
        cheek_puff_down_low to cheek_puff_down_high.
      - Lip tuck-in (max of mouthRoll/mouthPress) -> scroll UP, speed lerps
        from tuck_trigger_low to tuck_trigger_high.
    Each has its own 200 ms intent buffer to filter accidental activations.

    Asymmetric faces: ``smirk_baseline_left`` / ``smirk_baseline_right`` and
    ``tuck_baseline`` capture the user's natural at-rest activations from
    calibration. The runtime subtracts these so an asymmetric resting face
    reads as zero.
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
        tuck_release: float = TUCK_RELEASE,
        tuck_trigger_low: float = TUCK_TRIGGER_LOW,
        tuck_trigger_high: float = TUCK_TRIGGER_HIGH,
        tuck_baseline: float = 0.0,
        scroll_intent_delay_sec: float = SCROLL_INTENT_DELAY_SEC,
        scroll_min_units_per_sec: float = SCROLL_MIN_UNITS_PER_SEC,
        scroll_max_units_per_sec: float = SCROLL_MAX_UNITS_PER_SEC,
        scroll_tick_delta: int = SCROLL_TICK_DELTA,
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
        if not (0.0 <= tuck_release < tuck_trigger_low < tuck_trigger_high):
            raise ValueError(
                "tuck thresholds must satisfy: 0 <= release < trigger_low < trigger_high"
            )
        if tuck_baseline < 0.0:
            raise ValueError("tuck_baseline must be >= 0")
        if scroll_intent_delay_sec < 0.0:
            raise ValueError("scroll_intent_delay_sec must be >= 0")
        if scroll_min_units_per_sec <= 0.0 or scroll_max_units_per_sec < scroll_min_units_per_sec:
            raise ValueError("scroll speed bounds invalid")
        if scroll_tick_delta <= 0:
            raise ValueError("scroll_tick_delta must be > 0")
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
        self.tuck_release = float(tuck_release)
        self.tuck_trigger_low = float(tuck_trigger_low)
        self.tuck_trigger_high = float(tuck_trigger_high)
        self.tuck_baseline = float(tuck_baseline)
        self.scroll_intent_delay_sec = float(scroll_intent_delay_sec)
        self.scroll_min_units_per_sec = float(scroll_min_units_per_sec)
        self.scroll_max_units_per_sec = float(scroll_max_units_per_sec)
        self.scroll_tick_delta = int(scroll_tick_delta)
        self.scroll_min_tick_interval_sec = float(scroll_min_tick_interval_sec)

        self.click_enabled = True
        self.scroll_enabled = True

        # Click / hold state
        self._click_armed: bool = True
        self._held_button: Optional[str] = None      # 'left' | 'right' | None
        self._held_started_at: float = 0.0
        self._last_click_side: Optional[str] = None  # one-frame visualizer flash
        self._last_click_consumed: bool = False
        self._smirk_active: bool = False

        # Scroll state
        self._puff_intent_started_at: Optional[float] = None
        self._tuck_intent_started_at: Optional[float] = None
        self._scroll_last_tick_at: float = 0.0
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

    def _handle_smirk_click_or_hold(self, smirk_diff: float, now: float) -> None:
        # Re-arm only when the previous click has already fully released and
        # the user has relaxed.
        if self._held_button is None and not self._click_armed:
            if abs(smirk_diff) < self.smirk_relax_diff:
                self._click_armed = True
            return

        if self._held_button is not None:
            # Released: drop below relax threshold -> button up.
            if abs(smirk_diff) < self.smirk_relax_diff:
                self._release_held_button()
                self._click_armed = True
                return
            # Side switch: held=left and user is now smirking right (or vice versa).
            if self._held_button == "left" and (-smirk_diff) > self.smirk_trigger_diff:
                self._release_held_button()
                self._press_held_button("right", now)
            elif self._held_button == "right" and smirk_diff > self.smirk_trigger_diff:
                self._release_held_button()
                self._press_held_button("left", now)
            return

        # No button held and click armed: check for fresh trigger.
        if smirk_diff > self.smirk_trigger_diff:
            self._press_held_button("left", now)
            self._click_armed = False
        elif (-smirk_diff) > self.smirk_trigger_diff:
            self._press_held_button("right", now)
            self._click_armed = False

    # ---------------------------------------------------------------- scroll

    def _emit_scroll_tick(self, direction: str, speed: float, now: float) -> None:
        if (now - self._scroll_last_tick_at) < self.scroll_min_tick_interval_sec:
            return
        self._scroll_last_tick_at = now

        clamped_speed = speed
        if clamped_speed < self.scroll_min_units_per_sec:
            clamped_speed = self.scroll_min_units_per_sec
        elif clamped_speed > self.scroll_max_units_per_sec:
            clamped_speed = self.scroll_max_units_per_sec

        previous_speed = getattr(self.cursor, "scroll_units_per_sec", None)
        try:
            self.cursor.scroll_units_per_sec = clamped_speed
            delta = self.scroll_tick_delta if direction == "up" else -self.scroll_tick_delta
            self.cursor.scroll_with_speed(delta)
        finally:
            if previous_speed is not None:
                self.cursor.scroll_units_per_sec = previous_speed

    def _handle_puff_scroll_down(self, blendshapes: Dict[str, float], now: float) -> None:
        cp = puff_value(blendshapes)

        if cp <= self.cheek_puff_release:
            self._puff_intent_started_at = None
            if self.active_scroll_gesture == "scroll_down":
                self.active_scroll_gesture = None
            return

        if cp <= self.cheek_puff_down_low:
            # Hysteresis dead-zone: preserve state, no tick.
            return

        if self._puff_intent_started_at is None:
            self._puff_intent_started_at = now
            return
        if (now - self._puff_intent_started_at) < self.scroll_intent_delay_sec:
            return

        self.active_scroll_gesture = "scroll_down"
        speed = _lerp(
            cp,
            self.cheek_puff_down_low,
            self.cheek_puff_down_high,
            self.scroll_min_units_per_sec,
            self.scroll_max_units_per_sec,
        )
        self._emit_scroll_tick("down", speed, now)

    def _handle_tuck_scroll_up(self, blendshapes: Dict[str, float], now: float) -> None:
        raw = tuck_value(blendshapes)
        tv = raw - self.tuck_baseline
        if tv < 0.0:
            tv = 0.0

        if tv <= self.tuck_release:
            self._tuck_intent_started_at = None
            if self.active_scroll_gesture == "scroll_up":
                self.active_scroll_gesture = None
            return

        if tv <= self.tuck_trigger_low:
            return

        if self._tuck_intent_started_at is None:
            self._tuck_intent_started_at = now
            return
        if (now - self._tuck_intent_started_at) < self.scroll_intent_delay_sec:
            return

        self.active_scroll_gesture = "scroll_up"
        speed = _lerp(
            tv,
            self.tuck_trigger_low,
            self.tuck_trigger_high,
            self.scroll_min_units_per_sec,
            self.scroll_max_units_per_sec,
        )
        self._emit_scroll_tick("up", speed, now)

    # ---------------------------------------------------------------- public API

    def release_all(self) -> None:
        self._release_held_button()
        self._click_armed = True
        self._last_click_side = None
        self._last_click_consumed = False
        self._puff_intent_started_at = None
        self._tuck_intent_started_at = None
        self._scroll_last_tick_at = 0.0
        self.active_scroll_gesture = None
        self._smirk_active = False

    def handle_face_analysis(self, face_analysis, now: float) -> None:
        blendshapes = face_analysis.blendshapes or {}

        # --- Smirk + cursor freeze + click-and-hold ---
        adj_left, adj_right = self._adjusted_smirk(blendshapes)
        smirk_diff = adj_left - adj_right
        self._smirk_active = abs(smirk_diff) >= self.smirk_relax_diff

        held_long_enough = (
            self._held_button is not None
            and (now - self._held_started_at) >= self.click_hold_unfreeze_sec
        )
        cursor_frozen = self._smirk_active and not held_long_enough

        if face_analysis.screen_position is not None and not cursor_frozen:
            raw_tx, raw_ty = face_analysis.screen_position
            target_x = max(self.minx, min(self.maxx, raw_tx + self.minx))
            target_y = max(self.miny, min(self.maxy, raw_ty + self.miny))
            self.cursor.step_towards(target_x, target_y)

        if self.click_enabled:
            self._handle_smirk_click_or_hold(smirk_diff, now)
        elif self._held_button is not None:
            # Clicks just disabled mid-hold; release cleanly.
            self._release_held_button()
            self._click_armed = True

        # One-frame visualizer flash for the click event.
        if self._last_click_side is not None and self._last_click_consumed:
            self._last_click_side = None
        elif self._last_click_side is not None:
            self._last_click_consumed = True

        # --- Scroll: two independent lip gestures ---
        if self.scroll_enabled:
            self._handle_puff_scroll_down(blendshapes, now)
            self._handle_tuck_scroll_up(blendshapes, now)
        else:
            self._puff_intent_started_at = None
            self._tuck_intent_started_at = None
            self.active_scroll_gesture = None

    def shutdown(self) -> None:
        self.release_all()
