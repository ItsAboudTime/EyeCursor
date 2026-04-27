from typing import Optional


class GestureController:
    """Maps face signals to cursor movement and wink-based click actions."""

    def __init__(
        self,
        cursor,
        hold_trigger_seconds: float = 1.0,
        release_missed_frames: int = 5,
        wink_trigger_seconds: float = 0.1,
        both_eyes_open_threshold: float = 0.7,
        both_eyes_squint_threshold: float = 0.3,
        scroll_trigger_seconds: float = 1.0,
        scroll_delta: int = 120,
    ) -> None:
        if hold_trigger_seconds <= 0.0:
            raise ValueError("hold_trigger_seconds must be > 0")
        if release_missed_frames < 1:
            raise ValueError("release_missed_frames must be >= 1")
        if wink_trigger_seconds <= 0.0:
            raise ValueError("wink_trigger_seconds must be > 0")
        if both_eyes_squint_threshold < 0.0:
            raise ValueError("both_eyes_squint_threshold must be >= 0")
        if both_eyes_open_threshold <= both_eyes_squint_threshold:
            raise ValueError("both_eyes_open_threshold must be > both_eyes_squint_threshold")
        if scroll_trigger_seconds <= 0.0:
            raise ValueError("scroll_trigger_seconds must be > 0")
        if scroll_delta <= 0:
            raise ValueError("scroll_delta must be > 0")

        self.cursor = cursor
        minx, miny, maxx, maxy = self.cursor.get_virtual_bounds()
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy

        self.hold_trigger_seconds = float(hold_trigger_seconds)
        self.release_missed_frames = int(release_missed_frames)
        self.wink_trigger_seconds = float(wink_trigger_seconds)
        self.both_eyes_open_threshold = float(both_eyes_open_threshold)
        self.both_eyes_squint_threshold = float(both_eyes_squint_threshold)
        self.scroll_trigger_seconds = float(scroll_trigger_seconds)
        self.scroll_delta = int(scroll_delta)

        self.left_is_down = False
        self.right_is_down = False
        self.active_blink_side: Optional[str] = None
        self.blink_started_at = 0.0
        self.hold_mode = False
        self.missed_same_side_frames = 0
        self.pending_wink_side: Optional[str] = None
        self.pending_wink_started_at = 0.0
        self.confirmed_wink_side: Optional[str] = None

        self.click_enabled = True
        self.scroll_enabled = True
        self.active_scroll_gesture: Optional[str] = None
        self.scroll_gesture_started_at = 0.0
        self.last_scroll_at = 0.0

    def _resolve_scroll_gesture(self, left_eye_ratio: Optional[float], right_eye_ratio: Optional[float]) -> Optional[str]:
        if left_eye_ratio is None or right_eye_ratio is None:
            return None
        if (
            left_eye_ratio >= self.both_eyes_open_threshold
            and right_eye_ratio >= self.both_eyes_open_threshold
        ):
            return "both_open"
        if (
            left_eye_ratio <= self.both_eyes_squint_threshold
            and right_eye_ratio <= self.both_eyes_squint_threshold
        ):
            return "both_squint"
        return None

    def _handle_scroll_gesture(self, face_analysis, now: float) -> None:
        if not self.scroll_enabled:
            return
        gesture = self._resolve_scroll_gesture(
            left_eye_ratio=face_analysis.left_eye_ratio,
            right_eye_ratio=face_analysis.right_eye_ratio,
        )

        if gesture != self.active_scroll_gesture:
            self.active_scroll_gesture = gesture
            self.scroll_gesture_started_at = now
            self.last_scroll_at = 0.0
            return

        if gesture is None:
            return

        if (now - self.scroll_gesture_started_at) < self.scroll_trigger_seconds:
            return

        scroll_units_per_sec = float(getattr(self.cursor, "scroll_units_per_sec", 300.0))
        min_repeat_interval = abs(self.scroll_delta) / max(1e-6, scroll_units_per_sec)
        if self.last_scroll_at > 0.0 and (now - self.last_scroll_at) < min_repeat_interval:
            return

        if gesture == "both_open":
            self.cursor.scroll_with_speed(-self.scroll_delta)
        elif gesture == "both_squint":
            self.cursor.scroll_with_speed(self.scroll_delta)

        self.last_scroll_at = now

    @staticmethod
    def _wink_to_button(wink_side: Optional[str]) -> Optional[str]:
        # Left wink maps to right button, right wink maps to left button.
        if wink_side == "left":
            return "right"
        if wink_side == "right":
            return "left"
        return None

    def _release_active_button(self) -> list[str]:
        actions: list[str] = []
        if self.active_blink_side == "left" and self.left_is_down:
            actions.append("left_up")
            self.left_is_down = False
        elif self.active_blink_side == "right" and self.right_is_down:
            actions.append("right_up")
            self.right_is_down = False
        return actions

    def _apply_mouse_actions(self, actions: list[str]) -> None:
        for action in actions:
            if action == "left_down":
                self.cursor.left_down()
            elif action == "left_up":
                self.cursor.left_up()
            elif action == "right_down":
                self.cursor.right_down()
            elif action == "right_up":
                self.cursor.right_up()

    def _press_button(self, button: str) -> list[str]:
        actions: list[str] = []
        if button == "left" and not self.left_is_down:
            actions.append("left_down")
            self.left_is_down = True
        elif button == "right" and not self.right_is_down:
            actions.append("right_down")
            self.right_is_down = True
        return actions

    def _intentional_wink(self, wink_side: Optional[str], now: float) -> Optional[str]:
        if wink_side is None:
            self.pending_wink_side = None
            self.pending_wink_started_at = 0.0
            self.confirmed_wink_side = None
            return None

        if self.confirmed_wink_side == wink_side:
            return wink_side

        if self.pending_wink_side != wink_side:
            self.pending_wink_side = wink_side
            self.pending_wink_started_at = now
            self.confirmed_wink_side = None
            return None

        if (now - self.pending_wink_started_at) >= self.wink_trigger_seconds:
            self.confirmed_wink_side = wink_side
            return wink_side

        return None

    def update(self, wink_side: Optional[str], now: float) -> list[str]:
        actions: list[str] = []
        desired_button = self._wink_to_button(wink_side)

        if self.active_blink_side is None:
            if desired_button is not None:
                actions.extend(self._press_button(desired_button))
                self.active_blink_side = desired_button
                self.blink_started_at = now
                self.hold_mode = False
                self.missed_same_side_frames = 0
            return actions

        if desired_button == self.active_blink_side:
            self.missed_same_side_frames = 0
            if not self.hold_mode and (now - self.blink_started_at) >= self.hold_trigger_seconds:
                self.hold_mode = True
            return actions

        if self.hold_mode:
            self.missed_same_side_frames += 1
            should_release = self.missed_same_side_frames >= self.release_missed_frames
        else:
            should_release = True

        if should_release:
            actions.extend(self._release_active_button())

            self.active_blink_side = None
            self.blink_started_at = 0.0
            self.hold_mode = False
            self.missed_same_side_frames = 0

            # If the latest frame indicates the opposite wink,
            # start that click immediately after releasing.
            if desired_button is not None:
                actions.extend(self._press_button(desired_button))
                self.active_blink_side = desired_button
                self.blink_started_at = now

        return actions

    def release_all(self) -> list[str]:
        actions: list[str] = []
        if self.left_is_down:
            actions.append("left_up")
        if self.right_is_down:
            actions.append("right_up")

        self.left_is_down = False
        self.right_is_down = False
        self.active_blink_side = None
        self.blink_started_at = 0.0
        self.hold_mode = False
        self.missed_same_side_frames = 0
        self.pending_wink_side = None
        self.pending_wink_started_at = 0.0
        self.confirmed_wink_side = None
        self.active_scroll_gesture = None
        self.scroll_gesture_started_at = 0.0
        self.last_scroll_at = 0.0
        return actions

    def handle_face_analysis(self, face_analysis, now: float) -> None:
        if face_analysis.screen_position is not None:
            raw_tx, raw_ty = face_analysis.screen_position
            target_x = max(self.minx, min(self.maxx, raw_tx + self.minx))
            target_y = max(self.miny, min(self.maxy, raw_ty + self.miny))
            self.cursor.step_towards(target_x, target_y)

        if self.click_enabled:
            wink_side = self._intentional_wink(face_analysis.wink_direction, now)
            actions = self.update(
                wink_side=wink_side,
                now=now,
            )
            self._apply_mouse_actions(actions)
        self._handle_scroll_gesture(face_analysis=face_analysis, now=now)

    def shutdown(self) -> None:
        self._apply_mouse_actions(self.release_all())
