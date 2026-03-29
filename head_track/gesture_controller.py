from typing import Optional


class GestureController:
    """Maps face signals to cursor movement and wink-based click actions."""

    def __init__(
        self,
        cursor,
        hold_trigger_seconds: float = 1.0,
        release_missed_frames: int = 5,
    ) -> None:
        if hold_trigger_seconds <= 0.0:
            raise ValueError("hold_trigger_seconds must be > 0")
        if release_missed_frames < 1:
            raise ValueError("release_missed_frames must be >= 1")

        self.cursor = cursor
        minx, miny, maxx, maxy = self.cursor.get_virtual_bounds()
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy

        self.hold_trigger_seconds = float(hold_trigger_seconds)
        self.release_missed_frames = int(release_missed_frames)

        self.left_is_down = False
        self.right_is_down = False
        self.active_blink_side: Optional[str] = None
        self.blink_started_at = 0.0
        self.hold_mode = False
        self.missed_same_side_frames = 0

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
        return actions

    def handle_face_analysis(self, face_analysis, now: float) -> None:
        if face_analysis.screen_position is not None:
            raw_tx, raw_ty = face_analysis.screen_position
            target_x = max(self.minx, min(self.maxx, raw_tx + self.minx))
            target_y = max(self.miny, min(self.maxy, raw_ty + self.miny))
            self.cursor.step_towards(target_x, target_y)

        actions = self.update(
            wink_side=face_analysis.wink_direction,
            now=now,
        )
        self._apply_mouse_actions(actions)

    def shutdown(self) -> None:
        self._apply_mouse_actions(self.release_all())
