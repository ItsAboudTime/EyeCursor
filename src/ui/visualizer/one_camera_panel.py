from collections import deque
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.ui.visualizer._idle_overlay import dim_widget_for_idle, overlay_idle_on_label
from src.ui.visualizer.drawing import (
    bgr_to_qpixmap,
    draw_head_pose_arrow,
    draw_mediapipe_landmarks,
    forward_axis_from_matrix,
    render_screen_target_preview,
)


_GESTURE_ACTION_LABEL = {
    "left_click_down": "Pucker lips -> left click",
    "right_click_down": "Tuck lips in -> right click",
    "scroll_up": "Smirk left -> scroll up",
    "scroll_down": "Smirk right -> scroll down",
}


def _frame_label() -> QLabel:
    label = QLabel("No frame yet")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setMinimumSize(320, 240)
    label.setStyleSheet("background: #1c1f27; color: #cccccc; border-radius: 6px;")
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return label


class OneCameraPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._recent_actions: deque = deque(maxlen=5)
        self._last_action_signature: Optional[tuple] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        grid = QGridLayout()
        grid.setSpacing(10)

        self._raw_label = _frame_label()
        self._landmarks_label = _frame_label()
        self._pose_label = _frame_label()
        self._target_label = _frame_label()

        grid.addWidget(self._labeled_box("Raw frame", self._raw_label), 0, 0)
        grid.addWidget(self._labeled_box("Face landmarks (MediaPipe)", self._landmarks_label), 0, 1)
        grid.addWidget(self._labeled_box("Head pose vector", self._pose_label), 1, 0)
        grid.addWidget(self._labeled_box("Screen target", self._target_label), 1, 1)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        layout.addLayout(grid, 3)

        self._gesture_box = self._build_gesture_box()
        layout.addWidget(self._gesture_box, 1)

        caption = QLabel(
            "Stages of the one-camera head-pose pipeline. The screen target panel shows"
            " where the cursor is being driven on the virtual screen."
        )
        caption.setWordWrap(True)
        caption.setStyleSheet("color: #888a93; font-size: 11px;")
        layout.addWidget(caption)

    def _labeled_box(self, title: str, widget: QWidget) -> QWidget:
        box = QGroupBox(title)
        box.setStyleSheet(
            "QGroupBox { color: #dfe6e9; font-weight: bold; border: 1px solid #3a3f4b;"
            " border-radius: 6px; margin-top: 10px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.addWidget(widget)
        return box

    def _build_gesture_box(self) -> QWidget:
        box = QGroupBox("Gestures")
        box.setStyleSheet(
            "QGroupBox { color: #dfe6e9; font-weight: bold; border: 1px solid #3a3f4b;"
            " border-radius: 6px; margin-top: 10px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        outer = QHBoxLayout(box)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(20)

        left_col = QVBoxLayout()
        left_col.setSpacing(4)

        badges = QHBoxLayout()
        self._click_badge = QLabel("Click: ?")
        self._scroll_badge = QLabel("Scroll: ?")
        for badge in (self._click_badge, self._scroll_badge):
            badge.setStyleSheet(
                "padding: 4px 10px; border-radius: 10px; background: #3a3f4b;"
                " color: #dfe6e9; font-size: 11px;"
            )
            badges.addWidget(badge)
        badges.addStretch(1)
        left_col.addLayout(badges)

        self._last_click_label = QLabel("Last click: --")
        self._smirk_left_label = QLabel("Smirk L: --")
        self._smirk_right_label = QLabel("Smirk R: --")
        self._pucker_label = QLabel("Pucker: --")
        self._tuck_label = QLabel("Tuck: --")
        self._scroll_state_label = QLabel("Scroll state: idle")
        self._click_state_label = QLabel("Click state: armed")
        self._yaw_pitch_label = QLabel("Yaw: --  Pitch: --")
        for lbl in (
            self._last_click_label,
            self._smirk_left_label,
            self._smirk_right_label,
            self._pucker_label,
            self._tuck_label,
            self._click_state_label,
            self._scroll_state_label,
            self._yaw_pitch_label,
        ):
            lbl.setStyleSheet("color: #dfe6e9; font-size: 12px;")
            left_col.addWidget(lbl)
        left_col.addStretch(1)
        outer.addLayout(left_col, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #3a3f4b;")
        outer.addWidget(sep)

        right_col = QVBoxLayout()
        right_col.setSpacing(4)
        actions_title = QLabel("Recent actions")
        actions_title.setStyleSheet(
            "color: #b2bec3; font-size: 11px; font-weight: bold;"
        )
        right_col.addWidget(actions_title)
        self._actions_label = QLabel("(none)")
        self._actions_label.setStyleSheet("color: #dfe6e9; font-size: 11px;")
        self._actions_label.setWordWrap(True)
        right_col.addWidget(self._actions_label)
        right_col.addStretch(1)
        outer.addLayout(right_col, 1)
        return box

    def update_payload(self, payload: dict) -> None:
        frame_bgr = payload.get("frame_bgr")
        landmarks = payload.get("landmarks")
        frame_w = int(payload.get("frame_width") or 0)
        frame_h = int(payload.get("frame_height") or 0)
        ftm = payload.get("facial_transformation_matrix")
        yaw = payload.get("yaw_deg")
        pitch = payload.get("pitch_deg")
        gesture_state = payload.get("gesture_state") or {}
        idle = bool(payload.get("idle", False))

        if frame_bgr is not None:
            self._raw_label.setPixmap(
                bgr_to_qpixmap(frame_bgr).scaled(
                    self._raw_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            overlay_idle_on_label(self._raw_label, idle)

            if landmarks is not None and frame_w and frame_h:
                with_landmarks = draw_mediapipe_landmarks(
                    frame_bgr, landmarks, frame_w, frame_h
                )
                self._landmarks_label.setPixmap(
                    bgr_to_qpixmap(with_landmarks).scaled(
                        self._landmarks_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )

            forward = forward_axis_from_matrix(ftm)
            nose_xy = self._nose_pixel(landmarks, frame_w, frame_h)
            if forward is not None and nose_xy is not None:
                with_arrow = draw_head_pose_arrow(frame_bgr, nose_xy, forward)
                self._pose_label.setPixmap(
                    bgr_to_qpixmap(with_arrow).scaled(
                        self._pose_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )

        target = payload.get("target_screen_xy")
        screen_bounds = payload.get("screen_bounds")
        target_canvas = render_screen_target_preview(
            target_xy=tuple(target) if target is not None else None,
            screen_bounds=tuple(screen_bounds) if screen_bounds is not None else None,
            canvas_size=(
                max(320, self._target_label.width()),
                max(180, self._target_label.height()),
            ),
        )
        self._target_label.setPixmap(
            bgr_to_qpixmap(target_canvas).scaled(
                self._target_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        # Dim every secondary panel when idle; the raw frame already has the
        # dim + IDLE badge painted into its pixmap above.
        dim_widget_for_idle(self._landmarks_label, idle)
        dim_widget_for_idle(self._pose_label, idle)
        dim_widget_for_idle(self._target_label, idle)
        dim_widget_for_idle(self._gesture_box, idle)

        click_enabled = bool(gesture_state.get("click_enabled", False))
        scroll_enabled = bool(gesture_state.get("scroll_enabled", False))
        self._click_badge.setText(f"Click: {'on' if click_enabled else 'off'}")
        self._scroll_badge.setText(f"Scroll: {'on' if scroll_enabled else 'off'}")
        self._click_badge.setStyleSheet(
            "padding: 4px 10px; border-radius: 10px;"
            f" background: {'#00b894' if click_enabled else '#3a3f4b'};"
            " color: white; font-size: 11px;"
        )
        self._scroll_badge.setStyleSheet(
            "padding: 4px 10px; border-radius: 10px;"
            f" background: {'#00b894' if scroll_enabled else '#3a3f4b'};"
            " color: white; font-size: 11px;"
        )

        last_click = gesture_state.get("last_click_side")
        self._last_click_label.setText(f"Last click: {last_click if last_click else '--'}")
        smirk_l = gesture_state.get("smirk_left_activation")
        smirk_r = gesture_state.get("smirk_right_activation")
        pucker = gesture_state.get("pucker_value")
        self._smirk_left_label.setText(
            f"Smirk L: {smirk_l:.3f}" if smirk_l is not None else "Smirk L: --"
        )
        self._smirk_right_label.setText(
            f"Smirk R: {smirk_r:.3f}" if smirk_r is not None else "Smirk R: --"
        )
        self._pucker_label.setText(
            f"Pucker: {pucker:.3f}" if pucker is not None else "Pucker: --"
        )
        tuck = gesture_state.get("tuck_value")
        self._tuck_label.setText(
            f"Tuck: {tuck:.3f}" if tuck is not None else "Tuck: --"
        )
        is_held = bool(gesture_state.get("is_held", False))
        held_button = gesture_state.get("held_button")
        click_armed = gesture_state.get("click_armed", True)
        if is_held:
            self._click_state_label.setText(
                f"Click state: holding {held_button}" if held_button else "Click state: holding"
            )
        elif click_armed:
            self._click_state_label.setText("Click state: armed")
        else:
            self._click_state_label.setText("Click state: waiting for relax")
        active_scroll = gesture_state.get("active_scroll_gesture")
        self._scroll_state_label.setText(
            f"Scroll state: {active_scroll if active_scroll else 'idle'}"
        )
        if yaw is not None and pitch is not None:
            self._yaw_pitch_label.setText(f"Yaw: {yaw:+.1f}°  Pitch: {-pitch:+.1f}°")

        self._record_action(payload)

    def _nose_pixel(self, landmarks, frame_w: int, frame_h: int):
        if not landmarks or frame_w <= 0 or frame_h <= 0:
            return None
        try:
            pts = list(landmarks)
            if len(pts) <= 1:
                return None
            nose = pts[1]
            return int(float(nose.x) * frame_w), int(float(nose.y) * frame_h)
        except (AttributeError, TypeError):
            return None

    def _record_action(self, payload: dict) -> None:
        gesture_state = payload.get("gesture_state") or {}
        action = gesture_state.get("last_action")
        if not action:
            return
        sig = (action, gesture_state.get("last_action_at"))
        if sig == self._last_action_signature:
            return
        self._last_action_signature = sig
        label = _GESTURE_ACTION_LABEL.get(action, action)
        self._recent_actions.appendleft(label)
        self._actions_label.setText("\n".join(self._recent_actions) or "(none)")
