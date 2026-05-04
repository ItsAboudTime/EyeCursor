from collections import deque
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.ui.overlays.gaze_bubble_overlay import BUBBLE_RADIUS_PX
from src.ui.visualizer._idle_overlay import dim_widget_for_idle, overlay_idle_on_label
from src.ui.visualizer.drawing import (
    bgr_to_qpixmap,
    draw_dlib_landmarks,
    draw_gaze_arrow_on_patch,
    render_bubble_screen_preview,
    render_screen_target_preview,
    rgb_to_qpixmap,
)
from src.ui.visualizer.one_camera_panel import _GESTURE_ACTION_LABEL


def _frame_label(min_w: int = 320, min_h: int = 240) -> QLabel:
    label = QLabel("No frame yet")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setMinimumSize(min_w, min_h)
    label.setStyleSheet("background: #1c1f27; color: #cccccc; border-radius: 6px;")
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return label


def _labeled(title: str, widget: QWidget) -> QGroupBox:
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


class EyeGazePanel(QWidget):
    def __init__(self, show_bubble_indicator: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._show_bubble_indicator = show_bubble_indicator
        self._recent_actions = deque(maxlen=5)
        self._last_action_signature = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        grid = QGridLayout()
        grid.setSpacing(10)

        self._raw_label = _frame_label()
        self._detection_label = _frame_label()
        self._patch_label = _frame_label(min_w=240, min_h=240)
        self._target_label = _frame_label(min_w=320, min_h=180)

        target_title = "4. Bubble position" if self._show_bubble_indicator else "4. Screen target"
        grid.addWidget(_labeled("1. Raw frame", self._raw_label), 0, 0)
        grid.addWidget(_labeled("2. Face detection + dlib landmarks", self._detection_label), 0, 1)
        grid.addWidget(_labeled("3. Normalized face patch + gaze vector", self._patch_label), 1, 0)
        grid.addWidget(_labeled(target_title, self._target_label), 1, 1)

        if self._show_bubble_indicator:
            self._yaw_pitch_label = QLabel("Yaw: --  Pitch: --")
            self._yaw_pitch_label.setStyleSheet("color: #dfe6e9; font-size: 12px;")
            grid.addWidget(_labeled("5. Yaw / Pitch", self._yaw_pitch_label), 2, 0, 1, 2)

        if self._show_bubble_indicator:
            grid.setRowStretch(0, 1)
            grid.setRowStretch(1, 1)
            grid.setRowStretch(2, 0)
        else:
            for r in range(2):
                grid.setRowStretch(r, 1)
        for c in range(2):
            grid.setColumnStretch(c, 1)

        outer.addLayout(grid, 5)

        self._bubble_label = None
        if self._show_bubble_indicator:
            self._bubble_label = QLabel("Bubble overlay: inactive")
            self._bubble_label.setStyleSheet(
                "color: #b2bec3; font-size: 12px; padding-left: 8px;"
            )
        self._gesture_box = None
        if not self._show_bubble_indicator:
            self._gesture_box = self._build_gesture_box()
            outer.addWidget(self._gesture_box, 1)

    def update_payload(self, payload: dict) -> None:
        frame_bgr = payload.get("frame_bgr")
        face_box = payload.get("dlib_face_box")
        dlib_landmarks = payload.get("dlib_landmarks_68")
        face_patch_bgr = payload.get("face_patch_bgr")
        pitch_rad = payload.get("pitch_rad")
        yaw_rad = payload.get("yaw_rad")
        target = payload.get("target_screen_xy")
        screen_bounds = payload.get("screen_bounds")
        gesture_state = payload.get("gesture_state") or {}

        idle = bool(payload.get("idle", False))

        if frame_bgr is not None:
            self._set_pixmap_bgr(self._raw_label, frame_bgr)
            with_landmarks = draw_dlib_landmarks(frame_bgr, dlib_landmarks, face_box)
            self._set_pixmap_bgr(self._detection_label, with_landmarks)
            overlay_idle_on_label(self._raw_label, idle)

        if face_patch_bgr is not None and pitch_rad is not None and yaw_rad is not None:
            patch_rgb = cv2.cvtColor(face_patch_bgr, cv2.COLOR_BGR2RGB)
            with_arrow = draw_gaze_arrow_on_patch(patch_rgb, pitch_rad, yaw_rad)
            self._set_pixmap_rgb(self._patch_label, with_arrow)

        canvas_size = (
            max(320, self._target_label.width()),
            max(180, self._target_label.height()),
        )
        if self._show_bubble_indicator:
            canvas = render_bubble_screen_preview(
                target_xy=tuple(target) if target is not None else None,
                screen_bounds=tuple(screen_bounds) if screen_bounds is not None else None,
                bubble_radius_px=BUBBLE_RADIUS_PX,
                canvas_size=canvas_size,
            )
        else:
            canvas = render_screen_target_preview(
                target_xy=tuple(target) if target is not None else None,
                screen_bounds=tuple(screen_bounds) if screen_bounds is not None else None,
                canvas_size=canvas_size,
            )
        self._set_pixmap_bgr(self._target_label, canvas)

        if self._yaw_pitch_label is not None:
            if pitch_rad is not None and yaw_rad is not None:
                self._yaw_pitch_label.setText(
                    f"Yaw: {np.degrees(yaw_rad):+.1f}°  Pitch: {np.degrees(pitch_rad):+.1f}°"
                )
            else:
                self._yaw_pitch_label.setText("Yaw: --  Pitch: --")

        if self._bubble_label is not None:
            bubble_active = bool(payload.get("bubble_active", False))
            bubble_target = payload.get("bubble_target_xy")
            if bubble_active and bubble_target is not None:
                self._bubble_label.setText(
                    f"Bubble overlay: ACTIVE  target=({int(bubble_target[0])}, {int(bubble_target[1])})"
                )
                self._bubble_label.setStyleSheet(
                    "color: #00b894; font-size: 12px; padding-top: 4px; font-weight: bold;"
                )
            else:
                self._bubble_label.setText("Bubble overlay: inactive")
                self._bubble_label.setStyleSheet(
                    "color: #b2bec3; font-size: 12px; padding-top: 4px;"
                )

        if self._gesture_box is not None:
            self._update_gesture_panel(payload)

        # Dim all secondary panels uniformly when idle. The raw-frame label
        # already carries the dim + IDLE badge from overlay_idle_on_label above.
        dim_widget_for_idle(self._detection_label, idle)
        dim_widget_for_idle(self._patch_label, idle)
        dim_widget_for_idle(self._target_label, idle)
        if self._gesture_box is not None:
            dim_widget_for_idle(self._gesture_box, idle)
        if self._bubble_label is not None:
            dim_widget_for_idle(self._bubble_label, idle)

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
        if self._bubble_label is not None:
            right_col.addWidget(self._bubble_label)
        right_col.addStretch(1)
        outer.addLayout(right_col, 1)
        return box

    def _update_gesture_panel(self, payload: dict) -> None:
        gesture_state = payload.get("gesture_state") or {}
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
        tuck = gesture_state.get("tuck_value")
        self._smirk_left_label.setText(
            f"Smirk L: {smirk_l:.3f}" if smirk_l is not None else "Smirk L: --"
        )
        self._smirk_right_label.setText(
            f"Smirk R: {smirk_r:.3f}" if smirk_r is not None else "Smirk R: --"
        )
        self._pucker_label.setText(
            f"Pucker: {pucker:.3f}" if pucker is not None else "Pucker: --"
        )
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

        self._record_action(payload)

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

    def _set_pixmap_bgr(self, label: QLabel, frame_bgr) -> None:
        pix = bgr_to_qpixmap(frame_bgr).scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pix)

    def _set_pixmap_rgb(self, label: QLabel, frame_rgb) -> None:
        pix = rgb_to_qpixmap(frame_rgb).scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pix)
