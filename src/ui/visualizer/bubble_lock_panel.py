"""Visualizer panel for the Bubble Lock mode (hybrid_bubble_lock)."""

from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.ui.overlays.gaze_bubble_overlay import BUBBLE_RADIUS_PX
from src.ui.visualizer._idle_overlay import dim_widget_for_idle, overlay_idle_on_label
from src.ui.visualizer.drawing import (
    bgr_to_qpixmap,
    draw_gaze_arrow_on_patch,
    draw_head_pose_arrow,
    draw_mediapipe_landmarks,
    forward_axis_from_matrix,
    rgb_to_qpixmap,
)
from src.ui.visualizer.head_view_3d import HeadView3D
from src.ui.visualizer.one_camera_panel import _GESTURE_ACTION_LABEL


_STATE_COLORS = {
    "gaze_follow": "#0984e3",
    "frozen":      "#e17055",
}
_STATE_LABELS = {
    "gaze_follow": "GAZE FOLLOW",
    "frozen":      "FROZEN",
}


def _frame_label(min_w: int = 240, min_h: int = 180) -> QLabel:
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
    inner = QVBoxLayout(box)
    inner.setContentsMargins(8, 4, 8, 8)
    inner.addWidget(widget)
    return box


def _render_bubble_lock_screen(
    state: str,
    screen_bounds: Optional[Tuple],
    bubble_center_xy: Optional[Tuple[int, int]],
    cursor_xy: Optional[Tuple[int, int]],
    bubble_radius_px: int,
    canvas_size: Tuple[int, int] = (480, 270),
) -> np.ndarray:
    """Render the virtual screen showing the bubble circle and cursor crosshair."""
    cw, ch = canvas_size
    canvas = np.full((ch, cw, 3), 30, dtype=np.uint8)
    cv2.rectangle(canvas, (4, 4), (cw - 5, ch - 5), (90, 110, 160), 2)

    if screen_bounds is None or bubble_center_xy is None:
        cv2.putText(
            canvas, "no target",
            (16, ch // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA,
        )
        return canvas

    minx, miny, maxx, maxy = screen_bounds
    screen_w = max(1, maxx - minx + 1)
    screen_h = max(1, maxy - miny + 1)

    def to_canvas(sx: int, sy: int) -> Tuple[int, int]:
        nx = max(0.0, min(1.0, (sx - minx) / screen_w))
        ny = max(0.0, min(1.0, (sy - miny) / screen_h))
        return int(nx * (cw - 1)), int(ny * (ch - 1))

    bx, by = to_canvas(*bubble_center_xy)
    canvas_radius = max(4, int(bubble_radius_px * cw / screen_w))

    bubble_color = (60, 130, 230) if state == "gaze_follow" else (60, 130, 230)
    frozen_color = (40, 160, 230)
    gaze_color   = (40, 160, 230)
    circle_color = frozen_color if state == "frozen" else gaze_color
    cv2.circle(canvas, (bx, by), canvas_radius, circle_color, 2, cv2.LINE_AA)
    cv2.circle(canvas, (bx, by), 5, circle_color, -1, cv2.LINE_AA)

    if cursor_xy is not None:
        cx, cy = to_canvas(*cursor_xy)
        cv2.drawMarker(canvas, (cx, cy), (60, 200, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 6, (60, 200, 255), 2, cv2.LINE_AA)
        # Line from bubble center to cursor to show displacement.
        cv2.line(canvas, (bx, by), (cx, cy), (80, 80, 80), 1, cv2.LINE_AA)

    state_text = _STATE_LABELS.get(state, state.upper())
    cv2.putText(
        canvas, state_text,
        (12, ch - 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
    )
    return canvas


class BubbleLockPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._recent_actions: deque = deque(maxlen=5)
        self._last_action_signature: Optional[tuple] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        grid = QGridLayout()
        grid.setSpacing(10)

        self._patch_label     = _frame_label(min_w=220, min_h=220)
        self._landmarks_label = _frame_label()
        self._head_view       = HeadView3D()
        self._target_label    = _frame_label(min_w=320, min_h=180)

        grid.addWidget(_labeled("1. Gaze face patch + vector", self._patch_label), 0, 0)
        grid.addWidget(_labeled("2. MediaPipe landmarks + head pose", self._landmarks_label), 0, 1)
        grid.addWidget(_labeled("3. 3D head model (triangulated)", self._head_view), 1, 0)
        grid.addWidget(_labeled("4. Bubble & cursor position", self._target_label), 1, 1)

        for r in range(2):
            grid.setRowStretch(r, 1)
        for c in range(2):
            grid.setColumnStretch(c, 1)

        outer.addLayout(grid, 5)

        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        bottom.addWidget(self._build_state_box(), 1)
        bottom.addWidget(self._build_gesture_box(), 2)
        bottom.addWidget(self._build_actions_box(), 1)
        outer.addLayout(bottom, 1)

    # ------------------------------------------------------------------ #
    # Bottom-strip boxes                                                   #
    # ------------------------------------------------------------------ #

    def _build_state_box(self) -> QGroupBox:
        box = QGroupBox("Mode state")
        box.setStyleSheet(
            "QGroupBox { color: #dfe6e9; font-weight: bold; border: 1px solid #3a3f4b;"
            " border-radius: 6px; margin-top: 10px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        col = QVBoxLayout(box)
        col.setContentsMargins(12, 8, 12, 8)
        col.setSpacing(4)

        self._state_badge = QLabel("GAZE FOLLOW")
        self._state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state_badge.setStyleSheet(
            "padding: 5px 12px; border-radius: 10px; background: #0984e3;"
            " color: white; font-size: 12px; font-weight: bold;"
        )
        col.addWidget(self._state_badge)

        self._gaze_cnn_label    = QLabel("Gaze CNN: active")
        self._frozen_ctr_label  = QLabel("Frozen center: --")
        self._gaze_target_label = QLabel("Gaze target: --")
        self._head_angle_label  = QLabel("Head yaw: --  Pitch: --")
        self._gaze_angle_label  = QLabel("Gaze yaw: --  Pitch: --")
        for lbl in (
            self._gaze_cnn_label,
            self._frozen_ctr_label,
            self._gaze_target_label,
            self._head_angle_label,
            self._gaze_angle_label,
        ):
            lbl.setStyleSheet("color: #dfe6e9; font-size: 12px;")
            col.addWidget(lbl)
        col.addStretch(1)
        return box

    def _build_gesture_box(self) -> QGroupBox:
        box = QGroupBox("Gesture telemetry")
        box.setStyleSheet(
            "QGroupBox { color: #dfe6e9; font-weight: bold; border: 1px solid #3a3f4b;"
            " border-radius: 6px; margin-top: 10px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        col = QVBoxLayout(box)
        col.setContentsMargins(12, 8, 12, 8)
        col.setSpacing(4)

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
        col.addLayout(badges)

        self._last_click_label  = QLabel("Last click: --")
        self._click_state_label = QLabel("Click state: armed")
        self._pucker_label      = QLabel("Pucker: --")
        self._tuck_label        = QLabel("Tuck: --")
        self._smirk_left_label  = QLabel("Smirk L: --")
        self._smirk_right_label = QLabel("Smirk R: --")
        self._scroll_state_label = QLabel("Scroll state: idle")
        for lbl in (
            self._last_click_label,
            self._click_state_label,
            self._pucker_label,
            self._tuck_label,
            self._smirk_left_label,
            self._smirk_right_label,
            self._scroll_state_label,
        ):
            lbl.setStyleSheet("color: #dfe6e9; font-size: 12px;")
            col.addWidget(lbl)
        col.addStretch(1)
        return box

    def _build_actions_box(self) -> QGroupBox:
        box = QGroupBox("Recent actions")
        box.setStyleSheet(
            "QGroupBox { color: #dfe6e9; font-weight: bold; border: 1px solid #3a3f4b;"
            " border-radius: 6px; margin-top: 10px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        col = QVBoxLayout(box)
        col.setContentsMargins(12, 8, 12, 8)
        col.setSpacing(4)
        self._actions_label = QLabel("(none)")
        self._actions_label.setStyleSheet("color: #dfe6e9; font-size: 11px;")
        self._actions_label.setWordWrap(True)
        col.addWidget(self._actions_label)
        col.addStretch(1)
        return box

    # ------------------------------------------------------------------ #
    # Payload update                                                       #
    # ------------------------------------------------------------------ #

    def update_payload(self, payload: dict) -> None:
        frame_bgr  = payload.get("frame_bgr")
        landmarks  = payload.get("landmarks")
        frame_w    = int(payload.get("frame_width") or 0)
        frame_h    = int(payload.get("frame_height") or 0)
        ftm        = payload.get("facial_transformation_matrix")
        yaw_deg    = payload.get("yaw_deg")
        pitch_deg  = payload.get("pitch_deg")

        face_patch_bgr = payload.get("face_patch_bgr")
        pitch_rad      = payload.get("pitch_rad")
        yaw_rad        = payload.get("yaw_rad")

        state         = payload.get("bubble_lock_state", "gaze_follow")
        frozen_center = payload.get("bubble_lock_frozen_center")
        bubble_target = payload.get("bubble_target_xy")
        gaze_target   = payload.get("target_screen_xy")
        screen_bounds = payload.get("screen_bounds")
        bubble_r      = int(payload.get("bubble_lock_radius_px") or BUBBLE_RADIUS_PX)
        screen_pos    = payload.get("screen_position")
        screen_w      = int(payload.get("screen_width") or 1)
        screen_h      = int(payload.get("screen_height") or 1)
        points_3d     = payload.get("points_3d")
        depth         = payload.get("depth")
        gesture_state = payload.get("gesture_state") or {}

        # 1. Gaze face patch (skipped during FROZEN) -------------------
        if face_patch_bgr is not None and pitch_rad is not None and yaw_rad is not None:
            patch_rgb  = cv2.cvtColor(face_patch_bgr, cv2.COLOR_BGR2RGB)
            with_arrow = draw_gaze_arrow_on_patch(patch_rgb, pitch_rad, yaw_rad)
            self._set_rgb(self._patch_label, with_arrow)

        idle = bool(payload.get("idle", False))

        # 2. Landmarks + head-pose arrow --------------------------------
        if frame_bgr is not None:
            annotated = frame_bgr.copy()
            if landmarks is not None and frame_w and frame_h:
                annotated = draw_mediapipe_landmarks(annotated, landmarks, frame_w, frame_h)
            forward  = forward_axis_from_matrix(ftm)
            nose_xy  = self._nose_pixel(landmarks, frame_w, frame_h)
            if forward is not None and nose_xy is not None:
                annotated = draw_head_pose_arrow(annotated, nose_xy, forward)
            self._set_bgr(self._landmarks_label, annotated)
            overlay_idle_on_label(self._landmarks_label, idle)

        # 3. 3D head model ---------------------------------------------
        self._head_view.update_points(points_3d, depth)

        # 4. Bubble & cursor -------------------------------------------
        bubble_center = frozen_center if state == "frozen" else (bubble_target or gaze_target)

        cursor_xy: Optional[Tuple[int, int]] = None
        if state == "frozen" and frozen_center is not None and screen_pos is not None:
            if screen_w > 1 and screen_h > 1:
                norm_x = max(0.0, min(1.0, screen_pos[0] / float(screen_w - 1)))
                norm_y = max(0.0, min(1.0, screen_pos[1] / float(screen_h - 1)))
                cursor_xy = (
                    int(round(frozen_center[0] + (norm_x - 0.5) * 2.0 * bubble_r)),
                    int(round(frozen_center[1] + (norm_y - 0.5) * 2.0 * bubble_r)),
                )

        target_canvas = _render_bubble_lock_screen(
            state=state,
            screen_bounds=tuple(screen_bounds) if screen_bounds is not None else None,
            bubble_center_xy=bubble_center,
            cursor_xy=cursor_xy,
            bubble_radius_px=bubble_r,
            canvas_size=(
                max(320, self._target_label.width()),
                max(180, self._target_label.height()),
            ),
        )
        self._set_bgr(self._target_label, target_canvas)

        # State box ----------------------------------------------------
        color = _STATE_COLORS.get(state, "#636e72")
        self._state_badge.setText(_STATE_LABELS.get(state, state.upper()))
        self._state_badge.setStyleSheet(
            f"padding: 5px 12px; border-radius: 10px; background: {color};"
            " color: white; font-size: 12px; font-weight: bold;"
        )

        if state == "frozen":
            self._gaze_cnn_label.setText("Gaze CNN: paused")
            self._gaze_cnn_label.setStyleSheet("color: #e17055; font-size: 12px;")
        else:
            self._gaze_cnn_label.setText("Gaze CNN: active")
            self._gaze_cnn_label.setStyleSheet("color: #00b894; font-size: 12px;")

        if frozen_center is not None:
            self._frozen_ctr_label.setText(
                f"Frozen center: ({frozen_center[0]}, {frozen_center[1]})"
            )
        else:
            self._frozen_ctr_label.setText("Frozen center: --")

        if gaze_target is not None:
            self._gaze_target_label.setText(
                f"Gaze target: ({gaze_target[0]}, {gaze_target[1]})"
            )
        else:
            self._gaze_target_label.setText("Gaze target: --")

        if yaw_deg is not None and pitch_deg is not None:
            self._head_angle_label.setText(f"Head yaw: {yaw_deg:+.1f}°  Pitch: {-pitch_deg:+.1f}°")
        else:
            self._head_angle_label.setText("Head yaw: --  Pitch: --")

        if yaw_rad is not None and pitch_rad is not None:
            self._gaze_angle_label.setText(
                f"Gaze yaw: {np.degrees(yaw_rad):+.1f}°  Pitch: {np.degrees(pitch_rad):+.1f}°"
            )
        else:
            self._gaze_angle_label.setText("Gaze yaw: --  Pitch: --")

        # Gesture telemetry --------------------------------------------
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
        self._last_click_label.setText(
            f"Last click: {last_click if last_click else '--'}"
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

        pucker = gesture_state.get("pucker_value")
        tuck   = gesture_state.get("tuck_value")
        smirk_l = gesture_state.get("smirk_left_activation")
        smirk_r = gesture_state.get("smirk_right_activation")
        self._pucker_label.setText(
            f"Pucker: {pucker:.3f}" if pucker is not None else "Pucker: --"
        )
        self._tuck_label.setText(
            f"Tuck: {tuck:.3f}" if tuck is not None else "Tuck: --"
        )
        self._smirk_left_label.setText(
            f"Smirk L: {smirk_l:.3f}" if smirk_l is not None else "Smirk L: --"
        )
        self._smirk_right_label.setText(
            f"Smirk R: {smirk_r:.3f}" if smirk_r is not None else "Smirk R: --"
        )
        active_scroll = gesture_state.get("active_scroll_gesture")
        self._scroll_state_label.setText(
            f"Scroll state: {active_scroll if active_scroll else 'idle'}"
        )

        self._record_action(gesture_state)

        # Dim every other visual panel when idle. The landmarks label
        # already carries the dim + IDLE badge from overlay_idle_on_label above.
        dim_widget_for_idle(self._patch_label, idle)
        dim_widget_for_idle(self._head_view, idle)
        dim_widget_for_idle(self._target_label, idle)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _nose_pixel(self, landmarks, frame_w: int, frame_h: int) -> Optional[Tuple[int, int]]:
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

    def _record_action(self, gesture_state: dict) -> None:
        action = gesture_state.get("last_action")
        if not action:
            return
        sig = (action, gesture_state.get("last_action_at"))
        if sig == self._last_action_signature:
            return
        self._last_action_signature = sig
        self._recent_actions.appendleft(_GESTURE_ACTION_LABEL.get(action, action))
        self._actions_label.setText("\n".join(self._recent_actions) or "(none)")

    def _set_bgr(self, label: QLabel, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None:
            return
        pix = bgr_to_qpixmap(frame_bgr).scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pix)

    def _set_rgb(self, label: QLabel, frame_rgb: np.ndarray) -> None:
        if frame_rgb is None:
            return
        pix = rgb_to_qpixmap(frame_rgb).scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pix)
