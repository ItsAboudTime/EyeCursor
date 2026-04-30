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

from src.ui.visualizer.drawing import (
    bgr_to_qpixmap,
    draw_mediapipe_landmarks,
    render_displacement_panel,
    render_screen_target_preview,
)
from src.ui.visualizer.head_view_3d import HeadView3D


_DISPLACEMENT_INDICES = (1, 10, 33, 133, 152, 234, 263, 362, 454)


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
    layout = QVBoxLayout(box)
    layout.setContentsMargins(8, 4, 8, 8)
    layout.addWidget(widget)
    return box


class TwoCameraPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._recent_actions: deque = deque(maxlen=5)
        self._last_action_signature: Optional[tuple] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        grid = QGridLayout()
        grid.setSpacing(10)

        self._left_raw = _frame_label()
        self._right_raw = _frame_label()
        self._left_landmarks = _frame_label()
        self._right_landmarks = _frame_label()
        self._displacement = _frame_label()
        self._head_view = HeadView3D()

        grid.addWidget(_labeled("Left raw", self._left_raw), 0, 0)
        grid.addWidget(_labeled("Right raw", self._right_raw), 0, 1)
        grid.addWidget(_labeled("Left + landmarks", self._left_landmarks), 1, 0)
        grid.addWidget(_labeled("Right + landmarks", self._right_landmarks), 1, 1)
        grid.addWidget(_labeled("Displacement (left ↔ right)", self._displacement), 2, 0)
        grid.addWidget(_labeled("3D head model (triangulated)", self._head_view), 2, 1)

        for r in range(3):
            grid.setRowStretch(r, 1)
        for c in range(2):
            grid.setColumnStretch(c, 1)

        outer.addLayout(grid, 5)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)
        bottom_row.addWidget(self._build_status_box(), 2)
        self._target_label = _frame_label(min_w=280, min_h=160)
        bottom_row.addWidget(_labeled("Screen target", self._target_label), 1)
        outer.addLayout(bottom_row, 1)

    def _build_status_box(self) -> QWidget:
        box = QGroupBox("Stereo telemetry & gestures")
        box.setStyleSheet(
            "QGroupBox { color: #dfe6e9; font-weight: bold; border: 1px solid #3a3f4b;"
            " border-radius: 6px; margin-top: 10px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        outer = QHBoxLayout(box)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(20)

        left_col = QVBoxLayout()
        self._depth_label = QLabel("Depth: --")
        self._yaw_pitch_label = QLabel("Yaw: --  Pitch: --")
        self._screen_label = QLabel("Screen target: --")
        for lbl in (self._depth_label, self._yaw_pitch_label, self._screen_label):
            lbl.setStyleSheet("color: #dfe6e9; font-size: 12px;")
            left_col.addWidget(lbl)
        left_col.addStretch(1)
        outer.addLayout(left_col, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #3a3f4b;")
        outer.addWidget(sep)

        mid_col = QVBoxLayout()
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
        mid_col.addLayout(badges)
        self._wink_label = QLabel("Wink: none")
        self._click_state_label = QLabel("Click state: idle")
        self._scroll_state_label = QLabel("Scroll state: idle")
        for lbl in (self._wink_label, self._click_state_label, self._scroll_state_label):
            lbl.setStyleSheet("color: #dfe6e9; font-size: 12px;")
            mid_col.addWidget(lbl)
        mid_col.addStretch(1)
        outer.addLayout(mid_col, 1)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: #3a3f4b;")
        outer.addWidget(sep2)

        right_col = QVBoxLayout()
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
        left_bgr = payload.get("frame_left_bgr")
        right_bgr = payload.get("frame_right_bgr")
        left_lm = payload.get("landmarks_left")
        right_lm = payload.get("landmarks_right")
        left_w = int(payload.get("left_frame_width") or 0)
        left_h = int(payload.get("left_frame_height") or 0)
        right_w = int(payload.get("right_frame_width") or 0)
        right_h = int(payload.get("right_frame_height") or 0)
        points_3d = payload.get("points_3d")
        depth = payload.get("depth")

        self._set_pixmap(self._left_raw, left_bgr)
        self._set_pixmap(self._right_raw, right_bgr)

        if left_bgr is not None and left_lm is not None and left_w and left_h:
            self._set_pixmap(
                self._left_landmarks,
                draw_mediapipe_landmarks(left_bgr, left_lm, left_w, left_h),
            )
        if right_bgr is not None and right_lm is not None and right_w and right_h:
            self._set_pixmap(
                self._right_landmarks,
                draw_mediapipe_landmarks(right_bgr, right_lm, right_w, right_h),
            )

        if left_lm is not None and right_lm is not None:
            displacement = render_displacement_panel(
                left_lm,
                right_lm,
                left_w or 1,
                left_h or 1,
                right_w or 1,
                right_h or 1,
                indices=_DISPLACEMENT_INDICES,
                canvas_size=(self._displacement.width() or 480, self._displacement.height() or 320),
            )
            self._set_pixmap(self._displacement, displacement)

        self._head_view.update_points(points_3d, depth)

        target = payload.get("target_screen_xy")
        screen_bounds = payload.get("screen_bounds")
        target_canvas = render_screen_target_preview(
            target_xy=tuple(target) if target is not None else None,
            screen_bounds=tuple(screen_bounds) if screen_bounds is not None else None,
            canvas_size=(
                max(320, self._target_label.width()),
                max(160, self._target_label.height()),
            ),
        )
        self._set_pixmap(self._target_label, target_canvas)

        yaw = payload.get("yaw_deg")
        pitch = payload.get("pitch_deg")
        screen_position = payload.get("screen_position")
        gesture_state = payload.get("gesture_state") or {}

        if depth is not None:
            self._depth_label.setText(f"Depth: {depth:.3f} m")
        else:
            self._depth_label.setText("Depth: --")
        if yaw is not None and pitch is not None:
            self._yaw_pitch_label.setText(f"Yaw: {yaw:+.1f}°  Pitch: {pitch:+.1f}°")
        else:
            self._yaw_pitch_label.setText("Yaw: --  Pitch: --")
        if screen_position is not None:
            self._screen_label.setText(
                f"Screen target: ({screen_position[0]}, {screen_position[1]})"
            )
        else:
            self._screen_label.setText("Screen target: --")

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

        self._wink_label.setText(f"Wink: {payload.get('wink_direction') or 'none'}")
        active_blink = gesture_state.get("active_blink_side")
        self._click_state_label.setText(
            f"Click state: {active_blink + ' button held' if active_blink else 'idle'}"
        )
        active_scroll = gesture_state.get("active_scroll_gesture")
        self._scroll_state_label.setText(
            f"Scroll state: {active_scroll if active_scroll else 'idle'}"
        )

        self._record_action(gesture_state)

    def _record_action(self, gesture_state: dict) -> None:
        action = gesture_state.get("last_action")
        if not action:
            return
        sig = (action, gesture_state.get("last_action_at"))
        if sig == self._last_action_signature:
            return
        self._last_action_signature = sig
        from src.ui.visualizer.one_camera_panel import _GESTURE_ACTION_LABEL

        label = _GESTURE_ACTION_LABEL.get(action, action)
        self._recent_actions.appendleft(label)
        self._actions_label.setText("\n".join(self._recent_actions) or "(none)")

    def _set_pixmap(self, label: QLabel, frame_bgr) -> None:
        if frame_bgr is None:
            return
        pix = bgr_to_qpixmap(frame_bgr).scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pix)
