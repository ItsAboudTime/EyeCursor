from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.ui.visualizer.drawing import (
    bgr_to_qpixmap,
    draw_dlib_landmarks,
    draw_gaze_arrow_on_patch,
    render_screen_target_preview,
    rgb_to_qpixmap,
)


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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        grid = QGridLayout()
        grid.setSpacing(10)

        self._raw_label = _frame_label()
        self._detection_label = _frame_label()
        self._patch_label = _frame_label(min_w=240, min_h=240)
        self._target_label = _frame_label(min_w=320, min_h=180)

        grid.addWidget(_labeled("1. Raw frame", self._raw_label), 0, 0)
        grid.addWidget(_labeled("2. Face detection + dlib landmarks", self._detection_label), 0, 1)
        grid.addWidget(_labeled("3. Normalized face patch + gaze vector", self._patch_label), 1, 0)
        grid.addWidget(_labeled("4. Screen target", self._target_label), 1, 1)

        for r in range(2):
            grid.setRowStretch(r, 1)
        for c in range(2):
            grid.setColumnStretch(c, 1)

        outer.addLayout(grid, 5)

        readout = QGroupBox("Gaze readout")
        readout.setStyleSheet(
            "QGroupBox { color: #dfe6e9; font-weight: bold; border: 1px solid #3a3f4b;"
            " border-radius: 6px; margin-top: 10px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        readout_layout = QVBoxLayout(readout)
        readout_layout.setContentsMargins(12, 8, 12, 8)
        self._yaw_pitch_label = QLabel("Pitch: --   Yaw: --")
        self._target_text_label = QLabel("Target: --")
        for lbl in (self._yaw_pitch_label, self._target_text_label):
            lbl.setStyleSheet("color: #dfe6e9; font-size: 12px;")
            readout_layout.addWidget(lbl)
        if self._show_bubble_indicator:
            self._bubble_label = QLabel("Bubble overlay: inactive")
            self._bubble_label.setStyleSheet(
                "color: #b2bec3; font-size: 12px; padding-top: 4px;"
            )
            readout_layout.addWidget(self._bubble_label)
        else:
            self._bubble_label = None
        outer.addWidget(readout, 1)

    def update_payload(self, payload: dict) -> None:
        frame_bgr = payload.get("frame_bgr")
        face_box = payload.get("dlib_face_box")
        dlib_landmarks = payload.get("dlib_landmarks_68")
        face_patch_bgr = payload.get("face_patch_bgr")
        pitch_rad = payload.get("pitch_rad")
        yaw_rad = payload.get("yaw_rad")
        target = payload.get("target_screen_xy")
        screen_bounds = payload.get("screen_bounds")

        if frame_bgr is not None:
            self._set_pixmap_bgr(self._raw_label, frame_bgr)
            with_landmarks = draw_dlib_landmarks(frame_bgr, dlib_landmarks, face_box)
            self._set_pixmap_bgr(self._detection_label, with_landmarks)

        if face_patch_bgr is not None and pitch_rad is not None and yaw_rad is not None:
            patch_rgb = cv2.cvtColor(face_patch_bgr, cv2.COLOR_BGR2RGB)
            with_arrow = draw_gaze_arrow_on_patch(patch_rgb, pitch_rad, yaw_rad)
            self._set_pixmap_rgb(self._patch_label, with_arrow)

        canvas = render_screen_target_preview(
            target_xy=tuple(target) if target is not None else None,
            screen_bounds=tuple(screen_bounds) if screen_bounds is not None else None,
            canvas_size=(
                max(320, self._target_label.width()),
                max(180, self._target_label.height()),
            ),
        )
        self._set_pixmap_bgr(self._target_label, canvas)

        if pitch_rad is not None and yaw_rad is not None:
            self._yaw_pitch_label.setText(
                f"Pitch: {np.degrees(pitch_rad):+.1f}°   Yaw: {np.degrees(yaw_rad):+.1f}°"
            )
        else:
            self._yaw_pitch_label.setText("Pitch: --   Yaw: --")

        if target is not None:
            self._target_text_label.setText(
                f"Target: ({int(target[0])}, {int(target[1])})"
            )
        else:
            self._target_text_label.setText("Target: --")

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
