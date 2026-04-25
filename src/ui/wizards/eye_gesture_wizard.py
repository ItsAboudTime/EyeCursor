from typing import Optional

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

from src.core.calibration.eye_gesture_calibration import (
    NUM_CAPTURE_FRAMES,
    EyeGestureCalibrationSession,
)
from src.core.devices.camera_manager import CameraManager


STEPS = [
    ("open", "Open both eyes normally."),
    ("left_wink", "Close your LEFT eye (left wink)."),
    ("right_wink", "Close your RIGHT eye (right wink)."),
    ("squint", "Squint both eyes."),
    ("wide_open", "Open both eyes wide open."),
]


class EyeGestureCalibrationWizard(QDialog):
    def __init__(
        self,
        camera_index: int,
        camera_manager: CameraManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Eye Gesture Calibration")
        self.setMinimumSize(700, 550)
        self.setStyleSheet(
            "QDialog { background: #2d3436; }"
            "QLabel { color: white; }"
            "QProgressBar { color: white; }"
        )

        self._camera_index = camera_index
        self._camera_manager = camera_manager
        self._session = EyeGestureCalibrationSession()
        self._current_step = 0
        self._is_capturing = False
        self._result: Optional[dict] = None
        self._latest_ratios: Optional[tuple] = None

        self._setup_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        try:
            self._camera_manager.open_camera(self._camera_index)
            self._timer.start(33)
        except RuntimeError as e:
            self._instruction_label.setText(f"Camera error: {e}")

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self._instruction_label = QLabel(STEPS[0][1])
        self._instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._instruction_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px;")
        layout.addWidget(self._instruction_label)

        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(640, 480)
        self._preview_label.setStyleSheet("background: #2d3436; border-radius: 8px;")
        layout.addWidget(self._preview_label)

        self._ratio_label = QLabel("Eye ratios: --")
        self._ratio_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ratio_label.setStyleSheet("font-size: 14px; color: #b2bec3;")
        layout.addWidget(self._ratio_label)

        self._progress_label = QLabel(f"Step 1 / {len(STEPS)}")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, NUM_CAPTURE_FRAMES)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        btn_layout = QHBoxLayout()

        self._capture_btn = QPushButton("Capture  [Space]")
        self._capture_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._capture_btn.setStyleSheet(
            "QPushButton { background: #00b894; color: white; border: none; "
            "padding: 10px 30px; border-radius: 6px; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background: #00a381; }"
            "QPushButton:disabled { background: #b2bec3; }"
        )
        self._capture_btn.clicked.connect(self._on_capture)
        btn_layout.addWidget(self._capture_btn)

        self._retry_btn = QPushButton("Retry")
        self._retry_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._retry_btn.setStyleSheet(
            "QPushButton { background: #fdcb6e; color: #2d3436; border: none; "
            "padding: 10px 30px; border-radius: 6px; font-size: 14px; font-weight: bold; }"
        )
        self._retry_btn.clicked.connect(self._on_retry)
        self._retry_btn.setVisible(False)
        btn_layout.addWidget(self._retry_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._save_btn.setStyleSheet(
            "QPushButton { background: #0984e3; color: white; border: none; "
            "padding: 10px 30px; border-radius: 6px; font-size: 14px; font-weight: bold; }"
        )
        self._save_btn.clicked.connect(self._on_save)
        self._save_btn.setVisible(False)
        btn_layout.addWidget(self._save_btn)

        cancel_btn = QPushButton("Cancel  [Esc]")
        cancel_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        cancel_btn.setStyleSheet(
            "QPushButton { background: #636e72; color: white; border: none; "
            "padding: 10px 30px; border-radius: 6px; font-size: 14px; }"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            if self._capture_btn.isVisible() and self._capture_btn.isEnabled():
                self._on_capture()
        elif event.key() == Qt.Key.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)

    def _update_frame(self) -> None:
        frame = self._camera_manager.get_frame(self._camera_index)
        if frame is None:
            return

        if self._is_capturing and self._current_step < len(STEPS):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            step_name = STEPS[self._current_step][0]
            capture_fn = {
                "open": self._session.capture_open_eyes,
                "left_wink": self._session.capture_left_wink,
                "right_wink": self._session.capture_right_wink,
                "squint": self._session.capture_squint,
                "wide_open": self._session.capture_wide_open,
            }[step_name]

            ratios = capture_fn(rgb)
            if ratios:
                self._latest_ratios = ratios
            count = self._session.get_sample_count(step_name)
            self._progress_bar.setValue(count)

            if self._session.has_enough_samples(step_name):
                self._is_capturing = False
                self._advance_step()

        if self._latest_ratios:
            self._ratio_label.setText(
                f"Eye ratios: L={self._latest_ratios[0]:.3f}  R={self._latest_ratios[1]:.3f}"
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        scaled = QPixmap.fromImage(image).scaled(
            self._preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled)

    def _on_capture(self) -> None:
        if self._current_step < len(STEPS):
            self._is_capturing = True
            self._capture_btn.setEnabled(False)
            self._progress_bar.setValue(0)

    def _advance_step(self) -> None:
        self._current_step += 1
        if self._current_step < len(STEPS):
            step_name, instruction = STEPS[self._current_step]
            self._instruction_label.setText(instruction)
            self._progress_label.setText(f"Step {self._current_step + 1} / {len(STEPS)}")
            self._capture_btn.setEnabled(True)
            self._progress_bar.setValue(0)
        else:
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        self._capture_btn.setVisible(False)
        self._instruction_label.setText("Computing calibration...")
        self._result = self._session.compute_calibration()
        if self._result:
            quality = self._result["quality_label"]
            score = self._result["quality_score"]
            self._instruction_label.setText(
                f"Calibration complete! Quality: {quality} ({score:.0%})"
            )
            self._save_btn.setVisible(True)
            self._retry_btn.setVisible(True)
        else:
            self._instruction_label.setText("Calibration failed. Please retry.")
            self._retry_btn.setVisible(True)

    def _on_retry(self) -> None:
        self._session.reset()
        self._current_step = 0
        self._result = None
        self._latest_ratios = None
        self._capture_btn.setVisible(True)
        self._capture_btn.setEnabled(True)
        self._save_btn.setVisible(False)
        self._retry_btn.setVisible(False)
        self._progress_label.setText(f"Step 1 / {len(STEPS)}")
        self._progress_bar.setValue(0)
        self._instruction_label.setText(STEPS[0][1])

    def _on_save(self) -> None:
        self.accept()

    def get_result(self) -> Optional[dict]:
        return self._result

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._camera_manager.release_camera(self._camera_index)
        self._session.release()
        event.accept()

    def reject(self) -> None:
        self._timer.stop()
        self._camera_manager.release_camera(self._camera_index)
        self._session.release()
        super().reject()
