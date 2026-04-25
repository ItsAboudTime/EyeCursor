import pathlib
from typing import Optional

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

from src.core.calibration.gaze_calibration import (
    CALIBRATION_POINTS,
    NUM_CAPTURE_FRAMES,
    GazeCalibrationSession,
)
from src.core.devices.camera_manager import CameraManager


class GazeCalibrationWizard(QDialog):
    def __init__(
        self,
        camera_index: int,
        camera_manager: CameraManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gaze Calibration")
        self.setMinimumSize(800, 600)

        self._camera_index = camera_index
        self._camera_manager = camera_manager
        self._session = GazeCalibrationSession()
        self._inference = None
        self._current_target = 0
        self._is_capturing = False
        self._result: Optional[dict] = None
        self._weights_path: Optional[str] = None

        self._setup_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self._instruction_label = QLabel(
            "First, select the ETH-XGaze model weights file."
        )
        self._instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._instruction_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px;")
        self._instruction_label.setWordWrap(True)
        layout.addWidget(self._instruction_label)

        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(640, 480)
        self._preview_label.setStyleSheet("background: #2d3436; border-radius: 8px;")
        layout.addWidget(self._preview_label)

        self._progress_label = QLabel("")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, NUM_CAPTURE_FRAMES)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        btn_layout = QHBoxLayout()

        self._browse_btn = QPushButton("Browse Weights...")
        self._browse_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._browse_btn.setStyleSheet(
            "QPushButton { background: #6c5ce7; color: white; border: none; "
            "padding: 10px 24px; border-radius: 6px; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background: #5f3dc4; }"
        )
        self._browse_btn.clicked.connect(self._on_browse_weights)
        btn_layout.addWidget(self._browse_btn)

        self._capture_btn = QPushButton("Capture  [Space]")
        self._capture_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._capture_btn.setStyleSheet(
            "QPushButton { background: #00b894; color: white; border: none; "
            "padding: 10px 30px; border-radius: 6px; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background: #00a381; }"
            "QPushButton:disabled { background: #b2bec3; }"
        )
        self._capture_btn.setVisible(False)
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

    def _on_browse_weights(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ETH-XGaze Weights", "", "Model files (*.pth *.pth.tar *.pt);;All files (*)"
        )
        if not path:
            return
        self._weights_path = path
        self._instruction_label.setText("Loading model... (this may take a moment)")
        QTimer.singleShot(100, self._load_model)

    def _load_model(self) -> None:
        try:
            from src.eye_tracking.pipelines.eth_xgaze_inference import ETHXGazeInference
            self._inference = ETHXGazeInference(weights=pathlib.Path(self._weights_path))
            self._camera_manager.open_camera(self._camera_index)
            self._timer.start(33)
            self._browse_btn.setVisible(False)
            self._capture_btn.setVisible(True)
            self._progress_bar.setVisible(True)
            self._instruction_label.setText(
                f"Look at the target and press Capture. Target 1 / {len(CALIBRATION_POINTS)}"
            )
            self._progress_label.setText(f"Target 1 / {len(CALIBRATION_POINTS)}")
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", str(e))
            self._instruction_label.setText("Failed to load model. Try a different weights file.")

    def _update_frame(self) -> None:
        frame = self._camera_manager.get_frame(self._camera_index)
        if frame is None:
            return

        if self._is_capturing and self._inference and self._current_target < len(CALIBRATION_POINTS):
            result = self._inference.infer_from_frame(frame)
            if result is not None:
                pitch_rad, yaw_rad, _, _ = result
                self._session.capture_gaze_sample(pitch_rad, yaw_rad)
                count = self._session.get_capture_count()
                self._progress_bar.setValue(count)
                if self._session.has_enough_samples():
                    self._is_capturing = False
                    target = CALIBRATION_POINTS[self._current_target]
                    self._session.finalize_target(target)
                    self._advance_target()

        display = self._draw_target_overlay(frame)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        scaled = QPixmap.fromImage(image).scaled(
            self._preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled)

    def _draw_target_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self._current_target >= len(CALIBRATION_POINTS):
            return frame
        display = frame.copy()
        h, w = display.shape[:2]
        tx, ty = CALIBRATION_POINTS[self._current_target]
        cx = int(tx * w)
        cy = int(ty * h)
        cv2.circle(display, (cx, cy), 30, (0, 0, 255), 3)
        cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(display, (cx - 40, cy), (cx + 40, cy), (0, 0, 255), 1)
        cv2.line(display, (cx, cy - 40), (cx, cy + 40), (0, 0, 255), 1)
        return display

    def _on_capture(self) -> None:
        if self._current_target < len(CALIBRATION_POINTS):
            self._is_capturing = True
            self._capture_btn.setEnabled(False)
            self._progress_bar.setValue(0)

    def _advance_target(self) -> None:
        self._current_target += 1
        if self._current_target < len(CALIBRATION_POINTS):
            self._progress_label.setText(
                f"Target {self._current_target + 1} / {len(CALIBRATION_POINTS)}"
            )
            self._instruction_label.setText(
                f"Look at the target and press Capture. Target {self._current_target + 1} / {len(CALIBRATION_POINTS)}"
            )
            self._capture_btn.setEnabled(True)
            self._progress_bar.setValue(0)
        else:
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        self._capture_btn.setVisible(False)
        self._instruction_label.setText("Computing gaze calibration...")
        self._result = self._session.compute_calibration()
        if self._result:
            if self._weights_path:
                self._result["weights_path"] = self._weights_path
            quality = self._result["quality_label"]
            score = self._result["quality_score"]
            mean_err = self._result["mean_error"]
            self._instruction_label.setText(
                f"Calibration complete! Quality: {quality} ({score:.0%}, error: {mean_err:.4f})"
            )
            self._save_btn.setVisible(True)
            self._retry_btn.setVisible(True)
        else:
            self._instruction_label.setText("Calibration failed. Please retry.")
            self._retry_btn.setVisible(True)

    def _on_retry(self) -> None:
        self._session.reset()
        self._current_target = 0
        self._result = None
        self._capture_btn.setVisible(True)
        self._capture_btn.setEnabled(True)
        self._save_btn.setVisible(False)
        self._retry_btn.setVisible(False)
        self._progress_label.setText(f"Target 1 / {len(CALIBRATION_POINTS)}")
        self._progress_bar.setValue(0)
        self._instruction_label.setText(
            f"Look at the target and press Capture. Target 1 / {len(CALIBRATION_POINTS)}"
        )

    def _on_save(self) -> None:
        self.accept()

    def get_result(self) -> Optional[dict]:
        return self._result

    def _cleanup(self) -> None:
        self._timer.stop()
        self._camera_manager.release_camera(self._camera_index)

    def closeEvent(self, event) -> None:
        self._cleanup()
        event.accept()

    def reject(self) -> None:
        self._cleanup()
        super().reject()
