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
    QWidget,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPainter, QPen, QBrush, QColor, QPixmap

from src.core.calibration.head_pose_calibration import (
    CALIBRATION_POINTS,
    NUM_CAPTURE_FRAMES,
    HeadPoseCalibrationSession,
)
from src.core.devices.camera_manager import CameraManager

TARGET_DESCRIPTIONS = [
    "Center",
    "Top-Left",
    "Top-Center",
    "Top-Right",
    "Middle-Left",
    "Middle-Right",
    "Bottom-Left",
    "Bottom-Center",
    "Bottom-Right",
]


class HeadPoseCalibrationWizard(QDialog):
    def __init__(
        self,
        camera_index: int,
        camera_manager: CameraManager,
        screen_w: int = 1920,
        screen_h: int = 1080,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Head Pose Calibration")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
        )
        self.setStyleSheet("background: #1e272e;")

        self._camera_index = camera_index
        self._camera_manager = camera_manager
        self._session = HeadPoseCalibrationSession()
        self._session.set_screen_size(screen_w, screen_h)
        self._current_target = 0
        self._is_capturing = False
        self._result: Optional[dict] = None

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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setStyleSheet("background: #1e272e;")
        layout.addWidget(self._canvas, stretch=1)

        bottom_bar = QWidget()
        bottom_bar.setStyleSheet("background: rgba(0, 0, 0, 180);")
        bottom_bar.setFixedHeight(160)
        bottom_layout = QVBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(24, 10, 24, 10)
        bottom_layout.setSpacing(8)

        self._instruction_label = QLabel(self._get_instruction_text())
        self._instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._instruction_label.setWordWrap(True)
        self._instruction_label.setFont(QFont("", 20, QFont.Weight.Bold))
        self._instruction_label.setStyleSheet("color: white; padding: 4px;")
        bottom_layout.addWidget(self._instruction_label)

        self._progress_label = QLabel(f"Target 1 of {len(CALIBRATION_POINTS)}  —  {TARGET_DESCRIPTIONS[0]}")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setFont(QFont("", 14))
        self._progress_label.setStyleSheet("color: #b2bec3;")
        bottom_layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, NUM_CAPTURE_FRAMES)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background: #2d3436; border: none; border-radius: 4px; }"
            "QProgressBar::chunk { background: #00b894; border-radius: 4px; }"
        )
        bottom_layout.addWidget(self._progress_bar)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self._capture_btn = QPushButton("Capture  [Space]")
        self._capture_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._capture_btn.setFont(QFont("", 14, QFont.Weight.Bold))
        self._capture_btn.setStyleSheet(
            "QPushButton { background: #00b894; color: white; border: none; "
            "padding: 10px 36px; border-radius: 6px; }"
            "QPushButton:hover { background: #00a381; }"
            "QPushButton:disabled { background: #636e72; color: #b2bec3; }"
        )
        self._capture_btn.clicked.connect(self._on_capture)
        btn_layout.addWidget(self._capture_btn)

        self._retry_btn = QPushButton("Retry")
        self._retry_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._retry_btn.setFont(QFont("", 14, QFont.Weight.Bold))
        self._retry_btn.setStyleSheet(
            "QPushButton { background: #fdcb6e; color: #2d3436; border: none; "
            "padding: 10px 36px; border-radius: 6px; }"
            "QPushButton:hover { background: #feca57; }"
        )
        self._retry_btn.clicked.connect(self._on_retry)
        self._retry_btn.setVisible(False)
        btn_layout.addWidget(self._retry_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._save_btn.setFont(QFont("", 14, QFont.Weight.Bold))
        self._save_btn.setStyleSheet(
            "QPushButton { background: #0984e3; color: white; border: none; "
            "padding: 10px 36px; border-radius: 6px; }"
            "QPushButton:hover { background: #0652DD; }"
        )
        self._save_btn.clicked.connect(self._on_save)
        self._save_btn.setVisible(False)
        btn_layout.addWidget(self._save_btn)

        cancel_btn = QPushButton("Cancel  [Esc]")
        cancel_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        cancel_btn.setFont(QFont("", 14))
        cancel_btn.setStyleSheet(
            "QPushButton { background: #636e72; color: white; border: none; "
            "padding: 10px 36px; border-radius: 6px; }"
            "QPushButton:hover { background: #2d3436; }"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        bottom_layout.addLayout(btn_layout)
        layout.addWidget(bottom_bar)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.showFullScreen()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            if self._capture_btn.isVisible() and self._capture_btn.isEnabled():
                self._on_capture()
        elif event.key() == Qt.Key.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)

    def _get_instruction_text(self) -> str:
        if self._current_target == 0:
            return "Face the camera and turn your head toward the target dot. Press Capture."
        return "Turn your head toward the target dot, then press Capture."

    def _update_frame(self) -> None:
        frame = self._camera_manager.get_frame(self._camera_index)
        if frame is None:
            return

        if self._is_capturing and self._current_target < len(CALIBRATION_POINTS):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._session.capture_sample(rgb, self._current_target)
            count = self._session.get_capture_count(self._current_target)
            self._progress_bar.setValue(count)
            if self._session.has_enough_samples(self._current_target):
                self._is_capturing = False
                self._advance_target()

        self._draw_fullscreen_ui(frame)

    def _draw_fullscreen_ui(self, frame: np.ndarray) -> None:
        canvas_w = self._canvas.width()
        canvas_h = self._canvas.height()
        if canvas_w < 10 or canvas_h < 10:
            return

        pixmap = QPixmap(canvas_w, canvas_h)
        pixmap.fill(QColor("#1e272e"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw target dot on screen
        if self._current_target < len(CALIBRATION_POINTS):
            tx, ty = CALIBRATION_POINTS[self._current_target]
            target_x = int(tx * canvas_w)
            target_y = int(ty * canvas_h)

            # Outer glow
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(255, 107, 107, 60)))
            painter.drawEllipse(target_x - 40, target_y - 40, 80, 80)

            # Outer ring
            pen = QPen(QColor(255, 107, 107), 3)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(target_x - 25, target_y - 25, 50, 50)

            # Inner dot
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(255, 71, 87)))
            painter.drawEllipse(target_x - 8, target_y - 8, 16, 16)

            # Crosshair lines
            pen = QPen(QColor(255, 107, 107, 120), 1)
            painter.setPen(pen)
            painter.drawLine(target_x - 50, target_y, target_x - 30, target_y)
            painter.drawLine(target_x + 30, target_y, target_x + 50, target_y)
            painter.drawLine(target_x, target_y - 50, target_x, target_y - 30)
            painter.drawLine(target_x, target_y + 30, target_x, target_y + 50)

        # Draw camera preview in bottom-right, except move to bottom-left for the last target (bottom-right)
        preview_w = min(200, canvas_w // 5)
        preview_h = int(preview_w * frame.shape[0] / frame.shape[1])
        is_bottom_right_target = (
            self._current_target < len(CALIBRATION_POINTS)
            and CALIBRATION_POINTS[self._current_target][0] > 0.7
            and CALIBRATION_POINTS[self._current_target][1] > 0.7
        )
        if is_bottom_right_target:
            preview_x = 16
        else:
            preview_x = canvas_w - preview_w - 16
        preview_y = canvas_h - preview_h - 16

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        preview_pix = QPixmap.fromImage(q_img).scaled(
            preview_w, preview_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Preview border
        painter.setPen(QPen(QColor(255, 255, 255, 100), 2))
        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
        painter.drawRoundedRect(
            preview_x - 4, preview_y - 4,
            preview_pix.width() + 8, preview_pix.height() + 8,
            8, 8,
        )
        painter.drawPixmap(preview_x, preview_y, preview_pix)

        painter.end()
        self._canvas.setPixmap(pixmap)

    def _on_capture(self) -> None:
        if self._current_target < len(CALIBRATION_POINTS):
            self._is_capturing = True
            self._capture_btn.setEnabled(False)
            self._progress_bar.setValue(0)

    def _advance_target(self) -> None:
        self._current_target += 1
        if self._current_target < len(CALIBRATION_POINTS):
            desc = TARGET_DESCRIPTIONS[self._current_target] if self._current_target < len(TARGET_DESCRIPTIONS) else ""
            self._progress_label.setText(
                f"Target {self._current_target + 1} of {len(CALIBRATION_POINTS)}  —  {desc}"
            )
            self._instruction_label.setText(self._get_instruction_text())
            self._capture_btn.setEnabled(True)
            self._progress_bar.setValue(0)
        else:
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        self._capture_btn.setVisible(False)
        self._instruction_label.setText("Computing calibration...")
        self._progress_label.setText("")
        self._result = self._session.compute_calibration()
        if self._result:
            quality = self._result["quality_label"]
            score = self._result["quality_score"]
            self._instruction_label.setText(
                f"Calibration complete!  Quality: {quality} ({score:.0%})"
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
        self._progress_label.setText(f"Target 1 of {len(CALIBRATION_POINTS)}  —  {TARGET_DESCRIPTIONS[0]}")
        self._progress_bar.setValue(0)
        self._instruction_label.setText(self._get_instruction_text())

    def _on_save(self) -> None:
        self.accept()

    def get_result(self) -> Optional[dict]:
        return self._result

    def _cleanup(self) -> None:
        self._timer.stop()
        self._camera_manager.release_camera(self._camera_index)
        self._session.release()

    def closeEvent(self, event) -> None:
        self._cleanup()
        event.accept()

    def reject(self) -> None:
        self._cleanup()
        super().reject()
