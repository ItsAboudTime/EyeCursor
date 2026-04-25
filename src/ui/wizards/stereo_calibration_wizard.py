from typing import Optional

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot, QObject
from PySide6.QtGui import QImage, QPixmap

from src.core.calibration.stereo_calibration import StereoCalibrationSession
from src.core.devices.camera_manager import CameraManager


MIN_PAIRS = 15


class CalibrationComputer(QObject):
    finished = Signal(object)

    def __init__(self, session: StereoCalibrationSession) -> None:
        super().__init__()
        self._session = session

    @Slot()
    def run(self) -> None:
        result = self._session.compute_calibration()
        self.finished.emit(result)


class StereoCalibrationWizard(QDialog):
    def __init__(
        self,
        left_camera_index: int,
        right_camera_index: int,
        camera_manager: CameraManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stereo Calibration")
        self.setMinimumSize(900, 650)

        self._left_index = left_camera_index
        self._right_index = right_camera_index
        self._camera_manager = camera_manager
        self._session: Optional[StereoCalibrationSession] = None
        self._result: Optional[dict] = None
        self._compute_thread: Optional[QThread] = None

        self._setup_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frames)
        try:
            self._camera_manager.open_camera(self._left_index)
            self._camera_manager.open_camera(self._right_index)
        except RuntimeError as e:
            self._instruction_label.setText(f"Camera error: {e}")
            return

        self._create_session()
        self._timer.start(33)

    def _create_session(self) -> None:
        self._session = StereoCalibrationSession(
            left_camera_id=self._left_index,
            right_camera_id=self._right_index,
            board_rows=self._rows_spin.value(),
            board_cols=self._cols_spin.value(),
            square_size=self._square_spin.value() / 1000.0,
            min_pairs=MIN_PAIRS,
        )

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._instruction_label = QLabel(
            "Hold a checkerboard in view of both cameras. Press Capture when corners are detected."
        )
        self._instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._instruction_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 6px;")
        self._instruction_label.setWordWrap(True)
        layout.addWidget(self._instruction_label)

        preview_layout = QHBoxLayout()
        self._left_preview = QLabel("Left Camera")
        self._left_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._left_preview.setMinimumSize(400, 300)
        self._left_preview.setStyleSheet("background: #2d3436; border-radius: 4px; color: white;")
        preview_layout.addWidget(self._left_preview)

        self._right_preview = QLabel("Right Camera")
        self._right_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._right_preview.setMinimumSize(400, 300)
        self._right_preview.setStyleSheet("background: #2d3436; border-radius: 4px; color: white;")
        preview_layout.addWidget(self._right_preview)
        layout.addLayout(preview_layout)

        config_group = QGroupBox("Checkerboard Settings")
        config_layout = QFormLayout(config_group)
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(3, 20)
        self._rows_spin.setValue(7)
        config_layout.addRow("Rows (inner corners):", self._rows_spin)
        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(3, 20)
        self._cols_spin.setValue(9)
        config_layout.addRow("Cols (inner corners):", self._cols_spin)
        self._square_spin = QDoubleSpinBox()
        self._square_spin.setRange(5.0, 100.0)
        self._square_spin.setValue(20.0)
        self._square_spin.setSuffix(" mm")
        config_layout.addRow("Square size:", self._square_spin)
        layout.addWidget(config_group)

        self._status_label = QLabel(f"Pairs captured: 0 / {MIN_PAIRS}")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, MIN_PAIRS)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        btn_layout = QHBoxLayout()
        btn_style = (
            "QPushButton { background: #00b894; color: white; border: none; "
            "padding: 10px 24px; border-radius: 6px; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background: #00a381; }"
            "QPushButton:disabled { background: #b2bec3; }"
        )

        self._capture_btn = QPushButton("Capture  [Space]")
        self._capture_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._capture_btn.setStyleSheet(btn_style)
        self._capture_btn.clicked.connect(self._on_capture)
        btn_layout.addWidget(self._capture_btn)

        self._calibrate_btn = QPushButton("Calibrate")
        self._calibrate_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._calibrate_btn.setStyleSheet(
            btn_style.replace("#00b894", "#0984e3").replace("#00a381", "#0652DD")
        )
        self._calibrate_btn.setEnabled(False)
        self._calibrate_btn.clicked.connect(self._on_calibrate)
        btn_layout.addWidget(self._calibrate_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._save_btn.setStyleSheet(
            btn_style.replace("#00b894", "#0984e3").replace("#00a381", "#0652DD")
        )
        self._save_btn.setVisible(False)
        self._save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(self._save_btn)

        self._retry_btn = QPushButton("Retry")
        self._retry_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._retry_btn.setStyleSheet(
            "QPushButton { background: #fdcb6e; color: #2d3436; border: none; "
            "padding: 10px 24px; border-radius: 6px; font-size: 14px; font-weight: bold; }"
        )
        self._retry_btn.setVisible(False)
        self._retry_btn.clicked.connect(self._on_retry)
        btn_layout.addWidget(self._retry_btn)

        cancel_btn = QPushButton("Cancel  [Esc]")
        cancel_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        cancel_btn.setStyleSheet(
            "QPushButton { background: #636e72; color: white; border: none; "
            "padding: 10px 24px; border-radius: 6px; font-size: 14px; }"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self._latest_corners_l = None
        self._latest_corners_r = None
        self._latest_found_l = False
        self._latest_found_r = False

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            if self._capture_btn.isVisible() and self._capture_btn.isEnabled():
                self._on_capture()
        elif event.key() == Qt.Key.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)

    def _update_frames(self) -> None:
        left_frame = self._camera_manager.get_frame(self._left_index)
        right_frame = self._camera_manager.get_frame(self._right_index)
        if left_frame is None or right_frame is None:
            return

        if self._session:
            fl, fr, cl, cr = self._session.detect_corners(left_frame, right_frame)
            self._latest_found_l = fl
            self._latest_found_r = fr
            self._latest_corners_l = cl
            self._latest_corners_r = cr

            if fl:
                left_frame = self._session.draw_corners(left_frame, cl, fl)
            if fr:
                right_frame = self._session.draw_corners(right_frame, cr, fr)

        self._show_frame(left_frame, self._left_preview)
        self._show_frame(right_frame, self._right_preview)

    def _show_frame(self, frame: np.ndarray, label: QLabel) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        scaled = QPixmap.fromImage(image).scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)

    def _on_capture(self) -> None:
        if not self._session:
            self._create_session()

        left_frame = self._camera_manager.get_frame(self._left_index)
        right_frame = self._camera_manager.get_frame(self._right_index)
        if left_frame is None or right_frame is None:
            return

        found_l, found_r = self._session.add_frame_pair(left_frame, right_frame)
        count = self._session.get_pair_count()
        self._status_label.setText(f"Pairs captured: {count} / {MIN_PAIRS}")
        self._progress_bar.setValue(min(count, MIN_PAIRS))
        self._calibrate_btn.setEnabled(self._session.can_calibrate())

    def _on_calibrate(self) -> None:
        if not self._session:
            return
        self._timer.stop()
        self._capture_btn.setEnabled(False)
        self._calibrate_btn.setEnabled(False)
        self._instruction_label.setText("Computing stereo calibration... (this may take a moment)")

        self._compute_thread = QThread()
        worker = CalibrationComputer(self._session)
        worker.moveToThread(self._compute_thread)
        self._compute_thread.started.connect(worker.run)
        worker.finished.connect(self._on_calibration_done)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self._compute_thread.quit)
        self._compute_thread.start()

    @Slot(object)
    def _on_calibration_done(self, result) -> None:
        self._result = result
        if result:
            quality = result["quality_label"]
            rms = result["reprojection_error"]
            baseline_cm = result["baseline_meters"] * 100
            self._instruction_label.setText(
                f"Calibration complete! Quality: {quality} "
                f"(RMS: {rms:.3f}px, baseline: {baseline_cm:.1f}cm)"
            )
            self._save_btn.setVisible(True)
            self._retry_btn.setVisible(True)
        else:
            self._instruction_label.setText("Calibration failed. Please retry with more pairs.")
            self._retry_btn.setVisible(True)

    def _on_retry(self) -> None:
        self._create_session()
        self._result = None
        self._save_btn.setVisible(False)
        self._retry_btn.setVisible(False)
        self._capture_btn.setEnabled(True)
        self._calibrate_btn.setEnabled(False)
        self._status_label.setText(f"Pairs captured: 0 / {MIN_PAIRS}")
        self._progress_bar.setValue(0)
        self._instruction_label.setText(
            "Hold a checkerboard in view of both cameras. Press Capture when corners are detected."
        )
        self._timer.start(33)

    def _on_save(self) -> None:
        self.accept()

    def get_result(self) -> Optional[dict]:
        return self._result

    def _cleanup(self) -> None:
        self._timer.stop()
        if self._compute_thread and self._compute_thread.isRunning():
            self._compute_thread.quit()
            self._compute_thread.wait(5000)
        self._camera_manager.release_camera(self._left_index)
        self._camera_manager.release_camera(self._right_index)

    def closeEvent(self, event) -> None:
        self._cleanup()
        event.accept()

    def reject(self) -> None:
        self._cleanup()
        super().reject()
