import pathlib
import shutil
from typing import List, Optional, Tuple

import cv2
import numpy as np
from platformdirs import user_data_dir
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap

from src.core.calibration.gaze_calibration import (
    CALIBRATION_POINTS,
    NUM_CAPTURE_FRAMES,
    GazeCalibrationSession,
)
from src.core.devices.camera_manager import CameraManager


REQUIRED_FILES: List[Tuple[str, str, str, str]] = [
    (
        "weights",
        "ETH-XGaze model weights",
        "epoch_24_ckpt.pth.tar",
        "Model files (*.pth *.pth.tar *.pt);;All files (*)",
    ),
    (
        "predictor",
        "Dlib face landmark predictor",
        "shape_predictor_68_face_landmarks.dat",
        "Predictor files (*.dat);;All files (*)",
    ),
    (
        "face_model",
        "ETH-XGaze face model",
        "face_model.txt",
        "Text files (*.txt);;All files (*)",
    ),
]


class GazeCalibrationWizard(QDialog):
    PHASE_SETUP = 0
    PHASE_CALIBRATE = 1

    def __init__(
        self,
        camera_index: int,
        camera_manager: CameraManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gaze Calibration")
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint
        )
        self.setStyleSheet("background: #1e272e;")

        self._camera_index = camera_index
        self._camera_manager = camera_manager
        self._session = GazeCalibrationSession()
        self._inference = None
        self._current_target = 0
        self._is_capturing = False
        self._result: Optional[dict] = None
        self._weights_path: Optional[str] = None
        self._predictor_path: Optional[str] = None
        self._face_model_path: Optional[str] = None
        self._phase = self.PHASE_SETUP

        self._setup_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)

    def _setup_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_setup_page())
        self._stack.addWidget(self._build_calibration_page())
        self._stack.setCurrentIndex(self.PHASE_SETUP)
        root_layout.addWidget(self._stack, stretch=1)

        bottom_bar = QWidget()
        bottom_bar.setStyleSheet("background: rgba(0, 0, 0, 180);")
        bottom_bar.setFixedHeight(160)
        bottom_layout = QVBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(24, 10, 24, 10)
        bottom_layout.setSpacing(8)

        self._instruction_label = QLabel("Set up the 3 model files to begin gaze calibration.")
        self._instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._instruction_label.setWordWrap(True)
        self._instruction_label.setFont(QFont("", 20, QFont.Weight.Bold))
        self._instruction_label.setStyleSheet("color: white; padding: 4px;")
        bottom_layout.addWidget(self._instruction_label)

        self._progress_label = QLabel("")
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
        self._progress_bar.setVisible(False)
        bottom_layout.addWidget(self._progress_bar)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self._setup_btn = QPushButton("Set Up Model Files...")
        self._setup_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._setup_btn.setFont(QFont("", 14, QFont.Weight.Bold))
        self._setup_btn.setStyleSheet(
            "QPushButton { background: #6c5ce7; color: white; border: none; "
            "padding: 10px 36px; border-radius: 6px; }"
            "QPushButton:hover { background: #5f3dc4; }"
        )
        self._setup_btn.clicked.connect(self._on_setup_files)
        btn_layout.addWidget(self._setup_btn)

        self._capture_btn = QPushButton("Capture  [Space]")
        self._capture_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._capture_btn.setFont(QFont("", 14, QFont.Weight.Bold))
        self._capture_btn.setStyleSheet(
            "QPushButton { background: #00b894; color: white; border: none; "
            "padding: 10px 36px; border-radius: 6px; }"
            "QPushButton:hover { background: #00a381; }"
            "QPushButton:disabled { background: #636e72; color: #b2bec3; }"
        )
        self._capture_btn.setVisible(False)
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
        root_layout.addWidget(bottom_bar)

    def _build_setup_page(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background: #1e272e;")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(40, 40, 40, 40)
        outer.addStretch(1)

        center_row = QHBoxLayout()
        center_row.addStretch(1)

        card = QWidget()
        card.setStyleSheet(
            "background: #2d3436; border-radius: 12px;"
        )
        card.setFixedWidth(720)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(16)

        title = QLabel("Gaze Calibration Setup")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        card_layout.addWidget(title)

        subtitle = QLabel(
            "Before calibration starts, the wizard needs the 3 model files. "
            "Click the button below and select each file in order."
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setFont(QFont("", 13))
        subtitle.setStyleSheet("color: #dfe6e9;")
        card_layout.addWidget(subtitle)

        list_label = QLabel(
            "Files you'll select (in order):\n\n"
            f"  1.  {REQUIRED_FILES[0][1]}\n"
            f"        e.g.  {REQUIRED_FILES[0][2]}\n\n"
            f"  2.  {REQUIRED_FILES[1][1]}\n"
            f"        e.g.  {REQUIRED_FILES[1][2]}\n\n"
            f"  3.  {REQUIRED_FILES[2][1]}\n"
            f"        e.g.  {REQUIRED_FILES[2][2]}"
        )
        list_label.setFont(QFont("", 13))
        list_label.setStyleSheet(
            "color: #b2bec3; padding: 16px; background: #1e272e; border-radius: 8px;"
        )
        card_layout.addWidget(list_label)

        footer = QLabel(
            "Selected files are copied to the EyeCursor data directory and reused on every launch."
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setWordWrap(True)
        footer.setFont(QFont("", 11))
        footer.setStyleSheet("color: #95a5a6;")
        card_layout.addWidget(footer)

        center_row.addWidget(card)
        center_row.addStretch(1)
        outer.addLayout(center_row)
        outer.addStretch(1)
        return page

    def _build_calibration_page(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background: #1e272e;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setStyleSheet("background: #1e272e;")
        layout.addWidget(self._canvas, stretch=1)
        return page

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

    def _models_dir(self) -> pathlib.Path:
        models = pathlib.Path(user_data_dir("EyeCursor", "EyeCursorTeam")) / "models"
        models.mkdir(parents=True, exist_ok=True)
        return models

    def _on_setup_files(self) -> None:
        downloads_dir = pathlib.Path.home() / "Downloads"
        last_dir = str(downloads_dir) if downloads_dir.is_dir() else str(pathlib.Path.home())

        selected: List[Tuple[str, str, str]] = []
        for idx, (key, label, example, file_filter) in enumerate(REQUIRED_FILES, start=1):
            caption = f"Step {idx} of {len(REQUIRED_FILES)}: select {label}  (e.g. {example})"
            path, _ = QFileDialog.getOpenFileName(
                self, caption, last_dir, file_filter
            )
            if not path:
                self._instruction_label.setText(
                    f"Setup cancelled at step {idx}. Click 'Set Up Model Files' to try again."
                )
                return
            selected.append((key, path, example))
            last_dir = str(pathlib.Path(path).parent)

        try:
            models_dir = self._models_dir()
            resolved: dict = {}
            for key, src, canonical_name in selected:
                src_path = pathlib.Path(src)
                dest_path = models_dir / canonical_name
                if src_path.resolve() != dest_path.resolve():
                    shutil.copy2(src_path, dest_path)
                resolved[key] = str(dest_path)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Copy Error",
                f"Could not copy model files to {models_dir}:\n{e}",
            )
            return

        self._weights_path = resolved["weights"]
        self._predictor_path = resolved["predictor"]
        self._face_model_path = resolved["face_model"]

        self._instruction_label.setText("Loading model... (this may take a moment)")
        self._progress_label.setText("")
        self._setup_btn.setEnabled(False)
        QTimer.singleShot(100, self._load_model)

    def _load_model(self) -> None:
        try:
            from src.eye_tracking.pipelines.eth_xgaze_inference import ETHXGazeInference
            self._inference = ETHXGazeInference(
                weights=pathlib.Path(self._weights_path),
                predictor_path=pathlib.Path(self._predictor_path),
                face_model_path=pathlib.Path(self._face_model_path),
            )
            self._camera_manager.open_camera(self._camera_index)
            self._timer.start(33)
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", str(e))
            self._setup_btn.setEnabled(True)
            self._instruction_label.setText(
                "Failed to load model. Click 'Set Up Model Files' to try again."
            )
            return

        self._enter_calibration_phase()

    def _enter_calibration_phase(self) -> None:
        self._phase = self.PHASE_CALIBRATE
        self._stack.setCurrentIndex(self.PHASE_CALIBRATE)
        self._setup_btn.setVisible(False)
        self._capture_btn.setVisible(True)
        self._capture_btn.setEnabled(True)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._instruction_label.setText(self._get_instruction_text())
        self._progress_label.setText(
            f"Target 1 of {len(CALIBRATION_POINTS)}"
        )

    def _get_instruction_text(self) -> str:
        return "Look at the target dot (without moving your head), then press Capture."

    def _update_frame(self) -> None:
        frame = self._camera_manager.get_frame(self._camera_index)
        if frame is None:
            return

        if (
            self._phase == self.PHASE_CALIBRATE
            and self._is_capturing
            and self._inference is not None
            and self._current_target < len(CALIBRATION_POINTS)
        ):
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

        if self._phase == self.PHASE_CALIBRATE:
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

        if self._current_target < len(CALIBRATION_POINTS):
            tx, ty = CALIBRATION_POINTS[self._current_target]
            target_x = int(tx * canvas_w)
            target_y = int(ty * canvas_h)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(255, 107, 107, 60)))
            painter.drawEllipse(target_x - 40, target_y - 40, 80, 80)

            pen = QPen(QColor(255, 107, 107), 3)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(target_x - 25, target_y - 25, 50, 50)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(255, 71, 87)))
            painter.drawEllipse(target_x - 8, target_y - 8, 16, 16)

            pen = QPen(QColor(255, 107, 107, 120), 1)
            painter.setPen(pen)
            painter.drawLine(target_x - 50, target_y, target_x - 30, target_y)
            painter.drawLine(target_x + 30, target_y, target_x + 50, target_y)
            painter.drawLine(target_x, target_y - 50, target_x, target_y - 30)
            painter.drawLine(target_x, target_y + 30, target_x, target_y + 50)

        preview_w = min(200, canvas_w // 5)
        preview_h = int(preview_w * frame.shape[0] / frame.shape[1])
        is_bottom_right_target = (
            self._current_target < len(CALIBRATION_POINTS)
            and CALIBRATION_POINTS[self._current_target][0] > 0.7
            and CALIBRATION_POINTS[self._current_target][1] > 0.7
        )
        preview_x = 16 if is_bottom_right_target else canvas_w - preview_w - 16
        preview_y = canvas_h - preview_h - 16

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        preview_pix = QPixmap.fromImage(q_img).scaled(
            preview_w, preview_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

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
        if self._phase != self.PHASE_CALIBRATE:
            return
        if self._current_target < len(CALIBRATION_POINTS):
            self._is_capturing = True
            self._capture_btn.setEnabled(False)
            self._progress_bar.setValue(0)

    def _advance_target(self) -> None:
        self._current_target += 1
        if self._current_target < len(CALIBRATION_POINTS):
            self._progress_label.setText(
                f"Target {self._current_target + 1} of {len(CALIBRATION_POINTS)}"
            )
            self._instruction_label.setText(self._get_instruction_text())
            self._capture_btn.setEnabled(True)
            self._progress_bar.setValue(0)
        else:
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        self._capture_btn.setVisible(False)
        self._instruction_label.setText("Computing gaze calibration...")
        self._progress_label.setText("")
        self._result = self._session.compute_calibration()
        if self._result:
            if self._weights_path:
                self._result["weights_path"] = self._weights_path
            if self._predictor_path:
                self._result["predictor_path"] = self._predictor_path
            if self._face_model_path:
                self._result["face_model_path"] = self._face_model_path
            quality = self._result["quality_label"]
            score = self._result["quality_score"]
            mean_err = self._result["mean_error"]
            self._instruction_label.setText(
                f"Calibration complete!  Quality: {quality} ({score:.0%}, error: {mean_err:.4f})"
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
        self._progress_label.setText(f"Target 1 of {len(CALIBRATION_POINTS)}")
        self._progress_bar.setValue(0)
        self._instruction_label.setText(self._get_instruction_text())

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
