from typing import Optional

import cv2
import numpy as np
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

from src.core.devices.camera_manager import CameraManager


class CameraPreview(QLabel):
    def __init__(
        self,
        camera_manager: CameraManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._camera_manager = camera_manager
        self._camera_index: Optional[int] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background: #2d3436; border-radius: 4px;")
        self.setText("No Preview")

    def start_preview(self, camera_index: int) -> None:
        self.stop_preview()
        try:
            self._camera_manager.open_camera(camera_index)
            self._camera_index = camera_index
            self._timer.start(33)
        except RuntimeError:
            self.setText("Camera unavailable")

    def stop_preview(self) -> None:
        self._timer.stop()
        if self._camera_index is not None:
            self._camera_manager.release_camera(self._camera_index)
            self._camera_index = None
        self.clear()
        self.setText("No Preview")

    def _update_frame(self) -> None:
        if self._camera_index is None:
            return
        frame = self._camera_manager.get_frame(self._camera_index)
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled = QPixmap.fromImage(image).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)
