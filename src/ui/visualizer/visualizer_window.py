from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.ui.visualizer.bubble_lock_panel import BubbleLockPanel
from src.ui.visualizer.eye_gaze_panel import EyeGazePanel
from src.ui.visualizer.one_camera_panel import OneCameraPanel
from src.ui.visualizer.two_camera_panel import TwoCameraPanel


class VisualizerWindow(QMainWindow):
    closed = Signal()

    def __init__(self, mode_id: str, mode_display_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mode_id = mode_id

        self.setWindowTitle(f"EyeCursor — {mode_display_name} visualization")
        self.setStyleSheet(
            "QMainWindow { background: #14161c; }"
            " QLabel { color: #dfe6e9; }"
        )
        self.resize(1280, 800)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel(mode_display_name)
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        header.addWidget(title)
        header.addStretch(1)
        self._fps_label = QLabel("paused")
        self._fps_label.setStyleSheet("color: #b2bec3; font-size: 12px;")
        header.addWidget(self._fps_label)
        outer.addLayout(header)

        self._stack = QStackedWidget()
        self._panel = self._build_panel(mode_id)
        self._stack.addWidget(self._panel)
        outer.addWidget(self._stack, 1)

        hint = QLabel("Press Esc or close the window to stop visualization.")
        hint.setStyleSheet("color: #636e72; font-size: 11px;")
        outer.addWidget(hint)

        self._frame_count = 0
        self._last_payload_at = None

    def _build_panel(self, mode_id: str) -> QWidget:
        if mode_id == "one_camera_head_pose":
            return OneCameraPanel()
        if mode_id == "two_camera_head_pose":
            return TwoCameraPanel()
        if mode_id == "eye_gaze":
            return EyeGazePanel(show_bubble_indicator=False)
        if mode_id == "eye_gaze_bubble":
            return EyeGazePanel(show_bubble_indicator=True)
        if mode_id == "hybrid_bubble_lock":
            return BubbleLockPanel()
        placeholder = QLabel(f"No visualizer available for mode: {mode_id}")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #b2bec3; font-size: 14px;")
        return placeholder

    @Slot(dict)
    def update_payload(self, payload: dict) -> None:
        if payload.get("mode_id") and payload["mode_id"] != self._mode_id:
            return
        if hasattr(self._panel, "update_payload"):
            self._panel.update_payload(payload)
        self._frame_count += 1
        suffix = " · IDLE" if payload.get("idle") else ""
        self._fps_label.setText(f"frames: {self._frame_count}{suffix}")

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Escape,):
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)
