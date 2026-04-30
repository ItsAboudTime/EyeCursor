from typing import List, Optional

from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Signal

from src.core.devices.camera_manager import CameraManager
from src.core.devices.camera_model import CameraInfo
from src.ui.components.camera_preview import CameraPreview


class CameraCard(QFrame):
    select_one_camera = Signal(int)
    select_left = Signal(int)
    select_right = Signal(int)

    def __init__(
        self,
        camera_info: CameraInfo,
        camera_manager: CameraManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._info = camera_info
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "CameraCard { background: #ffffff; border: 1px solid #dcdde1; "
            "border-radius: 8px; padding: 12px; }"
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._preview = CameraPreview(camera_manager)
        self._preview.setFixedHeight(180)
        layout.addWidget(self._preview)

        info_layout = QGridLayout()
        info_layout.addWidget(QLabel("Index:"), 0, 0)
        info_layout.addWidget(QLabel(str(camera_info.index)), 0, 1)
        info_layout.addWidget(QLabel("Resolution:"), 1, 0)
        info_layout.addWidget(
            QLabel(f"{camera_info.width}x{camera_info.height}"), 1, 1
        )
        info_layout.addWidget(QLabel("Stable ID:"), 2, 0)
        sid_label = QLabel(camera_info.stable_id or "unknown")
        sid_label.setStyleSheet("font-family: monospace; font-size: 10px; color: #636e72;")
        sid_label.setWordWrap(True)
        sid_label.setToolTip(
            "Identifier the app uses to recognise this physical camera "
            "across reboots and USB replugs. Calibrations are bound to it."
        )
        info_layout.addWidget(sid_label, 2, 1)
        info_layout.addWidget(QLabel("Label:"), 3, 0)
        self._label_edit = QLineEdit(camera_info.label)
        self._label_edit.setPlaceholderText("Camera name...")
        self._label_edit.textChanged.connect(self._on_label_changed)
        info_layout.addWidget(self._label_edit, 3, 1)
        layout.addLayout(info_layout)

        btn_layout = QHBoxLayout()
        btn_style = (
            "QPushButton { background: #0984e3; color: white; border: none; "
            "padding: 6px 14px; border-radius: 4px; font-weight: bold; font-size: 12px; }"
            "QPushButton:hover { background: #0652DD; }"
        )

        select_btn = QPushButton("Select")
        select_btn.setStyleSheet(btn_style)
        select_btn.clicked.connect(lambda: self.select_one_camera.emit(self._info.index))
        btn_layout.addWidget(select_btn)

        left_btn = QPushButton("Set Left")
        left_btn.setStyleSheet(btn_style)
        left_btn.clicked.connect(lambda: self.select_left.emit(self._info.index))
        btn_layout.addWidget(left_btn)

        right_btn = QPushButton("Set Right")
        right_btn.setStyleSheet(btn_style)
        right_btn.clicked.connect(lambda: self.select_right.emit(self._info.index))
        btn_layout.addWidget(right_btn)

        layout.addLayout(btn_layout)

    def start_preview(self) -> None:
        self._preview.start_preview(self._info.index)

    def stop_preview(self) -> None:
        self._preview.stop_preview()

    def get_label(self) -> str:
        return self._label_edit.text()

    def _on_label_changed(self, text: str) -> None:
        self._info.label = text


class CamerasPage(QWidget):
    camera_selected = Signal(str, int)

    def __init__(self, camera_manager: CameraManager, parent=None) -> None:
        super().__init__(parent)
        self._camera_manager = camera_manager
        self._cards: List[CameraCard] = []
        self._cameras: List[CameraInfo] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Camera Setup")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        top_row = QHBoxLayout()
        self._scan_btn = QPushButton("Scan Cameras")
        self._scan_btn.setStyleSheet(
            "QPushButton { background: #0984e3; color: white; border: none; "
            "padding: 10px 24px; border-radius: 6px; font-weight: bold; font-size: 14px; }"
            "QPushButton:hover { background: #0652DD; }"
        )
        self._scan_btn.clicked.connect(self._on_scan)
        top_row.addWidget(self._scan_btn)

        self._status_label = QLabel("No cameras scanned yet.")
        self._status_label.setStyleSheet("color: #636e72; font-size: 13px;")
        top_row.addWidget(self._status_label)
        top_row.addStretch()
        layout.addLayout(top_row)

        selection_group = QGroupBox("Current Selection")
        selection_group.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 14px; border: 1px solid #dcdde1; "
            "border-radius: 8px; margin-top: 8px; padding-top: 16px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }"
        )
        sel_layout = QGridLayout(selection_group)
        sel_layout.addWidget(QLabel("One-camera mode:"), 0, 0)
        self._one_cam_label = QLabel("Not selected")
        self._one_cam_label.setStyleSheet("font-weight: bold;")
        sel_layout.addWidget(self._one_cam_label, 0, 1)

        sel_layout.addWidget(QLabel("Two-camera left:"), 1, 0)
        self._left_cam_label = QLabel("Not selected")
        self._left_cam_label.setStyleSheet("font-weight: bold;")
        sel_layout.addWidget(self._left_cam_label, 1, 1)

        sel_layout.addWidget(QLabel("Two-camera right:"), 2, 0)
        self._right_cam_label = QLabel("Not selected")
        self._right_cam_label.setStyleSheet("font-weight: bold;")
        sel_layout.addWidget(self._right_cam_label, 2, 1)

        swap_btn = QPushButton("Swap Left/Right")
        swap_btn.setStyleSheet(
            "QPushButton { background: #636e72; color: white; border: none; "
            "padding: 6px 14px; border-radius: 4px; }"
            "QPushButton:hover { background: #2d3436; }"
        )
        # Stereo modes now match calibration to physical cameras by USB
        # identity, so a /dev/videoN reshuffle is handled automatically.
        # This button is still useful as a manual override for users with
        # two identical-model cameras (whose USB serials may collide).
        swap_btn.setToolTip(
            "Swap which camera is treated as left vs right. Usually not "
            "needed — the app re-detects camera identity automatically. "
            "Use this only if your stereo image looks horizontally mirrored."
        )
        swap_btn.clicked.connect(self._on_swap)
        sel_layout.addWidget(swap_btn, 3, 0, 1, 2)

        layout.addWidget(selection_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._cards_widget = QWidget()
        self._cards_layout = QHBoxLayout(self._cards_widget)
        self._cards_layout.setSpacing(12)
        self._cards_layout.addStretch()
        scroll.setWidget(self._cards_widget)
        layout.addWidget(scroll)

    def _on_scan(self) -> None:
        self._stop_all_previews()
        self._clear_cards()
        self._scan_btn.setEnabled(False)
        self._status_label.setText("Scanning...")

        self._cameras = self._camera_manager.discover_cameras()
        self._status_label.setText(f"Found {len(self._cameras)} camera(s).")
        self._scan_btn.setEnabled(True)

        for cam in self._cameras:
            card = CameraCard(cam, self._camera_manager)
            card.select_one_camera.connect(lambda idx: self._on_select("one_camera", idx))
            card.select_left.connect(lambda idx: self._on_select("two_camera_left", idx))
            card.select_right.connect(lambda idx: self._on_select("two_camera_right", idx))
            insert_pos = self._cards_layout.count() - 1
            self._cards_layout.insertWidget(insert_pos, card)
            self._cards.append(card)
            card.start_preview()

    def _on_select(self, role: str, index: int) -> None:
        self.camera_selected.emit(role, index)
        self._update_selection_labels()

    def _on_swap(self) -> None:
        self.camera_selected.emit("swap_lr", -1)
        self._update_selection_labels()

    def update_selection_labels(
        self,
        one_camera: Optional[int],
        left: Optional[int],
        right: Optional[int],
    ) -> None:
        self._one_cam_label.setText(
            f"Camera {one_camera}" if one_camera is not None else "Not selected"
        )
        self._left_cam_label.setText(
            f"Camera {left}" if left is not None else "Not selected"
        )
        self._right_cam_label.setText(
            f"Camera {right}" if right is not None else "Not selected"
        )

    def _update_selection_labels(self) -> None:
        pass

    def _stop_all_previews(self) -> None:
        for card in self._cards:
            card.stop_preview()

    def _clear_cards(self) -> None:
        self._stop_all_previews()
        for card in self._cards:
            self._cards_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

    def stop_previews(self) -> None:
        self._stop_all_previews()
