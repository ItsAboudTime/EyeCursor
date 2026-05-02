from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)
from PySide6.QtCore import Signal

from src.ui.components.calibration_score import CalibrationScoreBadge


class ModeCard(QFrame):
    selected = Signal(str)

    def __init__(
        self,
        mode_id: str,
        display_name: str,
        description: str,
        required_cameras: int,
        calibration_label: str = "Not Calibrated",
        is_active: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._mode_id = mode_id
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "ModeCard { background: #ffffff; border: 1px solid #dcdde1; "
            "border-radius: 8px; padding: 16px; }"
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel(f"<b>{display_name}</b>")
        title.setStyleSheet("font-size: 16px; border: none;")
        header.addWidget(title)
        header.addStretch()

        self._badge = CalibrationScoreBadge(calibration_label)
        badge_width = 0
        for label in ("Not Calibrated", "Calibrated"):
            temp_badge = CalibrationScoreBadge(label)
            badge_width = max(badge_width, temp_badge.sizeHint().width())
        header.addWidget(self._badge)
        layout.addLayout(header)

        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #636e72; font-size: 13px; border: none;")
        layout.addWidget(desc)

        info_row = QHBoxLayout()
        camera_label = QLabel(f"Cameras: {required_cameras}")
        camera_label.setStyleSheet("color: #636e72; font-size: 12px; border: none;")
        info_row.addWidget(camera_label)
        info_row.addStretch()

        self._select_btn = QPushButton("Selected" if is_active else "Select")
        self._select_btn.setEnabled(not is_active)
        self._select_btn.setStyleSheet(
            "QPushButton { background: #0984e3; color: white; border: none; "
            "padding: 6px 20px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #0652DD; }"
            "QPushButton:disabled { background: #74b9ff; }"
        )
        current_text = self._select_btn.text()
        select_width = 0
        for text in ("Selected", "Select"):
            self._select_btn.setText(text)
            select_width = max(select_width, self._select_btn.sizeHint().width())
        self._select_btn.setText(current_text)

        target_width = max(badge_width, select_width)
        if target_width:
            self._badge.setFixedWidth(target_width)
            self._select_btn.setFixedWidth(target_width)
        self._select_btn.clicked.connect(lambda: self.selected.emit(self._mode_id))
        info_row.addWidget(self._select_btn)
        layout.addLayout(info_row)

    def set_active(self, active: bool) -> None:
        self._select_btn.setText("Selected" if active else "Select")
        self._select_btn.setEnabled(not active)

    def set_calibration_label(self, label: str) -> None:
        self._badge.set_label(label)
