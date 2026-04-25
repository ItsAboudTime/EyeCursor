from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from criteria.core.storage import APP_DATA_DIR
from criteria.ui.components.cards import card


class SettingsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        title = QLabel("Settings")
        title.setObjectName("Title")
        layout.addWidget(title)
        frame, frame_layout = card()
        text = QLabel(
            "MVP task defaults: Movement 15 trials, Accuracy 12 trials, "
            "Tracking 30 seconds at 30 Hz, Clicking 12 trials.\n\n"
            f"Data folder:\n{APP_DATA_DIR}"
        )
        text.setWordWrap(True)
        frame_layout.addWidget(text)
        layout.addWidget(frame)
        layout.addStretch(1)

