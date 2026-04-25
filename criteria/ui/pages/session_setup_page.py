from __future__ import annotations

import random

from PySide6.QtCore import Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from criteria.core.models import Session, TaskConfig
from criteria.ui.components.cards import card


class SessionSetupPage(QWidget):
    session_created = Signal(Session)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        title = QLabel("New Session")
        title.setObjectName("Title")
        layout.addWidget(title)

        frame, form_layout = card()
        form = QFormLayout()
        form.setSpacing(12)
        self.participant = QLineEdit()
        self.participant.setPlaceholderText("Participant name or ID")
        self.input_method = QComboBox()
        self.input_method.addItems(
            ["Mouse", "One-Camera Head Pose", "Two-Camera Head Pose", "Eye-Gaze Only", "Custom"]
        )
        self.custom_method = QLineEdit()
        self.custom_method.setPlaceholderText("Custom input method label")
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(random.randint(1, 999_999))
        self.notes = QTextEdit()
        self.notes.setMaximumHeight(90)
        screen = QGuiApplication.primaryScreen().geometry()
        self.screen_label = QLabel(f"{screen.width()} x {screen.height()} px")
        self.preset_label = QLabel("MVP Default")
        form.addRow("Participant", self.participant)
        form.addRow("Input Method", self.input_method)
        form.addRow("Custom Label", self.custom_method)
        form.addRow("Seed", self.seed)
        form.addRow("Screen Resolution", self.screen_label)
        form.addRow("Task Preset", self.preset_label)
        form.addRow("Notes", self.notes)
        form_layout.addLayout(form)
        layout.addWidget(frame)

        start = QPushButton("Start Fullscreen Test")
        start.clicked.connect(self._create_session)
        layout.addWidget(start)
        layout.addStretch(1)

    def _create_session(self) -> None:
        screen = QGuiApplication.primaryScreen().geometry()
        method = self.input_method.currentText()
        if method == "Custom":
            method = self.custom_method.text().strip() or "Custom"
        session = Session.create(
            participant_name=self.participant.text(),
            input_method=method,
            seed=self.seed.value(),
            screen_width=screen.width(),
            screen_height=screen.height(),
            notes=self.notes.toPlainText(),
            task_config=TaskConfig(),
        )
        self.session_created.emit(session)

