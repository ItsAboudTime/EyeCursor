from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from criteria.core.models import Session
from criteria.ui.components.cards import card


class ResumePage(QWidget):
    resume_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.sessions: list[Session] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        title = QLabel("Resume Session")
        title.setObjectName("Title")
        layout.addWidget(title)
        frame, frame_layout = card()
        self.list_widget = QListWidget()
        frame_layout.addWidget(self.list_widget)
        resume = QPushButton("Continue Selected Session")
        resume.clicked.connect(self._resume)
        frame_layout.addWidget(resume)
        layout.addWidget(frame)

    def set_sessions(self, sessions: list[Session]) -> None:
        self.sessions = [session for session in sessions if session.status != "complete"]
        self.list_widget.clear()
        for session in self.sessions:
            item = QListWidgetItem(
                f"{session.participant_name} | {session.input_method} | "
                f"next: {session.next_task} | seed: {session.seed}"
            )
            item.setData(32, session.session_id)
            self.list_widget.addItem(item)

    def _resume(self) -> None:
        item = self.list_widget.currentItem()
        if item:
            self.resume_selected.emit(item.data(32))

