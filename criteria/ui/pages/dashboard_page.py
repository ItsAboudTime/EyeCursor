from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from criteria.core.models import Session
from criteria.ui.components.cards import card


class DashboardPage(QWidget):
    new_requested = Signal()
    resume_requested = Signal()
    results_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        title = QLabel("EyeCursor TestLab")
        title.setObjectName("Title")
        subtitle = QLabel("Repeatable fullscreen cursor behavior tests for movement, accuracy, tracking, and clicking.")
        subtitle.setObjectName("Subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        actions, action_layout = card()
        start = QPushButton("Start New Session")
        resume = QPushButton("Resume Session")
        results = QPushButton("View Results")
        resume.setProperty("secondary", True)
        results.setProperty("secondary", True)
        start.clicked.connect(self.new_requested.emit)
        resume.clicked.connect(self.resume_requested.emit)
        results.clicked.connect(self.results_requested.emit)
        action_layout.addWidget(start)
        action_layout.addWidget(resume)
        action_layout.addWidget(results)
        layout.addWidget(actions)

        self.recent_label = QLabel("No sessions yet.")
        self.recent_label.setWordWrap(True)
        recent, recent_layout = card()
        recent_layout.addWidget(QLabel("Recent Session"))
        recent_layout.addWidget(self.recent_label)
        layout.addWidget(recent)
        layout.addStretch(1)

    def set_recent(self, sessions: list[Session]) -> None:
        if not sessions:
            self.recent_label.setText("No sessions yet.")
            return
        session = sessions[0]
        score = session.final_summary.get("final_score", "N/A")
        self.recent_label.setText(
            f"{session.participant_name} | {session.input_method} | "
            f"{session.status} | Score: {score}"
        )

