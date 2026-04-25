from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from criteria.core.models import Session
from criteria.ui.components.cards import card


class ResultsPage(QWidget):
    export_json_requested = Signal(str)
    export_csv_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.sessions: list[Session] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Results")
        title.setObjectName("Title")
        outer.addWidget(title)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setSpacing(14)
        self.layout.addStretch(1)
        scroll.setWidget(self.container)
        outer.addWidget(scroll)

    def set_sessions(self, sessions: list[Session]) -> None:
        self.sessions = sessions
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        if not sessions:
            self.layout.addWidget(QLabel("No sessions available."))
            self.layout.addStretch(1)
            return
        for session in sessions:
            frame, frame_layout = card()
            summary = session.final_summary or {}
            score = summary.get("final_score", "N/A")
            label = summary.get("quality_label", "N/A")
            heading = QLabel(f"{session.participant_name} | {session.input_method}")
            heading.setStyleSheet("font-size: 18px; font-weight: 700;")
            score_label = QLabel(f"Final Score: {score} / 100 | Rating: {label}")
            details = QLabel(
                f"Seed: {session.seed} | Screen: {session.screen_width}x{session.screen_height} | "
                f"Status: {session.status} | Started: {session.started_at}"
            )
            details.setWordWrap(True)
            frame_layout.addWidget(heading)
            frame_layout.addWidget(score_label)
            frame_layout.addWidget(details)
            for task_id in ("movement", "accuracy", "tracking", "clicking"):
                result = session.task_results.get(task_id)
                value = result.score if result else "N/A"
                frame_layout.addWidget(QLabel(f"{task_id.title()}: {value}"))
            json_button = QPushButton("Export JSON")
            csv_button = QPushButton("Export CSV Summary")
            csv_button.setProperty("secondary", True)
            json_button.clicked.connect(lambda _, sid=session.session_id: self.export_json_requested.emit(sid))
            csv_button.clicked.connect(lambda _, sid=session.session_id: self.export_csv_requested.emit(sid))
            frame_layout.addWidget(json_button)
            frame_layout.addWidget(csv_button)
            self.layout.addWidget(frame)
        self.layout.addStretch(1)

