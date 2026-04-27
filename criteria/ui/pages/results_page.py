from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from typing import Any

from criteria.core.advanced_metrics import compute_advanced_metrics
from criteria.core.models import Session
from criteria.ui.components.cards import card


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _format_advanced(adv: dict[str, dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    m = adv.get("movement", {})
    if m.get("nominal_throughput_bps") is not None:
        tp_str = f"Throughput {_fmt(m['nominal_throughput_bps'])} bits/s"
        if m.get("effective_throughput_bps") is not None:
            tp_str += f"  (effective: {_fmt(m['effective_throughput_bps'])} bits/s)"
        lines.append(f"Movement:  {tp_str}  |  Mean ID {_fmt(m['mean_index_of_difficulty'])}")
    a = adv.get("accuracy", {})
    if a.get("precision_2d_px") is not None:
        lines.append(
            f"Accuracy:  Precision {_fmt(a['precision_2d_px'])} px  |  "
            f"Bias {_fmt(a['bias_magnitude_px'])} px  |  "
            f"RMS Error {_fmt(a['rms_pixel_error'])} px"
        )
    t = adv.get("tracking", {})
    if t.get("pct_time_on_target") is not None:
        pct = t["pct_time_on_target"] * 100
        lines.append(
            f"Tracking:  On Target {pct:.1f}%  |  "
            f"Path Efficiency {_fmt(t['path_efficiency'])}  |  "
            f"Speed {_fmt(t['mean_cursor_speed_px_per_s'])} px/s"
        )
    c = adv.get("clicking", {})
    if c.get("click_scatter_2d_px") is not None:
        lines.append(
            f"Clicking:  Scatter {_fmt(c['click_scatter_2d_px'])} px  |  "
            f"Median RT {_fmt(c['median_time_to_click_ms'])} ms"
        )
    return lines


class ResultsPage(QWidget):
    export_json_requested = Signal(str)
    export_csv_requested = Signal(str)
    export_all_csv_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.sessions: list[Session] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Results")
        title.setObjectName("Title")
        outer.addWidget(title)
        export_all_btn = QPushButton("Export All Sessions as CSV")
        export_all_btn.setFixedWidth(260)
        export_all_btn.clicked.connect(self.export_all_csv_requested.emit)
        outer.addWidget(export_all_btn)
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
            adv = compute_advanced_metrics(session)
            adv_lines = _format_advanced(adv)
            if adv_lines:
                separator = QLabel("")
                separator.setFixedHeight(2)
                separator.setStyleSheet("background: #dfe6e9; margin: 4px 0;")
                frame_layout.addWidget(separator)
                adv_header = QLabel("Advanced Metrics")
                adv_header.setStyleSheet("font-size: 14px; font-weight: 600; color: #636e72;")
                frame_layout.addWidget(adv_header)
                for line in adv_lines:
                    lbl = QLabel(line)
                    lbl.setStyleSheet("color: #636e72; font-size: 13px;")
                    lbl.setWordWrap(True)
                    frame_layout.addWidget(lbl)
            json_button = QPushButton("Export JSON")
            csv_button = QPushButton("Export CSV Summary")
            csv_button.setProperty("secondary", True)
            json_button.clicked.connect(lambda _, sid=session.session_id: self.export_json_requested.emit(sid))
            csv_button.clicked.connect(lambda _, sid=session.session_id: self.export_csv_requested.emit(sid))
            frame_layout.addWidget(json_button)
            frame_layout.addWidget(csv_button)
            self.layout.addWidget(frame)
        self.layout.addStretch(1)

