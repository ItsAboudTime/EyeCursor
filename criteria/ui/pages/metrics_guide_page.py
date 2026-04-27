from __future__ import annotations

from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from criteria.ui.components.cards import card


_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Task Scores (0 – 100)",
        [
            (
                "Final Score",
                "Weighted average of four task scores: Movement 30%, Accuracy 30%, "
                "Tracking 25%, Clicking 15%.",
            ),
            (
                "Quality Labels",
                "Excellent (≥90) | Good (≥75) | Acceptable (≥60) | "
                "Poor (≥40) | Failed (<40).",
            ),
        ],
    ),
    (
        "Movement Metrics",
        [
            (
                "Throughput (bits/s)",
                "Pointing efficiency measured via Fitts’ Law (Shannon formulation). "
                "Higher is better. Mouse baseline ≈ 4.7 bits/s, eye+spacebar ≈ 3.8 bits/s, "
                "head tracker ≈ 1.5–2.5 bits/s. Nominal throughput uses the target width; "
                "effective throughput uses actual endpoint spread (W_e = 4.133 × SD), "
                "the ISO 9241-9 gold standard.",
            ),
            (
                "Index of Difficulty (ID)",
                "log₂(D/W + 1) — quantifies how hard each target is to reach based "
                "on distance (D) and target width (W).",
            ),
            (
                "Movement Time",
                "Time in milliseconds to acquire each target. Reported per target size "
                "(large / medium / small).",
            ),
            (
                "Fitts’ Regression",
                "Linear fit of Movement Time vs Index of Difficulty. The slope represents "
                "the inverse of throughput; R² shows how well Fitts’ Law models the data.",
            ),
        ],
    ),
    (
        "Accuracy Metrics",
        [
            (
                "Pixel Error",
                "Euclidean distance from cursor to target centre at trial end. Lower is better.",
            ),
            (
                "Spatial Precision",
                "Standard deviation of cursor endpoint positions. Low → consistent, "
                "high → jittery. Reported in X, Y, and combined 2D.",
            ),
            (
                "Endpoint Bias",
                "Mean systematic offset of the cursor from target centres. Reveals if the "
                "cursor consistently drifts in one direction.",
            ),
            (
                "RMS Error",
                "Root-mean-square pixel error. Penalises large deviations more heavily "
                "than the simple mean.",
            ),
        ],
    ),
    (
        "Tracking Metrics",
        [
            (
                "% Time on Target",
                "Fraction of samples where the cursor was inside the target circle. "
                "Higher is better.",
            ),
            (
                "Path Efficiency",
                "Ratio of target path length to cursor path length. A value of 1.0 means "
                "the cursor followed the target perfectly with no overshoot.",
            ),
            (
                "Mean Cursor Speed",
                "Average cursor movement speed in pixels per second over the 30-second "
                "tracking window.",
            ),
            (
                "Direction Change Rate",
                "Fraction of consecutive velocity samples that reversed direction. "
                "High values indicate jittery, unstable movement.",
            ),
        ],
    ),
    (
        "Clicking Metrics",
        [
            (
                "Click Scatter",
                "Standard deviation of click positions relative to the target centre. "
                "Shows how spread out clicks are.",
            ),
            (
                "Click Bias",
                "Mean offset of click positions from the target centre. Reveals if clicks "
                "systematically land to one side.",
            ),
            (
                "Reaction Time",
                "Median and standard deviation of time-to-click in milliseconds. "
                "Lower median → faster response.",
            ),
        ],
    ),
    (
        "Comparison Baselines (Published Literature)",
        [
            (
                "Standard Mouse",
                "≈ 4.7 bits/s throughput (Douglas et al., 1999).",
            ),
            (
                "Tobii Eye Tracker 5",
                "0.4–0.5° accuracy, >90% trackability (infrared-based).",
            ),
            (
                "Eye + Spacebar",
                "≈ 3.78 bits/s throughput (Zhang & MacKenzie, 2007).",
            ),
            (
                "eViacam (Head Tracker)",
                "NASA-TLX workload score 42.1, user satisfaction 3.4/5.",
            ),
            (
                "OptiKey (Eye Keyboard)",
                "15.8–21.7 characters/min, SUS usability score 90.6.",
            ),
        ],
    ),
]


class MetricsGuidePage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 24)

        title = QLabel("Metrics Guide")
        title.setObjectName("Title")
        outer.addWidget(title)

        subtitle = QLabel("Understanding EyeCursor TestLab measurements")
        subtitle.setObjectName("Subtitle")
        outer.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        self.layout_ = QVBoxLayout(container)
        self.layout_.setSpacing(14)
        scroll.setWidget(container)
        outer.addWidget(scroll)

        for section_title, items in _SECTIONS:
            frame, frame_layout = card()
            heading = QLabel(section_title)
            heading.setStyleSheet("font-size: 17px; font-weight: 700;")
            frame_layout.addWidget(heading)
            for name, description in items:
                name_label = QLabel(f"<b>{name}</b>")
                name_label.setStyleSheet("font-size: 14px; margin-top: 4px;")
                desc_label = QLabel(description)
                desc_label.setWordWrap(True)
                desc_label.setStyleSheet("color: #636e72; font-size: 13px;")
                frame_layout.addWidget(name_label)
                frame_layout.addWidget(desc_label)
            self.layout_.addWidget(frame)

        self.layout_.addStretch(1)
