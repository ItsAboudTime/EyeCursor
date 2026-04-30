from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, Signal

from src.ui.components.calibration_score import CalibrationScoreBadge


class DashboardPage(QWidget):
    start_requested = Signal()
    pause_requested = Signal()
    stop_requested = Signal()
    visualize_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Dashboard")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        info_group = QGroupBox("Current Session")
        info_group.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 14px; border: 1px solid #dcdde1; "
            "border-radius: 8px; margin-top: 8px; padding-top: 16px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }"
        )
        info_layout = QGridLayout(info_group)
        info_layout.setSpacing(8)

        info_layout.addWidget(QLabel("Profile:"), 0, 0)
        self._profile_label = QLabel("None")
        self._profile_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self._profile_label, 0, 1)

        info_layout.addWidget(QLabel("Mode:"), 1, 0)
        self._mode_label = QLabel("None")
        self._mode_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self._mode_label, 1, 1)

        info_layout.addWidget(QLabel("Cameras:"), 2, 0)
        self._cameras_label = QLabel("Not configured")
        info_layout.addWidget(self._cameras_label, 2, 1)

        layout.addWidget(info_group)

        calib_group = QGroupBox("Calibration Status")
        calib_group.setStyleSheet(info_group.styleSheet())
        self._calib_layout = QGridLayout(calib_group)
        self._calib_layout.setSpacing(8)

        self._calib_badges = {}
        calib_items = [
            ("head_pose", "Head Pose"),
            ("eye_gestures", "Facial Gestures"),
            ("stereo", "Stereo"),
            ("eye_gaze", "Eye Gaze"),
        ]
        for i, (key, label) in enumerate(calib_items):
            self._calib_layout.addWidget(QLabel(f"{label}:"), i, 0)
            badge = CalibrationScoreBadge("Not Calibrated")
            self._calib_badges[key] = badge
            self._calib_layout.addWidget(badge, i, 1)

        layout.addWidget(calib_group)

        self._status_label = QLabel("Ready")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #636e72; padding: 12px;"
        )
        layout.addWidget(self._status_label)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self._start_btn = QPushButton("Start Tracking")
        self._start_btn.setMinimumHeight(48)
        self._start_btn.setStyleSheet(
            "QPushButton { background: #00b894; color: white; border: none; "
            "border-radius: 8px; font-size: 16px; font-weight: bold; padding: 12px 32px; }"
            "QPushButton:hover { background: #00a381; }"
            "QPushButton:disabled { background: #b2bec3; }"
        )
        self._start_btn.clicked.connect(self.start_requested.emit)
        btn_layout.addWidget(self._start_btn)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setMinimumHeight(48)
        self._pause_btn.setVisible(False)
        self._pause_btn.setStyleSheet(
            "QPushButton { background: #fdcb6e; color: #2d3436; border: none; "
            "border-radius: 8px; font-size: 14px; font-weight: bold; padding: 12px 24px; }"
            "QPushButton:hover { background: #feca57; }"
        )
        self._pause_btn.clicked.connect(self.pause_requested.emit)

        self._visualize_btn = QPushButton("Visualize")
        self._visualize_btn.setMinimumHeight(48)
        self._visualize_btn.clicked.connect(self.visualize_requested.emit)
        self._visualize_btn.setStyleSheet(
            "QPushButton { background: #0984e3; color: white; border: none; "
            "border-radius: 8px; font-size: 14px; font-weight: bold; padding: 12px 24px; }"
            "QPushButton:hover { background: #0476cc; }"
            "QPushButton:disabled { background: #b2bec3; color: #f5f6fa; }"
        )
        self._visualize_btn.setEnabled(False)
        self._visualize_btn.setToolTip("Start tracking to enable visualization.")
        self._visualize_btn.setVisible(False)

        btn_layout.addWidget(self._visualize_btn)
        btn_layout.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setMinimumHeight(48)
        self._stop_btn.setVisible(False)
        self._stop_btn.setStyleSheet(
            "QPushButton { background: #d63031; color: white; border: none; "
            "border-radius: 8px; font-size: 14px; font-weight: bold; padding: 12px 24px; }"
            "QPushButton:hover { background: #e17055; }"
        )
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        btn_layout.addWidget(self._stop_btn)

        layout.addLayout(btn_layout)

        layout.addStretch()

    def set_profile_name(self, name: str) -> None:
        self._profile_label.setText(name)

    def set_mode_name(self, name: str) -> None:
        self._mode_label.setText(name)

    def set_cameras_info(self, info: str) -> None:
        self._cameras_label.setText(info)

    def update_calibration_status(self, statuses: dict) -> None:
        # Map each badge to the list of mode keys that contribute to it.
        # A badge is considered calibrated ("Good") if ANY of its contributing
        # modes is calibrated. This generalises the many-to-one case
        # (head_pose has two modes) so future additions don't reintroduce the
        # overwrite bug from a flat mode_key -> badge_key mapping.
        badge_to_modes = {
            "head_pose": ["one_camera_head_pose", "two_camera_head_pose"],
            "eye_gestures": ["eye_gestures"],
            "stereo": ["stereo"],
            "eye_gaze": ["eye_gaze"],
        }
        for badge_key, mode_ids in badge_to_modes.items():
            if badge_key not in self._calib_badges:
                continue
            present = [m for m in mode_ids if m in statuses]
            if not present:
                continue
            calibrated = any(statuses[m] for m in present)
            label = "Good" if calibrated else "Not Calibrated"
            self._calib_badges[badge_key].set_label(label)

    def set_tracking_state(self, state: str) -> None:
        if state == "active":
            self._status_label.setText("Tracking Active")
            self._status_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #00b894; padding: 12px;"
            )
            self._start_btn.setVisible(False)
            self._pause_btn.setVisible(True)
            self._pause_btn.setText("Pause")
            self._stop_btn.setVisible(True)
            self._visualize_btn.setVisible(True)
            self._visualize_btn.setEnabled(True)
            self._visualize_btn.setToolTip("Open the live visualizer for this mode.")
        elif state == "paused":
            self._status_label.setText("Tracking Paused")
            self._status_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #fdcb6e; padding: 12px;"
            )
            self._pause_btn.setText("Resume")
        elif state == "stopped":
            self._status_label.setText("Ready")
            self._status_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #636e72; padding: 12px;"
            )
            self._start_btn.setVisible(True)
            self._pause_btn.setVisible(False)
            self._stop_btn.setVisible(False)
            self._visualize_btn.setEnabled(False)
            self._visualize_btn.setToolTip("Start tracking to enable visualization.")
            self._visualize_btn.setVisible(False)

    def set_visualize_button_visible(self, visible: bool) -> None:
        self._visualize_btn.setVisible(visible)
