from typing import Optional

from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Signal

from src.core.devices.camera_identity import (
    annotate_single_camera_calibration,
    annotate_stereo_calibration,
)
from src.core.devices.camera_manager import CameraManager
from src.core.profiles.profile_manager import ProfileManager
from src.ui.components.calibration_score import CalibrationScoreBadge


class CalibrationEntry(QFrame):
    calibrate_clicked = Signal(str)
    reset_clicked = Signal(str)

    def __init__(self, cal_id: str, title: str, description: str, parent=None) -> None:
        super().__init__(parent)
        self._cal_id = cal_id
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "CalibrationEntry { background: #ffffff; border: 1px solid #dcdde1; "
            "border-radius: 8px; padding: 12px; }"
        )

        layout = QHBoxLayout(self)

        info_layout = QVBoxLayout()
        name_label = QLabel(f"<b>{title}</b>")
        name_label.setStyleSheet("font-size: 14px; border: none;")
        info_layout.addWidget(name_label)
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #636e72; font-size: 12px; border: none;")
        desc_label.setWordWrap(True)
        info_layout.addWidget(desc_label)
        layout.addLayout(info_layout, stretch=1)

        self._badge = CalibrationScoreBadge("Not Calibrated")
        self._badge.setFixedWidth(140)
        layout.addWidget(self._badge)

        btn_style = (
            "QPushButton { background: #0984e3; color: white; border: none; "
            "padding: 6px 16px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #0652DD; }"
        )
        reset_style = (
            "QPushButton { background: #d63031; color: white; border: none; "
            "padding: 6px 16px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #e17055; }"
        )

        cal_btn = QPushButton("Calibrate")
        cal_btn.setStyleSheet(btn_style)
        cal_btn.clicked.connect(lambda: self.calibrate_clicked.emit(self._cal_id))
        layout.addWidget(cal_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setStyleSheet(reset_style)
        reset_btn.clicked.connect(lambda: self.reset_clicked.emit(self._cal_id))
        layout.addWidget(reset_btn)

        target_height = max(
            self._badge.sizeHint().height(),
            cal_btn.sizeHint().height(),
            reset_btn.sizeHint().height(),
        )
        self._badge.setFixedHeight(target_height)
        cal_btn.setFixedHeight(target_height)
        reset_btn.setFixedHeight(target_height)

    def set_status(self, is_calibrated: bool, quality_label: str = "") -> None:
        if is_calibrated:
            label = quality_label if quality_label else "Calibrated"
            # Map to valid badge labels
            if label not in ("Excellent", "Good", "Acceptable", "Poor", "Failed"):
                label = "Good"
            self._badge.set_label(label)
        else:
            self._badge.set_label("Not Calibrated")


class CalibrationPage(QWidget):
    def __init__(
        self,
        camera_manager: Optional[CameraManager] = None,
        profile_manager: Optional[ProfileManager] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._camera_manager = camera_manager
        self._profile_manager = profile_manager
        self._active_profile_id: Optional[str] = None
        self._get_camera_index = None
        self._get_stereo_cameras = None
        self._entries = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Calibration")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Calibrate tracking modes for your profile. "
            "Each calibration adapts the system to your face, eyes, and setup."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #636e72; font-size: 13px;")
        layout.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        entries_widget = QWidget()
        self._entries_layout = QVBoxLayout(entries_widget)
        self._entries_layout.setSpacing(12)

        calibrations = [
            ("one_camera_head_pose", "Head Pose Calibration",
             "Maps head movement to cursor position. Required for one-camera and two-camera modes."),
            ("eye_gestures", "Facial Gesture Calibration",
             "Calibrates smirk and cheek-puff thresholds for clicks and scrolling."),
            ("stereo", "Stereo Calibration",
             "Calibrates two cameras for stereo depth. Required for two-camera mode."),
            ("eye_gaze", "Gaze Calibration",
             "Maps gaze direction to cursor position. Required for eye-gaze mode."),
        ]

        for cal_id, title_text, desc in calibrations:
            entry = CalibrationEntry(cal_id, title_text, desc)
            entry.calibrate_clicked.connect(self._on_calibrate)
            entry.reset_clicked.connect(self._on_reset)
            self._entries_layout.addWidget(entry)
            self._entries[cal_id] = entry

        self._entries_layout.addStretch()
        scroll.setWidget(entries_widget)
        layout.addWidget(scroll)

    def set_dependencies(
        self,
        camera_manager: CameraManager,
        profile_manager: ProfileManager,
        get_camera_index=None,
        get_stereo_cameras=None,
    ) -> None:
        self._camera_manager = camera_manager
        self._profile_manager = profile_manager
        self._get_camera_index = get_camera_index
        self._get_stereo_cameras = get_stereo_cameras

    def set_active_profile(self, profile_id: str) -> None:
        self._active_profile_id = profile_id
        self.refresh_status()

    def refresh_status(self) -> None:
        if not self._profile_manager or not self._active_profile_id:
            return
        statuses = self._profile_manager.get_calibration_status(self._active_profile_id)
        for cal_id, entry in self._entries.items():
            is_cal = statuses.get(cal_id, False)
            quality_label = ""
            if is_cal:
                if cal_id == "stereo":
                    cal_data = self._profile_manager.load_stereo_calibration(
                        self._active_profile_id
                    )
                else:
                    cal_data = self._profile_manager.load_calibration(
                        self._active_profile_id, cal_id
                    )
                if cal_data:
                    quality_label = cal_data.get("quality_label", "")
            entry.set_status(is_cal, quality_label)

    def _on_calibrate(self, cal_id: str) -> None:
        if not self._camera_manager or not self._profile_manager or not self._active_profile_id:
            QMessageBox.warning(self, "Error", "No active profile selected.")
            return

        camera_index = self._get_camera_index() if self._get_camera_index else 0

        if cal_id == "one_camera_head_pose":
            self._run_head_pose_calibration(camera_index)
        elif cal_id == "eye_gestures":
            self._run_eye_gesture_calibration(camera_index)
        elif cal_id == "stereo":
            self._run_stereo_calibration()
        elif cal_id == "eye_gaze":
            self._run_gaze_calibration(camera_index)

    def _run_head_pose_calibration(self, camera_index: int) -> None:
        from src.ui.wizards.head_pose_wizard import HeadPoseCalibrationWizard

        screen = self.screen()
        screen_geo = screen.geometry() if screen else None
        screen_w = screen_geo.width() if screen_geo else 1920
        screen_h = screen_geo.height() if screen_geo else 1080

        wizard = HeadPoseCalibrationWizard(
            camera_index=camera_index,
            camera_manager=self._camera_manager,
            screen_w=screen_w,
            screen_h=screen_h,
            parent=self,
        )
        if wizard.exec() == wizard.DialogCode.Accepted:
            result = wizard.get_result()
            if result:
                annotate_single_camera_calibration(
                    result, camera_index, self._camera_manager
                )
                self._profile_manager.save_calibration(
                    self._active_profile_id, "one_camera_head_pose", result
                )
                self.refresh_status()

    def _run_eye_gesture_calibration(self, camera_index: int) -> None:
        from src.ui.wizards.facial_gesture_wizard import FacialGestureCalibrationWizard

        wizard = FacialGestureCalibrationWizard(
            camera_index=camera_index,
            camera_manager=self._camera_manager,
            parent=self,
        )
        if wizard.exec() == wizard.DialogCode.Accepted:
            result = wizard.get_result()
            if result:
                annotate_single_camera_calibration(
                    result, camera_index, self._camera_manager
                )
                self._profile_manager.save_calibration(
                    self._active_profile_id, "eye_gestures", result
                )
                self.refresh_status()

    def _run_stereo_calibration(self) -> None:
        from src.ui.wizards.stereo_calibration_wizard import StereoCalibrationWizard

        if not self._get_stereo_cameras:
            QMessageBox.warning(
                self, "Cameras Required",
                "Please select left and right cameras on the Cameras page first."
            )
            return

        left, right = self._get_stereo_cameras()
        if left is None or right is None:
            QMessageBox.warning(
                self, "Cameras Required",
                "Please select left and right cameras on the Cameras page first."
            )
            return

        wizard = StereoCalibrationWizard(
            left_camera_index=left,
            right_camera_index=right,
            camera_manager=self._camera_manager,
            parent=self,
        )
        if wizard.exec() == wizard.DialogCode.Accepted:
            result = wizard.get_result()
            if result:
                annotate_stereo_calibration(
                    result, left, right, self._camera_manager
                )
                self._profile_manager.save_stereo_calibration(
                    self._active_profile_id, result
                )
                self.refresh_status()

    def _run_gaze_calibration(self, camera_index: int) -> None:
        from src.ui.wizards.gaze_calibration_wizard import GazeCalibrationWizard

        wizard = GazeCalibrationWizard(
            camera_index=camera_index,
            camera_manager=self._camera_manager,
            parent=self,
        )
        if wizard.exec() == wizard.DialogCode.Accepted:
            result = wizard.get_result()
            if result:
                annotate_single_camera_calibration(
                    result, camera_index, self._camera_manager
                )
                self._profile_manager.save_calibration(
                    self._active_profile_id, "eye_gaze", result
                )
                self.refresh_status()

    def _on_reset(self, cal_id: str) -> None:
        if not self._profile_manager or not self._active_profile_id:
            return
        result = QMessageBox.question(
            self,
            "Reset Calibration",
            f"Reset this calibration data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if result == QMessageBox.StandardButton.Yes:
            self._profile_manager.reset_calibration(self._active_profile_id, cal_id)
            self.refresh_status()
