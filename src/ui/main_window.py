import pathlib
from typing import Optional

from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QWidget,
)
from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtGui import QFont, QIcon

from src.core.devices.calibration_migration import migrate_all_profiles
from src.core.devices.camera_manager import CameraManager
from src.core.modes.base import TrackingMode
from src.core.modes.registry import ModeRegistry
from src.core.profiles.profile_manager import ProfileManager
from src.core.profiles.profile_model import ProfileModel
from src.ui.pages.calibration_page import CalibrationPage
from src.ui.pages.cameras_page import CamerasPage
from src.ui.pages.dashboard_page import DashboardPage
from src.ui.pages.modes_page import ModesPage
from src.ui.pages.profiles_page import ProfilesPage
from src.ui.pages.settings_page import SettingsPage


class MainWindow(QMainWindow):
    def __init__(
        self,
        profile_manager: ProfileManager,
        mode_registry: ModeRegistry,
        camera_manager: Optional[CameraManager] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._profile_manager = profile_manager
        self._mode_registry = mode_registry
        self._camera_manager = camera_manager or CameraManager()
        self._active_profile: Optional[ProfileModel] = None
        self._tracking_thread: Optional[QThread] = None
        self._tracking_mode: Optional[TrackingMode] = None
        self._tracking_worker = None
        self._is_tracking = False
        self._is_paused = False
        self._gaze_overlay = None
        self._gaze_signal_proxy = None
        self._visualizer_window = None
        self._viz_signal_proxy = None

        self.setWindowTitle("EyeCursor")
        self.setMinimumSize(960, 640)
        self.resize(1100, 720)

        icon_path = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "icon_256.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self._setup_ui()
        self._connect_signals()
        self._load_initial_profile()
        self._refresh_all()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(200)
        self._sidebar.setStyleSheet(
            "QListWidget { background: #2d3436; border: none; color: #dfe6e9; "
            "font-size: 14px; padding-top: 12px; }"
            "QListWidget::item { padding: 14px 20px; border: none; }"
            "QListWidget::item:selected { background: #636e72; color: white; }"
            "QListWidget::item:hover { background: #4a5568; }"
        )
        sidebar_font = QFont()
        sidebar_font.setPointSize(11)
        self._sidebar.setFont(sidebar_font)

        pages = ["Dashboard", "Modes", "Cameras", "Calibration", "Profiles", "Settings"]
        for page_name in pages:
            item = QListWidgetItem(page_name)
            self._sidebar.addItem(item)

        main_layout.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        self._stack.setStyleSheet(
            "QStackedWidget { background: #f5f6fa; }"
            "QStackedWidget QLabel { color: #2d3436; }"
            "QStackedWidget QGroupBox { color: #2d3436; }"
            "QStackedWidget QGroupBox::title { color: #2d3436; }"
            "QStackedWidget QPushButton { color: #2d3436; }"
            "QStackedWidget QLineEdit { color: #2d3436; background: white; }"
            "QStackedWidget QSpinBox { color: #2d3436; background: white; }"
            "QStackedWidget QDoubleSpinBox { color: #2d3436; background: white; }"
            "QStackedWidget QListWidget { color: #2d3436; background: white; }"
            "QStackedWidget QScrollArea { background: #f5f6fa; }"
        )

        self._dashboard_page = DashboardPage()
        self._modes_page = ModesPage()
        self._cameras_page = CamerasPage(self._camera_manager)
        self._calibration_page = CalibrationPage(
            camera_manager=self._camera_manager,
            profile_manager=self._profile_manager,
        )
        self._profiles_page = ProfilesPage()
        self._settings_page = SettingsPage()

        self._stack.addWidget(self._dashboard_page)
        self._stack.addWidget(self._modes_page)
        self._stack.addWidget(self._cameras_page)
        self._stack.addWidget(self._calibration_page)
        self._stack.addWidget(self._profiles_page)
        self._stack.addWidget(self._settings_page)

        main_layout.addWidget(self._stack)
        self._sidebar.setCurrentRow(0)

    def _connect_signals(self) -> None:
        self._sidebar.currentRowChanged.connect(self._stack.setCurrentIndex)

        self._modes_page.mode_selected.connect(self._on_mode_selected)

        self._profiles_page.set_callbacks(
            on_create=self._on_create_profile,
            on_rename=self._on_rename_profile,
            on_delete=self._on_delete_profile,
            on_reset=self._on_reset_calibration,
        )
        self._profiles_page.profile_switched.connect(self._on_switch_profile)

        self._cameras_page.camera_selected.connect(self._on_camera_selected)

        self._dashboard_page.start_requested.connect(self._on_start_tracking)
        self._dashboard_page.pause_requested.connect(self._on_pause_tracking)
        self._dashboard_page.stop_requested.connect(self._on_stop_tracking)
        self._dashboard_page.visualize_requested.connect(self._on_visualize_requested)

        self._settings_page.settings_changed.connect(self._on_settings_changed)

    def _get_active_camera_index(self) -> int:
        if self._active_profile:
            cam = self._active_profile.preferred_cameras.get("one_camera")
            if cam is not None:
                return cam
        return 0

    def _get_stereo_cameras(self):
        if self._active_profile:
            cams = self._active_profile.preferred_cameras
            return cams.get("two_camera_left"), cams.get("two_camera_right")
        return None, None

    def _load_initial_profile(self) -> None:
        profiles = self._profile_manager.list_profiles()
        if profiles:
            self._active_profile = profiles[0]
        else:
            self._active_profile = self._profile_manager.create_profile("Default User")

        # Best-effort backfill of stable camera IDs into legacy calibration
        # files. No-op when the cameras can't be fingerprinted right now.
        try:
            migrate_all_profiles(self._profile_manager, self._camera_manager)
        except Exception:
            # Migration must never block startup.
            pass

        self._calibration_page.set_dependencies(
            camera_manager=self._camera_manager,
            profile_manager=self._profile_manager,
            get_camera_index=self._get_active_camera_index,
            get_stereo_cameras=self._get_stereo_cameras,
        )

    def _refresh_all(self) -> None:
        self._refresh_dashboard()
        self._refresh_modes()
        self._refresh_profiles()
        self._refresh_cameras()
        self._refresh_calibration()

    def _refresh_calibration(self) -> None:
        if self._active_profile:
            self._calibration_page.set_active_profile(self._active_profile.id)

    def _refresh_dashboard(self) -> None:
        if self._active_profile is None:
            return
        self._dashboard_page.set_profile_name(self._active_profile.display_name)

        mode_cls = None
        try:
            mode_cls = self._mode_registry.get(self._active_profile.default_mode)
        except KeyError:
            pass
        mode_name = mode_cls.display_name if mode_cls else self._active_profile.default_mode
        self._dashboard_page.set_mode_name(mode_name)

        cal_status = self._profile_manager.get_calibration_status(self._active_profile.id)
        cal_details = {}
        for mode_id, is_calibrated in cal_status.items():
            quality_label = ""
            if is_calibrated:
                if mode_id == "stereo":
                    cal_data = self._profile_manager.load_stereo_calibration(
                        self._active_profile.id
                    )
                elif mode_id == "eye_gaze_bubble":
                    cal_data = self._profile_manager.load_calibration(
                        self._active_profile.id, "eye_gaze"
                    )
                else:
                    cal_data = self._profile_manager.load_calibration(
                        self._active_profile.id, mode_id
                    )
                if cal_data:
                    quality_label = cal_data.get("quality_label", "")
            cal_details[mode_id] = {
                "is_calibrated": is_calibrated,
                "quality_label": quality_label,
            }
        self._dashboard_page.update_calibration_status(cal_details)

        cams = self._active_profile.preferred_cameras
        one_cam = cams.get("one_camera")
        cam_info = f"Camera {one_cam}" if one_cam is not None else "Not configured"
        self._dashboard_page.set_cameras_info(cam_info)

    def _refresh_modes(self) -> None:
        active_mode_id = self._active_profile.default_mode if self._active_profile else None
        cal_status = {}
        if self._active_profile:
            cal_status = self._profile_manager.get_calibration_status(self._active_profile.id)
        self._modes_page.populate_modes(
            self._mode_registry.all_modes(), active_mode_id, cal_status
        )

    def _refresh_profiles(self) -> None:
        profiles = self._profile_manager.list_profiles()
        active_id = self._active_profile.id if self._active_profile else None
        self._profiles_page.populate_profiles(profiles, active_id)

    def _refresh_cameras(self) -> None:
        if self._active_profile is None:
            return
        cams = self._active_profile.preferred_cameras
        self._cameras_page.update_selection_labels(
            one_camera=cams.get("one_camera"),
            left=cams.get("two_camera_left"),
            right=cams.get("two_camera_right"),
        )

    @Slot(str, int)
    def _on_camera_selected(self, role: str, index: int) -> None:
        if self._active_profile is None:
            return
        cams = self._active_profile.preferred_cameras
        if role == "one_camera":
            cams["one_camera"] = index
            cams["eye_gaze"] = index
        elif role == "two_camera_left":
            cams["two_camera_left"] = index
        elif role == "two_camera_right":
            cams["two_camera_right"] = index
        elif role == "swap_lr":
            cams["two_camera_left"], cams["two_camera_right"] = (
                cams.get("two_camera_right"),
                cams.get("two_camera_left"),
            )
        self._active_profile.preferred_cameras = cams
        self._profile_manager.save_profile(self._active_profile)
        self._refresh_cameras()
        self._refresh_dashboard()

    @Slot(str)
    def _on_mode_selected(self, mode_id: str) -> None:
        if self._active_profile is None:
            return
        self._active_profile.default_mode = mode_id
        self._profile_manager.save_profile(self._active_profile)
        self._refresh_dashboard()

    def _on_create_profile(self, name: str) -> None:
        self._profile_manager.create_profile(name)
        self._refresh_profiles()

    def _on_rename_profile(self, profile_id: str, new_name: str) -> None:
        self._profile_manager.rename_profile(profile_id, new_name)
        if self._active_profile and self._active_profile.id == profile_id:
            self._active_profile.display_name = new_name
        self._refresh_profiles()
        self._refresh_dashboard()

    def _on_delete_profile(self, profile_id: str) -> None:
        self._profile_manager.delete_profile(profile_id)
        if self._active_profile and self._active_profile.id == profile_id:
            profiles = self._profile_manager.list_profiles()
            if profiles:
                self._active_profile = profiles[0]
            else:
                self._active_profile = self._profile_manager.create_profile("Default User")
        self._refresh_all()

    def _on_reset_calibration(self, profile_id: str) -> None:
        self._profile_manager.reset_all_calibrations(profile_id)
        self._refresh_all()

    @Slot(str)
    def _on_switch_profile(self, profile_id: str) -> None:
        profile = self._profile_manager.load_profile(profile_id)
        if profile:
            self._active_profile = profile
            self._refresh_all()

    @Slot()
    def _on_start_tracking(self) -> None:
        if self._is_tracking:
            return
        if self._active_profile is None:
            QMessageBox.warning(self, "Error", "No active profile.")
            return

        mode_id = self._active_profile.default_mode
        try:
            mode_cls = self._mode_registry.get(mode_id)
        except KeyError:
            QMessageBox.warning(self, "Error", f"Unknown mode: {mode_id}")
            return

        cams = self._active_profile.preferred_cameras
        if mode_cls.required_camera_count == 1:
            cam_idx = cams.get("one_camera")
            if cam_idx is None:
                QMessageBox.warning(self, "Camera Required", "Please select a camera first.")
                self._sidebar.setCurrentRow(2)
                return
            selected_cameras = [cam_idx]
        else:
            left = cams.get("two_camera_left")
            right = cams.get("two_camera_right")
            if left is None or right is None:
                QMessageBox.warning(self, "Cameras Required", "Please select both cameras first.")
                self._sidebar.setCurrentRow(2)
                return
            selected_cameras = [left, right]

        calibrations = {}
        for cal_id in ["one_camera_head_pose", "two_camera_head_pose", "eye_gaze", "facial_gestures"]:
            calibrations[cal_id] = self._profile_manager.load_calibration(
                self._active_profile.id, cal_id
            )
        calibrations["stereo"] = self._profile_manager.load_stereo_calibration(
            self._active_profile.id
        )

        mode_instance = mode_cls()
        ok, reason = mode_instance.validate_requirements(calibrations, selected_cameras)
        if not ok:
            QMessageBox.warning(self, "Requirements Not Met", reason)
            return

        self._cameras_page.stop_previews()
        self._camera_manager.release_all()

        self._pending_mode = mode_instance
        self._pending_calibrations = calibrations
        self._pending_cameras = selected_cameras
        self._countdown(3)

    def _countdown(self, remaining: int) -> None:
        if remaining > 0:
            self._dashboard_page.set_tracking_state("active")
            self._dashboard_page._status_label.setText(f"Starting in {remaining}...")
            QTimer.singleShot(1000, lambda: self._countdown(remaining - 1))
        else:
            self._actually_start_tracking()

    def _actually_start_tracking(self) -> None:
        from src.cursor import create_cursor
        from src.core.tracking_worker import TrackingWorker

        # Snapshot the current settings so the cursor and (later) the worker
        # both start with the user's configured values. update_settings()
        # below keeps everything live.
        current_settings = self._settings_page.get_settings()

        try:
            cursor = create_cursor(
                move_px_per_sec=float(current_settings.get("move_speed", 200)),
                frame_rate=int(current_settings.get("frame_rate", 30)),
                scroll_units_per_sec=float(current_settings.get("scroll_speed", 300)),
            )
        except Exception as e:
            QMessageBox.critical(self, "Cursor Error", f"Failed to initialize cursor: {e}")
            self._dashboard_page.set_tracking_state("stopped")
            return

        self._tracking_mode = self._pending_mode

        if getattr(self._tracking_mode, "id", None) == "eye_gaze_bubble":
            from src.ui.overlays.gaze_bubble_overlay import GazeBubbleOverlay
            from src.ui.overlays.gaze_signal_proxy import GazeSignalProxy

            self._gaze_signal_proxy = GazeSignalProxy()
            self._gaze_overlay = GazeBubbleOverlay(cursor.get_virtual_bounds())
            self._gaze_signal_proxy.gaze_target_changed.connect(
                self._gaze_overlay.update_position
            )
            self._tracking_mode.gaze_target_callback = (
                self._gaze_signal_proxy.gaze_target_changed.emit
            )
            self._gaze_overlay.show()

        worker = TrackingWorker(
            mode=self._tracking_mode,
            calibrations=self._pending_calibrations,
            cameras=self._pending_cameras,
            cursor=cursor,
            settings=current_settings,
        )

        self._tracking_thread = QThread()
        worker.moveToThread(self._tracking_thread)
        self._tracking_thread.started.connect(worker.run)
        worker.status_changed.connect(self._on_tracking_status_changed)
        worker.error_occurred.connect(self._on_tracking_error)
        worker.finished.connect(self._tracking_thread.quit)
        worker.finished.connect(worker.deleteLater)
        self._tracking_thread.finished.connect(self._on_tracking_stopped)

        self._is_tracking = True
        self._is_paused = False
        self._tracking_worker = worker
        self._tracking_thread.start()

    @Slot(str)
    def _on_tracking_status_changed(self, status: str) -> None:
        self._dashboard_page.set_tracking_state(status)

    @Slot(str)
    def _on_tracking_error(self, error: str) -> None:
        QMessageBox.critical(self, "Tracking Error", error)

    @Slot()
    def _on_pause_tracking(self) -> None:
        if self._tracking_mode and self._is_tracking:
            if self._is_paused:
                self._tracking_mode.resume()
                self._is_paused = False
                self._dashboard_page.set_tracking_state("active")
            else:
                self._tracking_mode.pause()
                self._is_paused = True
                self._dashboard_page.set_tracking_state("paused")

    @Slot(dict)
    def _on_settings_changed(self, settings: dict) -> None:
        # Forward settings to the active worker so it can update the running
        # mode. When idle, the next start picks up the latest snapshot via
        # SettingsPage.get_settings().
        if not self._is_tracking:
            return
        worker = getattr(self, "_tracking_worker", None)
        if worker is None:
            return
        worker.update_settings(settings)

    @Slot()
    def _on_stop_tracking(self) -> None:
        if self._tracking_mode and self._is_tracking:
            self._tracking_mode.stop()

    def _on_tracking_stopped(self) -> None:
        self._is_tracking = False
        self._is_paused = False
        # Tear down the visualizer BEFORE clearing _tracking_mode so the helper
        # can null out the callback on the live mode instance.
        self._teardown_visualizer()
        self._tracking_mode = None
        if self._tracking_thread:
            self._tracking_thread.quit()
            self._tracking_thread.wait(5000)
            self._tracking_thread = None
        # Drop the worker reference so a late settings_changed emit can't
        # reach a deleted QObject.
        self._tracking_worker = None
        self._teardown_gaze_overlay()
        self._dashboard_page.set_tracking_state("stopped")

    def _teardown_gaze_overlay(self) -> None:
        if self._gaze_overlay is not None:
            self._gaze_overlay.hide()
            self._gaze_overlay.deleteLater()
            self._gaze_overlay = None
        if self._gaze_signal_proxy is not None:
            self._gaze_signal_proxy.deleteLater()
            self._gaze_signal_proxy = None

    @Slot()
    def _on_visualize_requested(self) -> None:
        if not self._is_tracking or self._tracking_mode is None:
            return
        if self._active_profile is None:
            return

        mode_id = getattr(self._tracking_mode, "id", None)
        if mode_id == "balloon_ride":
            return

        if self._visualizer_window is not None:
            self._visualizer_window.raise_()
            self._visualizer_window.activateWindow()
            return

        from src.ui.overlays.visualization_signal_proxy import VisualizationSignalProxy
        from src.ui.visualizer.visualizer_window import VisualizerWindow

        try:
            mode_cls = self._mode_registry.get(mode_id)
            display_name = mode_cls.display_name
        except (KeyError, AttributeError):
            display_name = mode_id or "Unknown"

        self._viz_signal_proxy = VisualizationSignalProxy()
        self._visualizer_window = VisualizerWindow(
            mode_id=mode_id,
            mode_display_name=display_name,
            parent=self,
        )
        self._viz_signal_proxy.frame_ready.connect(
            self._visualizer_window.update_payload
        )
        self._visualizer_window.closed.connect(self._on_visualizer_closed)

        # Attaching the callback enables the throttled emit in the mode loop. Until the
        # window is opened, the mode skips this work entirely (callback is None).
        self._tracking_mode.visualization_callback = (
            self._viz_signal_proxy.frame_ready.emit
        )
        self._visualizer_window.show()
        self._visualizer_window.raise_()
        self._visualizer_window.activateWindow()

    @Slot()
    def _on_visualizer_closed(self) -> None:
        self._teardown_visualizer()

    def _teardown_visualizer(self) -> None:
        if self._tracking_mode is not None:
            self._tracking_mode.visualization_callback = None
        if self._visualizer_window is not None:
            try:
                self._visualizer_window.closed.disconnect(self._on_visualizer_closed)
            except (TypeError, RuntimeError):
                pass
            self._visualizer_window.hide()
            self._visualizer_window.deleteLater()
            self._visualizer_window = None
        if self._viz_signal_proxy is not None:
            self._viz_signal_proxy.deleteLater()
            self._viz_signal_proxy = None

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            if self._is_tracking:
                self._on_stop_tracking()
                return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self._tracking_mode and self._is_tracking:
            self._tracking_mode.stop()
        if self._tracking_thread:
            self._tracking_thread.quit()
            self._tracking_thread.wait(5000)
        self._teardown_visualizer()
        self._teardown_gaze_overlay()
        self._cameras_page.stop_previews()
        self._camera_manager.release_all()
        event.accept()
