from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QWidget,
)

from criteria.core.models import Session
from criteria.core.storage import StorageManager
from criteria.ui.pages.dashboard_page import DashboardPage
from criteria.ui.pages.results_page import ResultsPage
from criteria.ui.pages.resume_page import ResumePage
from criteria.ui.pages.session_setup_page import SessionSetupPage
from criteria.ui.pages.settings_page import SettingsPage
from criteria.ui.test_window import TestWindow


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.storage = StorageManager()
        self.test_window: TestWindow | None = None
        self.setWindowTitle("EyeCursor TestLab")
        self.setMinimumSize(1040, 680)
        self.resize(1160, 760)
        icon = Path(__file__).resolve().parents[2] / "assets" / "icon_256.png"
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))
        self._setup_ui()
        self._connect()
        self.refresh()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(210)
        for name in ["Dashboard", "New Session", "Resume Session", "Results", "Settings"]:
            self.sidebar.addItem(QListWidgetItem(name))
        layout.addWidget(self.sidebar)
        self.stack = QStackedWidget()
        self.dashboard = DashboardPage()
        self.setup_page = SessionSetupPage()
        self.resume_page = ResumePage()
        self.results_page = ResultsPage()
        self.settings_page = SettingsPage()
        for page in (self.dashboard, self.setup_page, self.resume_page, self.results_page, self.settings_page):
            self.stack.addWidget(page)
        layout.addWidget(self.stack)
        self.sidebar.setCurrentRow(0)

    def _connect(self) -> None:
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.dashboard.new_requested.connect(lambda: self.sidebar.setCurrentRow(1))
        self.dashboard.resume_requested.connect(lambda: self.sidebar.setCurrentRow(2))
        self.dashboard.results_requested.connect(lambda: self.sidebar.setCurrentRow(3))
        self.setup_page.session_created.connect(self.start_session)
        self.resume_page.resume_selected.connect(self.resume_session)
        self.results_page.export_json_requested.connect(self.export_json)
        self.results_page.export_csv_requested.connect(self.export_csv)

    def refresh(self) -> None:
        sessions = self.storage.list_sessions()
        self.dashboard.set_recent(sessions)
        self.resume_page.set_sessions(sessions)
        self.results_page.set_sessions(sessions)

    def start_session(self, session: Session) -> None:
        try:
            self.storage.save_session(session)
            self._open_test_window(session)
        except OSError:
            QMessageBox.critical(
                self,
                "Could not save session",
                "Could not save the session results. Check that the app has permission to write to the results folder.",
            )

    def resume_session(self, session_id: str) -> None:
        try:
            session = self.storage.load_session(session_id)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(self, "Could not resume session", str(exc))
            return
        self._open_test_window(session)

    def export_json(self, session_id: str) -> None:
        try:
            path = self.storage.export_json(self.storage.load_session(session_id))
        except OSError:
            QMessageBox.critical(self, "Export failed", "Could not export the raw JSON session.")
            return
        QMessageBox.information(self, "Export complete", f"JSON exported to:\n{path}")

    def export_csv(self, session_id: str) -> None:
        try:
            path = self.storage.export_summary_csv(self.storage.load_session(session_id))
        except OSError:
            QMessageBox.critical(self, "Export failed", "Could not export the CSV summary.")
            return
        QMessageBox.information(self, "Export complete", f"CSV exported to:\n{path}")

    def _open_test_window(self, session: Session) -> None:
        self.test_window = TestWindow(session, self.storage)
        self.test_window.closed.connect(self.refresh)
        self.test_window.results_requested.connect(lambda: self.sidebar.setCurrentRow(3))
        self.test_window.begin()

