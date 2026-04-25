from typing import List, Optional

from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Signal

from src.core.profiles.profile_model import ProfileModel


class ProfilesPage(QWidget):
    profile_switched = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("User Profiles")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Each profile stores its own calibration data. "
            "Create a profile for each user."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #636e72; font-size: 13px;")
        layout.addWidget(subtitle)

        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget { border: 1px solid #dcdde1; border-radius: 8px; padding: 4px; font-size: 14px; }"
            "QListWidget::item { padding: 10px; }"
            "QListWidget::item:selected { background: #dfe6e9; color: #2d3436; }"
        )
        self._list.currentRowChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        btn_style = (
            "QPushButton { background: #0984e3; color: white; border: none; "
            "padding: 8px 16px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #0652DD; }"
            "QPushButton:disabled { background: #b2bec3; }"
        )
        danger_style = (
            "QPushButton { background: #d63031; color: white; border: none; "
            "padding: 8px 16px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #e17055; }"
            "QPushButton:disabled { background: #b2bec3; }"
        )

        self._create_btn = QPushButton("Create")
        self._create_btn.setStyleSheet(btn_style)
        self._create_btn.clicked.connect(self._on_create)
        btn_layout.addWidget(self._create_btn)

        self._rename_btn = QPushButton("Rename")
        self._rename_btn.setStyleSheet(btn_style)
        self._rename_btn.setEnabled(False)
        self._rename_btn.clicked.connect(self._on_rename)
        btn_layout.addWidget(self._rename_btn)

        self._switch_btn = QPushButton("Switch To")
        self._switch_btn.setStyleSheet(btn_style)
        self._switch_btn.setEnabled(False)
        self._switch_btn.clicked.connect(self._on_switch)
        btn_layout.addWidget(self._switch_btn)

        self._reset_btn = QPushButton("Reset Calibration")
        self._reset_btn.setStyleSheet(danger_style)
        self._reset_btn.setEnabled(False)
        self._reset_btn.clicked.connect(self._on_reset_calibration)
        btn_layout.addWidget(self._reset_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setStyleSheet(danger_style)
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._on_delete)
        btn_layout.addWidget(self._delete_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self._profiles: List[ProfileModel] = []
        self._active_profile_id: Optional[str] = None
        self._on_create_callback = None
        self._on_rename_callback = None
        self._on_delete_callback = None
        self._on_reset_callback = None

    def set_callbacks(self, on_create, on_rename, on_delete, on_reset) -> None:
        self._on_create_callback = on_create
        self._on_rename_callback = on_rename
        self._on_delete_callback = on_delete
        self._on_reset_callback = on_reset

    def populate_profiles(
        self, profiles: List[ProfileModel], active_profile_id: Optional[str]
    ) -> None:
        self._profiles = profiles
        self._active_profile_id = active_profile_id
        self._list.clear()
        for p in profiles:
            suffix = " (active)" if p.id == active_profile_id else ""
            item = QListWidgetItem(f"{p.display_name}{suffix}")
            item.setData(256, p.id)
            self._list.addItem(item)

    def _selected_profile_id(self) -> Optional[str]:
        item = self._list.currentItem()
        if item is None:
            return None
        return item.data(256)

    def _on_selection_changed(self, row: int) -> None:
        has_selection = row >= 0
        self._rename_btn.setEnabled(has_selection)
        self._switch_btn.setEnabled(has_selection)
        self._delete_btn.setEnabled(has_selection)
        self._reset_btn.setEnabled(has_selection)

    def _on_create(self) -> None:
        name, ok = QInputDialog.getText(self, "Create Profile", "Display name:")
        if ok and name.strip():
            if self._on_create_callback:
                self._on_create_callback(name.strip())

    def _on_rename(self) -> None:
        profile_id = self._selected_profile_id()
        if profile_id is None:
            return
        current = next((p for p in self._profiles if p.id == profile_id), None)
        current_name = current.display_name if current else ""
        name, ok = QInputDialog.getText(
            self, "Rename Profile", "New name:", text=current_name
        )
        if ok and name.strip():
            if self._on_rename_callback:
                self._on_rename_callback(profile_id, name.strip())

    def _on_switch(self) -> None:
        profile_id = self._selected_profile_id()
        if profile_id and profile_id != self._active_profile_id:
            self.profile_switched.emit(profile_id)

    def _on_delete(self) -> None:
        profile_id = self._selected_profile_id()
        if profile_id is None:
            return
        result = QMessageBox.question(
            self,
            "Delete Profile",
            "Are you sure? This will delete all calibration data for this profile.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if result == QMessageBox.StandardButton.Yes:
            if self._on_delete_callback:
                self._on_delete_callback(profile_id)

    def _on_reset_calibration(self) -> None:
        profile_id = self._selected_profile_id()
        if profile_id is None:
            return
        result = QMessageBox.question(
            self,
            "Reset Calibration",
            "Reset all calibration data for this profile?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if result == QMessageBox.StandardButton.Yes:
            if self._on_reset_callback:
                self._on_reset_callback(profile_id)
