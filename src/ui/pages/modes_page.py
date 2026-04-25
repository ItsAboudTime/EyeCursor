from typing import List, Optional, Type

from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget
from PySide6.QtCore import Signal

from src.core.modes.base import TrackingMode
from src.ui.components.mode_card import ModeCard


class ModesPage(QWidget):
    mode_selected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Tracking Modes")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel("Select a tracking mode to use with EyeCursor.")
        subtitle.setStyleSheet("color: #636e72; font-size: 13px;")
        layout.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._cards_widget = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_widget)
        self._cards_layout.setSpacing(12)
        self._cards_layout.addStretch()
        scroll.setWidget(self._cards_widget)
        layout.addWidget(scroll)

        self._cards: dict[str, ModeCard] = {}

    def populate_modes(
        self,
        mode_classes: List[Type[TrackingMode]],
        active_mode_id: Optional[str],
        calibration_statuses: Optional[dict] = None,
    ) -> None:
        for card in self._cards.values():
            self._cards_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        for mode_cls in mode_classes:
            cal_label = "Not Calibrated"
            if calibration_statuses and calibration_statuses.get(mode_cls.id):
                cal_label = "Calibrated"

            card = ModeCard(
                mode_id=mode_cls.id,
                display_name=mode_cls.display_name,
                description=mode_cls.description,
                required_cameras=mode_cls.required_camera_count,
                calibration_label=cal_label,
                is_active=(mode_cls.id == active_mode_id),
            )
            card.selected.connect(self._on_mode_selected)
            insert_index = self._cards_layout.count() - 1
            self._cards_layout.insertWidget(insert_index, card)
            self._cards[mode_cls.id] = card

    def _on_mode_selected(self, mode_id: str) -> None:
        for mid, card in self._cards.items():
            card.set_active(mid == mode_id)
        self.mode_selected.emit(mode_id)

    def set_active_mode(self, mode_id: str) -> None:
        for mid, card in self._cards.items():
            card.set_active(mid == mode_id)
