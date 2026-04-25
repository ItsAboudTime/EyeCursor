from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt


QUALITY_COLORS = {
    "Excellent": "#2ecc71",
    "Good": "#27ae60",
    "Acceptable": "#f39c12",
    "Poor": "#e74c3c",
    "Failed": "#c0392b",
    "Not Calibrated": "#95a5a6",
}


class CalibrationScoreBadge(QLabel):
    def __init__(self, label: str = "Not Calibrated", parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_label(label)

    def set_label(self, label: str) -> None:
        color = QUALITY_COLORS.get(label, "#95a5a6")
        self.setText(label)
        self.setStyleSheet(
            f"background-color: {color}; color: white; padding: 4px 10px; "
            f"border-radius: 4px; font-weight: bold; font-size: 12px;"
        )
