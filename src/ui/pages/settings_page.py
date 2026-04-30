from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SettingsPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Settings")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        checkbox_style = (
            "QCheckBox { color: #2d3436; font-size: 13px; spacing: 8px; }"
            "QCheckBox::indicator { width: 20px; height: 20px; border: 2px solid #b2bec3; "
            "border-radius: 4px; background: white; }"
            "QCheckBox::indicator:checked { background: #0984e3; border-color: #0984e3; }"
            "QCheckBox::indicator:checked::after { color: white; }"
        )

        cursor_group = QGroupBox("Cursor Settings")
        cursor_group.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 14px; border: 1px solid #dcdde1; "
            "border-radius: 8px; margin-top: 8px; padding-top: 16px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }"
        )
        form = QFormLayout(cursor_group)

        self._move_speed = QSpinBox()
        self._move_speed.setRange(50, 2000)
        self._move_speed.setValue(200)
        self._move_speed.setSuffix(" px/sec")
        form.addRow("Move speed:", self._move_speed)

        self._frame_rate = QSpinBox()
        self._frame_rate.setRange(10, 120)
        self._frame_rate.setValue(30)
        self._frame_rate.setSuffix(" fps")
        form.addRow("Frame rate:", self._frame_rate)

        self._ema_alpha = QDoubleSpinBox()
        self._ema_alpha.setRange(0.01, 1.0)
        self._ema_alpha.setSingleStep(0.05)
        self._ema_alpha.setValue(0.1)
        form.addRow("Smoothing (EMA alpha):", self._ema_alpha)

        layout.addWidget(cursor_group)

        gestures_group = QGroupBox("Gesture Settings")
        gestures_group.setStyleSheet(cursor_group.styleSheet())
        gestures_form = QFormLayout(gestures_group)

        self._click_enabled = QCheckBox("Enable click gestures (smirk to click)")
        self._click_enabled.setChecked(True)
        self._click_enabled.setStyleSheet(checkbox_style)
        gestures_form.addRow(self._click_enabled)

        self._scroll_enabled = QCheckBox("Enable scroll gestures (cheek puff to scroll)")
        self._scroll_enabled.setChecked(True)
        self._scroll_enabled.setStyleSheet(checkbox_style)
        gestures_form.addRow(self._scroll_enabled)

        self._scroll_speed = QSpinBox()
        self._scroll_speed.setRange(10, 1000)
        self._scroll_speed.setValue(300)
        self._scroll_speed.setSuffix(" units/sec")
        gestures_form.addRow("Scroll speed:", self._scroll_speed)

        self._scroll_enabled.toggled.connect(self._scroll_speed.setEnabled)

        layout.addWidget(gestures_group)

        about_group = QGroupBox("About")
        about_group.setStyleSheet(cursor_group.styleSheet())
        about_layout = QVBoxLayout(about_group)
        about_label = QLabel(
            "<b>EyeCursor</b> v0.1.0<br>"
            "Control your cursor with head and eye movements.<br><br>"
            "Graduation Project - 2026"
        )
        about_label.setWordWrap(True)
        about_layout.addWidget(about_label)
        layout.addWidget(about_group)

        layout.addStretch()

    def get_settings(self) -> dict:
        return {
            "move_speed": self._move_speed.value(),
            "frame_rate": self._frame_rate.value(),
            "click_enabled": self._click_enabled.isChecked(),
            "scroll_enabled": self._scroll_enabled.isChecked(),
            "scroll_speed": self._scroll_speed.value(),
            "ema_alpha": self._ema_alpha.value(),
        }
