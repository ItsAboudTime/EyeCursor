from PySide6.QtCore import Qt, Signal
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
    # Emitted any time any setting changes. Payload is the full settings dict
    # (same shape as get_settings()).
    settings_changed = Signal(dict)

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

        # Move speed: 1..10000 px/sec. Floor of 1 lets users pick extremely
        # slow cursors (precision tasks); ceiling of 10000 well above any
        # realistic head/gaze tracking rate.
        self._move_speed = QSpinBox()
        self._move_speed.setRange(1, 10000)
        self._move_speed.setValue(1000)
        self._move_speed.setSuffix(" px/sec")
        form.addRow("Move speed:", self._move_speed)

        # Frame rate: 1..240 fps. 1 fps lets users debug; 240 fps covers
        # high-refresh-rate monitors and any reasonable webcam capture rate.
        self._frame_rate = QSpinBox()
        self._frame_rate.setRange(1, 240)
        self._frame_rate.setValue(60)
        self._frame_rate.setSuffix(" fps")
        form.addRow("Frame rate:", self._frame_rate)

        # EMA alpha: 0.001..1.0. The lower bound is the smallest practical
        # smoothing value that still permits change. 0.0 is excluded because
        # it freezes the cursor; downstream code asserts (0, 1].
        self._ema_alpha = QDoubleSpinBox()
        self._ema_alpha.setRange(0.001, 1.0)
        self._ema_alpha.setDecimals(3)
        self._ema_alpha.setSingleStep(0.05)
        self._ema_alpha.setValue(0.1)
        form.addRow("Smoothing (EMA alpha):", self._ema_alpha)

        layout.addWidget(cursor_group)

        gestures_group = QGroupBox("Gesture Settings")
        gestures_group.setStyleSheet(cursor_group.styleSheet())
        gestures_form = QFormLayout(gestures_group)

        self._click_enabled = QCheckBox("Enable click gestures (pucker / lip-tuck)")
        self._click_enabled.setChecked(True)
        self._click_enabled.setStyleSheet(checkbox_style)
        gestures_form.addRow(self._click_enabled)
        _click_scroll_spacer = QWidget()
        _click_scroll_spacer.setFixedHeight(8)
        gestures_form.addRow(_click_scroll_spacer)

        self._scroll_enabled = QCheckBox("Enable scroll gestures (smirk left / right)")
        self._scroll_enabled.setChecked(True)
        self._scroll_enabled.setStyleSheet(checkbox_style)
        gestures_form.addRow(self._scroll_enabled)

        # Scroll speed: 1..10000 units/sec. Same logic as move speed.
        self._scroll_speed = QSpinBox()
        self._scroll_speed.setRange(1, 10000)
        self._scroll_speed.setValue(20)
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

        # Wire change notifications. Each control emits the full snapshot so
        # consumers don't need to know which control fired.
        self._move_speed.valueChanged.connect(self._emit_settings_changed)
        self._frame_rate.valueChanged.connect(self._emit_settings_changed)
        self._ema_alpha.valueChanged.connect(self._emit_settings_changed)
        self._scroll_speed.valueChanged.connect(self._emit_settings_changed)
        self._click_enabled.stateChanged.connect(self._emit_settings_changed)
        self._scroll_enabled.stateChanged.connect(self._emit_settings_changed)

    def _emit_settings_changed(self, *_args) -> None:
        self.settings_changed.emit(self.get_settings())

    def get_settings(self) -> dict:
        return {
            "move_speed": self._move_speed.value(),
            "frame_rate": self._frame_rate.value(),
            "click_enabled": self._click_enabled.isChecked(),
            "scroll_enabled": self._scroll_enabled.isChecked(),
            "scroll_speed": self._scroll_speed.value(),
            "ema_alpha": self._ema_alpha.value(),
        }
