from __future__ import annotations

from direct.gui.DirectGui import DirectButton, DirectFrame, DirectLabel
from panda3d.core import NodePath, TextNode, Vec4

from game.core import settings
from game.scenes.base_scene import BaseScene


BG_COLOR = Vec4(0.10, 0.10, 0.10, 1.0)
PANEL_COLOR = Vec4(0.12, 0.12, 0.12, 1.0)
TEXT_COLOR = Vec4(0.91, 0.91, 0.81, 1.0)
ACCENT = Vec4(0.353, 0.710, 0.322, 1.0)


TRIGGER_LABELS = {
    "left_click": "LEFT CLICK",
    "spacebar": "SPACEBAR",
    "right_click": "RIGHT CLICK",
}


BACK_KEY = "__back__"
ROW_TITLES = {
    "photo_trigger": "PHOTO TRIGGER",
    "countdown_duration": "COUNTDOWN",
    "cart_speed": "CART SPEED",
}


class SettingsScreen(BaseScene):
    name = "settings"
    cursor_visible = True

    def __init__(self, app) -> None:
        super().__init__(app)
        self.root: NodePath | None = None
        self._widgets: list = []
        self._labels: dict = {}
        self._title_labels: dict = {}
        self._items: list = []
        self._focus_idx: int = 0
        self._accept_keys: list = []
        self._back_btn = None

    def enter(self) -> None:
        self.root = NodePath("settings_root")
        self.root.reparentTo(self.app.base.aspect2d)

        font = self.app.font

        DirectFrame(
            parent=self.root,
            frameColor=BG_COLOR,
            frameSize=(-2.0, 2.0, -1.5, 1.5),
        )

        DirectLabel(
            parent=self.root,
            text="SETTINGS",
            text_fg=ACCENT,
            text_scale=0.13,
            text_font=font,
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, 0.65),
        )

        self._make_row("photo_trigger", (0, 0, 0.30))
        self._make_row("countdown_duration", (0, 0, 0.10))
        self._make_row("cart_speed", (0, 0, -0.10))

        self._back_btn = DirectButton(
            parent=self.root,
            text="BACK",
            command=self._back,
            text_fg=TEXT_COLOR,
            text_scale=0.07,
            text_font=font,
            frameColor=(PANEL_COLOR, ACCENT, ACCENT, PANEL_COLOR),
            frameSize=(-0.3, 0.3, -0.10, 0.10),
            borderWidth=(0.005, 0.005),
            relief=1,
            pos=(0, 0, -0.45),
        )
        self._widgets.append(self._back_btn)

        self._items = ["photo_trigger", "countdown_duration", "cart_speed", BACK_KEY]
        self._focus_idx = 0
        self._refresh_labels()
        self._refresh_focus()
        self._bind_keys()

    def _make_row(self, key: str, pos) -> None:
        font = self.app.font
        title_label = DirectLabel(
            parent=self.root,
            text=ROW_TITLES[key],
            text_fg=TEXT_COLOR,
            text_scale=0.055,
            text_font=font,
            text_align=TextNode.ALeft,
            frameColor=(0, 0, 0, 0),
            pos=(-1.10, 0, pos[2]),
        )
        self._title_labels[key] = title_label
        self._widgets.append(title_label)
        btn = DirectButton(
            parent=self.root,
            text="...",
            command=self._cycle,
            extraArgs=[key],
            text_fg=TEXT_COLOR,
            text_scale=0.055,
            text_font=font,
            frameColor=(PANEL_COLOR, ACCENT, ACCENT, PANEL_COLOR),
            frameSize=(-0.45, 0.45, -0.08, 0.08),
            borderWidth=(0.005, 0.005),
            relief=1,
            pos=(0.55, 0, pos[2]),
        )
        self._widgets.append(btn)
        self._labels[key] = btn

    def _bind_keys(self) -> None:
        base = self.app.base
        base.accept("arrow_up", self._focus_prev)
        base.accept("arrow_down", self._focus_next)
        base.accept("arrow_left", self._cycle_left)
        base.accept("arrow_right", self._cycle_right)
        base.accept("enter", self._activate)
        base.accept("return", self._activate)
        self._accept_keys = [
            "arrow_up", "arrow_down", "arrow_left", "arrow_right", "enter", "return",
        ]

    def _refresh_focus(self) -> None:
        for key, lbl in self._title_labels.items():
            focused = self._items[self._focus_idx] == key
            prefix = "> " if focused else "  "
            lbl["text"] = prefix + ROW_TITLES[key]
            lbl["text_fg"] = ACCENT if focused else TEXT_COLOR
        if self._back_btn is not None:
            focused = self._items[self._focus_idx] == BACK_KEY
            prefix = "> " if focused else "  "
            self._back_btn["text"] = prefix + "BACK"
            self._back_btn["text_fg"] = ACCENT if focused else TEXT_COLOR

    def _focus_prev(self) -> None:
        if not self._items:
            return
        self._focus_idx = (self._focus_idx - 1) % len(self._items)
        self._refresh_focus()

    def _focus_next(self) -> None:
        if not self._items:
            return
        self._focus_idx = (self._focus_idx + 1) % len(self._items)
        self._refresh_focus()

    def _cycle_left(self) -> None:
        item = self._items[self._focus_idx]
        if item == BACK_KEY:
            return
        self._cycle_value(item, forward=True)

    def _cycle_right(self) -> None:
        item = self._items[self._focus_idx]
        if item == BACK_KEY:
            return
        self._cycle_value(item, forward=False)

    def _activate(self) -> None:
        item = self._items[self._focus_idx]
        if item == BACK_KEY:
            self._back()
            return
        self._cycle_value(item, forward=True)

    def _cycle_value(self, key: str, forward: bool) -> None:
        cur = self.app.config.get(key)
        if key == "photo_trigger":
            values = settings.VALID_TRIGGERS
        elif key == "countdown_duration":
            values = settings.VALID_DURATIONS
        elif key == "cart_speed":
            values = settings.VALID_SPEEDS
        else:
            return
        new = settings.cycle(values, cur) if forward else settings.cycle_back(values, cur)
        self.app.config[key] = new
        settings.save(self.app.config)
        self._refresh_labels()

    def _value_label(self, key: str) -> str:
        v = self.app.config.get(key)
        if key == "photo_trigger":
            return f"[ {TRIGGER_LABELS.get(v, str(v).upper())} ]"
        if key == "countdown_duration":
            return f"[ {v:.1f}s ]"
        if key == "cart_speed":
            return f"[ {str(v).upper()} ]"
        return f"[ {v} ]"

    def _refresh_labels(self) -> None:
        for key, btn in self._labels.items():
            btn["text"] = self._value_label(key)

    def _cycle(self, key: str) -> None:
        self._cycle_value(key, forward=True)

    def _back(self) -> None:
        self.app.scene_manager.switch("main_menu")

    def exit(self) -> None:
        for k in self._accept_keys:
            self.app.base.ignore(k)
        self._accept_keys = []
        for w in self._widgets:
            try:
                w.destroy()
            except Exception:
                pass
        self._widgets = []
        self._labels = {}
        self._title_labels = {}
        self._items = []
        self._back_btn = None
        if self.root is not None:
            self.root.removeNode()
            self.root = None
