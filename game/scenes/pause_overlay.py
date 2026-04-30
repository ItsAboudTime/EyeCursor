from __future__ import annotations

from direct.gui.DirectGui import DirectButton, DirectFrame, DirectLabel
from panda3d.core import NodePath, TextNode, Vec4

from game.scenes.base_scene import BaseScene


PANEL_COLOR = Vec4(0.10, 0.10, 0.10, 0.92)
DIM_COLOR = Vec4(0.0, 0.0, 0.0, 0.55)
TEXT_COLOR = Vec4(0.91, 0.91, 0.81, 1.0)
ACCENT = Vec4(0.353, 0.710, 0.322, 1.0)


class PauseOverlay(BaseScene):
    name = "pause"
    cursor_visible = True

    def __init__(self, app) -> None:
        super().__init__(app)
        self.root: NodePath | None = None
        self._widgets: list = []
        self._buttons: list = []
        self._button_texts: list = []
        self._commands: list = []
        self._focus_idx: int = 0
        self._accept_keys: list = []

    def enter(self, **kwargs) -> None:
        self.root = NodePath("pause_root")
        self.root.reparentTo(self.app.base.aspect2d)

        DirectFrame(
            parent=self.root,
            frameColor=DIM_COLOR,
            frameSize=(-2.0, 2.0, -1.5, 1.5),
        )

        font = self.app.font

        DirectFrame(
            parent=self.root,
            frameColor=PANEL_COLOR,
            frameSize=(-0.7, 0.7, -0.5, 0.5),
            borderWidth=(0.01, 0.01),
        )

        DirectLabel(
            parent=self.root,
            text="-- PAUSED --",
            text_fg=ACCENT,
            text_scale=0.10,
            text_font=font,
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, 0.30),
        )

        self._button_texts = ["RESUME", "SETTINGS", "QUIT TO MENU"]
        self._commands = [self._resume, self._settings, self._quit_to_menu]
        positions = [(0, 0, 0.10), (0, 0, -0.05), (0, 0, -0.20)]
        self._buttons = [self._btn(t, c, p) for t, c, p in zip(self._button_texts, self._commands, positions)]
        self._widgets.extend(self._buttons)
        self._focus_idx = 0
        self._refresh_focus()
        self._bind_keys()

    def _bind_keys(self) -> None:
        base = self.app.base
        base.accept("arrow_up", self._focus_prev)
        base.accept("arrow_down", self._focus_next)
        base.accept("enter", self._activate)
        base.accept("return", self._activate)
        self._accept_keys = ["arrow_up", "arrow_down", "enter", "return"]

    def _refresh_focus(self) -> None:
        for i, btn in enumerate(self._buttons):
            focused = i == self._focus_idx
            prefix = "> " if focused else "  "
            btn["text"] = prefix + self._button_texts[i]
            btn["text_fg"] = ACCENT if focused else TEXT_COLOR

    def _focus_prev(self) -> None:
        if not self._buttons:
            return
        self._focus_idx = (self._focus_idx - 1) % len(self._buttons)
        self._refresh_focus()

    def _focus_next(self) -> None:
        if not self._buttons:
            return
        self._focus_idx = (self._focus_idx + 1) % len(self._buttons)
        self._refresh_focus()

    def _activate(self) -> None:
        if 0 <= self._focus_idx < len(self._commands):
            self._commands[self._focus_idx]()

    def _btn(self, text, command, pos):
        return DirectButton(
            parent=self.root,
            text=text,
            command=command,
            text_fg=TEXT_COLOR,
            text_scale=0.06,
            text_font=self.app.font,
            frameColor=(Vec4(0.12, 0.12, 0.12, 1), ACCENT, ACCENT, Vec4(0.12, 0.12, 0.12, 1)),
            frameSize=(-0.55, 0.55, -0.07, 0.07),
            borderWidth=(0.005, 0.005),
            relief=1,
            pos=pos,
        )

    def on_escape(self) -> None:
        self._resume()

    def _resume(self) -> None:
        self.app.play_click()
        self.app.scene_manager.pop_overlay()

    def _settings(self) -> None:
        self.app.play_click()
        self.app.scene_manager.swap_overlay("settings")

    def _quit_to_menu(self) -> None:
        self.app.play_click()
        self.app.scene_manager.pop_overlay()
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
        self._buttons = []
        self._button_texts = []
        self._commands = []
        if self.root is not None:
            self.root.removeNode()
            self.root = None
