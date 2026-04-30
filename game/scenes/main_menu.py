from __future__ import annotations

from direct.gui.DirectGui import DirectButton, DirectFrame, DirectLabel
from panda3d.core import NodePath, TextNode, Vec4

from game.scenes.base_scene import BaseScene


BG_COLOR = Vec4(0.10, 0.10, 0.10, 1.0)
PANEL_COLOR = Vec4(0.12, 0.12, 0.12, 1.0)
BORDER_COLOR = Vec4(0.30, 0.30, 0.30, 1.0)
TEXT_COLOR = Vec4(0.91, 0.91, 0.81, 1.0)
ACCENT = Vec4(0.353, 0.710, 0.322, 1.0)


def make_button(parent, text, command, pos, font, width=0.6, height=0.10):
    btn = DirectButton(
        parent=parent,
        text=text,
        command=command,
        text_fg=TEXT_COLOR,
        text_scale=0.07,
        text_font=font,
        frameColor=(PANEL_COLOR, ACCENT, ACCENT, PANEL_COLOR),
        frameSize=(-width, width, -height, height),
        relief=1,
        borderWidth=(0.005, 0.005),
        pos=pos,
    )
    btn["text_fg"] = TEXT_COLOR
    return btn


class MainMenu(BaseScene):
    name = "main_menu"
    cursor_visible = True

    def __init__(self, app) -> None:
        super().__init__(app)
        self.root: NodePath | None = None
        self._buttons: list = []
        self._button_texts: list = []
        self._commands: list = []
        self._focus_idx: int = 0
        self._accept_keys: list = []

    def enter(self, **kwargs) -> None:
        self.root = NodePath("main_menu_root")
        self.root.reparentTo(self.app.base.aspect2d)

        bg = DirectFrame(
            parent=self.root,
            frameColor=BG_COLOR,
            frameSize=(-2.0, 2.0, -1.5, 1.5),
        )
        self._bg = bg

        font = self.app.font

        title = DirectLabel(
            parent=self.root,
            text="HORSIN' AROUND",
            text_fg=ACCENT,
            text_scale=0.16,
            text_font=font,
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, 0.55),
        )
        self._title = title

        sub = DirectLabel(
            parent=self.root,
            text="EYECURSOR DEMO",
            text_fg=TEXT_COLOR,
            text_scale=0.05,
            text_font=font,
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, 0.40),
        )
        self._sub = sub

        self._button_texts = ["MAPS", "DEX", "GALLERY", "SETTINGS", "QUIT"]
        self._commands = [self._maps, self._dex, self._gallery, self._settings, self._quit]
        positions = [(0, 0, 0.20), (0, 0, 0.05), (0, 0, -0.10), (0, 0, -0.25), (0, 0, -0.40)]
        self._buttons = [
            make_button(self.root, txt, cmd, pos, font)
            for txt, cmd, pos in zip(self._button_texts, self._commands, positions)
        ]
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

    def _maps(self) -> None:
        self.app.play_click()
        self.app.scene_manager.switch("map_select")

    def _dex(self) -> None:
        self.app.play_click()
        self.app.scene_manager.switch("dex")

    def _gallery(self) -> None:
        self.app.play_click()
        self.app.scene_manager.switch("gallery")

    def _settings(self) -> None:
        self.app.play_click()
        self.app.scene_manager.switch("settings")

    def _quit(self) -> None:
        self.app.play_click()
        self.app.shutdown()

    def exit(self) -> None:
        for k in self._accept_keys:
            self.app.base.ignore(k)
        self._accept_keys = []
        for b in self._buttons:
            b.destroy()
        self._buttons = []
        self._button_texts = []
        self._commands = []
        if self.root is not None:
            self.root.removeNode()
            self.root = None
