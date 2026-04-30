from __future__ import annotations

from direct.gui.DirectGui import DirectButton, DirectFrame, DirectLabel
from panda3d.core import NodePath, TextNode, Vec4

from game.core.maps import MAPS
from game.scenes.base_scene import BaseScene


BG_COLOR = Vec4(0.10, 0.10, 0.10, 1.0)
PANEL_COLOR = Vec4(0.12, 0.12, 0.12, 1.0)
TEXT_COLOR = Vec4(0.91, 0.91, 0.81, 1.0)
DIM_TEXT = Vec4(0.65, 0.65, 0.60, 1.0)
ACCENT = Vec4(0.353, 0.710, 0.322, 1.0)


CARD_W = 0.65
CARD_H = 0.55
CARD_SPACING = 0.20


class MapSelectScene(BaseScene):
    name = "map_select"
    cursor_visible = True

    def __init__(self, app) -> None:
        super().__init__(app)
        self.root: NodePath | None = None
        self._widgets: list = []
        self._cards: list = []
        self._focus_idx: int = 0
        self._accept_keys: list = []

    def enter(self, **kwargs) -> None:
        self.root = NodePath("map_select_root")
        self.root.reparentTo(self.app.base.aspect2d)
        font = self.app.font

        DirectFrame(
            parent=self.root,
            frameColor=BG_COLOR,
            frameSize=(-2.0, 2.0, -1.5, 1.5),
        )

        DirectLabel(
            parent=self.root,
            text="SELECT MAP",
            text_fg=ACCENT,
            text_scale=0.13,
            text_font=font,
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, 0.65),
        )

        n = len(MAPS)
        total_w = n * CARD_W + (n - 1) * CARD_SPACING
        x_origin = -(total_w / 2.0) + CARD_W / 2.0
        for i, m in enumerate(MAPS):
            x = x_origin + i * (CARD_W + CARD_SPACING)
            self._build_card(m, x)

        back_btn = DirectButton(
            parent=self.root,
            text="BACK",
            command=self._back,
            text_fg=TEXT_COLOR,
            text_scale=0.06,
            text_font=font,
            frameColor=(PANEL_COLOR, ACCENT, ACCENT, PANEL_COLOR),
            frameSize=(-0.20, 0.20, -0.07, 0.07),
            borderWidth=(0.005, 0.005),
            relief=1,
            pos=(0, 0, -0.85),
        )
        self._widgets.append(back_btn)

        self._focus_idx = 0
        self._refresh_focus()
        self._bind_keys()

    def _build_card(self, m, x: float) -> None:
        font = self.app.font
        card_root = self.root.attachNewNode("card")
        card_root.setPos(x, 0, 0.0)

        btn = DirectButton(
            parent=card_root,
            text="",
            command=self._pick,
            extraArgs=[m.id],
            frameColor=(PANEL_COLOR, ACCENT, ACCENT, PANEL_COLOR),
            frameSize=(-CARD_W / 2, CARD_W / 2, -CARD_H / 2, CARD_H / 2),
            borderWidth=(0.006, 0.006),
            relief=1,
            pos=(0, 0, 0),
        )
        self._widgets.append(btn)

        swatch = DirectFrame(
            parent=card_root,
            frameColor=m.swatch_color,
            frameSize=(-CARD_W / 2 + 0.04, CARD_W / 2 - 0.04, 0.04, 0.04 + 0.18),
        )
        self._widgets.append(swatch)

        title = DirectLabel(
            parent=card_root,
            text=m.title,
            text_fg=TEXT_COLOR,
            text_scale=0.045,
            text_font=font,
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, -0.05),
        )
        self._widgets.append(title)

        desc = DirectLabel(
            parent=card_root,
            text=m.description,
            text_fg=DIM_TEXT,
            text_scale=0.035,
            text_font=font,
            text_align=TextNode.ACenter,
            text_wordwrap=18,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, -0.16),
        )
        self._widgets.append(desc)

        self._cards.append({"map": m, "btn": btn, "title": title})

    def _bind_keys(self) -> None:
        base = self.app.base
        base.accept("arrow_left", self._focus_prev)
        base.accept("arrow_right", self._focus_next)
        base.accept("arrow_up", self._focus_prev)
        base.accept("arrow_down", self._focus_next)
        base.accept("enter", self._activate)
        base.accept("return", self._activate)
        self._accept_keys = [
            "arrow_left", "arrow_right", "arrow_up", "arrow_down",
            "enter", "return",
        ]

    def on_escape(self) -> None:
        self._back()

    def _refresh_focus(self) -> None:
        for i, card in enumerate(self._cards):
            focused = i == self._focus_idx
            card["title"]["text_fg"] = ACCENT if focused else TEXT_COLOR
            prefix = "> " if focused else "  "
            card["title"]["text"] = prefix + card["map"].title

    def _focus_prev(self) -> None:
        if not self._cards:
            return
        self._focus_idx = (self._focus_idx - 1) % len(self._cards)
        self._refresh_focus()

    def _focus_next(self) -> None:
        if not self._cards:
            return
        self._focus_idx = (self._focus_idx + 1) % len(self._cards)
        self._refresh_focus()

    def _activate(self) -> None:
        if 0 <= self._focus_idx < len(self._cards):
            self._pick(self._cards[self._focus_idx]["map"].id)

    def _pick(self, map_id: str) -> None:
        self.app.play_click()
        self.app.scene_manager.switch("game", map_id=map_id)

    def _back(self) -> None:
        self.app.play_click()
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
        self._cards = []
        if self.root is not None:
            self.root.removeNode()
            self.root = None
