from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from direct.gui.DirectGui import DirectButton, DirectFrame, DirectLabel, OnscreenImage
from panda3d.core import NodePath, SamplerState, TextNode, TransparencyAttrib, Vec4

from game.core.horse_roster import ROSTER, HorseSpecies, Rarity
from game.core.photo_manager import PHOTOS_DIR
from game.scenes.base_scene import BaseScene


BG_COLOR = Vec4(0.10, 0.10, 0.10, 1.0)
PANEL_COLOR = Vec4(0.12, 0.12, 0.12, 1.0)
PANEL_DIM = Vec4(0.08, 0.08, 0.08, 1.0)
SILHOUETTE = Vec4(0.18, 0.18, 0.20, 1.0)
TEXT_COLOR = Vec4(0.91, 0.91, 0.81, 1.0)
DIM_TEXT = Vec4(0.50, 0.50, 0.50, 1.0)
ACCENT = Vec4(0.353, 0.710, 0.322, 1.0)

RARITY_COLORS = {
    Rarity.COMMON: Vec4(0.70, 0.70, 0.70, 1.0),
    Rarity.RARE: Vec4(0.40, 0.65, 0.95, 1.0),
    Rarity.LEGENDARY: Vec4(0.95, 0.78, 0.30, 1.0),
}
RARITY_LABELS = {
    Rarity.COMMON: "COMMON",
    Rarity.RARE: "RARE",
    Rarity.LEGENDARY: "LEGENDARY",
}

CARD_W = 0.34
CARD_H = 0.36
CARD_SPACING_X = 0.08
CARD_SPACING_Y = 0.06
COLS = 4


def _scan_captured() -> Dict[str, Path]:
    """species_id -> oldest sidecar's PNG that captured it (first in dex)."""
    out: Dict[str, Path] = {}
    if not PHOTOS_DIR.exists():
        return out
    for sidecar in sorted(PHOTOS_DIR.glob("*.json")):
        try:
            data = json.loads(sidecar.read_text())
            ids = data.get("species_ids") or []
        except Exception:
            continue
        png = sidecar.with_suffix(".png")
        if not png.exists():
            continue
        for sid in ids:
            if sid not in out:
                out[sid] = png
    return out


class DexScene(BaseScene):
    name = "dex"
    cursor_visible = True

    def __init__(self, app) -> None:
        super().__init__(app)
        self.root: NodePath | None = None
        self._widgets: list = []
        self._cards: list = []
        self._focus_idx: int = 0
        self._accept_keys: list = []
        self._fullscreen_root: NodePath | None = None

    def enter(self, **kwargs) -> None:
        self.root = NodePath("dex_root")
        self.root.reparentTo(self.app.base.aspect2d)
        font = self.app.font

        DirectFrame(
            parent=self.root,
            frameColor=BG_COLOR,
            frameSize=(-2.0, 2.0, -1.5, 1.5),
        )

        DirectLabel(
            parent=self.root,
            text="DEX",
            text_fg=ACCENT,
            text_scale=0.13,
            text_font=font,
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, 0.85),
        )

        captured = _scan_captured()

        x_origin = -((COLS - 1) / 2.0) * (CARD_W + CARD_SPACING_X)
        y_origin = 0.50
        for idx, species in enumerate(ROSTER):
            r = idx // COLS
            c = idx % COLS
            x = x_origin + c * (CARD_W + CARD_SPACING_X)
            y = y_origin - r * (CARD_H + CARD_SPACING_Y)
            self._build_card(species, captured.get(species.id), x, y)

        captured_count = sum(1 for s in ROSTER if s.id in captured)
        counter = DirectLabel(
            parent=self.root,
            text=f"CAPTURED  {captured_count}/{len(ROSTER)}",
            text_fg=DIM_TEXT,
            text_scale=0.04,
            text_font=font,
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, -0.78),
        )
        self._widgets.append(counter)

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
            pos=(0, 0, -0.92),
        )
        self._widgets.append(back_btn)

        self._focus_idx = 0
        self._refresh_focus()
        self._bind_keys()

    def _build_card(self, species: HorseSpecies, photo: Optional[Path], x: float, y: float) -> None:
        font = self.app.font
        is_captured = photo is not None
        card_root = self.root.attachNewNode("card")
        card_root.setPos(x, 0, y)

        frame_idle = PANEL_COLOR if is_captured else PANEL_DIM
        btn = DirectButton(
            parent=card_root,
            text="",
            command=self._open_photo,
            extraArgs=[photo],
            frameColor=(frame_idle, ACCENT, ACCENT, frame_idle),
            frameSize=(-CARD_W / 2, CARD_W / 2, -CARD_H / 2, CARD_H / 2),
            borderWidth=(0.005, 0.005),
            relief=1,
            pos=(0, 0, 0),
        )
        self._widgets.append(btn)

        img_w = CARD_W / 2 - 0.02
        img_h = 0.10
        img_z = 0.05
        if is_captured:
            try:
                tex = self.app.base.loader.loadTexture(str(photo))
                tex.setMagfilter(SamplerState.FT_linear)
                tex.setMinfilter(SamplerState.FT_linear)
                img = OnscreenImage(
                    parent=card_root,
                    image=tex,
                    scale=(img_w, 1, img_h),
                    pos=(0, 0, img_z),
                )
                self._widgets.append(img)
            except Exception:
                pass
        else:
            sil = DirectFrame(
                parent=card_root,
                frameColor=SILHOUETTE,
                frameSize=(-img_w, img_w, img_z - img_h, img_z + img_h),
            )
            self._widgets.append(sil)

        title_text = species.name if is_captured else "???"
        title = DirectLabel(
            parent=card_root,
            text=title_text,
            text_fg=TEXT_COLOR if is_captured else DIM_TEXT,
            text_scale=0.030,
            text_font=font,
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, -0.085),
        )
        self._widgets.append(title)

        rarity = DirectLabel(
            parent=card_root,
            text=f"[{RARITY_LABELS[species.rarity]}]",
            text_fg=RARITY_COLORS[species.rarity] if is_captured else DIM_TEXT,
            text_scale=0.028,
            text_font=font,
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, -0.13),
        )
        self._widgets.append(rarity)

        self._cards.append({
            "species": species,
            "photo": photo,
            "title": title,
            "title_text": title_text,
        })

    def _bind_keys(self) -> None:
        base = self.app.base
        base.accept("arrow_left", self._focus_left)
        base.accept("arrow_right", self._focus_right)
        base.accept("arrow_up", self._focus_up)
        base.accept("arrow_down", self._focus_down)
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
            is_captured = card["photo"] is not None
            prefix = "> " if focused else "  "
            card["title"]["text"] = prefix + card["title_text"]
            if focused:
                card["title"]["text_fg"] = ACCENT
            else:
                card["title"]["text_fg"] = TEXT_COLOR if is_captured else DIM_TEXT

    def _focus_left(self) -> None:
        self._focus_step(-1)

    def _focus_right(self) -> None:
        self._focus_step(1)

    def _focus_up(self) -> None:
        self._focus_step(-COLS)

    def _focus_down(self) -> None:
        self._focus_step(COLS)

    def _focus_step(self, delta: int) -> None:
        if not self._cards:
            return
        self._focus_idx = (self._focus_idx + delta) % len(self._cards)
        self._refresh_focus()

    def _activate(self) -> None:
        if 0 <= self._focus_idx < len(self._cards):
            self._open_photo(self._cards[self._focus_idx]["photo"])

    def _open_photo(self, photo: Optional[Path]) -> None:
        if photo is None:
            return
        self.app.play_click()
        self._close_full()
        self._fullscreen_root = NodePath("dex_full")
        self._fullscreen_root.reparentTo(self.app.base.aspect2d)
        DirectFrame(
            parent=self._fullscreen_root,
            frameColor=Vec4(0, 0, 0, 0.95),
            frameSize=(-2.0, 2.0, -1.5, 1.5),
        )
        try:
            tex = self.app.base.loader.loadTexture(str(photo))
        except Exception:
            tex = None
        if tex is not None:
            img = OnscreenImage(
                parent=self._fullscreen_root,
                image=tex,
                scale=(1.4, 1, 0.8),
                pos=(0, 0, 0.05),
            )
            img.setTransparency(TransparencyAttrib.MAlpha)
        DirectButton(
            parent=self._fullscreen_root,
            text="CLOSE",
            command=self._close_full,
            text_fg=TEXT_COLOR,
            text_scale=0.05,
            text_font=self.app.font,
            frameColor=(PANEL_COLOR, ACCENT, ACCENT, PANEL_COLOR),
            frameSize=(-0.20, 0.20, -0.07, 0.07),
            borderWidth=(0.005, 0.005),
            relief=1,
            pos=(0, 0, -0.85),
        )

    def _close_full(self) -> None:
        if self._fullscreen_root is not None:
            self.app.play_click()
            self._fullscreen_root.removeNode()
            self._fullscreen_root = None

    def _back(self) -> None:
        if self._fullscreen_root is not None:
            self._close_full()
            return
        self.app.play_click()
        self.app.scene_manager.switch("main_menu")

    def exit(self) -> None:
        for k in self._accept_keys:
            self.app.base.ignore(k)
        self._accept_keys = []
        self._close_full()
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
