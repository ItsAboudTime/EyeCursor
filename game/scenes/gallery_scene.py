from __future__ import annotations

from pathlib import Path
from typing import List

from direct.gui.DirectGui import DirectButton, DirectFrame, DirectLabel, OnscreenImage
from panda3d.core import NodePath, SamplerState, TextNode, TransparencyAttrib, Vec4

from game.core.photo_manager import list_photos
from game.scenes.base_scene import BaseScene


BG_COLOR = Vec4(0.10, 0.10, 0.10, 1.0)
PANEL_COLOR = Vec4(0.12, 0.12, 0.12, 1.0)
TEXT_COLOR = Vec4(0.91, 0.91, 0.81, 1.0)
ACCENT = Vec4(0.353, 0.710, 0.322, 1.0)

THUMB_W = 0.40
THUMB_H = 0.24
THUMB_SPACING_X = 0.10
THUMB_SPACING_Y = 0.10


class GalleryScene(BaseScene):
    name = "gallery"
    cursor_visible = True

    def __init__(self, app) -> None:
        super().__init__(app)
        self.root: NodePath | None = None
        self._widgets: list = []
        self._fullscreen_widget = None
        self._fullscreen_root: NodePath | None = None
        self._photos: List[Path] = []
        self._accept_keys: list = []

    def enter(self) -> None:
        self.root = NodePath("gallery_root")
        self.root.reparentTo(self.app.base.aspect2d)
        font = self.app.font

        DirectFrame(
            parent=self.root,
            frameColor=BG_COLOR,
            frameSize=(-2.0, 2.0, -1.5, 1.5),
        )

        DirectLabel(
            parent=self.root,
            text="GALLERY",
            text_fg=ACCENT,
            text_scale=0.10,
            text_font=font,
            text_align=TextNode.ACenter,
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, 0.85),
        )

        back = DirectButton(
            parent=self.root,
            text="BACK",
            command=self._back,
            text_fg=TEXT_COLOR,
            text_scale=0.05,
            text_font=font,
            frameColor=(PANEL_COLOR, ACCENT, ACCENT, PANEL_COLOR),
            frameSize=(-0.18, 0.18, -0.07, 0.07),
            borderWidth=(0.005, 0.005),
            relief=1,
            pos=(-1.10, 0, 0.85),
        )
        self._widgets.append(back)

        self._photos = list_photos()

        if not self._photos:
            DirectLabel(
                parent=self.root,
                text="NO PHOTOS YET",
                text_fg=TEXT_COLOR,
                text_scale=0.08,
                text_font=font,
                text_align=TextNode.ACenter,
                frameColor=(0, 0, 0, 0),
                pos=(0, 0, 0.0),
            )
        else:
            self._build_grid()

        self.app.base.accept("escape", self._back)
        self._accept_keys.append("escape")

    def _build_grid(self) -> None:
        cols = 3
        x_origin = -(cols - 1) / 2.0 * (THUMB_W + THUMB_SPACING_X)
        y_origin = 0.55
        for i, p in enumerate(self._photos[:24]):
            r = i // cols
            c = i % cols
            x = x_origin + c * (THUMB_W + THUMB_SPACING_X)
            y = y_origin - r * (THUMB_H + THUMB_SPACING_Y)
            self._make_thumb(p, x, y)

    def _make_thumb(self, path: Path, x: float, y: float) -> None:
        font = self.app.font
        tex = None
        try:
            tex = self.app.base.loader.loadTexture(str(path))
            tex.setMagfilter(SamplerState.FT_linear)
            tex.setMinfilter(SamplerState.FT_linear)
        except Exception:
            tex = None
        btn = DirectButton(
            parent=self.root,
            text="",
            command=self._open_full,
            extraArgs=[path],
            frameColor=(PANEL_COLOR, ACCENT, ACCENT, PANEL_COLOR),
            frameSize=(-THUMB_W / 2, THUMB_W / 2, -THUMB_H / 2, THUMB_H / 2),
            borderWidth=(0.004, 0.004),
            relief=1,
            pos=(x, 0, y),
        )
        if tex is not None:
            img = OnscreenImage(
                parent=btn,
                image=tex,
                scale=(THUMB_W / 2 - 0.01, 1, THUMB_H / 2 - 0.01),
                pos=(0, 0, 0),
            )
            self._widgets.append(img)
        else:
            DirectLabel(
                parent=btn,
                text=path.name,
                text_fg=TEXT_COLOR,
                text_scale=0.03,
                text_font=font,
                frameColor=(0, 0, 0, 0),
                pos=(0, 0, 0),
            )
        self._widgets.append(btn)

    def _open_full(self, path: Path) -> None:
        self._close_full()
        self._fullscreen_root = NodePath("gallery_full")
        self._fullscreen_root.reparentTo(self.app.base.aspect2d)
        DirectFrame(
            parent=self._fullscreen_root,
            frameColor=Vec4(0, 0, 0, 0.95),
            frameSize=(-2.0, 2.0, -1.5, 1.5),
        )
        try:
            tex = self.app.base.loader.loadTexture(str(path))
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
            self._fullscreen_widget = img
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
            self._fullscreen_root.removeNode()
            self._fullscreen_root = None
            self._fullscreen_widget = None

    def _back(self) -> None:
        if self._fullscreen_root is not None:
            self._close_full()
            return
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
        if self.root is not None:
            self.root.removeNode()
            self.root = None
