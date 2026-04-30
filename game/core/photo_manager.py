from __future__ import annotations

import datetime as dt
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

from panda3d.core import (
    CardMaker,
    Filename,
    LineSegs,
    NodePath,
    SamplerState,
    TransparencyAttrib,
    Vec4,
)


PHOTOS_DIR = Path(__file__).resolve().parent.parent / "photos"

FOV_MIN = 30.0
FOV_MAX = 90.0
FOV_DEFAULT = 90.0
DELTA_MAX = 0.2
FLASH_DURATION = 0.1
FOV_RECOVER_DURATION = 0.3
COOLDOWN = 0.5

PREVIEW_HEIGHT_HALF = 0.16
PREVIEW_MARGIN = 0.06
PREVIEW_SLIDE_IN = 0.3
PREVIEW_HOLD = 2.5
PREVIEW_SLIDE_OUT = 0.4
PREVIEW_BORDER_RGBA = (0.35, 0.71, 0.32, 1.0)


class PhotoState(Enum):
    IDLE = "idle"
    HOLDING = "holding"
    FLASHING = "flashing"
    COOLDOWN = "cooldown"


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def list_photos() -> List[Path]:
    if not PHOTOS_DIR.exists():
        return []
    paths = sorted(PHOTOS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


class PhotoManager:
    def __init__(
        self,
        base,
        countdown_duration: float,
        get_depth: Callable[[], Optional[float]],
        hud_root: NodePath,
        progress_setter: Callable[[float], None],
        get_paused: Callable[[], bool],
    ) -> None:
        self.base = base
        self.countdown_duration = float(countdown_duration)
        self.get_depth = get_depth
        self.hud_root = hud_root
        self.progress_setter = progress_setter
        self.get_paused = get_paused

        self.state = PhotoState.IDLE
        self.elapsed = 0.0
        self.cooldown_left = 0.0
        self.flash_left = 0.0
        self.fov_recover_left = 0.0
        self.current_fov = FOV_DEFAULT
        self._trigger_held = False

        self._baseline_depth: Optional[float] = None

        self.preview_node: Optional[NodePath] = None
        self.preview_state = "hidden"
        self.preview_elapsed = 0.0
        self._preview_rest_z = 0.0
        self._preview_off_z = 0.0
        self._preview_x = 0.0

        PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

        cm = CardMaker("photo_flash")
        cm.setFrameFullscreenQuad()
        self.flash_card = NodePath(cm.generate())
        self.flash_card.reparentTo(base.render2d)
        self.flash_card.setColor(Vec4(1, 1, 1, 0))
        self.flash_card.setTransparency(TransparencyAttrib.MAlpha)
        self.flash_card.hide()

    def set_trigger_held(self, held: bool) -> None:
        self._trigger_held = held

    def _depth_to_fov(self) -> float:
        cur = self.get_depth()
        if cur is None or self._baseline_depth is None:
            return FOV_MAX
        delta = abs(self._baseline_depth) - abs(cur)
        t = _clamp(delta / DELTA_MAX, 0.0, 1.0)
        return _lerp(FOV_MAX, FOV_MIN, t)

    def _capture(self) -> Optional[Path]:
        was_hidden = self.hud_root.isHidden()
        self.hud_root.hide()
        flash_was_hidden = self.flash_card.isHidden()
        self.flash_card.hide()
        saved_path: Optional[Path] = None
        try:
            self.base.graphicsEngine.renderFrame()
            ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = PHOTOS_DIR / f"{ts}.png"
            try:
                self.base.win.saveScreenshot(Filename.fromOsSpecific(str(path)))
                saved_path = path
            except Exception as e:
                print(f"[photo_manager] save failed: {e}")
        finally:
            if not was_hidden:
                self.hud_root.show()
            if not flash_was_hidden:
                self.flash_card.show()
        return saved_path

    def _dispose_preview(self) -> None:
        if self.preview_node is not None and not self.preview_node.isEmpty():
            self.preview_node.removeNode()
        self.preview_node = None
        self.preview_state = "hidden"
        self.preview_elapsed = 0.0

    def _show_preview(self, path: Path) -> None:
        self._dispose_preview()
        try:
            tex = self.base.loader.loadTexture(Filename.fromOsSpecific(str(path)))
        except Exception:
            return
        if tex is None:
            return

        tex.setMagfilter(SamplerState.FT_linear)
        tex.setMinfilter(SamplerState.FT_linear)

        h = PREVIEW_HEIGHT_HALF
        tw, th = tex.getXSize(), tex.getYSize()
        if tw > 0 and th > 0:
            w = h * (tw / th)
        else:
            w = h * (16.0 / 9.0)

        try:
            aspect = float(self.base.getAspectRatio())
        except Exception:
            aspect = 16.0 / 9.0
        if aspect <= 0.0:
            aspect = 16.0 / 9.0

        pivot = NodePath("photo_preview")
        pivot.reparentTo(self.hud_root)

        cm = CardMaker("preview_card")
        cm.setFrame(-w, w, -h, h)
        card = pivot.attachNewNode(cm.generate())
        card.setTexture(tex)

        ls = LineSegs()
        ls.setThickness(2.5)
        ls.setColor(*PREVIEW_BORDER_RGBA)
        ls.moveTo(-w, 0, -h)
        ls.drawTo(w, 0, -h)
        ls.drawTo(w, 0, h)
        ls.drawTo(-w, 0, h)
        ls.drawTo(-w, 0, -h)
        pivot.attachNewNode(ls.create())

        self._preview_x = aspect - PREVIEW_MARGIN - w
        self._preview_rest_z = -1.0 + PREVIEW_MARGIN + h
        self._preview_off_z = -1.0 - h - 0.05

        pivot.setPos(self._preview_x, 0, self._preview_off_z)

        self.preview_node = pivot
        self.preview_state = "sliding_in"
        self.preview_elapsed = 0.0

    def _tick_preview(self, dt_: float) -> None:
        if self.preview_node is None or self.preview_state == "hidden":
            return
        self.preview_elapsed += dt_
        rest_z = self._preview_rest_z
        off_z = self._preview_off_z

        if self.preview_state == "sliding_in":
            t = _clamp(self.preview_elapsed / PREVIEW_SLIDE_IN, 0.0, 1.0)
            ease = 1.0 - (1.0 - t) * (1.0 - t)
            z = _lerp(off_z, rest_z, ease)
            self.preview_node.setPos(self._preview_x, 0, z)
            if t >= 1.0:
                self.preview_state = "showing"
                self.preview_elapsed = 0.0
        elif self.preview_state == "showing":
            self.preview_node.setPos(self._preview_x, 0, rest_z)
            if self.preview_elapsed >= PREVIEW_HOLD:
                self.preview_state = "sliding_out"
                self.preview_elapsed = 0.0
        elif self.preview_state == "sliding_out":
            t = _clamp(self.preview_elapsed / PREVIEW_SLIDE_OUT, 0.0, 1.0)
            ease = t * t
            z = _lerp(rest_z, off_z, ease)
            self.preview_node.setPos(self._preview_x, 0, z)
            if t >= 1.0:
                self._dispose_preview()

    def _enter_flashing(self) -> None:
        self.state = PhotoState.FLASHING
        self.flash_left = FLASH_DURATION
        self.fov_recover_left = FOV_RECOVER_DURATION
        self.flash_card.setColor(Vec4(1, 1, 1, 1))
        self.flash_card.show()
        self.progress_setter(0.0)
        self._baseline_depth = None

    def _enter_cooldown(self) -> None:
        self.state = PhotoState.COOLDOWN
        self.cooldown_left = COOLDOWN
        self.flash_card.hide()
        self.flash_card.setColor(Vec4(1, 1, 1, 0))

    def tick(self, dt_: float) -> None:
        if self.get_paused():
            return

        self._tick_preview(dt_)

        if self.state is PhotoState.IDLE:
            self.current_fov = _lerp(self.current_fov, FOV_DEFAULT, dt_ * 5.0)
            self.base.camLens.setFov(self.current_fov)
            if self._trigger_held:
                self.state = PhotoState.HOLDING
                self.elapsed = 0.0
                self._baseline_depth = self.get_depth()
                self.progress_setter(0.0)
                print(f"[photo] hold start: baseline_depth={self._baseline_depth}", flush=True)
            else:
                self.progress_setter(0.0)
            return

        if self.state is PhotoState.HOLDING:
            self.elapsed += dt_
            progress = _clamp(self.elapsed / max(self.countdown_duration, 1e-3), 0.0, 1.0)
            self.progress_setter(progress)
            if self._baseline_depth is None:
                cur = self.get_depth()
                if cur is not None:
                    self._baseline_depth = cur
                    print(f"[photo] lazy baseline captured: {cur}", flush=True)
            target = self._depth_to_fov()
            self.current_fov = _lerp(self.current_fov, target, _clamp(dt_ * 12.0, 0.0, 1.0))
            self.base.camLens.setFov(self.current_fov)
            if self.elapsed >= self.countdown_duration:
                cur = self.get_depth()
                print(
                    f"[photo] firing: baseline={self._baseline_depth} cur={cur} fov={self.current_fov:.1f}",
                    flush=True,
                )
                saved_path = self._capture()
                self._enter_flashing()
                if saved_path is not None:
                    self._show_preview(saved_path)
            return

        if self.state is PhotoState.FLASHING:
            self.flash_left = max(0.0, self.flash_left - dt_)
            alpha = _clamp(self.flash_left / FLASH_DURATION, 0.0, 1.0)
            self.flash_card.setColor(Vec4(1, 1, 1, alpha))
            self.fov_recover_left = max(0.0, self.fov_recover_left - dt_)
            recover_t = 1.0 - _clamp(self.fov_recover_left / FOV_RECOVER_DURATION, 0.0, 1.0)
            self.current_fov = _lerp(self.current_fov, FOV_DEFAULT, recover_t)
            self.base.camLens.setFov(self.current_fov)
            if self.flash_left <= 0.0 and self.fov_recover_left <= 0.0:
                self._enter_cooldown()
            return

        if self.state is PhotoState.COOLDOWN:
            self.cooldown_left = max(0.0, self.cooldown_left - dt_)
            self.current_fov = _lerp(self.current_fov, FOV_DEFAULT, dt_ * 5.0)
            self.base.camLens.setFov(self.current_fov)
            if self.cooldown_left <= 0.0:
                self.state = PhotoState.IDLE
            return

    def cleanup(self) -> None:
        self._dispose_preview()
        if not self.flash_card.isEmpty():
            self.flash_card.removeNode()
