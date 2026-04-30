from __future__ import annotations

import sys
from typing import Dict, List, Optional

from panda3d.core import (
    DynamicTextFont,
    GraphicsPipeSelection,
    WindowProperties,
    loadPrcFileData,
)


loadPrcFileData("", "window-title Horsin' Around")
# Borderless windowed at native desktop res. Exclusive fullscreen on X11
# triggers a videomode switch that invalidates absolute-screen pointer mappings.
loadPrcFileData("", "fullscreen #f")
loadPrcFileData("", "undecorated #t")
loadPrcFileData("", "win-origin 0 0")
loadPrcFileData("", "sync-video #t")
loadPrcFileData("", "show-frame-rate-meter #f")

_pipe = GraphicsPipeSelection.getGlobalPtr().makeDefaultPipe()
if _pipe is not None:
    _w, _h = _pipe.getDisplayWidth(), _pipe.getDisplayHeight()
    if _w > 0 and _h > 0:
        loadPrcFileData("", f"win-size {_w} {_h}")


from direct.showbase.ShowBase import ShowBase
from direct.task import Task

from game.core import asset_gen, settings
from game.core.depth_client import DepthClient
from game.scenes.base_scene import BaseScene
from game.scenes.dex_scene import DexScene
from game.scenes.game_scene import GameScene
from game.scenes.gallery_scene import GalleryScene
from game.scenes.main_menu import MainMenu
from game.scenes.map_select import MapSelectScene
from game.scenes.pause_overlay import PauseOverlay
from game.scenes.settings_screen import SettingsScreen


class SceneManager:
    def __init__(self, app: "App") -> None:
        self.app = app
        self.scenes: Dict[str, BaseScene] = {}
        self.current: Optional[BaseScene] = None
        self.overlay_stack: List[BaseScene] = []

    def register(self, scene: BaseScene) -> None:
        self.scenes[scene.name] = scene

    def _set_cursor(self, visible: bool) -> None:
        props = WindowProperties()
        props.setCursorHidden(not visible)
        try:
            self.app.base.win.requestProperties(props)
        except Exception:
            pass

    def switch(self, name: str, **kwargs) -> None:
        if name not in self.scenes:
            return
        prev_name: Optional[str] = None
        if self.overlay_stack:
            prev_name = self.overlay_stack[-1].name
        elif self.current is not None:
            prev_name = self.current.name
        while self.overlay_stack:
            self.pop_overlay()
        prev = self.current
        if prev is not None:
            prev.exit()
        self.app.previous_scene_name = prev_name
        self.current = self.scenes[name]
        self.current.enter(**kwargs)
        self._set_cursor(self.current.cursor_visible)

    def push_overlay(self, name: str) -> None:
        if name not in self.scenes:
            return
        if self.current is not None:
            self.current.frozen = True
        overlay = self.scenes[name]
        overlay.enter()
        self.overlay_stack.append(overlay)
        self._set_cursor(overlay.cursor_visible)

    def pop_overlay(self) -> None:
        if not self.overlay_stack:
            return
        top = self.overlay_stack.pop()
        top.exit()
        if self.overlay_stack:
            self._set_cursor(self.overlay_stack[-1].cursor_visible)
        else:
            if self.current is not None:
                self.current.frozen = False
                self._set_cursor(self.current.cursor_visible)

    def swap_overlay(self, name: str) -> None:
        # Replace the topmost overlay without unfreezing self.current. Used by
        # pause↔settings so the underlying GameScene survives the round-trip.
        if name not in self.scenes:
            return
        if not self.overlay_stack:
            self.push_overlay(name)
            return
        top = self.overlay_stack.pop()
        top.exit()
        overlay = self.scenes[name]
        overlay.enter()
        self.overlay_stack.append(overlay)
        self._set_cursor(overlay.cursor_visible)

    def dispatch_escape(self) -> None:
        # Single source of truth for ESC. Top overlay wins so e.g. settings
        # overlay's BACK fires instead of GameScene re-pushing pause on top.
        if self.overlay_stack:
            self.overlay_stack[-1].on_escape()
            return
        if self.current is not None:
            self.current.on_escape()

    def update(self, dt: float) -> None:
        if self.current is not None:
            self.current.update(dt)
        for ov in self.overlay_stack:
            ov.update(dt)


class App:
    def __init__(self) -> None:
        asset_gen.ensure_assets()

        self.base = ShowBase()
        self.base.setBackgroundColor(0.10, 0.10, 0.10, 1.0)
        self.base.disableMouse()

        self.config = settings.load()
        settings.save(self.config)

        self.font = self._load_font()

        self._ui_click = self._load_sfx("ui_click.ogg")
        self._ambient_music = self._load_music("ambient.ogg")
        self._apply_volumes()
        if self._ambient_music is not None:
            self._ambient_music.setLoop(True)
            self._ambient_music.play()

        self.depth_client = DepthClient()
        self.depth_client.start()

        self.previous_scene_name: Optional[str] = None
        self.scene_manager = SceneManager(self)
        self.scene_manager.register(MainMenu(self))
        self.scene_manager.register(SettingsScreen(self))
        self.scene_manager.register(GameScene(self))
        self.scene_manager.register(PauseOverlay(self))
        self.scene_manager.register(GalleryScene(self))
        self.scene_manager.register(MapSelectScene(self))
        self.scene_manager.register(DexScene(self))

        self.base.taskMgr.add(self._tick, "horsin_tick")

        # ESC is bound here exactly once and never re-bound, so per-scene
        # accept('escape', ...) calls can never overwrite each other. Scenes
        # opt in by overriding `on_escape`.
        self.base.accept("escape", self.scene_manager.dispatch_escape)

        self.scene_manager.switch("main_menu")

    def _load_font(self):
        fp = asset_gen.font_path()
        if fp is None:
            return None
        try:
            font = DynamicTextFont(str(fp))
            font.setPixelsPerUnit(64)
            font.setMinfilter(2)
            font.setMagfilter(0)
            return font
        except Exception:
            return None

    def _load_sfx(self, name: str):
        p = asset_gen.audio_path(name)
        if p is None:
            return None
        try:
            return self.base.loader.loadSfx(str(p))
        except Exception:
            return None

    def _load_music(self, name: str):
        p = asset_gen.audio_path(name)
        if p is None:
            return None
        try:
            return self.base.loader.loadMusic(str(p))
        except Exception:
            return None

    def _apply_volumes(self) -> None:
        sfx_v = float(self.config.get("sfx_volume", 0.75))
        music_v = float(self.config.get("music_volume", 0.5))
        try:
            for mgr in self.base.sfxManagerList:
                mgr.setVolume(sfx_v)
        except Exception:
            pass
        try:
            if self.base.musicManager is not None:
                self.base.musicManager.setVolume(music_v)
        except Exception:
            pass

    def play_click(self) -> None:
        if self._ui_click is None:
            return
        try:
            self._ui_click.play()
        except Exception:
            pass

    def _tick(self, task: "Task.Task") -> int:
        dt = globalClock.getDt()
        self.scene_manager.update(dt)
        return Task.cont

    def shutdown(self) -> None:
        try:
            if self._ambient_music is not None:
                self._ambient_music.stop()
        except Exception:
            pass
        try:
            self.depth_client.stop()
        except Exception:
            pass
        try:
            self.base.userExit()
        except Exception:
            sys.exit(0)

    def run(self) -> None:
        self.base.run()


def main() -> None:
    app = App()
    app.run()


if __name__ == "__main__":
    main()
