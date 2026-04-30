from __future__ import annotations

import sys
from typing import Dict, List, Optional

from panda3d.core import (
    DynamicTextFont,
    WindowProperties,
    loadPrcFileData,
)


loadPrcFileData("", "window-title Horsin' Around")
loadPrcFileData("", "fullscreen #t")
loadPrcFileData("", "sync-video #t")
loadPrcFileData("", "show-frame-rate-meter #f")
loadPrcFileData("", "audio-library-name null")


from direct.showbase.ShowBase import ShowBase
from direct.task import Task

from game.core import asset_gen, settings
from game.core.depth_client import DepthClient
from game.scenes.base_scene import BaseScene
from game.scenes.game_scene import GameScene
from game.scenes.gallery_scene import GalleryScene
from game.scenes.main_menu import MainMenu
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
        self.current.enter()
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

        self.depth_client = DepthClient()
        self.depth_client.start()

        self.previous_scene_name: Optional[str] = None
        self.scene_manager = SceneManager(self)
        self.scene_manager.register(MainMenu(self))
        self.scene_manager.register(SettingsScreen(self))
        self.scene_manager.register(GameScene(self))
        self.scene_manager.register(PauseOverlay(self))
        self.scene_manager.register(GalleryScene(self))

        self.base.taskMgr.add(self._tick, "horsin_tick")

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

    def _tick(self, task: "Task.Task") -> int:
        dt = globalClock.getDt()
        self.scene_manager.update(dt)
        return Task.cont

    def shutdown(self) -> None:
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
