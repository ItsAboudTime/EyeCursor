from __future__ import annotations

from typing import Optional


class BaseScene:
    name: str = "base"
    cursor_visible: bool = True

    def __init__(self, app) -> None:
        self.app = app
        self.frozen: bool = False

    def enter(self, **kwargs) -> None:
        raise NotImplementedError

    def exit(self) -> None:
        raise NotImplementedError

    def update(self, dt: float) -> None:
        return None

    def on_escape(self) -> None:
        # Override to react to ESC. Routed by SceneManager — top overlay first,
        # then current scene. Default is no-op so leaf scenes (e.g. main menu)
        # can simply ignore the key.
        return None
