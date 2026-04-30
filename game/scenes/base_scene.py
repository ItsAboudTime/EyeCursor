from __future__ import annotations

from typing import Optional


class BaseScene:
    name: str = "base"
    cursor_visible: bool = True

    def __init__(self, app) -> None:
        self.app = app
        self.frozen: bool = False

    def enter(self) -> None:
        raise NotImplementedError

    def exit(self) -> None:
        raise NotImplementedError

    def update(self, dt: float) -> None:
        return None
