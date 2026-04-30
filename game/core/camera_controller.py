from __future__ import annotations


class CameraController:
    def __init__(self, base, camera_pivot) -> None:
        self.base = base
        self.pivot = camera_pivot
        self.yaw_range = 90.0
        self.pitch_range = 45.0

    def update(self, dt: float) -> None:
        mw = self.base.mouseWatcherNode
        if mw is None or not mw.hasMouse():
            return
        mx = mw.getMouseX()
        my = mw.getMouseY()
        yaw = -mx * self.yaw_range
        pitch = my * self.pitch_range
        self.pivot.setHpr(yaw, pitch, 0)
