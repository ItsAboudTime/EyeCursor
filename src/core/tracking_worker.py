from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Signal, Slot

from src.core.modes.base import TrackingMode


class TrackingWorker(QObject):
    status_changed = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(
        self,
        mode: TrackingMode,
        calibrations: Dict[str, Optional[dict]],
        cameras: List[int],
        cursor,
        settings: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._calibrations = calibrations
        self._cameras = cameras
        self._cursor = cursor
        self._settings = settings or {}

    @Slot()
    def run(self) -> None:
        self.status_changed.emit("active")
        try:
            self._mode.start(
                self._calibrations, self._cameras, self._cursor,
                settings=self._settings,
            )
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.status_changed.emit("stopped")
            self.finished.emit()
