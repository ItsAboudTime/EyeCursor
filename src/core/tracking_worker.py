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

    @Slot(dict)
    def update_settings(self, settings: dict) -> None:
        """Forward live settings updates to the running mode.

        Called from the GUI thread while the mode loop runs on the tracking
        thread. Modes only mutate simple attributes (floats / ints / bools)
        on the cursor and gesture controller, which is safe across threads
        thanks to the GIL making each individual attribute write atomic.

        Safe to call regardless of run state: the mode is responsible for
        applying the new settings to its long-lived collaborators (cursor,
        gesture controller, gaze controller). Mode implementations should
        treat this as best-effort and never raise.
        """
        if not isinstance(settings, dict):
            return
        # Keep the latest snapshot so a subsequent restart picks up the
        # current values too.
        self._settings = dict(settings)
        try:
            self._mode.update_settings(self._settings)
        except Exception as exc:
            # Never let a settings-update failure kill the tracking loop.
            print(f"warning: mode.update_settings raised: {exc}")
