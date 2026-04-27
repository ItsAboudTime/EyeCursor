from PySide6.QtCore import QObject, Signal


class GazeSignalProxy(QObject):
    gaze_target_changed = Signal(int, int)
