from PySide6.QtCore import QObject, Signal


class VisualizationSignalProxy(QObject):
    frame_ready = Signal(dict)
