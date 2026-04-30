from typing import Dict, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


_LANDMARK_INDICES: Dict[str, int] = {
    "front": 1,
    "top": 10,
    "bottom": 152,
    "left": 234,
    "right": 454,
}

_FACE_TRIANGLES = [
    ("top", "left", "front"),
    ("top", "right", "front"),
    ("front", "left", "bottom"),
    ("front", "right", "bottom"),
]

_EDGES = [
    ("top", "left"),
    ("top", "right"),
    ("left", "bottom"),
    ("right", "bottom"),
    ("top", "front"),
    ("left", "front"),
    ("right", "front"),
    ("bottom", "front"),
    ("left", "right"),
]

_AXIS_FLIP = np.array([1.0, -1.0, -1.0], dtype=np.float64)
_EMA_ALPHA = 0.25


class HeadView3D(QtWidgets.QWidget):
    """5-vertex head model rendered with QPainter.

    Adapted from src/examples/two_camera_head_visualizer.py so the production
    `points_3d` dict (keyed by MediaPipe landmark indices) can drive it directly
    without re-running stereo triangulation.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(240, 240)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(20, 22, 28))
        self.setPalette(palette)

        self._smoothed: Optional[Dict[str, np.ndarray]] = None
        self._depth: Optional[float] = None
        self._status_message = "Waiting for stereo..."

    def update_points(
        self,
        points_3d: Optional[Dict[int, np.ndarray]],
        depth: Optional[float] = None,
    ) -> None:
        if not points_3d:
            self._status_message = "Waiting for face detection..."
            self.update()
            return

        named: Dict[str, np.ndarray] = {}
        for name, idx in _LANDMARK_INDICES.items():
            pt = points_3d.get(idx)
            if pt is None:
                self._status_message = "Waiting for face detection..."
                self.update()
                return
            named[name] = np.asarray(pt, dtype=np.float64) * _AXIS_FLIP

        if self._smoothed is None:
            self._smoothed = named
        else:
            self._smoothed = {
                k: _EMA_ALPHA * named[k] + (1.0 - _EMA_ALPHA) * self._smoothed[k]
                for k in named
            }
        self._depth = depth
        self.update()

    def reset(self) -> None:
        self._smoothed = None
        self._depth = None
        self._status_message = "Waiting for stereo..."
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect()

        if self._smoothed is None:
            self._draw_status(painter, rect, self._status_message)
            painter.end()
            return

        named_points = self._smoothed
        cx = rect.width() / 2.0
        cy = rect.height() / 2.0
        focal = min(rect.width(), rect.height() * 1.6) * 0.9

        all_points = np.stack(list(named_points.values()))
        centroid = all_points.mean(axis=0)
        view_z = float(centroid[2])
        if view_z < 0.05:
            view_z = 0.05

        def project(point: np.ndarray) -> Tuple[QtCore.QPointF, float]:
            x = float(point[0]) - float(centroid[0])
            y = float(point[1]) - float(centroid[1])
            z = float(point[2])
            sx = cx + focal * x / view_z
            sy = cy + focal * y / view_z
            return QtCore.QPointF(sx, sy), z

        light_dir = np.array([0.35, 0.45, 1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)
        to_light = -light_dir

        triangles = []
        for a_name, b_name, c_name in _FACE_TRIANGLES:
            a = named_points[a_name]
            b = named_points[b_name]
            c = named_points[c_name]
            normal = np.cross(b - a, c - a)
            norm_len = float(np.linalg.norm(normal))
            if norm_len < 1e-9:
                continue
            normal = normal / norm_len
            if normal[2] > 0.0:
                normal = -normal
            shade = float(np.dot(normal, to_light))
            shade = max(0.3, min(1.0, shade))
            avg_z = float((a[2] + b[2] + c[2]) / 3.0)
            triangles.append((avg_z, [a, b, c], shade))

        triangles.sort(key=lambda item: -item[0])

        skin_base = np.array([238.0, 198.0, 168.0])
        for _, verts, shade in triangles:
            poly = QtGui.QPolygonF([project(v)[0] for v in verts])
            color = (skin_base * shade).clip(0, 255).astype(int)
            painter.setBrush(QtGui.QColor(int(color[0]), int(color[1]), int(color[2])))
            painter.setPen(QtGui.QPen(QtGui.QColor(40, 30, 30, 180), 1))
            painter.drawPolygon(poly)

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 110), 1, QtCore.Qt.DashLine))
        for a_name, b_name in _EDGES:
            pa, _ = project(named_points[a_name])
            pb, _ = project(named_points[b_name])
            painter.drawLine(pa, pb)

        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        for name, point in named_points.items():
            sp, _ = project(point)
            painter.setBrush(QtGui.QColor(230, 70, 70))
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
            painter.drawEllipse(sp, 5.0, 5.0)
            painter.setPen(QtGui.QColor(255, 215, 100))
            painter.drawText(
                QtCore.QPointF(sp.x() + 8.0, sp.y() - 6.0),
                f"{name}",
            )

        painter.setPen(QtGui.QColor(220, 220, 220))
        font.setPointSize(11)
        painter.setFont(font)
        if self._depth is not None:
            painter.drawText(
                QtCore.QPointF(15.0, 25.0),
                f"Depth (Z): {self._depth:.3f} m",
            )

        painter.end()

    @staticmethod
    def _draw_status(
        painter: QtGui.QPainter,
        rect: QtCore.QRect,
        message: str,
    ) -> None:
        painter.setPen(QtGui.QColor(200, 200, 200))
        font = painter.font()
        font.setPointSize(13)
        painter.setFont(font)
        painter.drawText(rect, QtCore.Qt.AlignCenter, message)
