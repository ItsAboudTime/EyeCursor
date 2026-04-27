"""
Stereo head-depth visualizer.

Reuses the triangulation pipeline from two_camera_final.py but skips cursor
control and gestures. Displays:
  - top: a 3D head built from the 5 triangulated landmarks (nose, forehead,
    chin, left side, right side), shaded and rendered with perspective so the
    user's distance from the cameras shows up as size + parallax.
  - bottom third: a rolling depth-vs-time plot of the nose-Z value (sign-
    flipped to natural convention so Z > 0 = farther from camera).

Fullscreen PySide6 window. Press Esc or Q to quit.
"""

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from src.face_tracking.pipelines.stereo_face_analysis import (
    StereoCalibration,
    StereoTriangulator,
)
from src.face_tracking.providers.face_landmarks import FaceLandmarksProvider


LEFT_CAMERA_INDEX = 4
RIGHT_CAMERA_INDEX = 6

K1 = np.array([
    [542.975661, 0.000000, 347.621721],
    [0.000000, 542.580855, 266.597383],
    [0.000000, 0.000000, 1.000000],
], dtype=np.float64)

D1 = np.array([
    [0.139821, -0.092846, 0.013859, 0.018398, -1.438426],
], dtype=np.float64)

K2 = np.array([
    [550.591389, 0.000000, 354.946646],
    [0.000000, 547.426744, 257.201464],
    [0.000000, 0.000000, 1.000000],
], dtype=np.float64)

D2 = np.array([
    [0.061811, 0.096200, 0.009729, 0.016548, -0.325093],
], dtype=np.float64)

R = np.array([
    [0.999955, -0.009529, -0.000044],
    [0.009528, 0.999776, 0.018882],
    [-0.000136, -0.018881, 0.999822],
], dtype=np.float64)

T = np.array([
    [-0.078688],
    [-0.000478],
    [0.004312],
], dtype=np.float64)

EMA_ALPHA = 0.08
DEPTH_WINDOW_SECONDS = 30.0

LANDMARK_INDICES: Dict[str, int] = {
    "front": 1,
    "top": 10,
    "bottom": 152,
    "left": 234,
    "right": 454,
}
ALL_INDICES = sorted(LANDMARK_INDICES.values())

# Triangles that approximate the front-facing surface of a head from the 5 vertices.
FACE_TRIANGLES = [
    ("top", "left", "front"),
    ("top", "right", "front"),
    ("front", "left", "bottom"),
    ("front", "right", "bottom"),
]

EDGES = [
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


@dataclass
class Sample:
    timestamp: float
    points_3d: Dict[int, np.ndarray]
    depth: float


class TrackingWorker(QtCore.QObject):
    sample_ready = QtCore.Signal(object)
    error = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    @QtCore.Slot()
    def run(self) -> None:
        left_camera = cv2.VideoCapture(LEFT_CAMERA_INDEX)
        right_camera = cv2.VideoCapture(RIGHT_CAMERA_INDEX)

        if not left_camera.isOpened() or not right_camera.isOpened():
            self.error.emit(
                "Could not open stereo cameras "
                f"(left={LEFT_CAMERA_INDEX}, right={RIGHT_CAMERA_INDEX})."
            )
            left_camera.release()
            right_camera.release()
            self.finished.emit()
            return

        try:
            calibration = StereoCalibration(k1=K1, d1=D1, k2=K2, d2=D2, r=R, t=T)
            triangulator = StereoTriangulator(
                calibration=calibration,
                landmark_indices=ALL_INDICES,
            )
            left_provider = FaceLandmarksProvider()
            right_provider = FaceLandmarksProvider()
        except Exception as exc:
            self.error.emit(f"Failed to initialize stereo pipeline: {exc}")
            left_camera.release()
            right_camera.release()
            self.finished.emit()
            return

        ema_depth: Optional[float] = None
        front_index = LANDMARK_INDICES["front"]

        try:
            while not self._stop_event.is_set():
                ok_left, left_frame = left_camera.read()
                ok_right, right_frame = right_camera.read()
                if not ok_left or not ok_right:
                    continue

                left_rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
                right_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)

                left_obs = left_provider.get_primary_face_observation(left_rgb)
                right_obs = right_provider.get_primary_face_observation(right_rgb)
                if left_obs is None or right_obs is None:
                    continue

                points_3d = triangulator.triangulate_from_landmarks(
                    left_landmarks=left_obs.landmarks,
                    right_landmarks=right_obs.landmarks,
                    left_frame_width=left_frame.shape[1],
                    left_frame_height=left_frame.shape[0],
                    right_frame_width=right_frame.shape[1],
                    right_frame_height=right_frame.shape[0],
                )
                if not points_3d or front_index not in points_3d:
                    continue

                # This calibration produces negative Z and Y-up for points in front
                # of the cameras. Flip both to the natural convention:
                #   Y > 0 = down (matches screen coords)
                #   Z > 0 = farther from the camera
                # Net flip is a 180° rotation about X (det = +1), so handedness
                # is preserved and Umeyama still gives a proper rotation.
                axis_flip = np.array([1.0, -1.0, -1.0], dtype=np.float64)
                snapshot = {
                    int(idx): np.asarray(pt, dtype=np.float64) * axis_flip
                    for idx, pt in points_3d.items()
                }
                raw_depth = float(snapshot[front_index][2])
                if ema_depth is None:
                    ema_depth = raw_depth
                else:
                    ema_depth = EMA_ALPHA * raw_depth + (1.0 - EMA_ALPHA) * ema_depth

                self.sample_ready.emit(
                    Sample(
                        timestamp=time.monotonic(),
                        points_3d=snapshot,
                        depth=float(ema_depth),
                    )
                )
        finally:
            try:
                left_provider.release()
            finally:
                right_provider.release()
            left_camera.release()
            right_camera.release()
            self.finished.emit()


class HeadView3D(QtWidgets.QWidget):
    """Renders the 5-vertex head with QPainter using a perspective projection."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(20, 22, 28))
        self.setPalette(palette)
        self._sample: Optional[Sample] = None

    def update_sample(self, sample: Sample) -> None:
        self._sample = sample
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect()

        if self._sample is None:
            self._draw_status(painter, rect, "Waiting for stereo cameras...")
            painter.end()
            return

        named_points: Dict[str, np.ndarray] = {}
        for name, idx in LANDMARK_INDICES.items():
            point = self._sample.points_3d.get(idx)
            if point is None:
                self._draw_status(painter, rect, "Waiting for face detection...")
                painter.end()
                return
            named_points[name] = point

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

        # Light traveling direction in camera coords (Y is +down, Z is +away).
        # Source: above-left of the user, pointing into the scene.
        light_dir = np.array([0.35, 0.45, 1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)
        to_light = -light_dir

        triangles = []
        for a_name, b_name, c_name in FACE_TRIANGLES:
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
        for a_name, b_name in EDGES:
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
                f"{name} ({LANDMARK_INDICES[name]})",
            )

        painter.setPen(QtGui.QColor(220, 220, 220))
        font.setPointSize(11)
        painter.setFont(font)
        painter.drawText(
            QtCore.QPointF(15.0, 25.0),
            f"Depth (Z, nose): {self._sample.depth:.3f} m",
        )

        painter.end()

    @staticmethod
    def _draw_status(painter: QtGui.QPainter, rect: QtCore.QRect, message: str) -> None:
        painter.setPen(QtGui.QColor(200, 200, 200))
        font = painter.font()
        font.setPointSize(13)
        painter.setFont(font)
        painter.drawText(rect, QtCore.Qt.AlignCenter, message)


class DepthPlot(QtWidgets.QWidget):
    """Rolling depth-vs-time plot using matplotlib's Qt backend."""

    def __init__(
        self,
        window_seconds: float = DEPTH_WINDOW_SECONDS,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._window = float(window_seconds)
        self._times: deque = deque()
        self._depths: deque = deque()
        self._t0: Optional[float] = None

        self._figure = Figure(facecolor="#14161c")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        self._ax.set_facecolor("#1c1f27")
        (self._line,) = self._ax.plot([], [], color="#4dd0e1", linewidth=2.0)
        self._ax.tick_params(colors="#cccccc")
        for spine in self._ax.spines.values():
            spine.set_color("#444a5a")
        self._ax.set_xlabel("Time (s)", color="#cccccc")
        self._ax.set_ylabel("Depth Z (m)", color="#cccccc")
        self._ax.grid(True, color="#2c3140", linewidth=0.5)
        self._figure.tight_layout()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def add_sample(self, sample: Sample) -> None:
        if self._t0 is None:
            self._t0 = sample.timestamp
        t = sample.timestamp - self._t0
        self._times.append(t)
        self._depths.append(sample.depth)

        cutoff = t - self._window
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
            self._depths.popleft()

        xs = list(self._times)
        ys = list(self._depths)
        self._line.set_data(xs, ys)
        self._ax.set_xlim(max(0.0, t - self._window), max(self._window, t))
        if ys:
            ymin, ymax = min(ys), max(ys)
            margin = max(0.02, (ymax - ymin) * 0.2)
            self._ax.set_ylim(ymin - margin, ymax + margin)
        self._canvas.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stereo Head Depth Visualizer")
        self.setStyleSheet(
            "QMainWindow { background: #0f1115; }"
            " QLabel { color: #e0e0e0; }"
        )

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        top_container = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        title_top = QtWidgets.QLabel("3D head from the 5 triangulated landmarks")
        title_top.setStyleSheet("font-size: 15px; font-weight: bold;")
        top_layout.addWidget(title_top)
        self._head_view = HeadView3D()
        top_layout.addWidget(self._head_view, 1)

        bottom_container = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        header_layout = QtWidgets.QHBoxLayout()
        title_bot = QtWidgets.QLabel(
            f"Depth (nose Z) over the last {int(DEPTH_WINDOW_SECONDS)} seconds"
        )
        title_bot.setStyleSheet("font-size: 15px; font-weight: bold;")
        self._stats_label = QtWidgets.QLabel("Depth: --   |   FPS: --")
        self._stats_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._stats_label.setStyleSheet("font-size: 13px;")
        header_layout.addWidget(title_bot)
        header_layout.addStretch(1)
        header_layout.addWidget(self._stats_label)
        bottom_layout.addLayout(header_layout)
        self._plot = DepthPlot(window_seconds=DEPTH_WINDOW_SECONDS)
        bottom_layout.addWidget(self._plot, 1)

        outer.addWidget(top_container, 2)
        outer.addWidget(bottom_container, 1)

        self._frames_since_tick = 0
        self._fps = 0.0
        self._fps_timer = QtCore.QElapsedTimer()
        self._fps_timer.start()

        self._thread = QtCore.QThread(self)
        self._worker = TrackingWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.sample_ready.connect(self._on_sample, QtCore.Qt.QueuedConnection)
        self._worker.error.connect(self._on_error, QtCore.Qt.QueuedConnection)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    @QtCore.Slot(object)
    def _on_sample(self, sample: Sample) -> None:
        self._head_view.update_sample(sample)
        self._plot.add_sample(sample)

        self._frames_since_tick += 1
        elapsed_ms = self._fps_timer.elapsed()
        if elapsed_ms >= 500:
            self._fps = self._frames_since_tick / (elapsed_ms / 1000.0)
            self._frames_since_tick = 0
            self._fps_timer.restart()

        self._stats_label.setText(
            f"Depth: {sample.depth:.3f} m   |   FPS: {self._fps:.1f}"
        )

    @QtCore.Slot(str)
    def _on_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Stereo Visualizer", message)
        self.close()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q):
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._worker.stop()
        self._thread.quit()
        self._thread.wait(2000)
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.showFullScreen()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
