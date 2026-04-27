import math
import time
from collections import deque
from typing import Optional, Tuple

from PySide6.QtCore import Qt, QPointF, QTimer, Slot
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget


TRAIL_LIFETIME_SEC = 0.5
TRAIL_MAX_LEN = 20
TRAIL_DOT_RADIUS_PX = 6
BUBBLE_RADIUS_PX = 120
BUBBLE_RING_PX = 3

REPAINT_HZ = 60

# Spring-damper. Critically damped so the bubble lags smoothly toward the
# gaze target without overshooting or oscillating. Adjust SPRING_K to change
# how quickly it catches up; keep damping ~= 2*sqrt(K) for no jiggle.
SPRING_K = 60.0
SPRING_DAMPING = 15.5


class GazeBubbleOverlay(QWidget):
    def __init__(
        self,
        virtual_bounds: Tuple[int, int, int, int],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        minx, miny, maxx, maxy = virtual_bounds
        self._origin = (minx, miny)
        self.setGeometry(minx, miny, maxx - minx + 1, maxy - miny + 1)

        self._target: Optional[Tuple[float, float]] = None
        self._pos: Optional[Tuple[float, float]] = None
        self._vel: Tuple[float, float] = (0.0, 0.0)
        self._last_tick = time.monotonic()

        self._trail: deque = deque(maxlen=TRAIL_MAX_LEN)
        self._last_trail_pos: Optional[Tuple[float, float]] = None

        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / REPAINT_HZ))
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    @Slot(int, int)
    def update_position(self, x: int, y: int) -> None:
        self._target = (float(x), float(y))
        if self._pos is None:
            self._pos = (float(x), float(y))
            self._vel = (0.0, 0.0)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now
        if dt > 0.05:
            dt = 0.05

        if self._pos is not None and self._target is not None:
            px, py = self._pos
            tx, ty = self._target
            vx, vy = self._vel

            ax = SPRING_K * (tx - px) - SPRING_DAMPING * vx
            ay = SPRING_K * (ty - py) - SPRING_DAMPING * vy

            vx += ax * dt
            vy += ay * dt
            px += vx * dt
            py += vy * dt

            self._pos = (px, py)
            self._vel = (vx, vy)

            if self._last_trail_pos is None:
                self._trail.append((px, py, now))
                self._last_trail_pos = (px, py)
            else:
                lx, ly = self._last_trail_pos
                if (px - lx) * (px - lx) + (py - ly) * (py - ly) > 25.0:
                    self._trail.append((px, py, now))
                    self._last_trail_pos = (px, py)

        self.update()

    def paintEvent(self, event) -> None:
        if self._pos is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        ox, oy = self._origin
        now = time.monotonic()

        bubble_x, bubble_y = self._pos
        bubble_clip_sq = (BUBBLE_RADIUS_PX + TRAIL_DOT_RADIUS_PX) ** 2

        painter.setPen(Qt.PenStyle.NoPen)
        for x, y, t in self._trail:
            age = now - t
            if age >= TRAIL_LIFETIME_SEC:
                continue
            dx = x - bubble_x
            dy = y - bubble_y
            if dx * dx + dy * dy < bubble_clip_sq:
                continue
            life = 1.0 - (age / TRAIL_LIFETIME_SEC)
            alpha = int(110 * life)
            radius = max(2, int(TRAIL_DOT_RADIUS_PX * (0.4 + 0.6 * life)))
            painter.setBrush(QColor(230, 230, 240, alpha))
            painter.drawEllipse(
                int(x - ox - radius),
                int(y - oy - radius),
                radius * 2,
                radius * 2,
            )

        cx = bubble_x - ox
        cy = bubble_y - oy
        r = BUBBLE_RADIUS_PX

        painter.save()
        painter.translate(cx, cy)

        # 1) Frosted glass fill: faintly violet, brighter toward upper-left.
        #    Reads as a glass tint without obscuring whatever is behind.
        fill = QRadialGradient(-r * 0.20, -r * 0.30, r * 1.3)
        fill.setColorAt(0.00, QColor(255, 255, 255, 50))
        fill.setColorAt(0.55, QColor(210, 210, 235, 30))
        fill.setColorAt(0.90, QColor(160, 160, 200, 20))
        fill.setColorAt(1.00, QColor(120, 120, 170, 14))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawEllipse(-r, -r, r * 2, r * 2)

        # 2) Fresnel band: a soft dark ring just inside the rim. Sells the
        #    "thickness of curved glass" without darkening the center.
        fresnel = QRadialGradient(0.0, 0.0, r)
        fresnel.setColorAt(0.00, QColor(0, 0, 0, 0))
        fresnel.setColorAt(0.78, QColor(0, 0, 0, 0))
        fresnel.setColorAt(0.94, QColor(20, 10, 40, 55))
        fresnel.setColorAt(1.00, QColor(20, 10, 40, 0))
        painter.setBrush(fresnel)
        painter.drawEllipse(-r, -r, r * 2, r * 2)

        # 3) Big crescent reflection on the upper-left: built from two
        #    offset circles via odd-even fill, gives a curved highlight
        #    that wraps the inside of the rim like a real reflection.
        outer_r = r - 2
        inner_offset = r * 0.22
        inner_r = r * 0.96
        crescent = QPainterPath()
        crescent.setFillRule(Qt.FillRule.OddEvenFill)
        crescent.addEllipse(QPointF(0.0, 0.0), outer_r, outer_r)
        crescent.addEllipse(
            QPointF(inner_offset, inner_offset), inner_r, inner_r
        )
        crescent_grad = QLinearGradient(
            -r * 0.85, -r * 0.85, -r * 0.05, -r * 0.05
        )
        crescent_grad.setColorAt(0.00, QColor(255, 255, 255, 220))
        crescent_grad.setColorAt(0.55, QColor(255, 255, 255, 60))
        crescent_grad.setColorAt(1.00, QColor(255, 255, 255, 0))
        painter.fillPath(crescent, crescent_grad)

        # 4) Smaller secondary crescent on the lower-right rim to hint at
        #    light wrapping around the back of the sphere. Subtle.
        outer_r2 = r - 4
        inner_r2 = r * 0.94
        crescent2 = QPainterPath()
        crescent2.setFillRule(Qt.FillRule.OddEvenFill)
        crescent2.addEllipse(QPointF(0.0, 0.0), outer_r2, outer_r2)
        crescent2.addEllipse(
            QPointF(-r * 0.18, -r * 0.18), inner_r2, inner_r2
        )
        crescent2_grad = QLinearGradient(r * 0.85, r * 0.85, r * 0.1, r * 0.1)
        crescent2_grad.setColorAt(0.00, QColor(220, 255, 240, 110))
        crescent2_grad.setColorAt(0.65, QColor(220, 255, 240, 25))
        crescent2_grad.setColorAt(1.00, QColor(220, 255, 240, 0))
        painter.fillPath(crescent2, crescent2_grad)

        # 5) Chromatic rim. Most of the ring is bright white; color
        #    concentrates in two hot zones on opposite sides — pink/purple
        #    on the upper-left, green/cyan on the lower-right. Closer to
        #    real soap-film dispersion than a uniform rainbow.
        rim = QConicalGradient(0.0, 0.0, 135.0)
        rim.setColorAt(0.00, QColor(255, 235, 245, 230))
        rim.setColorAt(0.05, QColor(255, 200, 230, 235))
        rim.setColorAt(0.12, QColor(220, 170, 255, 235))
        rim.setColorAt(0.22, QColor(200, 200, 255, 200))
        rim.setColorAt(0.40, QColor(235, 240, 255, 150))
        rim.setColorAt(0.55, QColor(190, 255, 220, 200))
        rim.setColorAt(0.65, QColor(220, 255, 190, 220))
        rim.setColorAt(0.80, QColor(245, 250, 235, 190))
        rim.setColorAt(1.00, QColor(255, 235, 245, 230))
        rim_pen = QPen(QBrush(rim), BUBBLE_RING_PX + 1)
        rim_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(rim_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(-r, -r, r * 2, r * 2)

        # 6) Bright window-reflection blob: the punchy hot spot near the
        #    top-left, like a lamp reflected on a wet sphere.
        glint = QRadialGradient(-r * 0.50, -r * 0.55, r * 0.32)
        glint.setColorAt(0.00, QColor(255, 255, 255, 235))
        glint.setColorAt(0.40, QColor(255, 255, 255, 90))
        glint.setColorAt(1.00, QColor(255, 255, 255, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glint)
        painter.drawEllipse(
            QPointF(-r * 0.50, -r * 0.55), r * 0.30, r * 0.20
        )

        # 7) Tiny pinpoint specular dot — the brightest single pixel.
        painter.setBrush(QColor(255, 255, 255, 240))
        painter.drawEllipse(
            QPointF(-r * 0.46, -r * 0.62), r * 0.05, r * 0.05
        )

        painter.restore()
