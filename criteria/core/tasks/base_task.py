from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QPointF, QRect
from PySide6.QtGui import QColor, QPainter, QPen

from criteria.core.models import TaskConfig, TaskResult, utcish_now


@dataclass
class Target:
    x: float
    y: float
    radius: float


class TestTask:
    id = "base"
    display_name = "Base Task"
    description = ""

    def __init__(
        self,
        seed: int,
        config: TaskConfig,
        screen_width: int,
        screen_height: int,
        screen_diagonal_px: float,
    ) -> None:
        self.seed = seed
        self.config = config
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_diagonal_px = screen_diagonal_px
        self.random = random.Random(f"{seed}:{self.id}")
        self.elapsed_ms = 0
        self.raw: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}
        self.score = 0.0
        self.completed = False
        self.stopped = False
        self.paused = False

    def start(self, bounds: QRect) -> None:
        self.bounds = bounds

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def stop(self) -> None:
        self.stopped = True
        self.completed = True

    def update(self, elapsed_ms: int, cursor: QPointF) -> None:
        self.elapsed_ms = elapsed_ms

    def mouse_press(self, elapsed_ms: int, pos: QPointF, button: str) -> None:
        self.elapsed_ms = elapsed_ms

    def paint(self, painter: QPainter, rect: QRect) -> None:
        painter.fillRect(rect, QColor("#f5f6fa"))

    def result(self) -> TaskResult:
        return TaskResult(
            task_id=self.id,
            display_name=self.display_name,
            status="stopped" if self.stopped else "complete",
            score=self.score,
            summary=self.summary,
            raw=self.raw,
            completed_at=utcish_now(),
        )

    def target_from_rng(self, radius: float) -> Target:
        margin = int(radius + 40)
        return Target(
            x=self.random.randint(margin, max(margin, self.screen_width - margin)),
            y=self.random.randint(margin, max(margin, self.screen_height - margin)),
            radius=radius,
        )

    @staticmethod
    def draw_target(
        painter: QPainter,
        target: Target,
        fill: str = "#0984e3",
        outline: str = "#ffffff",
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QColor(fill))
        painter.setPen(QPen(QColor(outline), 3))
        painter.drawEllipse(QPointF(target.x, target.y), target.radius, target.radius)

    @staticmethod
    def point_inside(target: Target, point: QPointF) -> bool:
        return ((point.x() - target.x) ** 2 + (point.y() - target.y) ** 2) ** 0.5 <= target.radius

