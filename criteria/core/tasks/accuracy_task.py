from __future__ import annotations

from PySide6.QtCore import QPointF, QRect
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from criteria.core.metrics import avg, distance, med
from criteria.core.scoring import accuracy_score
from criteria.core.tasks.base_task import TestTask


class AccuracyTask(TestTask):
    id = "accuracy"
    display_name = "Accuracy"
    description = "Get as close as possible to the center before time expires."

    radius = 20
    timeout_ms = 1000

    def start(self, bounds: QRect) -> None:
        super().start(bounds)
        self.trial_index = 0
        self.current_started_ms = 0
        self.target = self.target_from_rng(self.radius)
        self.last_cursor = QPointF(self.screen_width / 2, self.screen_height / 2)

    def update(self, elapsed_ms: int, cursor: QPointF) -> None:
        super().update(elapsed_ms, cursor)
        self.last_cursor = cursor
        if self.completed or self.paused:
            return
        if elapsed_ms - self.current_started_ms >= self.timeout_ms:
            self._finish_trial(elapsed_ms)

    def paint(self, painter: QPainter, rect: QRect) -> None:
        painter.fillRect(rect, QColor("#f5f6fa"))
        self.draw_target(painter, self.target, "#0984e3")
        painter.setPen(QPen(QColor("#2d3436")))
        painter.setFont(QFont("Arial", 16))
        painter.drawText(28, 42, f"Accuracy {self.trial_index + 1}/{self.config.accuracy_trials}")
        painter.drawText(28, 72, f"{max(0, self.timeout_ms - (self.elapsed_ms - self.current_started_ms)) / 1000:.1f}s")

    def _finish_trial(self, elapsed_ms: int) -> None:
        pixel_error = distance(self.last_cursor.x(), self.last_cursor.y(), self.target.x, self.target.y)
        self.raw.append(
            {
                "task": self.id,
                "trial_index": self.trial_index,
                "target_x": round(self.target.x, 2),
                "target_y": round(self.target.y, 2),
                "target_radius": self.radius,
                "cursor_x": round(self.last_cursor.x(), 2),
                "cursor_y": round(self.last_cursor.y(), 2),
                "pixel_error": round(pixel_error, 3),
                "radius_normalized_error": round(pixel_error / self.radius, 5),
                "screen_normalized_error": round(pixel_error / self.screen_diagonal_px, 7),
                "timeout_ms": self.timeout_ms,
                "end_time_ms": elapsed_ms,
            }
        )
        self.trial_index += 1
        self.current_started_ms = elapsed_ms
        if self.trial_index >= self.config.accuracy_trials:
            self._summarize()
            self.completed = True
        else:
            self.target = self.target_from_rng(self.radius)

    def _summarize(self) -> None:
        pixel_errors = [row["pixel_error"] for row in self.raw]
        radius_errors = [row["radius_normalized_error"] for row in self.raw]
        screen_errors = [row["screen_normalized_error"] for row in self.raw]
        avg_radius = avg(radius_errors)
        self.score = accuracy_score(avg_radius)
        self.summary = {
            "average_pixel_error": round(avg(pixel_errors), 3),
            "median_pixel_error": round(med(pixel_errors), 3),
            "average_radius_normalized_error": round(avg_radius, 5),
            "median_radius_normalized_error": round(med(radius_errors), 5),
            "average_screen_normalized_error": round(avg(screen_errors), 7),
            "accuracy_score": self.score,
        }

