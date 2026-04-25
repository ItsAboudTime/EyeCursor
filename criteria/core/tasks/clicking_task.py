from __future__ import annotations

from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from criteria.core.scoring import clicking_score
from criteria.core.tasks.base_task import TestTask


class ClickingTask(TestTask):
    id = "clicking"
    display_name = "Clicking"
    description = "Click the target using the requested left or right click."

    radius = 35
    timeout_ms = 5000

    def start(self, bounds: QRect) -> None:
        super().start(bounds)
        self.trial_index = 0
        self.current_started_ms = 0
        self.target = self.target_from_rng(self.radius)
        self.requested_click = self._next_click_type()

    def update(self, elapsed_ms: int, cursor: QPointF) -> None:
        super().update(elapsed_ms, cursor)
        if self.completed or self.paused:
            return
        if elapsed_ms - self.current_started_ms >= self.timeout_ms:
            self._record_trial(elapsed_ms, QPointF(-1, -1), None, "timeout", False)

    def mouse_press(self, elapsed_ms: int, pos: QPointF, button: str) -> None:
        super().mouse_press(elapsed_ms, pos, button)
        if self.completed or self.paused:
            return
        inside = self.point_inside(self.target, pos)
        if inside and button == self.requested_click:
            result = "success"
        elif inside:
            result = "fail_inside"
        else:
            result = "fail_outside"
        self._record_trial(elapsed_ms, pos, button, result, inside)

    def paint(self, painter: QPainter, rect: QRect) -> None:
        painter.fillRect(rect, QColor("#f5f6fa"))
        fill = "#00cec9" if self.requested_click == "left" else "#e17055"
        self.draw_target(painter, self.target, fill)
        painter.setPen(QPen(QColor("#2d3436")))
        painter.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        painter.drawText(28, 44, f"{self.requested_click.title()} Click")
        painter.setFont(QFont("Arial", 16))
        painter.drawText(28, 76, f"Clicking {self.trial_index + 1}/{self.config.clicking_trials}")
        painter.drawText(28, 106, f"{max(0, self.timeout_ms - (self.elapsed_ms - self.current_started_ms)) / 1000:.1f}s")

    def _next_click_type(self) -> str:
        return "left" if self.random.random() < 0.5 else "right"

    def _record_trial(
        self,
        elapsed_ms: int,
        pos: QPointF,
        button: str | None,
        result: str,
        inside: bool,
    ) -> None:
        self.raw.append(
            {
                "task": self.id,
                "trial_index": self.trial_index,
                "target_x": round(self.target.x, 2),
                "target_y": round(self.target.y, 2),
                "target_radius": self.radius,
                "requested_click": self.requested_click,
                "actual_click": button,
                "click_x": round(pos.x(), 2) if button else None,
                "click_y": round(pos.y(), 2) if button else None,
                "inside_target": inside,
                "result": result,
                "time_to_click_ms": elapsed_ms - self.current_started_ms if button else None,
            }
        )
        self.trial_index += 1
        self.current_started_ms = elapsed_ms
        if self.trial_index >= self.config.clicking_trials:
            self._summarize()
            self.completed = True
        else:
            self.target = self.target_from_rng(self.radius)
            self.requested_click = self._next_click_type()

    def _summarize(self) -> None:
        total = len(self.raw) or 1
        success = sum(1 for row in self.raw if row["result"] == "success")
        fail_inside = sum(1 for row in self.raw if row["result"] == "fail_inside")
        fail_outside = sum(1 for row in self.raw if row["result"] == "fail_outside")
        timeouts = sum(1 for row in self.raw if row["result"] == "timeout")
        success_rate = success / total
        fail_inside_rate = fail_inside / total
        fail_outside_rate = fail_outside / total
        timeout_rate = timeouts / total
        self.score = clicking_score(success_rate, fail_inside_rate, fail_outside_rate, timeout_rate)
        self.summary = {
            "trial_count": len(self.raw),
            "success_count": success,
            "fail_inside_count": fail_inside,
            "fail_outside_count": fail_outside,
            "timeout_count": timeouts,
            "success_rate": round(success_rate, 4),
            "wrong_click_rate": round(fail_inside_rate, 4),
            "outside_click_rate": round(fail_outside_rate, 4),
            "clicking_score": self.score,
        }

