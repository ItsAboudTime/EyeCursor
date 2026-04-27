from __future__ import annotations

from PySide6.QtCore import QPointF, QRect
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from criteria.core.metrics import avg, distance, med
from criteria.core.scoring import movement_score
from criteria.core.sounds import play as sfx
from criteria.core.tasks.base_task import Target, TestTask


class MovementTask(TestTask):
    id = "movement"
    display_name = "Movement"
    description = "Move into each target and hold for 100 ms."

    dwell_required_ms = 100
    timeout_ms = 3000
    radii = [60, 35, 20]

    def start(self, bounds: QRect) -> None:
        super().start(bounds)
        self.trial_index = 0
        self.current_started_ms = 0
        self.dwell_started_ms: int | None = None
        self.previous_center: tuple[float, float] | None = None
        self._was_inside = False
        self._warning_played = False
        self.target = self._next_target()

    def update(self, elapsed_ms: int, cursor: QPointF) -> None:
        super().update(elapsed_ms, cursor)
        self._last_cursor = cursor
        if self.completed or self.paused:
            return
        inside = self.point_inside(self.target, cursor)
        if inside:
            if not self._was_inside:
                sfx("ding")
                self._was_inside = True
            if self.dwell_started_ms is None:
                self.dwell_started_ms = elapsed_ms
            if elapsed_ms - self.dwell_started_ms >= self.dwell_required_ms:
                sfx("success")
                self._finish_trial(elapsed_ms, completed=True, timed_out=False)
        else:
            self._was_inside = False
            self.dwell_started_ms = None
        trial_remaining = self.timeout_ms - (elapsed_ms - self.current_started_ms)
        if not self._warning_played and trial_remaining <= 1000:
            sfx("warning")
            self._warning_played = True
        if elapsed_ms - self.current_started_ms >= self.timeout_ms:
            self._finish_trial(elapsed_ms, completed=False, timed_out=True)

    def paint(self, painter: QPainter, rect: QRect) -> None:
        painter.fillRect(rect, QColor("#f5f6fa"))
        self.draw_target(painter, self.target, "#00b894")
        painter.setPen(QPen(QColor("#2d3436")))
        painter.setFont(QFont("Arial", 16))
        painter.drawText(28, 42, f"Movement {self.trial_index + 1}/{self.config.movement_trials}")
        remaining = max(0, self.timeout_ms - (self.elapsed_ms - self.current_started_ms))
        painter.drawText(28, 72, f"{remaining / 1000:.1f}s")

    def _next_target(self) -> Target:
        radius = self.radii[self.trial_index % len(self.radii)]
        return self.target_from_rng(radius)

    def _finish_trial(self, elapsed_ms: int, completed: bool, timed_out: bool) -> None:
        movement_time = elapsed_ms - self.current_started_ms
        start_x, start_y = self.previous_center or (self.screen_width / 2, self.screen_height / 2)
        target_distance = distance(start_x, start_y, self.target.x, self.target.y)
        cursor = self._last_cursor
        self.raw.append(
            {
                "task": self.id,
                "trial_index": self.trial_index,
                "target_x": round(self.target.x, 2),
                "target_y": round(self.target.y, 2),
                "target_radius": self.target.radius,
                "cursor_x": round(cursor.x(), 2),
                "cursor_y": round(cursor.y(), 2),
                "start_time_ms": self.current_started_ms,
                "end_time_ms": elapsed_ms,
                "movement_time_ms": movement_time if completed else None,
                "dwell_time_required_ms": self.dwell_required_ms,
                "completed": completed,
                "timed_out": timed_out,
                "target_distance_px": round(target_distance, 2),
            }
        )
        self.previous_center = (self.target.x, self.target.y)
        self.trial_index += 1
        self.dwell_started_ms = None
        self._was_inside = False
        self._warning_played = False
        self.current_started_ms = elapsed_ms
        if self.trial_index >= self.config.movement_trials:
            self._summarize()
            self.completed = True
        else:
            self.target = self._next_target()

    def _summarize(self) -> None:
        completed_trials = [row for row in self.raw if row["completed"]]
        movement_times = [row["movement_time_ms"] for row in completed_trials if row["movement_time_ms"] is not None]
        distances = [row["target_distance_px"] for row in self.raw]
        completion_rate = len(completed_trials) / len(self.raw) if self.raw else 0.0
        average_time = avg(movement_times)
        self.score = movement_score(completion_rate, average_time)
        self.summary = {
            "completion_count": len(completed_trials),
            "timeout_count": sum(1 for row in self.raw if row["timed_out"]),
            "average_movement_time_ms": round(average_time, 2),
            "median_movement_time_ms": round(med(movement_times), 2),
            "average_target_distance_px": round(avg(distances), 2),
            "completion_rate": round(completion_rate, 4),
            "movement_score": self.score,
        }

