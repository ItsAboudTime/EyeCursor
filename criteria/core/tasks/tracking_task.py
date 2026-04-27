from __future__ import annotations

from PySide6.QtCore import QPointF, QRect
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from criteria.core.metrics import avg, distance, med, stddev
from criteria.core.scoring import tracking_score
from criteria.core.sounds import play as sfx
from criteria.core.tasks.base_task import Target, TestTask


class TrackingTask(TestTask):
    id = "tracking"
    display_name = "Tracking"
    description = "Follow the moving target as closely as possible."

    radius = 30
    ready_delay_ms = 500

    def start(self, bounds: QRect) -> None:
        super().start(bounds)
        self.sample_interval_ms = max(1, round(1000 / self.config.tracking_sample_hz))
        self.last_sample_ms = -self.sample_interval_ms
        self.waypoints = [self.target_from_rng(self.radius) for _ in range(9)]
        self.segment_duration_ms = max(700, self.config.tracking_duration_ms // (len(self.waypoints) - 1))
        self.target = self.waypoints[0]
        self._waiting = True
        self._entered_ms: int | None = None
        self._tracking_start_ms: int | None = None
        self._dinged_enter = False
        self._warning_played = False

    def update(self, elapsed_ms: int, cursor: QPointF) -> None:
        super().update(elapsed_ms, cursor)
        if self.completed or self.paused:
            return

        if self._waiting:
            inside = self.point_inside(self.target, cursor)
            if inside:
                if self._entered_ms is None:
                    self._entered_ms = elapsed_ms
                    if not self._dinged_enter:
                        sfx("ding")
                        self._dinged_enter = True
                elif elapsed_ms - self._entered_ms >= self.ready_delay_ms:
                    sfx("success")
                    self._waiting = False
                    self._tracking_start_ms = elapsed_ms
            else:
                self._entered_ms = None
                self._dinged_enter = False
            return

        tracking_elapsed = elapsed_ms - self._tracking_start_ms
        self.target = self._target_at(tracking_elapsed)
        if tracking_elapsed - self.last_sample_ms >= self.sample_interval_ms:
            self._record_sample(tracking_elapsed, cursor)
            self.last_sample_ms = tracking_elapsed
        remaining = self.config.tracking_duration_ms - tracking_elapsed
        if not self._warning_played and remaining <= 3000:
            sfx("warning")
            self._warning_played = True
        if tracking_elapsed >= self.config.tracking_duration_ms:
            self._summarize()
            self.completed = True

    def paint(self, painter: QPainter, rect: QRect) -> None:
        painter.fillRect(rect, QColor("#f5f6fa"))
        self.draw_target(painter, self.target, "#6c5ce7")
        painter.setPen(QPen(QColor("#2d3436")))
        painter.setFont(QFont("Arial", 16))

        if self._waiting:
            painter.drawText(28, 42, "Tracking — move cursor into the circle to begin")
            if self._entered_ms is not None:
                hold = self.elapsed_ms - self._entered_ms
                remaining_hold = max(0, self.ready_delay_ms - hold)
                painter.drawText(28, 72, f"Starting in {remaining_hold / 1000:.1f}s")
        else:
            painter.drawText(28, 42, "Tracking")
            tracking_elapsed = self.elapsed_ms - self._tracking_start_ms
            remaining = max(0, self.config.tracking_duration_ms - tracking_elapsed)
            painter.drawText(28, 72, f"{remaining / 1000:.1f}s")

    def _target_at(self, elapsed_ms: int) -> Target:
        max_index = len(self.waypoints) - 1
        segment = min(max_index - 1, elapsed_ms // self.segment_duration_ms)
        progress = (elapsed_ms % self.segment_duration_ms) / self.segment_duration_ms
        start = self.waypoints[segment]
        end = self.waypoints[segment + 1]
        eased = progress * progress * (3 - 2 * progress)
        return Target(
            x=start.x + (end.x - start.x) * eased,
            y=start.y + (end.y - start.y) * eased,
            radius=self.radius,
        )

    def _record_sample(self, elapsed_ms: int, cursor: QPointF) -> None:
        pixel_error = distance(cursor.x(), cursor.y(), self.target.x, self.target.y)
        self.raw.append(
            {
                "task": self.id,
                "sample_index": len(self.raw),
                "timestamp_ms": elapsed_ms,
                "target_x": round(self.target.x, 2),
                "target_y": round(self.target.y, 2),
                "target_radius": self.radius,
                "cursor_x": round(cursor.x(), 2),
                "cursor_y": round(cursor.y(), 2),
                "pixel_error": round(pixel_error, 3),
                "radius_normalized_error": round(pixel_error / self.radius, 5),
                "screen_normalized_error": round(pixel_error / self.screen_diagonal_px, 7),
            }
        )

    def _summarize(self) -> None:
        pixel_errors = [row["pixel_error"] for row in self.raw]
        radius_errors = [row["radius_normalized_error"] for row in self.raw]
        screen_errors = [row["screen_normalized_error"] for row in self.raw]
        radius_std = stddev(radius_errors)
        self.score = tracking_score(avg(radius_errors), radius_std)
        self.summary = {
            "sample_count": len(self.raw),
            "average_pixel_error": round(avg(pixel_errors), 3),
            "median_pixel_error": round(med(pixel_errors), 3),
            "average_radius_normalized_error": round(avg(radius_errors), 5),
            "median_radius_normalized_error": round(med(radius_errors), 5),
            "average_screen_normalized_error": round(avg(screen_errors), 7),
            "maximum_error": round(max(pixel_errors), 3) if pixel_errors else 0.0,
            "error_std_dev": round(radius_std, 5),
            "stability_score": round(max(0, min(1, 1 - radius_std / 3)) * 20, 2),
            "tracking_score": self.score,
        }

