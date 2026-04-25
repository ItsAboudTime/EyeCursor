from __future__ import annotations

from PySide6.QtCore import QElapsedTimer, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QKeyEvent, QMouseEvent, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from criteria.core.models import Session, utcish_now
from criteria.core.storage import StorageManager
from criteria.core.tasks import TASK_IDS, TASK_SEQUENCE
from criteria.core.tasks.base_task import TestTask


TASK_BY_ID = {task.id: task for task in TASK_SEQUENCE}


class TestCanvas(QWidget):
    clicked = Signal(QPointF, str)

    def __init__(self) -> None:
        super().__init__()
        self.task: TestTask | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_task(self, task: TestTask | None) -> None:
        self.task = task
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if self.task:
            self.task.paint(painter, self.rect())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self.task:
            return
        button = "left" if event.button() == Qt.MouseButton.LeftButton else "right"
        self.clicked.emit(event.position(), button)


class TransitionPage(QWidget):
    continue_clicked = Signal()
    pause_clicked = Signal()
    end_clicked = Signal()
    results_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(80, 70, 80, 70)
        layout.setSpacing(18)
        self.title = QLabel("Ready")
        self.title.setObjectName("Title")
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-size: 17px; color: #636e72;")
        layout.addStretch(1)
        layout.addWidget(self.title)
        layout.addWidget(self.summary)
        buttons = QHBoxLayout()
        self.continue_button = QPushButton("Continue")
        pause = QPushButton("Pause Session")
        end = QPushButton("End Session")
        results = QPushButton("View Results")
        pause.setProperty("secondary", True)
        results.setProperty("secondary", True)
        end.setProperty("danger", True)
        self.continue_button.clicked.connect(self.continue_clicked.emit)
        pause.clicked.connect(self.pause_clicked.emit)
        end.clicked.connect(self.end_clicked.emit)
        results.clicked.connect(self.results_clicked.emit)
        for button in (self.continue_button, pause, end, results):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Keyboard: Esc pauses, Q stops."))
        layout.addStretch(1)

    def set_content(self, title: str, summary: str, continue_text: str = "Continue") -> None:
        self.title.setText(title)
        self.summary.setText(summary)
        self.continue_button.setText(continue_text)


class TestWindow(QWidget):
    closed = Signal()
    results_requested = Signal()

    def __init__(self, session: Session, storage: StorageManager) -> None:
        super().__init__()
        self.session = session
        self.storage = storage
        self.current_task: TestTask | None = None
        self.elapsed = QElapsedTimer()
        self.paused_elapsed_ms = 0
        self.active = False

        self.setWindowTitle("EyeCursor TestLab")
        self.stack = QStackedWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)
        self.transition = TransitionPage()
        self.canvas = TestCanvas()
        self.stack.addWidget(self.transition)
        self.stack.addWidget(self.canvas)

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.canvas.clicked.connect(self.handle_canvas_click)
        self.transition.continue_clicked.connect(self._continue)
        self.transition.pause_clicked.connect(self._pause_from_transition)
        self.transition.end_clicked.connect(self._end_session)
        self.transition.results_clicked.connect(self._view_results)

        self._show_next_transition("Session Ready", f"Next task: {self.session.next_task.title()}", "Start Task")

    def begin(self) -> None:
        self.showFullScreen()
        self.raise_()
        self.activateWindow()

    def handle_canvas_click(self, pos: QPointF, button: str) -> None:
        if self.current_task and self.active:
            self.current_task.mouse_press(self._elapsed_ms(), pos, button)
            self._check_completion()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._pause_current_task()
        elif event.key() == Qt.Key.Key_Q:
            self._stop_current_task()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.closed.emit()
        super().closeEvent(event)

    def _continue(self) -> None:
        if self.current_task and self.current_task.paused:
            self._resume_current_task()
            return
        self._start_next_task()

    def _start_next_task(self) -> None:
        task_cls = TASK_BY_ID.get(self.session.next_task)
        if task_cls is None:
            self._complete_session()
            return
        self.current_task = task_cls(
            self.session.seed,
            self.session.task_config,
            self.session.screen_width,
            self.session.screen_height,
            self.session.screen_diagonal_px,
        )
        self.current_task.start(self.canvas.rect())
        self.canvas.set_task(self.current_task)
        self.paused_elapsed_ms = 0
        self.elapsed.restart()
        self.active = True
        self.stack.setCurrentWidget(self.canvas)
        self.canvas.setFocus()
        self.timer.start()

    def _tick(self) -> None:
        if not self.current_task or not self.active:
            return
        local = self.canvas.mapFromGlobal(QCursor.pos())
        self.current_task.update(self._elapsed_ms(), QPointF(local))
        self.canvas.update()
        self._check_completion()

    def _check_completion(self) -> None:
        if not self.current_task or not self.current_task.completed:
            return
        self.timer.stop()
        self.active = False
        result = self.current_task.result()
        self.session.task_results[result.task_id] = result
        if result.task_id not in self.session.completed_tasks and result.status == "complete":
            self.session.completed_tasks.append(result.task_id)
        self.session.next_task = self._next_task_after(result.task_id)
        self.session.status = "in_progress"
        self.storage.save_session(self.session)
        self._show_next_transition(
            f"{result.display_name} Task Complete",
            self._summary_text(result.summary),
            "Continue" if self.session.next_task != "final" else "View Final Results",
        )
        self.current_task = None

    def _pause_current_task(self) -> None:
        if not self.current_task or not self.active:
            return
        self.paused_elapsed_ms = self._elapsed_ms()
        self.timer.stop()
        self.active = False
        self.current_task.pause()
        self.session.status = "paused"
        self.storage.save_session(self.session)
        self._show_next_transition(
            f"{self.current_task.display_name} Paused",
            "The current task is paused. Continue resumes the same task.",
            "Continue Task",
        )

    def _resume_current_task(self) -> None:
        if not self.current_task:
            return
        self.current_task.resume()
        self.elapsed.restart()
        self.active = True
        self.session.status = "in_progress"
        self.storage.save_session(self.session)
        self.stack.setCurrentWidget(self.canvas)
        self.canvas.setFocus()
        self.timer.start()

    def _stop_current_task(self) -> None:
        if self.current_task:
            self.current_task.stop()
            self.session.status = "stopped"
            self.storage.save_session(self.session)
        self._end_session()

    def _pause_from_transition(self) -> None:
        self.session.status = "paused"
        self.storage.save_session(self.session)
        self.close()

    def _end_session(self) -> None:
        self.timer.stop()
        self.session.status = "stopped"
        self.storage.save_session(self.session)
        self.close()

    def _view_results(self) -> None:
        self.storage.save_session(self.session)
        self.results_requested.emit()
        self.close()

    def _complete_session(self) -> None:
        self.session.status = "complete"
        self.session.completed_at = utcish_now()
        self.session.next_task = "final"
        self.storage.save_session(self.session)
        self.results_requested.emit()
        self.close()

    def _show_next_transition(self, title: str, summary: str, continue_text: str) -> None:
        self.transition.set_content(title, summary, continue_text)
        self.stack.setCurrentWidget(self.transition)
        self.transition.setFocus()

    def _elapsed_ms(self) -> int:
        return self.paused_elapsed_ms + self.elapsed.elapsed()

    def _next_task_after(self, task_id: str) -> str:
        index = TASK_IDS.index(task_id)
        if index + 1 >= len(TASK_IDS):
            return "final"
        return TASK_IDS[index + 1]

    @staticmethod
    def _summary_text(summary: dict) -> str:
        if not summary:
            return "No summary available."
        parts = []
        for key, value in summary.items():
            parts.append(f"{key.replace('_', ' ').title()}: {value}")
        return "\n".join(parts)
