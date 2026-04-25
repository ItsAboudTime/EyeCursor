from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import hypot
from typing import Any
from uuid import uuid4


def utcish_now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


@dataclass
class TaskConfig:
    movement_trials: int = 15
    accuracy_trials: int = 12
    clicking_trials: int = 12
    tracking_duration_ms: int = 30_000
    tracking_sample_hz: int = 30
    preset: str = "MVP Default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TaskConfig":
        if not data:
            return cls()
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class TaskResult:
    task_id: str
    display_name: str
    status: str = "incomplete"
    score: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)
    raw: list[dict[str, Any]] = field(default_factory=list)
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskResult":
        return cls(**data)


@dataclass
class Session:
    session_id: str
    participant_name: str
    input_method: str
    seed: int
    screen_width: int
    screen_height: int
    screen_diagonal_px: float
    notes: str = ""
    task_config: TaskConfig = field(default_factory=TaskConfig)
    started_at: str = field(default_factory=utcish_now)
    completed_at: str | None = None
    status: str = "in_progress"
    completed_tasks: list[str] = field(default_factory=list)
    next_task: str = "movement"
    task_results: dict[str, TaskResult] = field(default_factory=dict)
    final_summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        participant_name: str,
        input_method: str,
        seed: int,
        screen_width: int,
        screen_height: int,
        notes: str = "",
        task_config: TaskConfig | None = None,
    ) -> "Session":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return cls(
            session_id=f"session_{timestamp}_{uuid4().hex[:6]}",
            participant_name=participant_name.strip() or "Unnamed Participant",
            input_method=input_method.strip() or "Mouse",
            seed=int(seed),
            screen_width=int(screen_width),
            screen_height=int(screen_height),
            screen_diagonal_px=hypot(screen_width, screen_height),
            notes=notes.strip(),
            task_config=task_config or TaskConfig(),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["task_config"] = self.task_config.to_dict()
        data["task_results"] = {
            task_id: result.to_dict() for task_id, result in self.task_results.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        payload = dict(data)
        payload["task_config"] = TaskConfig.from_dict(payload.get("task_config"))
        payload["task_results"] = {
            task_id: TaskResult.from_dict(result)
            for task_id, result in payload.get("task_results", {}).items()
        }
        return cls(**payload)

