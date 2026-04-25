from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

from criteria.core.models import Session
from criteria.core.scoring import final_summary


APP_DATA_DIR = Path(user_data_dir("EyeCursor TestLab", "EyeCursorTeam"))


class StorageManager:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or APP_DATA_DIR
        self.sessions_dir = self.base_dir / "sessions"
        self.exports_dir = self.base_dir / "exports"
        self.logs_dir = self.base_dir / "logs"
        for path in (self.sessions_dir, self.exports_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def save_session(self, session: Session) -> None:
        path = self.session_dir(session.session_id)
        path.mkdir(parents=True, exist_ok=True)
        summary = final_summary(session)
        session.final_summary = summary
        self._write_json(path / "session.json", session.to_dict())
        self._write_json(path / "summary.json", summary)
        self._write_json(path / "raw_events.json", self._raw_payload(session))
        self._write_task_csvs(session)

    def load_session(self, session_id: str) -> Session:
        path = self.session_dir(session_id) / "session.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing session file: {path}")
        with path.open("r", encoding="utf-8") as file:
            return Session.from_dict(json.load(file))

    def list_sessions(self) -> list[Session]:
        sessions: list[Session] = []
        for session_file in sorted(self.sessions_dir.glob("*/session.json"), reverse=True):
            try:
                with session_file.open("r", encoding="utf-8") as file:
                    sessions.append(Session.from_dict(json.load(file)))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
        return sessions

    def export_json(self, session: Session) -> Path:
        self.save_session(session)
        output = self.exports_dir / f"{session.session_id}_raw_session.json"
        shutil.copy2(self.session_dir(session.session_id) / "session.json", output)
        return output

    def export_summary_csv(self, session: Session) -> Path:
        self.save_session(session)
        output = self.exports_dir / f"{session.session_id}_summary.csv"
        summary = final_summary(session)
        row: dict[str, Any] = {
            "session_id": session.session_id,
            "participant_name": session.participant_name,
            "input_method": session.input_method,
            "seed": session.seed,
            "screen_width": session.screen_width,
            "screen_height": session.screen_height,
            "started_at": session.started_at,
            "completed_at": session.completed_at or "",
            **summary,
        }
        for task_id, result in session.task_results.items():
            row[f"{task_id}_status"] = result.status
            row[f"{task_id}_score"] = result.score
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return output

    def _write_task_csvs(self, session: Session) -> None:
        path = self.session_dir(session.session_id)
        filenames = {
            "movement": "movement_trials.csv",
            "accuracy": "accuracy_trials.csv",
            "tracking": "tracking_samples.csv",
            "clicking": "clicking_trials.csv",
        }
        for task_id, filename in filenames.items():
            result = session.task_results.get(task_id)
            if not result:
                continue
            self._write_csv(path / filename, result.raw)

    @staticmethod
    def _raw_payload(session: Session) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "tasks": {
                task_id: {
                    "summary": result.summary,
                    "raw": result.raw,
                }
                for task_id, result in session.task_results.items()
            },
        }

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

