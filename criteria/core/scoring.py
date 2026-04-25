from __future__ import annotations

from criteria.core.metrics import clamp
from criteria.core.models import Session


WEIGHTS = {
    "movement": 0.30,
    "accuracy": 0.30,
    "tracking": 0.25,
    "clicking": 0.15,
}


def rating_label(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Acceptable"
    if score >= 40:
        return "Poor"
    return "Failed"


def movement_score(completion_rate: float, average_movement_time_ms: float) -> float:
    completion_component = completion_rate * 70
    speed_component = clamp(1 - average_movement_time_ms / 3000) * 30
    return round(clamp(completion_component + speed_component, 0, 100), 2)


def accuracy_score(average_radius_normalized_error: float) -> float:
    return round(clamp(1 - average_radius_normalized_error / 3) * 100, 2)


def tracking_score(average_radius_normalized_error: float, error_std_dev: float) -> float:
    error_component = clamp(1 - average_radius_normalized_error / 4) * 80
    stability_component = clamp(1 - error_std_dev / 3) * 20
    return round(clamp(error_component + stability_component, 0, 100), 2)


def clicking_score(
    success_rate: float,
    fail_inside_rate: float,
    fail_outside_rate: float,
    timeout_rate: float,
) -> float:
    score = (
        success_rate * 100
        - fail_inside_rate * 35
        - fail_outside_rate * 50
        - timeout_rate * 25
    )
    return round(clamp(score, 0, 100), 2)


def final_summary(session: Session) -> dict[str, float | str | None]:
    task_scores = {
        task_id: session.task_results.get(task_id).score
        if task_id in session.task_results
        else None
        for task_id in WEIGHTS
    }
    completed = {task_id: score for task_id, score in task_scores.items() if score is not None}
    if not completed:
        final = 0.0
    else:
        weight_total = sum(WEIGHTS[task_id] for task_id in completed)
        final = sum(completed[task_id] * WEIGHTS[task_id] for task_id in completed) / weight_total
    final = round(final, 2)
    return {
        "session_id": session.session_id,
        "movement_score": task_scores["movement"],
        "accuracy_score": task_scores["accuracy"],
        "tracking_score": task_scores["tracking"],
        "clicking_score": task_scores["clicking"],
        "final_score": final,
        "quality_label": rating_label(final),
    }

