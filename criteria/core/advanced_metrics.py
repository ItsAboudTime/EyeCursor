from __future__ import annotations

from math import hypot, log2, sqrt
from typing import Any

from criteria.core.metrics import avg, med, stddev
from criteria.core.models import Session


def compute_advanced_metrics(session: Session) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for task_id, func in _TASK_FUNCS.items():
        task_result = session.task_results.get(task_id)
        if task_result and task_result.raw:
            result[task_id] = func(task_result.raw)
        else:
            result[task_id] = {}
    return result


def _movement_metrics(raw: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        r for r in raw
        if r.get("completed") and r.get("movement_time_ms") and r["movement_time_ms"] > 0
    ]
    if not completed:
        return _null_movement()

    ids: list[float] = []
    tps: list[float] = []
    mts: list[float] = []

    by_radius: dict[float, list[float]] = {}
    mt_by_radius: dict[float, list[float]] = {}

    for trial in completed:
        d = trial["target_distance_px"]
        w = 2 * trial["target_radius"]
        mt_s = trial["movement_time_ms"] / 1000.0

        if d <= 0 or w <= 0:
            continue

        trial_id = log2(d / w + 1)
        trial_tp = trial_id / mt_s

        ids.append(trial_id)
        tps.append(trial_tp)
        mts.append(trial["movement_time_ms"])

        r = trial["target_radius"]
        by_radius.setdefault(r, []).append(trial_tp)
        mt_by_radius.setdefault(r, []).append(trial["movement_time_ms"])

    if not tps:
        return _null_movement()

    slope, intercept, r_sq = _linear_regression(ids, mts)

    effective_tp, effective_we, endpoint_sd = _effective_throughput(completed)

    return {
        "nominal_throughput_bps": round(avg(tps), 4),
        "median_throughput_bps": round(med(tps), 4),
        "effective_throughput_bps": round(effective_tp, 4) if effective_tp is not None else None,
        "effective_width_px": round(effective_we, 4) if effective_we is not None else None,
        "endpoint_sd_px": round(endpoint_sd, 4) if endpoint_sd is not None else None,
        "mean_index_of_difficulty": round(avg(ids), 4),
        "throughput_by_radius": {r: round(avg(v), 4) for r, v in sorted(by_radius.items())},
        "mean_mt_by_radius": {r: round(avg(v), 2) for r, v in sorted(mt_by_radius.items())},
        "fitts_regression_slope": round(slope, 4) if slope is not None else None,
        "fitts_regression_intercept": round(intercept, 4) if intercept is not None else None,
        "fitts_regression_r_squared": round(r_sq, 4) if r_sq is not None else None,
    }


def _effective_throughput(
    completed: list[dict[str, Any]],
) -> tuple[float | None, float | None, float | None]:
    has_cursor = [t for t in completed if "cursor_x" in t and "cursor_y" in t]
    if len(has_cursor) < 2:
        return None, None, None

    dx = [t["cursor_x"] - t["target_x"] for t in has_cursor]
    sd_x = stddev(dx)
    we = 4.133 * sd_x if sd_x > 0 else None
    if we is None or we < 1:
        return None, None, sd_x

    de = avg([t["target_distance_px"] for t in has_cursor])
    ide = log2(de / we + 1)
    mean_mt = avg([t["movement_time_ms"] for t in has_cursor]) / 1000.0
    if mean_mt <= 0:
        return None, we, sd_x

    return ide / mean_mt, we, sd_x


def _null_movement() -> dict[str, Any]:
    return {
        "nominal_throughput_bps": None,
        "median_throughput_bps": None,
        "effective_throughput_bps": None,
        "effective_width_px": None,
        "endpoint_sd_px": None,
        "mean_index_of_difficulty": None,
        "throughput_by_radius": {},
        "mean_mt_by_radius": {},
        "fitts_regression_slope": None,
        "fitts_regression_intercept": None,
        "fitts_regression_r_squared": None,
    }


def _accuracy_metrics(raw: list[dict[str, Any]]) -> dict[str, Any]:
    if not raw:
        return _null_accuracy()

    offsets_x = [r["cursor_x"] - r["target_x"] for r in raw]
    offsets_y = [r["cursor_y"] - r["target_y"] for r in raw]
    errors = [r["pixel_error"] for r in raw]

    prec_x = stddev(offsets_x)
    prec_y = stddev(offsets_y)

    return {
        "precision_x_px": round(prec_x, 3),
        "precision_y_px": round(prec_y, 3),
        "precision_2d_px": round(sqrt(prec_x ** 2 + prec_y ** 2), 3),
        "bias_x_px": round(avg(offsets_x), 3),
        "bias_y_px": round(avg(offsets_y), 3),
        "bias_magnitude_px": round(hypot(avg(offsets_x), avg(offsets_y)), 3),
        "rms_pixel_error": round(sqrt(avg([e ** 2 for e in errors])), 3),
    }


def _null_accuracy() -> dict[str, Any]:
    return {
        "precision_x_px": None,
        "precision_y_px": None,
        "precision_2d_px": None,
        "bias_x_px": None,
        "bias_y_px": None,
        "bias_magnitude_px": None,
        "rms_pixel_error": None,
    }


def _tracking_metrics(raw: list[dict[str, Any]]) -> dict[str, Any]:
    if len(raw) < 2:
        return _null_tracking()

    on_target = sum(1 for r in raw if r["pixel_error"] <= r["target_radius"])
    total = len(raw)

    cursor_path = 0.0
    target_path = 0.0
    direction_changes = 0
    prev_dx = 0.0
    prev_dy = 0.0

    for i in range(1, len(raw)):
        cx = raw[i]["cursor_x"] - raw[i - 1]["cursor_x"]
        cy = raw[i]["cursor_y"] - raw[i - 1]["cursor_y"]
        cursor_path += hypot(cx, cy)

        tx = raw[i]["target_x"] - raw[i - 1]["target_x"]
        ty = raw[i]["target_y"] - raw[i - 1]["target_y"]
        target_path += hypot(tx, ty)

        if i > 1 and (cx * prev_dx < 0 or cy * prev_dy < 0):
            direction_changes += 1

        prev_dx, prev_dy = cx, cy

    velocity_count = len(raw) - 1
    duration_s = (raw[-1]["timestamp_ms"] - raw[0]["timestamp_ms"]) / 1000.0
    path_eff = target_path / cursor_path if cursor_path > 0 else 0.0

    return {
        "pct_time_on_target": round(on_target / total, 4),
        "on_target_sample_count": on_target,
        "total_sample_count": total,
        "cursor_path_length_px": round(cursor_path, 2),
        "target_path_length_px": round(target_path, 2),
        "path_efficiency": round(path_eff, 4),
        "direction_change_rate": round(direction_changes / max(velocity_count - 1, 1), 4),
        "mean_cursor_speed_px_per_s": round(cursor_path / duration_s, 2) if duration_s > 0 else 0.0,
    }


def _null_tracking() -> dict[str, Any]:
    return {
        "pct_time_on_target": None,
        "on_target_sample_count": None,
        "total_sample_count": None,
        "cursor_path_length_px": None,
        "target_path_length_px": None,
        "path_efficiency": None,
        "direction_change_rate": None,
        "mean_cursor_speed_px_per_s": None,
    }


def _clicking_metrics(raw: list[dict[str, Any]]) -> dict[str, Any]:
    clicked = [r for r in raw if r.get("click_x") is not None]
    if not clicked:
        return _null_clicking()

    offsets_x = [r["click_x"] - r["target_x"] for r in clicked]
    offsets_y = [r["click_y"] - r["target_y"] for r in clicked]
    distances = [hypot(ox, oy) for ox, oy in zip(offsets_x, offsets_y)]

    scatter_x = stddev(offsets_x)
    scatter_y = stddev(offsets_y)

    times = [r["time_to_click_ms"] for r in clicked if r.get("time_to_click_ms") is not None]

    return {
        "click_scatter_x_px": round(scatter_x, 3),
        "click_scatter_y_px": round(scatter_y, 3),
        "click_scatter_2d_px": round(sqrt(scatter_x ** 2 + scatter_y ** 2), 3),
        "click_bias_x_px": round(avg(offsets_x), 3),
        "click_bias_y_px": round(avg(offsets_y), 3),
        "click_bias_magnitude_px": round(hypot(avg(offsets_x), avg(offsets_y)), 3),
        "mean_click_distance_px": round(avg(distances), 3),
        "median_time_to_click_ms": round(med(times), 2),
        "stddev_time_to_click_ms": round(stddev(times), 2),
    }


def _null_clicking() -> dict[str, Any]:
    return {
        "click_scatter_x_px": None,
        "click_scatter_y_px": None,
        "click_scatter_2d_px": None,
        "click_bias_x_px": None,
        "click_bias_y_px": None,
        "click_bias_magnitude_px": None,
        "mean_click_distance_px": None,
        "median_time_to_click_ms": None,
        "stddev_time_to_click_ms": None,
    }


def _linear_regression(
    xs: list[float], ys: list[float],
) -> tuple[float | None, float | None, float | None]:
    n = len(xs)
    if n < 2:
        return None, None, None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    if ss_xx == 0:
        return None, None, None
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    ss_yy = sum((y - mean_y) ** 2 for y in ys)
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    r_sq = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 0 else 0.0
    return slope, intercept, r_sq


_TASK_FUNCS: dict[str, Any] = {
    "movement": _movement_metrics,
    "accuracy": _accuracy_metrics,
    "tracking": _tracking_metrics,
    "clicking": _clicking_metrics,
}
