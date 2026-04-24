from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from src.eye_tracking.controllers.gaze_cursor_controller import GazeCursorController


CALIB_WINDOW_NAME = "Gaze Calibration"


def draw_calibration_screen(
    screen_size: Tuple[int, int],
    target_norm: Tuple[float, float],
    step_index: int,
    step_total: int,
    msg: str,
) -> np.ndarray:
    width, height = screen_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    tx = int(round(target_norm[0] * (width - 1)))
    ty = int(round(target_norm[1] * (height - 1)))

    cv2.circle(canvas, (tx, ty), 34, (80, 80, 80), 2, cv2.LINE_AA)
    cv2.circle(canvas, (tx, ty), 16, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.line(canvas, (tx - 45, ty), (tx + 45, ty), (0, 180, 220), 2, cv2.LINE_AA)
    cv2.line(canvas, (tx, ty - 45), (tx, ty + 45), (0, 180, 220), 2, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"Calibration {step_index}/{step_total}",
        (36, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Look at the marker, then press SPACE to capture.",
        (36, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "ESC/q: quit  |  s: skip calibration",
        (36, 126),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (170, 170, 170),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        msg,
        (36, height - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def prepare_calibration_window(screen_w: int, screen_h: int) -> None:
    cv2.namedWindow(CALIB_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CALIB_WINDOW_NAME, screen_w, screen_h)
    cv2.moveWindow(CALIB_WINDOW_NAME, 0, 0)
    try:
        cv2.setWindowProperty(CALIB_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass


def capture_gaze_average(
    cap: cv2.VideoCapture,
    infer_gaze_from_frame: Callable[[np.ndarray], Optional[Tuple[float, float, np.ndarray, np.ndarray]]],
    sample_count: int = 24,
    max_frames: int = 140,
) -> Optional[Tuple[float, float, float, float]]:
    samples: List[Tuple[float, float]] = []
    for _ in range(max_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        try:
            result = infer_gaze_from_frame(frame_bgr)
        except Exception:
            result = None
        if result is None:
            continue

        pitch_rad, yaw_rad, _, _ = result
        yaw_adj = yaw_rad
        pitch_adj = pitch_rad
        samples.append((yaw_adj, pitch_adj))

        if len(samples) >= sample_count:
            break

    if len(samples) < max(8, sample_count // 2):
        return None

    arr = np.asarray(samples, dtype=np.float64)
    med = np.median(arr, axis=0)
    delta = arr - med.reshape(1, 2)
    dist = np.linalg.norm(delta, axis=1)
    keep = dist <= np.percentile(dist, 80)
    robust = arr[keep]
    if robust.shape[0] < 6:
        robust = arr

    mean_yaw = float(np.mean(robust[:, 0]))
    mean_pitch = float(np.mean(robust[:, 1]))
    std_yaw = float(np.std(robust[:, 0]))
    std_pitch = float(np.std(robust[:, 1]))
    return mean_yaw, mean_pitch, std_yaw, std_pitch


def run_cursor_calibration(
    cap: cv2.VideoCapture,
    infer_gaze_from_frame: Callable[[np.ndarray], Optional[Tuple[float, float, np.ndarray, np.ndarray]]],
    cursor_controller: GazeCursorController,
) -> bool:
    if (
        not cursor_controller.cursor_enabled
        or cursor_controller.cursor is None
        or cursor_controller.cursor_bounds is None
    ):
        return False

    minx, miny, maxx, maxy = cursor_controller.cursor_bounds
    screen_w = maxx - minx + 1
    screen_h = maxy - miny + 1
    if screen_w <= 10 or screen_h <= 10:
        print("warning: invalid screen bounds for calibration; skipping")
        return False

    target_norms = cursor_controller.calibration_points()
    collected_gaze: List[Tuple[float, float]] = []
    collected_targets: List[Tuple[float, float]] = []

    prepare_calibration_window(screen_w, screen_h)
    print("Starting startup gaze calibration. Follow the on-screen points.")

    for idx, target_norm in enumerate(target_norms, start=1):
        msg = "Hold gaze on target, press SPACE."
        while True:
            canvas = draw_calibration_screen(
                screen_size=(screen_w, screen_h),
                target_norm=target_norm,
                step_index=idx,
                step_total=len(target_norms),
                msg=msg,
            )
            cv2.imshow(CALIB_WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return False
            if key == ord("s"):
                print("Calibration skipped by user.")
                cv2.destroyWindow(CALIB_WINDOW_NAME)
                return False
            if key != ord(" "):
                continue

            sample = capture_gaze_average(
                cap=cap,
                infer_gaze_from_frame=infer_gaze_from_frame,
            )
            if sample is None:
                msg = "Face/gaze not stable. Re-align and press SPACE again."
                continue

            mean_yaw, mean_pitch, std_yaw, std_pitch = sample
            stability = max(std_yaw, std_pitch)
            if stability > 0.06:
                msg = f"Too noisy (stability={stability:.3f}). Keep still and retry."
                continue

            collected_gaze.append((mean_yaw, mean_pitch))
            collected_targets.append(target_norm)

            target_abs = cursor_controller.target_abs_point(cursor_controller.cursor_bounds, target_norm)
            cursor_controller.cursor.step_towards(*target_abs)

            msg = f"Captured: yaw={mean_yaw:.3f}, pitch={mean_pitch:.3f}"
            break

    cv2.destroyWindow(CALIB_WINDOW_NAME)

    gaze_np = np.asarray(collected_gaze, dtype=np.float64)
    target_np = np.asarray(collected_targets, dtype=np.float64)
    if gaze_np.shape[0] < 6:
        print("warning: insufficient calibration points; falling back to span mapping")
        return False

    ok = cursor_controller.fit_calibration(gaze_samples=gaze_np, target_points=target_np)
    if not ok:
        print("warning: calibration quality was not sufficient; falling back to span mapping")
        return False
    return True
