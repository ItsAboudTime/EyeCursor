"""
Linux-only eye gaze demo using L2CS-Net.

Reads face input from the webcam, predicts gaze with L2CS-Net, and maps the
predicted gaze direction to cursor coordinates.

Controls:
  - c: calibrate center gaze
  - m: toggle real mouse movement
  - q / ESC: quit

Example:
  python -m cursor.experiments.eye_gaze_l2cs_net --weights /path/to/L2CSNet_gaze360.pkl
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from cursor import create_cursor


def _build_pipeline(weights_path: Path, prefer_cuda: bool) -> tuple[Any, Any, Optional[Any]]:
    """Create L2CS pipeline and optional render helper."""
    from l2cs import Pipeline

    render_fn = None
    try:
        from l2cs import render as render_fn
    except Exception:
        render_fn = None

    import torch

    device = torch.device("cuda:0" if prefer_cuda and torch.cuda.is_available() else "cpu")
    pipeline = Pipeline(weights=weights_path, arch="ResNet50", device=device)
    return pipeline, device, render_fn


def _extract_angles_and_anchor(results: Any, frame_w: int, frame_h: int) -> Optional[Tuple[float, float, float, float]]:
    """Extract first-face yaw/pitch and an anchor point from L2CS results."""

    def _first_val(val: Any) -> Optional[float]:
        if val is None:
            return None
        if isinstance(val, (float, int)):
            return float(val)
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return None
            return float(val.flatten()[0])
        if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            if not val:
                return None
            return float(val[0])
        return None

    yaw = pitch = None
    anchor_x = frame_w * 0.5
    anchor_y = frame_h * 0.5

    if results is None:
        return None

    if isinstance(results, dict):
        yaw = _first_val(results.get("yaw") or results.get("yaws"))
        pitch = _first_val(results.get("pitch") or results.get("pitches"))
        bboxes = results.get("bbox") or results.get("bboxes")
        if bboxes is not None:
            box = None
            if isinstance(bboxes, np.ndarray) and bboxes.size >= 4:
                box = bboxes.reshape(-1, 4)[0]
            elif isinstance(bboxes, Sequence) and len(bboxes) > 0:
                box = bboxes[0]
            if box is not None:
                x1, y1, x2, y2 = [float(v) for v in list(box)[:4]]
                anchor_x = 0.5 * (x1 + x2)
                anchor_y = 0.5 * (y1 + y2)

    if yaw is None and hasattr(results, "yaw"):
        yaw = _first_val(getattr(results, "yaw"))
    if pitch is None and hasattr(results, "pitch"):
        pitch = _first_val(getattr(results, "pitch"))

    if hasattr(results, "bboxes"):
        bboxes = getattr(results, "bboxes")
        if isinstance(bboxes, np.ndarray) and bboxes.size >= 4:
            x1, y1, x2, y2 = [float(v) for v in bboxes.reshape(-1, 4)[0]]
            anchor_x = 0.5 * (x1 + x2)
            anchor_y = 0.5 * (y1 + y2)

    if yaw is None or pitch is None:
        return None

    return yaw, pitch, anchor_x, anchor_y


class L2CSGazeTracker:
    def __init__(
        self,
        weights_path: Path,
        camera_index: int = 0,
        smooth_len: int = 6,
        yaw_span: float = 0.9,
        pitch_span: float = 0.9,
        prefer_cuda: bool = False,
    ) -> None:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("L2CS-Net tracker currently supports Linux only.")

        self.camera_index = int(camera_index)
        self.center_yaw = 0.0
        self.center_pitch = 0.0
        self.yaw_span = float(max(1e-3, yaw_span))
        self.pitch_span = float(max(1e-3, pitch_span))

        self._pipeline, self.device, self._render_fn = _build_pipeline(weights_path=weights_path, prefer_cuda=prefer_cuda)
        self._cap: Optional[cv2.VideoCapture] = None
        self._smoothed_xy: deque[Tuple[float, float]] = deque(maxlen=max(1, int(smooth_len)))

    def start(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam (index {self.camera_index})")
        self._cap = cap

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()

    def calibrate_center(self, yaw: float, pitch: float) -> None:
        self.center_yaw = float(yaw)
        self.center_pitch = float(pitch)

    def next_position(
        self,
        screen_w: int,
        screen_h: int,
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray, Optional[Tuple[float, float]]]:
        if self._cap is None:
            raise RuntimeError("Tracker not started. Call start() first.")

        ok, frame = self._cap.read()
        if not ok:
            return None, np.zeros((1, 1, 3), dtype=np.uint8), None

        results = self._pipeline.step(frame)

        display = frame
        if self._render_fn is not None:
            try:
                display = self._render_fn(frame.copy(), results)
            except Exception:
                display = frame

        frame_h, frame_w, _ = frame.shape
        extracted = _extract_angles_and_anchor(results, frame_w=frame_w, frame_h=frame_h)
        if extracted is None:
            return None, display, None

        yaw, pitch, _, _ = extracted

        norm_x = np.clip(0.5 + (yaw - self.center_yaw) / self.yaw_span, 0.0, 1.0)
        norm_y = np.clip(0.5 + (pitch - self.center_pitch) / self.pitch_span, 0.0, 1.0)

        sx = float(norm_x * (screen_w - 1))
        sy = float(norm_y * (screen_h - 1))

        self._smoothed_xy.append((sx, sy))
        avg_x = float(np.mean([p[0] for p in self._smoothed_xy]))
        avg_y = float(np.mean([p[1] for p in self._smoothed_xy]))

        return (int(avg_x), int(avg_y)), display, (yaw, pitch)


def run_demo(weights_path: Path, camera_index: int, prefer_cuda: bool) -> int:
    if not sys.platform.startswith("linux"):
        print("This demo currently supports Linux only.")
        return 1

    if not weights_path.exists():
        print(f"L2CS-Net weights file not found: {weights_path}")
        return 1

    try:
        cur = create_cursor()
    except Exception as exc:
        print(f"Failed to initialize cursor backend: {exc}")
        return 1

    try:
        tracker = L2CSGazeTracker(
            weights_path=weights_path,
            camera_index=camera_index,
            smooth_len=8,
            yaw_span=0.9,
            pitch_span=0.9,
            prefer_cuda=prefer_cuda,
        )
    except ImportError:
        print("L2CS-Net dependencies are not installed. Install with: pip install l2cs torch torchvision")
        return 1

    minx, miny, maxx, maxy = cur.get_virtual_bounds()
    screen_w = maxx - minx + 1
    screen_h = maxy - miny + 1

    tracker.start()
    move_mouse = False

    print("L2CS-Net eye gaze demo started.")
    print("Controls: c=calibrate center, m=toggle mouse move, q=quit")
    print(f"Running on device: {tracker.device}")

    try:
        while True:
            pos, frame, angles = tracker.next_position(screen_w, screen_h)

            if pos is not None:
                gx, gy = pos
                target_x = max(minx, min(maxx, gx + minx))
                target_y = max(miny, min(maxy, gy + miny))

                if move_mouse:
                    cur.step_towards(target_x, target_y)

                fh, fw, _ = frame.shape
                pad = 16
                bubble_x = int(np.clip((gx / max(1, screen_w - 1)) * (fw - 2 * pad) + pad, pad, fw - pad))
                bubble_y = int(np.clip((gy / max(1, screen_h - 1)) * (fh - 2 * pad) + pad, pad, fh - pad))
                cv2.circle(frame, (bubble_x, bubble_y), 18, (0, 255, 255), -1)
                cv2.circle(frame, (bubble_x, bubble_y), 24, (0, 255, 255), 2)

                cv2.putText(
                    frame,
                    f"screen=({target_x},{target_y})",
                    (10, frame.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

            mode_txt = "MOUSE: ON" if move_mouse else "MOUSE: OFF"
            cv2.putText(frame, mode_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
            cv2.putText(
                frame,
                "c: calibrate center  m: toggle mouse  q: quit",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow("L2CS-Net Gaze Pointer (Linux)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                move_mouse = not move_mouse
            if key == ord("c") and angles is not None:
                yaw, pitch = angles
                tracker.calibrate_center(yaw, pitch)
                print("Calibration updated.")
    finally:
        tracker.stop()

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linux eye gaze cursor control using L2CS-Net")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to L2CS-Net weights file (e.g., L2CSNet_gaze360.pkl)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if available",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return run_demo(weights_path=args.weights, camera_index=args.camera, prefer_cuda=args.cuda)


if __name__ == "__main__":
    sys.exit(main())
