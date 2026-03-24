"""
Eye-gaze cursor control using L2CS-Net.

This demo reads gaze angles from L2CS and maps them to screen coordinates,
then moves the OS mouse cursor.

Controls:
  - c: calibrate current gaze as center
  - m: toggle mouse movement on/off
  - q / ESC: quit

Usage:
  python -m examples.eye_gaze_l2cs_cursor --weights /path/to/L2CSNet_gaze360.pkl
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from cursor import create_cursor


class L2CSGazeTracker:
    def __init__(
        self,
        weights: pathlib.Path,
        arch: str = "ResNet50",
        device: str = "cpu",
        camera_index: int = 0,
        confidence_threshold: float = 0.6,
        smooth_len: int = 8,
        yaw_span_deg: float = 35.0,
        pitch_span_deg: float = 22.0,
    ) -> None:
        self.weights = pathlib.Path(weights).expanduser().resolve()
        self.arch = arch
        self.device_name = device
        self.camera_index = int(camera_index)
        self.confidence_threshold = float(confidence_threshold)
        self._smoothed_xy: deque[Tuple[float, float]] = deque(maxlen=max(1, int(smooth_len)))

        if not self.weights.exists():
            raise FileNotFoundError(
                f"Weights file not found: {self.weights}\n"
                "Download L2CSNet_gaze360.pkl and pass its path with --weights."
            )

        try:
            import torch
            from l2cs import Pipeline
        except ImportError as exc:
            raise ImportError(
                "Missing dependencies. Install with: pip install l2cs torch torchvision"
            ) from exc

        self._torch = torch
        self._pipeline = Pipeline(
            weights=self.weights,
            arch=self.arch,
            device=self._select_device(device),
            include_detector=True,
            confidence_threshold=self.confidence_threshold,
        )

        self._cap: Optional[cv2.VideoCapture] = None

        # Calibrated neutral gaze center (radians).
        self.center_yaw = 0.0
        self.center_pitch = 0.0

        # Range from center that maps to screen edges.
        self.yaw_span = float(np.deg2rad(yaw_span_deg))
        self.pitch_span = float(np.deg2rad(pitch_span_deg))

    def _select_device(self, name: str):
        name = name.strip().lower()
        if name == "auto":
            if self._torch.cuda.is_available():
                return self._torch.device("cuda:0")
            return self._torch.device("cpu")
        if name == "cuda" and self._torch.cuda.is_available():
            return self._torch.device("cuda:0")
        return self._torch.device("cpu")

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

    @staticmethod
    def _pick_primary_face(results) -> Optional[int]:
        if results is None:
            return None
        if not hasattr(results, "bboxes") or not hasattr(results, "yaw") or not hasattr(results, "pitch"):
            return None

        bboxes = np.asarray(results.bboxes)
        yaw = np.asarray(results.yaw)
        pitch = np.asarray(results.pitch)

        if bboxes.ndim != 2 or bboxes.shape[0] == 0:
            return None
        if yaw.size == 0 or pitch.size == 0:
            return None

        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        if areas.size == 0:
            return None
        return int(np.argmax(areas))

    @staticmethod
    def _extract_primary_angles(results, idx: int) -> Optional[Tuple[float, float]]:
        try:
            yaw_vals = np.asarray(getattr(results, "yaw", np.empty((0,))))
            pitch_vals = np.asarray(getattr(results, "pitch", np.empty((0,))))
            yaw = float(yaw_vals.reshape(-1)[idx])
            pitch = float(pitch_vals.reshape(-1)[idx])
            return yaw, pitch
        except Exception:
            return None

    def next_position(
        self,
        screen_w: int,
        screen_h: int,
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray, Optional[Tuple[float, float]], object]:
        if self._cap is None:
            raise RuntimeError("Tracker not started. Call start() first.")

        ok, frame = self._cap.read()
        if not ok:
            return None, np.zeros((1, 1, 3), dtype=np.uint8), None, None

        try:
            results = self._pipeline.step(frame)
        except Exception:
            # Some l2cs versions can error when no faces are found in frame.
            results = None

        if results is None:
            return None, frame, None, None

        idx = self._pick_primary_face(results)
        if idx is None:
            return None, frame, None, results

        angles = self._extract_primary_angles(results, idx)
        if angles is None:
            return None, frame, None, results
        yaw, pitch = angles

        norm_x = np.clip(0.5 + (yaw - self.center_yaw) / self.yaw_span, 0.0, 1.0)
        norm_y = np.clip(0.5 + (pitch - self.center_pitch) / self.pitch_span, 0.0, 1.0)

        sx = float(norm_x * max(0, screen_w - 1))
        sy = float(norm_y * max(0, screen_h - 1))

        self._smoothed_xy.append((sx, sy))
        avg_x = float(np.mean([p[0] for p in self._smoothed_xy]))
        avg_y = float(np.mean([p[1] for p in self._smoothed_xy]))

        return (int(avg_x), int(avg_y)), frame, (yaw, pitch), results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control the cursor with L2CS gaze angles.")
    parser.add_argument(
        "--weights",
        type=pathlib.Path,
        required=True,
        help="Path to the pretrained L2CS weights file (e.g. L2CSNet_gaze360.pkl).",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument("--arch", type=str, default="ResNet50", help="Backbone architecture.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument("--smooth-len", type=int, default=8, help="Gaze smoothing window size.")
    parser.add_argument("--yaw-span", type=float, default=35.0, help="Yaw span in degrees.")
    parser.add_argument("--pitch-span", type=float, default=22.0, help="Pitch span in degrees.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Face detector confidence threshold.",
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="Start with mouse movement disabled (toggle with 'm').",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        cur = create_cursor()
    except Exception as exc:
        print(f"Failed to initialize cursor backend: {exc}")
        return 1

    try:
        tracker = L2CSGazeTracker(
            weights=args.weights,
            arch=args.arch,
            device=args.device,
            camera_index=args.camera,
            confidence_threshold=args.confidence_threshold,
            smooth_len=args.smooth_len,
            yaw_span_deg=args.yaw_span,
            pitch_span_deg=args.pitch_span,
        )
    except Exception as exc:
        print(f"Failed to initialize L2CS tracker: {exc}")
        return 1

    minx, miny, maxx, maxy = cur.get_virtual_bounds()
    screen_w = maxx - minx + 1
    screen_h = maxy - miny + 1

    move_mouse = not args.start_paused

    try:
        tracker.start()
    except Exception as exc:
        print(f"Could not start webcam tracker: {exc}")
        return 1

    print("L2CS eye-gaze cursor demo started.")
    print("Controls: c=calibrate center, m=toggle mouse movement, q=quit")

    try:
        while True:
            pos, frame, gaze, results = tracker.next_position(screen_w, screen_h)

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
                cv2.circle(frame, (bubble_x, bubble_y), 14, (0, 255, 255), -1)
                cv2.circle(frame, (bubble_x, bubble_y), 20, (0, 255, 255), 2)

                if gaze is not None:
                    yaw, pitch = gaze
                    cv2.putText(
                        frame,
                        f"yaw={yaw:+.3f} pitch={pitch:+.3f}",
                        (10, frame.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )

            if results is not None:
                bboxes = np.asarray(getattr(results, "bboxes", np.empty((0, 4))))
                for bbox in bboxes:
                    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 0), 1)

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

            cv2.imshow("L2CS Eye Gaze Cursor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                move_mouse = not move_mouse
            if key == ord("c") and gaze is not None:
                tracker.calibrate_center(gaze[0], gaze[1])
                print("Calibration updated.")
    finally:
        tracker.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
