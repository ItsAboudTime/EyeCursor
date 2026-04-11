#!/usr/bin/env python3
"""
Low-latency stereo camera preview with CLAHE preprocessing.

Shows a single 2x2 view:
  Top-left:   Left camera raw
  Top-right:  Right camera raw
  Bottom-left: Left camera after CLAHE
  Bottom-right: Right camera after CLAHE

Controls:
  - q or Esc: quit

Example:
  python hardware/stereo_clahe_preview.py --left-index 0 --right-index 1
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraConfig:
    index: int
    width: int
    height: int
    fps: int
    mjpeg: bool


class CameraReader:
    """Continuously grabs frames in a background thread and keeps only the latest."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = cv2.VideoCapture(config.index)
        self._configure_capture()

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {config.index}")

        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread = None

    def _configure_capture(self) -> None:
        if self.config.mjpeg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.001)
                continue

            with self._lock:
                self._frame = frame

    def latest(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()


def apply_clahe_bgr(frame: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
    """Apply CLAHE only on luminance for speed and color stability."""
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def put_label(frame: np.ndarray, text: str) -> None:
    cv2.putText(
        frame,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo CLAHE preview (optimized for Raspberry Pi 4)")
    parser.add_argument("--left-index", type=int, default=0, help="Left camera index")
    parser.add_argument("--right-index", type=int, default=1, help="Right camera index")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS")
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument("--tile-size", type=int, default=8, help="CLAHE tile size (NxN)")
    parser.add_argument("--no-mjpeg", action="store_true", help="Disable MJPEG camera mode")
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="Scale final preview window (e.g., 0.75 for smaller display)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    left_cfg = CameraConfig(
        index=args.left_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        mjpeg=not args.no_mjpeg,
    )
    right_cfg = CameraConfig(
        index=args.right_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        mjpeg=not args.no_mjpeg,
    )

    left_cam = None
    right_cam = None

    try:
        left_cam = CameraReader(left_cfg)
        right_cam = CameraReader(right_cfg)
        left_cam.start()
        right_cam.start()

        clahe = cv2.createCLAHE(clipLimit=args.clip_limit, tileGridSize=(args.tile_size, args.tile_size))

        cv2.namedWindow("Stereo CLAHE Preview", cv2.WINDOW_NORMAL)

        print("Running stereo CLAHE preview...")
        print("Press 'q' or Esc to quit.")
        print(f"Left index={args.left_index}, Right index={args.right_index}, {args.width}x{args.height}@{args.fps}")

        while True:
            left_raw = left_cam.latest()
            right_raw = right_cam.latest()

            if left_raw is None or right_raw is None:
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue

            if left_raw.shape != right_raw.shape:
                right_raw = cv2.resize(right_raw, (left_raw.shape[1], left_raw.shape[0]), interpolation=cv2.INTER_LINEAR)

            left_proc = apply_clahe_bgr(left_raw, clahe)
            right_proc = apply_clahe_bgr(right_raw, clahe)

            put_label(left_raw, "LEFT RAW")
            put_label(right_raw, "RIGHT RAW")
            put_label(left_proc, "LEFT CLAHE")
            put_label(right_proc, "RIGHT CLAHE")

            top = cv2.hconcat([left_raw, right_raw])
            bottom = cv2.hconcat([left_proc, right_proc])
            preview = cv2.vconcat([top, bottom])

            if args.display_scale != 1.0:
                preview = cv2.resize(
                    preview,
                    None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow("Stereo CLAHE Preview", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    finally:
        if left_cam is not None:
            left_cam.stop()
        if right_cam is not None:
            right_cam.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
