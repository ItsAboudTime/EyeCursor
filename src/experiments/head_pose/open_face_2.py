"""
Linux-only demo: control the mouse cursor with OpenFace head pose.

Requires OpenFace FeatureExtraction executable installed on the system.

Example:
  python -m examples.openface_head_cursor --feature-extraction /path/to/FeatureExtraction
"""

import argparse
import csv
import math
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.cursor import create_cursor
from src.ui.settings import SettingsWindow


class OpenFaceHeadPoseTracker:
    """
    Linux-only head pose tracker backed by OpenFace FeatureExtraction.

    It runs the OpenFace executable on webcam input and reads pose_Rx/pose_Ry
    from the generated CSV stream.
    """

    def __init__(
        self,
        feature_extraction_path: str = "FeatureExtraction",
        camera_index: int = 0,
        yaw_span: float = 20.0,
        pitch_span: float = 10.0,
        smooth_len: int = 8,
    ) -> None:
        self.feature_extraction_path = feature_extraction_path
        self.camera_index = int(camera_index)
        self.yaw_span = float(max(1e-3, yaw_span))
        self.pitch_span = float(max(1e-3, pitch_span))
        self.smooth_len = max(1, int(smooth_len))

        self.center_yaw = 0.0
        self.center_pitch = 0.0

        self._tmp_dir_obj: Optional[tempfile.TemporaryDirectory] = None
        self._csv_path: Optional[Path] = None
        self._proc: Optional[subprocess.Popen] = None
        self._csv_fp = None
        self._csv_reader = None
        self._idx_pose_rx: Optional[int] = None
        self._idx_pose_ry: Optional[int] = None
        self._idx_success: Optional[int] = None

        self._smoothed_angles: list[Tuple[float, float]] = []

    def start(self) -> None:
        if shutil.which(self.feature_extraction_path) is None and not Path(self.feature_extraction_path).exists():
            raise RuntimeError(
                "OpenFace FeatureExtraction executable was not found. "
                "Install OpenFace and ensure FeatureExtraction is in PATH, "
                "or pass its full path."
            )

        self._tmp_dir_obj = tempfile.TemporaryDirectory(prefix="openface_live_")
        out_dir = Path(self._tmp_dir_obj.name)
        self._csv_path = out_dir / "live.csv"

        cmd = [
            self.feature_extraction_path,
            "-device",
            str(self.camera_index),
            "-pose",
            "-gaze",
            "-out_dir",
            str(out_dir),
            "-of",
            "live",
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        deadline = time.time() + 8.0
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("OpenFace process exited early. Check OpenFace installation/camera access.")
            if self._csv_path.exists() and self._csv_path.stat().st_size > 0:
                break
            time.sleep(0.05)

        if not self._csv_path.exists():
            raise RuntimeError("OpenFace did not produce a CSV output file.")

    def stop(self) -> None:
        if self._csv_fp is not None:
            try:
                self._csv_fp.close()
            except Exception:
                pass
            self._csv_fp = None
            self._csv_reader = None

        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

        if self._tmp_dir_obj is not None:
            self._tmp_dir_obj.cleanup()
            self._tmp_dir_obj = None

        cv2.destroyAllWindows()

    def calibrate_center(self, yaw_deg: float, pitch_deg: float) -> None:
        self.center_yaw = float(yaw_deg)
        self.center_pitch = float(pitch_deg)

    def _ensure_csv_reader(self) -> None:
        if self._csv_reader is not None:
            return

        if self._csv_path is None:
            raise RuntimeError("Tracker not started.")

        self._csv_fp = open(self._csv_path, "r", newline="")
        self._csv_reader = csv.reader(self._csv_fp)

        header = None
        deadline = time.time() + 4.0
        while time.time() < deadline:
            try:
                header = next(self._csv_reader)
                if header:
                    break
            except StopIteration:
                time.sleep(0.03)
                continue

        if not header:
            raise RuntimeError("Unable to read OpenFace CSV header.")

        idx = {name.strip(): i for i, name in enumerate(header)}
        self._idx_pose_rx = idx.get("pose_Rx")
        self._idx_pose_ry = idx.get("pose_Ry")
        self._idx_success = idx.get("success")

        if self._idx_pose_rx is None or self._idx_pose_ry is None:
            raise RuntimeError("OpenFace CSV is missing pose_Rx/pose_Ry columns.")

    def _read_latest_angles_deg(self) -> Optional[Tuple[float, float]]:
        self._ensure_csv_reader()

        latest = None
        for row in self._csv_reader:
            if not row:
                continue
            if self._idx_success is not None:
                try:
                    if int(float(row[self._idx_success])) != 1:
                        continue
                except Exception:
                    continue

            try:
                pitch_rad = float(row[self._idx_pose_rx])
                yaw_rad = float(row[self._idx_pose_ry])
            except Exception:
                continue

            pitch_deg = math.degrees(pitch_rad)
            yaw_deg = math.degrees(yaw_rad)
            latest = (yaw_deg, pitch_deg)

        return latest

    def next_position(
        self,
        screen_w: int,
        screen_h: int,
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray, Optional[Tuple[float, float]]]:
        if self._proc is None:
            raise RuntimeError("Tracker not started. Call start() first.")

        if self._proc.poll() is not None:
            return None, self._status_frame("OpenFace process stopped"), None

        angles = self._read_latest_angles_deg()
        if angles is None:
            return None, self._status_frame("Waiting for OpenFace face tracking..."), None

        self._smoothed_angles.append(angles)
        if len(self._smoothed_angles) > self.smooth_len:
            self._smoothed_angles.pop(0)

        yaw = sum(v[0] for v in self._smoothed_angles) / len(self._smoothed_angles)
        pitch = sum(v[1] for v in self._smoothed_angles) / len(self._smoothed_angles)

        norm_x = np.clip(0.5 + (yaw - self.center_yaw) / (2.0 * self.yaw_span), 0.0, 1.0)
        norm_y = np.clip(0.5 - (pitch - self.center_pitch) / (2.0 * self.pitch_span), 0.0, 1.0)

        sx = int(norm_x * (screen_w - 1))
        sy = int(norm_y * (screen_h - 1))

        frame = self._status_frame(f"yaw={yaw:.2f} pitch={pitch:.2f}")
        return (sx, sy), frame, (yaw, pitch)

    @staticmethod
    def _status_frame(line2: str) -> np.ndarray:
        frame = np.zeros((240, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "OpenFace head-pose control", (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        cv2.putText(frame, line2, (16, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        cv2.putText(frame, "c: calibrate  q: quit", (16, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 120), 2)
        return frame


def run_tracking_loop(cur, tracker, stop_queue):
    minx, miny, maxx, maxy = cur.get_virtual_bounds()
    screen_w = maxx - minx + 1
    screen_h = maxy - miny + 1

    tracker.start()
    print("OpenFace head-cursor demo running. Press 'q' to quit, 'c' to calibrate.")

    while True:
        if not stop_queue.empty():
            if stop_queue.get() == "QUIT":
                break

        pos, frame, angles = tracker.next_position(screen_w, screen_h)

        if pos is not None:
            raw_tx, raw_ty = pos
            target_x = max(minx, min(maxx, raw_tx + minx))
            target_y = max(miny, min(maxy, raw_ty + miny))
            cur.step_towards(target_x, target_y)

        cv2.imshow("OpenFace Head Cursor (Linux)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            stop_queue.put("QUIT")
            break
        if key == ord("c") and angles is not None:
            yaw, pitch = angles
            tracker.calibrate_center(yaw, pitch)
            print("Calibration updated.")

    tracker.stop()
    cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cursor control using OpenFace head pose")
    parser.add_argument(
        "--feature-extraction",
        default="FeatureExtraction",
        help="Path to OpenFace FeatureExtraction executable (default: FeatureExtraction from PATH)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--yaw-span", type=float, default=20.0, help="Yaw span in degrees from center")
    parser.add_argument("--pitch-span", type=float, default=10.0, help="Pitch span in degrees from center")
    parser.add_argument("--smooth-len", type=int, default=8, help="Smoothing window length")
    return parser.parse_args()


def main():
    args = _parse_args()

    cur = create_cursor()
    tracker = OpenFaceHeadPoseTracker(
        feature_extraction_path=args.feature_extraction,
        camera_index=args.camera,
        yaw_span=args.yaw_span,
        pitch_span=args.pitch_span,
        smooth_len=args.smooth_len,
    )

    msg_queue = queue.Queue()

    try:
        root = SettingsWindow.create_app(cursor=cur)
    except Exception as e:
        print(f"Fatal Error: Could not start Tkinter: {e}")
        return 1

    def check_queue():
        try:
            msg = msg_queue.get_nowait()
            if msg == "QUIT":
                root.quit()
                root.destroy()
                return
        except queue.Empty:
            pass
        root.after(100, check_queue)

    t = threading.Thread(target=run_tracking_loop, args=(cur, tracker, msg_queue), daemon=True)
    t.start()

    check_queue()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
