"""
Eye gaze demo using MediaPipe Tasks.

Tracks iris landmarks from webcam input, maps gaze to screen coordinates,
and visualizes a streamer-style gaze bubble on the camera preview.

Controls:
  - c: calibrate center gaze
  - m: toggle real mouse movement
  - q / ESC: quit
"""

import sys
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from cursor import create_cursor
from head_track.tasks_face_landmarks import FaceLandmarksProvider


class EyeGazeTracker:
	def __init__(
		self,
		camera_index: int = 0,
		smooth_len: int = 8,
		face_model_path: Optional[str] = None,
	) -> None:
		self.camera_index = int(camera_index)
		self.smooth_len = int(smooth_len)

		self._landmarks_provider = FaceLandmarksProvider(face_model_path=face_model_path)

		self._cap: Optional[cv2.VideoCapture] = None
		self._smoothed_xy: deque[Tuple[float, float]] = deque(maxlen=self.smooth_len)

		# Calibration center in normalized eye-space.
		self.center_h = 0.5
		self.center_v = 0.5

		# Sensitivity: smaller span means stronger movement.
		self.h_span = 0.18
		self.v_span = 0.16

		# MediaPipe eye landmarks (with refine_landmarks=True).
		self._EYE = {
			"right": {"outer": 33, "inner": 133, "upper": 159, "lower": 145, "iris": 468},
			"left": {"outer": 362, "inner": 263, "upper": 386, "lower": 374, "iris": 473},
		}

	def start(self) -> None:
		cap = cv2.VideoCapture(self.camera_index)
		if not cap.isOpened():
			raise RuntimeError(f"Could not open webcam (index {self.camera_index})")
		self._cap = cap

	def stop(self) -> None:
		if self._cap is not None:
			self._cap.release()
			self._cap = None
		self._landmarks_provider.release()
		cv2.destroyAllWindows()

	def calibrate_center(self, h_ratio: float, v_ratio: float) -> None:
		self.center_h = float(h_ratio)
		self.center_v = float(v_ratio)

	@staticmethod
	def _lmk_to_px(lmk, w: int, h: int) -> np.ndarray:
		return np.array([lmk.x * w, lmk.y * h], dtype=np.float32)

	@staticmethod
	def _compute_eye_ratio(eye_pts: dict, lm, w: int, h: int) -> Tuple[float, float]:
		outer = EyeGazeTracker._lmk_to_px(lm[eye_pts["outer"]], w, h)
		inner = EyeGazeTracker._lmk_to_px(lm[eye_pts["inner"]], w, h)
		upper = EyeGazeTracker._lmk_to_px(lm[eye_pts["upper"]], w, h)
		lower = EyeGazeTracker._lmk_to_px(lm[eye_pts["lower"]], w, h)
		iris = EyeGazeTracker._lmk_to_px(lm[eye_pts["iris"]], w, h)

		h_ratio = (iris[0] - outer[0]) / ((inner[0] - outer[0]) + 1e-6)
		v_ratio = (iris[1] - upper[1]) / ((lower[1] - upper[1]) + 1e-6)

		return float(np.clip(h_ratio, 0.0, 1.0)), float(np.clip(v_ratio, 0.0, 1.0))

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

		h, w, _ = frame.shape
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		lm = self._landmarks_provider.get_primary_face_landmarks(rgb)
		if lm is None:
			return None, frame, None

		rh, rv = self._compute_eye_ratio(self._EYE["right"], lm, w, h)
		lh, lv = self._compute_eye_ratio(self._EYE["left"], lm, w, h)

		gaze_h = (rh + lh) * 0.5
		gaze_v = (rv + lv) * 0.5

		norm_x = np.clip(0.5 + (gaze_h - self.center_h) / self.h_span, 0.0, 1.0)
		norm_y = np.clip(0.5 + (gaze_v - self.center_v) / self.v_span, 0.0, 1.0)

		sx = float(norm_x * (screen_w - 1))
		sy = float(norm_y * (screen_h - 1))

		self._smoothed_xy.append((sx, sy))
		avg_x = float(np.mean([p[0] for p in self._smoothed_xy]))
		avg_y = float(np.mean([p[1] for p in self._smoothed_xy]))

		return (int(avg_x), int(avg_y)), frame, (gaze_h, gaze_v)


def run_demo() -> int:
	cur = create_cursor()
	tracker = EyeGazeTracker(camera_index=0, smooth_len=8)

	minx, miny, maxx, maxy = cur.get_virtual_bounds()
	screen_w = maxx - minx + 1
	screen_h = maxy - miny + 1

	tracker.start()
	move_mouse = False

	print("Eye gaze demo started.")
	print("Controls: c=calibrate, m=toggle mouse move, q=quit")

	try:
		while True:
			pos, frame, gaze = tracker.next_position(screen_w, screen_h)

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
				"c: calibrate  m: toggle mouse  q: quit",
				(10, 58),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.55,
				(255, 255, 255),
				2,
			)

			cv2.imshow("Eye Gaze Pointer", frame)
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


def main() -> int:
	return run_demo()


if __name__ == "__main__":
	sys.exit(main())
