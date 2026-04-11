"""
Linux-only eye gaze demo using EyeGestures.

Tracks webcam frames through EyeGestures and maps predicted gaze points to the
system cursor area.

Controls:
  - c: toggle calibration mode
  - m: toggle real mouse movement
  - q / ESC: quit
"""

import sys
from collections import deque
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from src.cursor import create_cursor


def _create_eyegestures_engine() -> Any:
	"""Build the newest available EyeGestures engine."""
	try:
		from eyeGestures import EyeGestures_v3

		return EyeGestures_v3()
	except Exception:
		from eyeGestures import EyeGestures_v2

		return EyeGestures_v2()


class EyeGesturesTracker:
	def __init__(self, camera_index: int = 0, smooth_len: int = 6, context: str = "linux_eye_cursor") -> None:
		self.camera_index = int(camera_index)
		self.context = context
		self.calibrating = True

		self._cap: Optional[cv2.VideoCapture] = None
		self._engine = _create_eyegestures_engine()
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

	def next_position(self, screen_w: int, screen_h: int) -> Tuple[Optional[Tuple[int, int]], np.ndarray, Optional[Any]]:
		if self._cap is None:
			raise RuntimeError("Tracker not started. Call start() first.")

		ok, frame = self._cap.read()
		if not ok:
			return None, np.zeros((1, 1, 3), dtype=np.uint8), None

		event, _ = self._engine.step(
			frame,
			self.calibrating,
			screen_w,
			screen_h,
			context=self.context,
		)

		if event is None or not hasattr(event, "point") or event.point is None:
			return None, frame, event

		sx = float(np.clip(event.point[0], 0, max(0, screen_w - 1)))
		sy = float(np.clip(event.point[1], 0, max(0, screen_h - 1)))

		self._smoothed_xy.append((sx, sy))
		avg_x = float(np.mean([p[0] for p in self._smoothed_xy]))
		avg_y = float(np.mean([p[1] for p in self._smoothed_xy]))

		return (int(avg_x), int(avg_y)), frame, event


def run_demo() -> int:
	try:
		cur = create_cursor()
	except Exception as exc:
		print(f"Failed to initialize cursor backend: {exc}")
		return 1

	try:
		tracker = EyeGesturesTracker(camera_index=0, smooth_len=6)
	except ImportError:
		print("EyeGestures is not installed. Install it with: pip install eyeGestures")
		return 1

	minx, miny, maxx, maxy = cur.get_virtual_bounds()
	screen_w = maxx - minx + 1
	screen_h = maxy - miny + 1

	tracker.start()
	move_mouse = False

	print("EyeGestures demo started.")
	print("Controls: c=toggle calibration, m=toggle mouse move, q=quit")

	try:
		while True:
			pos, frame, event = tracker.next_position(screen_w, screen_h)

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

				bubble_color = (0, 255, 255)
				if event is not None and getattr(event, "fixation", False):
					bubble_color = (255, 200, 0)

				cv2.circle(frame, (bubble_x, bubble_y), 18, bubble_color, -1)
				cv2.circle(frame, (bubble_x, bubble_y), 24, bubble_color, 2)

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
			cal_txt = "CALIBRATION: ON" if tracker.calibrating else "CALIBRATION: OFF"
			cv2.putText(frame, mode_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
			cv2.putText(frame, cal_txt, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 220), 2)
			cv2.putText(
				frame,
				"c: calibration  m: toggle mouse  q: quit",
				(10, 86),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.55,
				(255, 255, 255),
				2,
			)

			cv2.imshow("EyeGestures Gaze Pointer (Linux)", frame)
			key = cv2.waitKey(1) & 0xFF
			if key in (27, ord("q")):
				break
			if key == ord("m"):
				move_mouse = not move_mouse
			if key == ord("c"):
				tracker.calibrating = not tracker.calibrating
				state = "ON" if tracker.calibrating else "OFF"
				print(f"Calibration mode: {state}")
	finally:
		tracker.stop()

	return 0


def main() -> int:
	return run_demo()


if __name__ == "__main__":
	sys.exit(main())
