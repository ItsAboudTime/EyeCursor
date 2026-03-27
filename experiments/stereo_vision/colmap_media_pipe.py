from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class ColmapScreenTarget:
	image_path: Path
	database_path: Path
	keypoint_count: int
	feature_x: float
	feature_y: float
	normalized_x: float
	normalized_y: float


class ColmapOneShotMapper:
	"""
	Capture one webcam frame, run COLMAP feature extraction, then map the
	extracted 2D features to a normalized pointer target.
	"""

	def __init__(
		self,
		camera_index: int = 0,
		colmap_binary: str = "colmap",
	) -> None:
		self.camera_index = int(camera_index)
		self.colmap_binary = colmap_binary

	def compute_normalized_target(
		self,
		work_dir: Optional[Path] = None,
	) -> ColmapScreenTarget:
		self._ensure_colmap_available()

		if work_dir is None:
			output_dir = Path(tempfile.mkdtemp(prefix="colmap_cursor_"))
		else:
			output_dir = Path(work_dir)
			output_dir.mkdir(parents=True, exist_ok=True)

		image_dir = output_dir / "images"
		image_dir.mkdir(parents=True, exist_ok=True)

		image_path = image_dir / "capture.jpg"
		database_path = output_dir / "database.db"

		width, height = self._capture_single_frame(image_path)
		self._run_feature_extractor(database_path=database_path, image_dir=image_dir)
		keypoints = self._load_keypoints(database_path=database_path, image_name=image_path.name)

		fx, fy, count = self._estimate_feature_focus(keypoints=keypoints, width=width, height=height)
		nx = 0.0 if width <= 1 else fx / float(width - 1)
		ny = 0.0 if height <= 1 else fy / float(height - 1)
		nx = float(np.clip(nx, 0.0, 1.0))
		ny = float(np.clip(ny, 0.0, 1.0))

		return ColmapScreenTarget(
			image_path=image_path,
			database_path=database_path,
			keypoint_count=count,
			feature_x=fx,
			feature_y=fy,
			normalized_x=nx,
			normalized_y=ny,
		)

	def to_screen_coords(
		self,
		normalized_x: float,
		normalized_y: float,
		bounds: Tuple[int, int, int, int],
	) -> Tuple[int, int]:
		minx, miny, maxx, maxy = bounds
		sx = int(round(minx + normalized_x * (maxx - minx)))
		sy = int(round(miny + normalized_y * (maxy - miny)))
		sx = max(minx, min(maxx, sx))
		sy = max(miny, min(maxy, sy))
		return sx, sy

	def _ensure_colmap_available(self) -> None:
		if shutil.which(self.colmap_binary) is None:
			raise RuntimeError(
				"COLMAP binary not found. Install COLMAP and ensure 'colmap' is in PATH."
			)

	def _capture_single_frame(self, image_path: Path) -> Tuple[int, int]:
		try:
			import cv2
		except ImportError as exc:
			raise RuntimeError("OpenCV is required to capture camera frames (pip install opencv-python).") from exc

		cap = cv2.VideoCapture(self.camera_index)
		if not cap.isOpened():
			raise RuntimeError(f"Could not open camera index {self.camera_index}")

		width = 0
		height = 0
		frame = None
		try:
			for _ in range(8):
				ok, warmed = cap.read()
				if ok:
					frame = warmed
			if frame is None:
				ok, frame = cap.read()
				if not ok:
					raise RuntimeError("Failed to capture frame from camera")

			height, width = frame.shape[:2]
			if not cv2.imwrite(str(image_path), frame):
				raise RuntimeError(f"Failed to save captured image to {image_path}")
		finally:
			cap.release()

		return int(width), int(height)

	def _run_feature_extractor(self, database_path: Path, image_dir: Path) -> None:
		cmd = [
			self.colmap_binary,
			"feature_extractor",
			"--database_path",
			str(database_path),
			"--image_path",
			str(image_dir),
			"--ImageReader.single_camera",
			"1",
			"--FeatureExtraction.use_gpu",
			"0",
		]

		proc = subprocess.run(cmd, capture_output=True, text=True)
		if proc.returncode != 0 and "unrecognised option '--FeatureExtraction.use_gpu'" in (
			proc.stderr or ""
		):
			legacy_cmd = cmd.copy()
			option_index = legacy_cmd.index("--FeatureExtraction.use_gpu")
			legacy_cmd[option_index] = "--SiftExtraction.use_gpu"
			proc = subprocess.run(legacy_cmd, capture_output=True, text=True)

		if proc.returncode != 0:
			err = proc.stderr.strip() or proc.stdout.strip() or "Unknown COLMAP error"
			raise RuntimeError(f"COLMAP feature_extractor failed: {err}")

	def _load_keypoints(self, database_path: Path, image_name: str) -> Optional[np.ndarray]:
		conn = sqlite3.connect(str(database_path))
		try:
			cur = conn.cursor()
			cur.execute(
				"""
				SELECT kp.rows, kp.cols, kp.data
				FROM keypoints AS kp
				JOIN images AS im ON kp.image_id = im.image_id
				WHERE im.name = ?
				LIMIT 1
				""",
				(image_name,),
			)
			row = cur.fetchone()
			if row is None:
				return None

			rows, cols, blob = row
			if rows is None or cols is None or blob is None:
				return None

			arr = np.frombuffer(blob, dtype=np.float32)
			if rows <= 0 or cols <= 0:
				return None
			expected = int(rows) * int(cols)
			if arr.size < expected:
				return None
			return arr[:expected].reshape(int(rows), int(cols))
		finally:
			conn.close()

	def _estimate_feature_focus(
		self,
		keypoints: Optional[np.ndarray],
		width: int,
		height: int,
	) -> Tuple[float, float, int]:
		if width <= 0 or height <= 0:
			raise RuntimeError("Captured image has invalid dimensions")

		if keypoints is None or keypoints.shape[0] == 0:
			return float(width / 2.0), float(height / 2.0), 0

		xy = keypoints[:, :2]
		if keypoints.shape[1] >= 3:
			weights = np.clip(keypoints[:, 2], 1e-6, None)
			fx = float(np.average(xy[:, 0], weights=weights))
			fy = float(np.average(xy[:, 1], weights=weights))
		else:
			fx = float(np.mean(xy[:, 0]))
			fy = float(np.mean(xy[:, 1]))

		fx = float(np.clip(fx, 0.0, float(width - 1)))
		fy = float(np.clip(fy, 0.0, float(height - 1)))
		return fx, fy, int(keypoints.shape[0])


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="One-shot COLMAP cursor control")
	parser.add_argument(
		"--mode",
		choices=("colmap", "headpose"),
		default="colmap",
		help="Tracking mode: 'colmap' (file/database workflow) or 'headpose' (real-time stream)",
	)
	parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0)")
	parser.add_argument(
		"--work-dir",
		type=str,
		default="",
		help="Optional directory to store captured image and COLMAP database",
	)
	parser.add_argument(
		"--keep-artifacts",
		action="store_true",
		help="Keep generated files when using implicit temporary workspace",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Compute target only, do not move mouse cursor",
	)
	parser.add_argument(
		"--left-click",
		action="store_true",
		help="Perform a left click after moving",
	)
	parser.add_argument(
		"--loop",
		action="store_true",
		help="Run continuously until interrupted (Ctrl+C)",
	)
	parser.add_argument(
		"--interval-sec",
		type=float,
		default=0.15,
		help="Delay between loop iterations in seconds (default: 0.15)",
	)
	parser.add_argument(
		"--yaw-span",
		type=float,
		default=20.0,
		help="Horizontal head-pose span in degrees for full-screen mapping (headpose mode)",
	)
	parser.add_argument(
		"--pitch-span",
		type=float,
		default=10.0,
		help="Vertical head-pose span in degrees for full-screen mapping (headpose mode)",
	)
	parser.add_argument(
		"--ema-alpha",
		type=float,
		default=0.25,
		help="EMA alpha for head-pose smoothing in (0, 1] (headpose mode)",
	)
	parser.add_argument(
		"--show-preview",
		action="store_true",
		help="Show camera preview window in headpose mode",
	)
	parser.add_argument(
		"--face-model-path",
		type=str,
		default="",
		help="Path to a local MediaPipe Face Landmarker .task model file (headpose mode)",
	)
	return parser


def _run_headpose_mode(cur, args) -> int:
	try:
		import cv2 as cv2_module
		from head_track.face_analysis_pipeline import FaceAnalysisPipeline
	except Exception as exc:
		raise RuntimeError(
			"Head-pose mode requires head_track dependencies (opencv-python, mediapipe)."
		) from exc

	face_analysis_pipeline = FaceAnalysisPipeline(
		yaw_span=args.yaw_span,
		pitch_span=args.pitch_span,
		ema_alpha=args.ema_alpha,
		face_model_path=args.face_model_path.strip() or None,
	)
	cap = cv2_module.VideoCapture(args.camera_index)
	if not cap.isOpened():
		raise RuntimeError(f"Could not open webcam (index {args.camera_index})")

	minx, miny, maxx, maxy = cur.get_virtual_bounds()
	screen_w = maxx - minx + 1
	screen_h = maxy - miny + 1

	interval_sec = max(0.0, float(args.interval_sec))
	cv2 = cv2_module if args.show_preview else None

	print("Real-time head-pose mode enabled. Press Ctrl+C to stop.")
	if args.show_preview:
		print("Preview enabled. Press 'q' in preview window to stop.")

	try:
		while True:
			ok, frame = cap.read()
			if not ok:
				continue

			rgb = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
			face_analysis = face_analysis_pipeline.analyze(
				rgb_frame=rgb,
				frame_width=frame.shape[1],
				frame_height=frame.shape[0],
				screen_width=screen_w,
				screen_height=screen_h,
			)
			if face_analysis is not None and face_analysis.screen_position is not None:
				raw_tx, raw_ty = face_analysis.screen_position
				target_x = max(minx, min(maxx, raw_tx + minx))
				target_y = max(miny, min(maxy, raw_ty + miny))
				cur.step_towards(target_x, target_y)

			if args.show_preview:
				assert cv2 is not None
				cv2.imshow("Head Pose Cursor", frame)
				key = cv2.waitKey(1) & 0xFF
				if key in (27, ord("q")):
					break

			if interval_sec > 0.0:
				time.sleep(interval_sec)
	except KeyboardInterrupt:
		print("Stopped head-pose mode.")
	finally:
		cap.release()
		face_analysis_pipeline.release()
		if args.show_preview and cv2 is not None:
			try:
				cv2.destroyAllWindows()
			except Exception:
				pass

	return 0


def main() -> int:
	from cursor import create_cursor

	parser = _build_parser()
	args = parser.parse_args()

	cur = create_cursor()
	if args.mode == "headpose":
		return _run_headpose_mode(cur=cur, args=args)

	minx, miny, maxx, maxy = cur.get_virtual_bounds()

	mapper = ColmapOneShotMapper(camera_index=args.camera_index)

	explicit_work_dir = bool(args.work_dir.strip())
	work_dir = Path(args.work_dir).expanduser().resolve() if explicit_work_dir else None

	def run_once(iteration: int = 0) -> None:
		iteration_work_dir: Optional[Path]
		if explicit_work_dir and args.loop:
			assert work_dir is not None
			iteration_work_dir = work_dir / f"iter_{iteration:06d}"
		else:
			iteration_work_dir = work_dir

		result = mapper.compute_normalized_target(work_dir=iteration_work_dir)
		sx, sy = mapper.to_screen_coords(
			normalized_x=result.normalized_x,
			normalized_y=result.normalized_y,
			bounds=(minx, miny, maxx, maxy),
		)

		mode_label = "COLMAP loop target" if args.loop else "COLMAP one-shot target"
		print(mode_label)
		print(f"- Virtual bounds: x [{minx}..{maxx}], y [{miny}..{maxy}]")
		print(f"- Captured image: {result.image_path}")
		print(f"- Database path: {result.database_path}")
		print(f"- Keypoints found: {result.keypoint_count}")
		print(f"- Feature focus (image px): ({result.feature_x:.1f}, {result.feature_y:.1f})")
		print(f"- Normalized target: ({result.normalized_x:.4f}, {result.normalized_y:.4f})")
		print(f"- Screen target: ({sx}, {sy})")

		if not args.dry_run:
			cur.move_to_with_speed(sx, sy)
			if args.left_click:
				cur.left_click()

		if not explicit_work_dir and not args.keep_artifacts:
			shutil.rmtree(result.image_path.parent.parent, ignore_errors=True)

	if args.loop:
		interval_sec = max(0.0, float(args.interval_sec))
		print("Continuous mode enabled. Press Ctrl+C to stop.")
		iteration = 0
		try:
			while True:
				run_once(iteration=iteration)
				iteration += 1
				if interval_sec > 0.0:
					time.sleep(interval_sec)
		except KeyboardInterrupt:
			print("Stopped continuous mode.")
	else:
		run_once(iteration=0)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
