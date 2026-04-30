from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.face_tracking.pipelines.face_analysis import FaceAnalysisResult
from src.face_tracking.providers.face_landmarks import FaceLandmarksProvider
from src.face_tracking.signals.blendshapes import extract_blendshapes
from src.face_tracking.signals.wink import detect_wink_direction_from_ratios, get_eye_aspect_ratios


@dataclass
class StereoCalibration:
    k1: np.ndarray
    d1: np.ndarray
    k2: np.ndarray
    d2: np.ndarray
    r: np.ndarray
    t: np.ndarray

    @classmethod
    def from_npz(cls, calibration_path: str) -> "StereoCalibration":
        path = Path(calibration_path).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(
                f"Stereo calibration file not found at: {path}. "
                "Create one with K1, D1, K2, D2, R, T arrays."
            )

        with np.load(path) as data:
            required = ("K1", "D1", "K2", "D2", "R", "T")
            missing = [name for name in required if name not in data]
            if missing:
                raise RuntimeError(
                    f"Stereo calibration file is missing keys: {', '.join(missing)}"
                )

            k1 = np.asarray(data["K1"], dtype=np.float64)
            d1 = np.asarray(data["D1"], dtype=np.float64)
            k2 = np.asarray(data["K2"], dtype=np.float64)
            d2 = np.asarray(data["D2"], dtype=np.float64)
            r = np.asarray(data["R"], dtype=np.float64)
            t = np.asarray(data["T"], dtype=np.float64).reshape(3, 1)

        if k1.shape != (3, 3) or k2.shape != (3, 3):
            raise RuntimeError("K1 and K2 must each have shape (3, 3).")
        if r.shape != (3, 3):
            raise RuntimeError("R must have shape (3, 3).")
        if t.shape != (3, 1):
            raise RuntimeError("T must have shape (3, 1).")

        return cls(k1=k1, d1=d1, k2=k2, d2=d2, r=r, t=t)


class StereoHeadPoseDepthMapper:
    """Maps triangulated 3D landmarks into yaw, pitch, depth and screen coordinates."""

    def __init__(
        self,
        yaw_span: float = 20.0,
        pitch_span: float = 10.0,
        ema_alpha: float = 0.25,
    ) -> None:
        self.yaw_span = float(yaw_span)
        self.pitch_span = float(pitch_span)
        self.ema_alpha = float(ema_alpha)

        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in the range (0, 1].")

        self._ema_direction: Optional[np.ndarray] = None
        self._ema_depth: Optional[float] = None

        self._calibration_yaw = 0.0
        self._calibration_pitch = 0.0

        self._landmark_indices = {
            "left": 234,
            "right": 454,
            "top": 10,
            "bottom": 152,
            "front": 1,
        }

    def calibrate_to_center(self, yaw: float, pitch: float) -> None:
        self._calibration_yaw = -float(yaw)
        self._calibration_pitch = -float(pitch)

    def estimate_screen_position(
        self,
        points_3d: Dict[int, np.ndarray],
        screen_width: int,
        screen_height: int,
        facial_transformation_matrix: Optional[Sequence[Sequence[float]]] = None,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[float, float], float]]:
        needed = [self._landmark_indices[name] for name in ("left", "right", "top", "bottom", "front")]
        if any(index not in points_3d for index in needed):
            return None

        front = points_3d[self._landmark_indices["front"]]

        forward_axis = self._forward_axis_from_matrix(facial_transformation_matrix)
        if forward_axis is None:
            forward_axis = self._forward_axis_from_points(points_3d)

        if self._ema_direction is None:
            self._ema_direction = forward_axis
        else:
            self._ema_direction = (
                self.ema_alpha * forward_axis + (1.0 - self.ema_alpha) * self._ema_direction
            )
            self._ema_direction /= np.linalg.norm(self._ema_direction) + 1e-9

        yaw, pitch = self._compute_angles(self._ema_direction)
        sx, sy = self._map_to_screen(yaw=yaw, pitch=pitch, screen_width=screen_width, screen_height=screen_height)

        depth = float(front[2])
        if self._ema_depth is None:
            self._ema_depth = depth
        else:
            self._ema_depth = self.ema_alpha * depth + (1.0 - self.ema_alpha) * self._ema_depth

        return (sx, sy), (yaw, pitch), float(self._ema_depth)

    def _forward_axis_from_matrix(
        self,
        facial_transformation_matrix: Optional[Sequence[Sequence[float]]],
    ) -> Optional[np.ndarray]:
        if facial_transformation_matrix is None:
            return None

        matrix = np.asarray(facial_transformation_matrix, dtype=float)
        if matrix.shape == (16,):
            matrix = matrix.reshape(4, 4)
        if matrix.shape != (4, 4):
            return None

        rotation = matrix[:3, :3]
        try:
            u, _, vh = np.linalg.svd(rotation)
            rotation = u @ vh
            if np.linalg.det(rotation) < 0:
                u[:, -1] *= -1.0
                rotation = u @ vh
        except Exception:
            pass

        forward_axis = -rotation[:, 2]
        norm = np.linalg.norm(forward_axis)
        if norm < 1e-9:
            return None

        forward_axis = forward_axis / norm
        if forward_axis[2] > 0.0:
            forward_axis = -forward_axis
        return forward_axis

    def _forward_axis_from_points(self, points_3d: Dict[int, np.ndarray]) -> np.ndarray:
        left = points_3d[self._landmark_indices["left"]]
        right = points_3d[self._landmark_indices["right"]]
        top = points_3d[self._landmark_indices["top"]]
        bottom = points_3d[self._landmark_indices["bottom"]]

        right_axis = right - left
        right_axis /= np.linalg.norm(right_axis) + 1e-9

        up_axis = top - bottom
        up_axis = up_axis - np.dot(up_axis, right_axis) * right_axis
        up_axis /= np.linalg.norm(up_axis) + 1e-9

        forward_axis = np.cross(right_axis, up_axis)
        forward_axis /= np.linalg.norm(forward_axis) + 1e-9
        forward_axis = -forward_axis

        if forward_axis[2] > 0.0:
            forward_axis = -forward_axis

        return forward_axis

    def _compute_angles(self, direction: np.ndarray) -> Tuple[float, float]:
        x, y, z = float(direction[0]), float(direction[1]), float(direction[2])

        yaw = float(np.degrees(np.arctan2(x, -z)))
        pitch = float(np.degrees(np.arctan2(y, -z)))

        yaw += self._calibration_yaw
        pitch += self._calibration_pitch

        return yaw, pitch

    def _map_to_screen(
        self,
        yaw: float,
        pitch: float,
        screen_width: int,
        screen_height: int,
    ) -> Tuple[int, int]:
        norm_x = (yaw + self.yaw_span) / (2.0 * self.yaw_span)
        norm_y = (pitch + self.pitch_span) / (2.0 * self.pitch_span)

        sx = int(norm_x * screen_width)
        sy = int(norm_y * screen_height)

        sx = max(0, min(screen_width - 1, sx))
        sy = max(0, min(screen_height - 1, sy))
        return sx, sy


class StereoTriangulator:
    """Triangulates selected face landmarks from synchronized left/right detections."""

    def __init__(self, calibration: StereoCalibration, landmark_indices: Sequence[int]) -> None:
        self._calibration = calibration
        self._landmark_indices = list(landmark_indices)

        self._p1 = np.hstack((np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)))
        self._p2 = np.hstack((self._calibration.r, self._calibration.t))

    def triangulate_from_landmarks(
        self,
        left_landmarks: Iterable,
        right_landmarks: Iterable,
        left_frame_width: int,
        left_frame_height: int,
        right_frame_width: int,
        right_frame_height: int,
    ) -> Dict[int, np.ndarray]:
        left_list = list(left_landmarks)
        right_list = list(right_landmarks)

        left_points = []
        right_points = []
        valid_indices = []

        for index in self._landmark_indices:
            if index >= len(left_list) or index >= len(right_list):
                continue

            lp = left_list[index]
            rp = right_list[index]
            left_points.append([float(lp.x) * left_frame_width, float(lp.y) * left_frame_height])
            right_points.append([float(rp.x) * right_frame_width, float(rp.y) * right_frame_height])
            valid_indices.append(index)

        if len(valid_indices) < 5:
            return {}

        left_arr = np.asarray(left_points, dtype=np.float64).reshape(-1, 1, 2)
        right_arr = np.asarray(right_points, dtype=np.float64).reshape(-1, 1, 2)

        undistorted_left = cv2.undistortPoints(
            left_arr,
            self._calibration.k1,
            self._calibration.d1,
        ).reshape(-1, 2)
        undistorted_right = cv2.undistortPoints(
            right_arr,
            self._calibration.k2,
            self._calibration.d2,
        ).reshape(-1, 2)

        points_4d = cv2.triangulatePoints(
            self._p1,
            self._p2,
            undistorted_left.T,
            undistorted_right.T,
        )
        points_3d = (points_4d[:3] / (points_4d[3] + 1e-9)).T

        return {index: points_3d[i] for i, index in enumerate(valid_indices)}


class StereoFaceAnalysisPipeline:
    """Stereo variant of FaceAnalysisPipeline using triangulated 3D landmarks."""

    def __init__(
        self,
        stereo_calibration: Optional[StereoCalibration] = None,
        stereo_calibration_path: Optional[str] = None,
        yaw_span: float = 20.0,
        pitch_span: float = 10.0,
        ema_alpha: float = 0.25,
        face_model_path: Optional[str] = None,
    ) -> None:
        self._left_provider = FaceLandmarksProvider(face_model_path=face_model_path)
        self._right_provider = FaceLandmarksProvider(face_model_path=face_model_path)

        if stereo_calibration is not None:
            calibration = stereo_calibration
        else:
            if not stereo_calibration_path:
                raise RuntimeError(
                    "Stereo calibration is required. Provide either stereo_calibration "
                    "(direct matrices) or stereo_calibration_path (.npz file)."
                )
            calibration = StereoCalibration.from_npz(stereo_calibration_path)

        important_indices = {
            1,
            10,
            33,
            133,
            145,
            152,
            159,
            160,
            234,
            263,
            362,
            373,
            374,
            385,
            386,
            387,
            454,
        }

        self._triangulator = StereoTriangulator(
            calibration=calibration,
            landmark_indices=sorted(important_indices),
        )
        self._pose_mapper = StereoHeadPoseDepthMapper(
            yaw_span=yaw_span,
            pitch_span=pitch_span,
            ema_alpha=ema_alpha,
        )

    def analyze(
        self,
        left_rgb_frame,
        right_rgb_frame,
        left_frame_width: int,
        left_frame_height: int,
        right_frame_width: int,
        right_frame_height: int,
        screen_width: int,
        screen_height: int,
    ) -> Optional[FaceAnalysisResult]:
        left_observation = self._left_provider.get_primary_face_observation(left_rgb_frame)
        right_observation = self._right_provider.get_primary_face_observation(right_rgb_frame)
        if left_observation is None or right_observation is None:
            return None

        points_3d = self._triangulator.triangulate_from_landmarks(
            left_landmarks=left_observation.landmarks,
            right_landmarks=right_observation.landmarks,
            left_frame_width=left_frame_width,
            left_frame_height=left_frame_height,
            right_frame_width=right_frame_width,
            right_frame_height=right_frame_height,
        )

        mapped = self._pose_mapper.estimate_screen_position(
            points_3d=points_3d,
            screen_width=screen_width,
            screen_height=screen_height,
            facial_transformation_matrix=left_observation.facial_transformation_matrix,
        )

        screen_position: Optional[Tuple[int, int]] = None
        angles: Optional[Tuple[float, float]] = None
        depth: Optional[float] = None
        if mapped is not None:
            screen_position, angles, depth = mapped

        # Gesture detection sources blendshapes and EAR from the left camera so
        # downstream consumers see the same observation that drives head pose.
        blendshapes = extract_blendshapes(left_observation.blendshapes)
        left_eye_ratio, right_eye_ratio = get_eye_aspect_ratios(left_observation.landmarks)
        wink_direction = detect_wink_direction_from_ratios(
            left_ratio=float(left_eye_ratio),
            right_ratio=float(right_eye_ratio),
        )

        return FaceAnalysisResult(
            landmarks=left_observation.landmarks,
            screen_position=screen_position,
            angles=angles,
            wink_direction=wink_direction,
            left_eye_ratio=float(left_eye_ratio),
            right_eye_ratio=float(right_eye_ratio),
            facial_transformation_matrix=left_observation.facial_transformation_matrix,
            depth=depth,
            blendshapes=blendshapes,
            right_landmarks=right_observation.landmarks,
            points_3d=points_3d,
        )

    def calibrate_to_center(self, yaw: float, pitch: float) -> None:
        self._pose_mapper.calibrate_to_center(yaw=yaw, pitch=pitch)

    def release(self) -> None:
        self._left_provider.release()
        self._right_provider.release()
