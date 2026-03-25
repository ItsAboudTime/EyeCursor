import math
from collections import deque
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


class HeadPoseSignalMapper:
    """Maps face landmarks to head-pose angles and screen coordinates."""

    def __init__(self, yaw_span: float = 20.0, pitch_span: float = 10.0, smooth_len: int = 8) -> None:
        self.yaw_span = float(yaw_span)
        self.pitch_span = float(pitch_span)
        self._ray_dirs: deque[np.ndarray] = deque(maxlen=int(smooth_len))

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
        self._calibration_yaw = 180.0 - yaw
        self._calibration_pitch = 180.0 - pitch

    def estimate_head_pose(self, landmarks, frame_width: int, frame_height: int) -> Optional[Tuple[float, float]]:
        if landmarks is None:
            return None

        def landmark_to_numpy(landmark_index: int) -> np.ndarray:
            point = landmarks[landmark_index]
            return np.array([point.x * frame_width, point.y * frame_height, point.z * frame_width], dtype=float)

        left = landmark_to_numpy(self._landmark_indices["left"])
        right = landmark_to_numpy(self._landmark_indices["right"])
        top = landmark_to_numpy(self._landmark_indices["top"])
        bottom = landmark_to_numpy(self._landmark_indices["bottom"])

        right_axis = right - left
        right_axis /= np.linalg.norm(right_axis) + 1e-9

        up_axis = top - bottom
        up_axis /= np.linalg.norm(up_axis) + 1e-9

        forward_axis = np.cross(right_axis, up_axis)
        forward_axis /= np.linalg.norm(forward_axis) + 1e-9
        forward_axis = -forward_axis

        self._ray_dirs.append(forward_axis)
        averaged_direction = np.mean(self._ray_dirs, axis=0)
        averaged_direction /= np.linalg.norm(averaged_direction) + 1e-9

        yaw, pitch = self._compute_angles(averaged_direction)
        return yaw, pitch

    def get_x_and_y_on_screen(self, yaw: float, pitch: float, screen_width: int, screen_height: int) -> Tuple[int, int]:
        screen_x = int(((yaw - (180.0 - self.yaw_span)) / (2.0 * self.yaw_span)) * screen_width)
        screen_y = int(((180.0 + self.pitch_span - pitch) / (2.0 * self.pitch_span)) * screen_height)

        screen_x = max(0, min(screen_width - 1, screen_x))
        screen_y = max(0, min(screen_height - 1, screen_y))
        return screen_x, screen_y

    def estimate_screen_position(
        self,
        landmarks,
        frame_width: int,
        frame_height: int,
        screen_width: int,
        screen_height: int,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[float, float]]]:
        angles = self.estimate_head_pose(
            landmarks=landmarks,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if angles is None:
            return None

        yaw, pitch = angles
        screen_position = self.get_x_and_y_on_screen(
            yaw=yaw,
            pitch=pitch,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        return screen_position, angles

    def _compute_angles(self, averaged_direction: np.ndarray) -> Tuple[float, float]:
        reference_forward = np.array([0.0, 0.0, -1.0])

        xz_projection = np.array([averaged_direction[0], 0.0, averaged_direction[2]])
        xz_projection /= np.linalg.norm(xz_projection) + 1e-9
        yaw = math.degrees(math.acos(np.clip(np.dot(reference_forward, xz_projection), -1.0, 1.0)))
        if averaged_direction[0] < 0:
            yaw = -yaw

        yz_projection = np.array([0.0, averaged_direction[1], averaged_direction[2]])
        yz_projection /= np.linalg.norm(yz_projection) + 1e-9
        pitch = math.degrees(math.acos(np.clip(np.dot(reference_forward, yz_projection), -1.0, 1.0)))
        if averaged_direction[1] > 0:
            pitch = -pitch

        if yaw < 0:
            yaw = abs(yaw)
        elif yaw < 180:
            yaw = 360 - yaw

        if pitch < 0:
            pitch = 360 + pitch

        yaw += self._calibration_yaw
        pitch += self._calibration_pitch
        return yaw, pitch


def calculate_eye_aspect_ratio(eye_points: Sequence[Tuple[float, float]]) -> float:
    vertical_1 = ((eye_points[1][0] - eye_points[5][0]) ** 2 + (eye_points[1][1] - eye_points[5][1]) ** 2) ** 0.5
    vertical_2 = ((eye_points[2][0] - eye_points[4][0]) ** 2 + (eye_points[2][1] - eye_points[4][1]) ** 2) ** 0.5
    horizontal = ((eye_points[0][0] - eye_points[3][0]) ** 2 + (eye_points[0][1] - eye_points[3][1]) ** 2) ** 0.5
    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-9)


def detect_wink_direction(
    landmarks: Iterable,
    left_eye_indices: Sequence[int] = LEFT_EYE_INDICES,
    right_eye_indices: Sequence[int] = RIGHT_EYE_INDICES,
    closed_threshold: float = 0.2,
    open_threshold: float = 0.3,
) -> Optional[str]:
    """
    Returns:
      - "left" if left wink is detected
      - "right" if right wink is detected
      - None if no wink is detected
    """

    def to_xy(point) -> Tuple[float, float]:
        if hasattr(point, "x") and hasattr(point, "y"):
            return float(point.x), float(point.y)
        return float(point[0]), float(point[1])

    landmarks_list = list(landmarks)
    left_eye_points = [to_xy(landmarks_list[i]) for i in left_eye_indices]
    right_eye_points = [to_xy(landmarks_list[i]) for i in right_eye_indices]

    left_ratio = calculate_eye_aspect_ratio(left_eye_points)
    right_ratio = calculate_eye_aspect_ratio(right_eye_points)

    if left_ratio < closed_threshold and right_ratio > open_threshold:
        return "left"
    if right_ratio < closed_threshold and left_ratio > open_threshold:
        return "right"
    return None
