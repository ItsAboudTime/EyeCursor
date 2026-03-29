from typing import Optional, Sequence, Tuple

import numpy as np


class HeadPoseSignalMapper:
    """Maps face landmarks to head-pose angles and screen coordinates."""

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
        self._calibration_yaw = -yaw
        self._calibration_pitch = -pitch

    def estimate_head_pose(
        self,
        landmarks=None,
        frame_width: int = 1,
        frame_height: int = 1,
        facial_transformation_matrix: Optional[Sequence[Sequence[float]]] = None,
    ) -> Optional[Tuple[float, float]]:
        forward_axis = self._forward_axis_from_matrix(facial_transformation_matrix)
        if forward_axis is None:
            if landmarks is None:
                return None
            forward_axis = self._forward_axis_from_landmarks(
                landmarks=landmarks,
                frame_width=frame_width,
                frame_height=frame_height,
            )

        if self._ema_direction is None:
            self._ema_direction = forward_axis
        else:
            self._ema_direction = (
                self.ema_alpha * forward_axis + (1.0 - self.ema_alpha) * self._ema_direction
            )
            self._ema_direction /= np.linalg.norm(self._ema_direction) + 1e-9

        averaged_direction = self._ema_direction

        yaw, pitch = self._compute_angles(averaged_direction)
        return yaw, pitch

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

        # Use a single axis convention to avoid frame-to-frame flips.
        forward_axis = -rotation[:, 2]
        norm = np.linalg.norm(forward_axis)
        if norm < 1e-9:
            return None
        forward_axis = forward_axis / norm

        # Keep neutral orientation roughly aligned to looking "forward" along -Z.
        if forward_axis[2] > 0.0:
            forward_axis = -forward_axis

        return forward_axis

    def _forward_axis_from_landmarks(self, landmarks, frame_width: int, frame_height: int) -> np.ndarray:
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
        return -forward_axis

    def get_x_and_y_on_screen(self, yaw: float, pitch: float, screen_width: int, screen_height: int) -> Tuple[int, int]:
        norm_x = (yaw + self.yaw_span) / (2.0 * self.yaw_span)
        norm_y = (pitch + self.pitch_span) / (2.0 * self.pitch_span)
        screen_x = int(norm_x * screen_width)
        screen_y = int(norm_y * screen_height)

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
        facial_transformation_matrix: Optional[Sequence[Sequence[float]]] = None,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[float, float]]]:
        angles = self.estimate_head_pose(
            landmarks=landmarks,
            frame_width=frame_width,
            frame_height=frame_height,
            facial_transformation_matrix=facial_transformation_matrix,
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
        x, y, z = float(averaged_direction[0]), float(averaged_direction[1]), float(averaged_direction[2])

        # Zero-centered angles around forward (-Z):
        # yaw  > 0 => looking right, pitch > 0 => looking down.
        yaw = float(np.degrees(np.arctan2(x, -z)))
        pitch = float(np.degrees(np.arctan2(y, -z)))

        yaw += self._calibration_yaw
        pitch += self._calibration_pitch

        return yaw, pitch
