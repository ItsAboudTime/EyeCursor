from dataclasses import dataclass
import time
from typing import Iterable, Optional, Tuple

from src.face_tracking.providers.face_landmarks import FaceLandmarksProvider
from src.face_tracking.signals.head_pose import HeadPoseSignalMapper
from src.face_tracking.signals.wink import detect_wink_direction, get_eye_aspect_ratios


@dataclass
class FaceAnalysisResult:
    landmarks: Iterable
    screen_position: Optional[Tuple[int, int]]
    angles: Optional[Tuple[float, float]]
    wink_direction: Optional[str]
    left_eye_ratio: Optional[float] = None
    right_eye_ratio: Optional[float] = None
    facial_transformation_matrix: Optional[object] = None
    depth: Optional[float] = None


class FaceAnalysisPipeline:
    """Coordinates landmarks acquisition and face signal extraction."""

    def __init__(
        self,
        yaw_span: float = 20.0,
        pitch_span: float = 10.0,
        ema_alpha: float = 0.25,
        wink_freeze_seconds: float = 1.0,
        face_model_path: Optional[str] = None,
    ) -> None:
        self._landmarks_provider = FaceLandmarksProvider(face_model_path=face_model_path)
        self._head_pose_mapper = HeadPoseSignalMapper(
            yaw_span=yaw_span,
            pitch_span=pitch_span,
            ema_alpha=ema_alpha,
        )
        self._last_screen_position: Optional[Tuple[int, int]] = None
        self._last_angles: Optional[Tuple[float, float]] = None
        self._wink_freeze_seconds = float(wink_freeze_seconds)
        if self._wink_freeze_seconds < 0.0:
            raise ValueError("wink_freeze_seconds must be >= 0")
        self._wink_started_at: Optional[float] = None
        self._last_wink_direction: Optional[str] = None

    def analyze(
        self,
        rgb_frame,
        frame_width: int,
        frame_height: int,
        screen_width: int,
        screen_height: int,
    ) -> Optional[FaceAnalysisResult]:
        observation = self._landmarks_provider.get_primary_face_observation(rgb_frame)
        if observation is None:
            return None

        landmarks = observation.landmarks
        facial_transformation_matrix = observation.facial_transformation_matrix

        position_and_angles = self._head_pose_mapper.estimate_screen_position(
            landmarks=landmarks,
            frame_width=frame_width,
            frame_height=frame_height,
            screen_width=screen_width,
            screen_height=screen_height,
            facial_transformation_matrix=facial_transformation_matrix,
        )

        if position_and_angles is not None:
            screen_position, angles = position_and_angles
        else:
            screen_position = None
            angles = None

        left_eye_ratio, right_eye_ratio = get_eye_aspect_ratios(landmarks)
        wink_direction = detect_wink_direction(landmarks)

        if wink_direction is None:
            self._wink_started_at = None
            self._last_wink_direction = None
        else:
            now = time.monotonic()
            if self._last_wink_direction != wink_direction:
                self._wink_started_at = now
                self._last_wink_direction = wink_direction

        if wink_direction is not None and self._last_screen_position is not None and self._last_angles is not None:
            should_freeze = True
            if self._wink_started_at is not None and self._wink_freeze_seconds > 0.0:
                elapsed = time.monotonic() - self._wink_started_at
                if elapsed >= self._wink_freeze_seconds:
                    should_freeze = False
            elif self._wink_freeze_seconds == 0.0:
                should_freeze = False

            if should_freeze:
                screen_position = self._last_screen_position
                angles = self._last_angles
            else:
                self._last_screen_position = screen_position
                self._last_angles = angles
        else:
            self._last_screen_position = screen_position
            self._last_angles = angles

        return FaceAnalysisResult(
            landmarks=landmarks,
            facial_transformation_matrix=facial_transformation_matrix,
            screen_position=screen_position,
            angles=angles,
            wink_direction=wink_direction,
            left_eye_ratio=left_eye_ratio,
            right_eye_ratio=right_eye_ratio,
        )

    def calibrate_to_center(self, yaw: float, pitch: float) -> None:
        self._head_pose_mapper.calibrate_to_center(yaw, pitch)

    def release(self) -> None:
        self._landmarks_provider.release()
