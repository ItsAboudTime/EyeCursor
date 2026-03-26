from dataclasses import dataclass
from typing import Optional, Tuple

from head_track.face_signals import HeadPoseSignalMapper, detect_wink_direction
from head_track.tasks_face_landmarks import FaceLandmarksProvider


@dataclass
class FaceAnalysisResult:
    landmarks: object
    screen_position: Optional[Tuple[int, int]]
    angles: Optional[Tuple[float, float]]
    wink_direction: Optional[str]


class FaceAnalysisPipeline:
    """Coordinates landmarks acquisition and face signal extraction."""

    def __init__(
        self,
        yaw_span: float = 20.0,
        pitch_span: float = 10.0,
        smooth_len: int = 8,
        face_model_path: Optional[str] = None,
    ) -> None:
        self._landmarks_provider = FaceLandmarksProvider(face_model_path=face_model_path)
        self._head_pose_mapper = HeadPoseSignalMapper(
            yaw_span=yaw_span,
            pitch_span=pitch_span,
            smooth_len=smooth_len,
        )

    def analyze(
        self,
        rgb_frame,
        frame_width: int,
        frame_height: int,
        screen_width: int,
        screen_height: int,
    ) -> Optional[FaceAnalysisResult]:
        landmarks = self._landmarks_provider.get_primary_face_landmarks(rgb_frame)
        if landmarks is None:
            return None

        position_and_angles = self._head_pose_mapper.estimate_screen_position(
            landmarks=landmarks,
            frame_width=frame_width,
            frame_height=frame_height,
            screen_width=screen_width,
            screen_height=screen_height,
        )

        if position_and_angles is not None:
            screen_position, angles = position_and_angles
        else:
            screen_position = None
            angles = None

        wink_direction = detect_wink_direction(landmarks)
        return FaceAnalysisResult(
            landmarks=landmarks,
            screen_position=screen_position,
            angles=angles,
            wink_direction=wink_direction,
        )

    def calibrate_to_center(self, yaw: float, pitch: float) -> None:
        self._head_pose_mapper.calibrate_to_center(yaw, pitch)

    def release(self) -> None:
        self._landmarks_provider.release()
