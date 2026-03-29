from typing import Iterable, Optional, Sequence, Tuple


LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


def calculate_eye_aspect_ratio(eye_points: Sequence[Tuple[float, float]]) -> float:
    vertical_1 = ((eye_points[1][0] - eye_points[5][0]) ** 2 + (eye_points[1][1] - eye_points[5][1]) ** 2) ** 0.5
    vertical_2 = ((eye_points[2][0] - eye_points[4][0]) ** 2 + (eye_points[2][1] - eye_points[4][1]) ** 2) ** 0.5
    horizontal = ((eye_points[0][0] - eye_points[3][0]) ** 2 + (eye_points[0][1] - eye_points[3][1]) ** 2) ** 0.5
    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-9)


def detect_wink_direction(
    landmarks: Iterable,
    left_eye_indices: Sequence[int] = LEFT_EYE_INDICES,
    right_eye_indices: Sequence[int] = RIGHT_EYE_INDICES,
    closed_threshold: float = 0.3,
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
