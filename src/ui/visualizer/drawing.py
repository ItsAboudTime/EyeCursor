from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def bgr_to_qpixmap(frame_bgr: np.ndarray) -> QPixmap:
    if frame_bgr is None or frame_bgr.size == 0:
        return QPixmap()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def rgb_to_qpixmap(frame_rgb: np.ndarray) -> QPixmap:
    if frame_rgb is None or frame_rgb.size == 0:
        return QPixmap()
    rgb = np.ascontiguousarray(frame_rgb)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


_MEDIAPIPE_LANDMARK_SUBSET = (
    1,    # front
    10,   # top
    152,  # bottom
    234,  # left
    454,  # right
)


def draw_mediapipe_landmarks(
    frame_bgr: np.ndarray,
    landmarks: Iterable,
    frame_width: int,
    frame_height: int,
    subset_only: bool = True,
    color: Tuple[int, int, int] = (60, 220, 60),
    radius: int = 10,
) -> np.ndarray:
    out = frame_bgr.copy()
    pts = list(landmarks) if landmarks is not None else []
    if not pts:
        return out
    indices = _MEDIAPIPE_LANDMARK_SUBSET if subset_only else range(len(pts))
    for idx in indices:
        if idx >= len(pts):
            continue
        lm = pts[idx]
        x = int(float(lm.x) * frame_width)
        y = int(float(lm.y) * frame_height)
        cv2.circle(out, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def draw_dlib_landmarks(
    frame_bgr: np.ndarray,
    points_xy: np.ndarray,
    face_box: Optional[Sequence[int]] = None,
    color: Tuple[int, int, int] = (80, 200, 255),
    box_color: Tuple[int, int, int] = (255, 180, 60),
    radius: int = 2,
) -> np.ndarray:
    out = frame_bgr.copy()
    if face_box is not None and len(face_box) == 4:
        x1, y1, x2, y2 = (int(v) for v in face_box)
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
    if points_xy is not None and len(points_xy) > 0:
        for p in points_xy:
            x = int(round(float(p[0])))
            y = int(round(float(p[1])))
            cv2.circle(out, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def draw_head_pose_arrow(
    frame_bgr: np.ndarray,
    nose_xy: Tuple[int, int],
    forward_axis_3d: np.ndarray,
    length_px: int = 120,
    color: Tuple[int, int, int] = (40, 80, 255),
    thickness: int = 3,
) -> np.ndarray:
    out = frame_bgr.copy()
    if forward_axis_3d is None:
        return out
    fx, fy, _ = (
        float(forward_axis_3d[0]),
        float(forward_axis_3d[1]),
        float(forward_axis_3d[2]),
    )
    ox, oy = int(nose_xy[0]), int(nose_xy[1])
    # Forward axis points away from the camera (-Z). Draw the projection of the
    # axis direction in image space; flip Y because image Y grows downward.
    tip = (int(ox + fx * length_px), int(oy - fy * length_px))
    cv2.arrowedLine(out, (ox, oy), tip, color, thickness, tipLength=0.25, line_type=cv2.LINE_AA)
    return out


def render_displacement_panel(
    landmarks_left: Iterable,
    landmarks_right: Iterable,
    left_frame_width: int,
    left_frame_height: int,
    right_frame_width: int,
    right_frame_height: int,
    indices: Sequence[int],
    canvas_size: Tuple[int, int] = (640, 480),
) -> np.ndarray:
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    left_list = list(landmarks_left) if landmarks_left is not None else []
    right_list = list(landmarks_right) if landmarks_right is not None else []
    if not left_list or not right_list:
        cv2.putText(
            canvas,
            "Waiting for stereo landmarks...",
            (24, canvas_size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        return canvas

    out_w, out_h = canvas_size

    def to_canvas(x_norm: float, y_norm: float) -> Tuple[int, int]:
        cx = int(x_norm * out_w)
        cy = int(y_norm * out_h)
        cx = max(0, min(out_w - 1, cx))
        cy = max(0, min(out_h - 1, cy))
        return cx, cy

    cyan = (255, 220, 80)
    magenta = (200, 80, 240)
    line_color = (100, 100, 100)
    for idx in indices:
        if idx >= len(left_list) or idx >= len(right_list):
            continue
        lp = left_list[idx]
        rp = right_list[idx]
        lx, ly = to_canvas(float(lp.x), float(lp.y))
        rx, ry = to_canvas(float(rp.x), float(rp.y))
        cv2.line(canvas, (lx, ly), (rx, ry), line_color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (lx, ly), 3, cyan, -1, cv2.LINE_AA)
        cv2.circle(canvas, (rx, ry), 3, magenta, -1, cv2.LINE_AA)

    cv2.putText(canvas, "left", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cyan, 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "right",
        (12, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        magenta,
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_gaze_arrow_on_patch(
    patch_rgb: np.ndarray,
    pitch_rad: float,
    yaw_rad: float,
    length_ratio: float = 0.45,
    color: Tuple[int, int, int] = (255, 80, 80),
    thickness: int = 3,
) -> np.ndarray:
    out = patch_rgb.copy()
    h, w = out.shape[:2]
    cx = w // 2
    cy = h // 2
    length = int(min(w, h) * length_ratio)
    # Project the gaze unit vector (pitch, yaw) onto image axes.
    # x_image = sin(yaw) * cos(pitch); y_image = -sin(pitch) (image Y grows down).
    dx = float(np.sin(yaw_rad) * np.cos(pitch_rad))
    dy = float(-np.sin(pitch_rad))
    tip = (int(cx + dx * length), int(cy + dy * length))
    cv2.arrowedLine(out, (cx, cy), tip, color, thickness, tipLength=0.25, line_type=cv2.LINE_AA)
    return out


def render_screen_target_preview(
    target_xy: Optional[Tuple[int, int]],
    screen_bounds: Optional[Tuple[int, int, int, int]],
    canvas_size: Tuple[int, int] = (480, 270),
) -> np.ndarray:
    canvas = np.full((canvas_size[1], canvas_size[0], 3), 30, dtype=np.uint8)
    cv2.rectangle(
        canvas,
        (4, 4),
        (canvas_size[0] - 5, canvas_size[1] - 5),
        (90, 110, 160),
        2,
    )
    if target_xy is None or screen_bounds is None:
        cv2.putText(
            canvas,
            "no target",
            (16, canvas_size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        return canvas

    minx, miny, maxx, maxy = screen_bounds
    width = max(1, maxx - minx + 1)
    height = max(1, maxy - miny + 1)
    nx = (target_xy[0] - minx) / width
    ny = (target_xy[1] - miny) / height
    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))
    px = int(nx * (canvas_size[0] - 1))
    py = int(ny * (canvas_size[1] - 1))
    cv2.drawMarker(canvas, (px, py), (60, 200, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    cv2.circle(canvas, (px, py), 8, (60, 200, 255), 2, cv2.LINE_AA)
    label = f"({target_xy[0]}, {target_xy[1]})"
    cv2.putText(
        canvas,
        label,
        (12, canvas_size[1] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def forward_axis_from_matrix(matrix: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if matrix is None:
        return None
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape == (16,):
        m = m.reshape(4, 4)
    if m.shape != (4, 4):
        return None
    rotation = m[:3, :3]
    forward = -rotation[:, 2]
    n = float(np.linalg.norm(forward))
    if n < 1e-9:
        return None
    forward = forward / n
    if forward[2] > 0.0:
        forward = -forward
    return forward
