import cv2
import numpy as np


def draw_gaze_arrow(image: np.ndarray, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    length = float(min(h, w)) * 0.45
    center = np.array([w * 0.5, h * 0.5], dtype=np.float32)

    dx = -length * np.sin(yaw_rad) * np.cos(pitch_rad)
    dy = -length * np.sin(pitch_rad)

    start = tuple(np.round(center).astype(np.int32))
    end = tuple(np.round(center + np.array([dx, dy], dtype=np.float32)).astype(np.int32))
    cv2.arrowedLine(out, start, end, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.2)
    return out
