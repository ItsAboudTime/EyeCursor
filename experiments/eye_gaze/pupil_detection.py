"""
Pupil localization demo using classical computer vision.

Implements three pupil-localization methods per eye:
1) Threshold + contour filtering (Upgraded with Image Moments)
2) Hough Circle Transform
3) Hybrid fusion (prefers threshold, falls back to Hough)

The demo uses MediaPipe face landmarks to crop eye ROIs and then localizes
left/right pupils in each frame.

Controls:
  - 1: threshold mode
  - 2: hough mode
  - 3: hybrid mode
  - q / ESC: quit
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import cv2
import numpy as np

# Allow running this file directly (python experiments/eye_gaze/pupil_detect.py)
# while still resolving project-local imports from repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_tracking.providers.face_landmarks import FaceLandmarksProvider

PupilMethod = Literal["threshold", "hough", "hybrid"]


@dataclass
class PupilDetection:
    center: tuple[int, int]
    radius: int
    confidence: float
    method: str


@dataclass
class EyeRoi:
    image: np.ndarray
    x0: int
    y0: int


class PupilDetector:
    """Classical CV pupil detector operating on eye ROIs."""

    # Eye contour landmarks (MediaPipe face mesh indexing).
    _LEFT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]
    _RIGHT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]

    def __init__(self, min_radius: int = 3, max_radius: int = 28) -> None:
        self.min_radius = int(min_radius)
        self.max_radius = int(max_radius)

    @staticmethod
    def _to_px(landmark, width: int, height: int) -> np.ndarray:
        return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)

    def _extract_eye_roi(
        self,
        frame_bgr: np.ndarray,
        landmarks: Sequence,
        eye_indices: list[int],
        margin: float = 0.25,
    ) -> Optional[EyeRoi]:
        h, w = frame_bgr.shape[:2]
        pts = np.array([self._to_px(landmarks[i], w, h) for i in eye_indices], dtype=np.float32)
        if pts.size == 0:
            return None

        x, y, rw, rh = cv2.boundingRect(pts.astype(np.int32))
        
        # IMPROVEMENT: Base the vertical padding on the eye's WIDTH, not height.
        # This prevents the bounding box from turning into a tiny slit when squinting.
        pad_x = int(rw * margin)
        pad_y = int(rw * margin * 0.6) 

        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + rw + pad_x)
        y1 = min(h, y + rh + pad_y)

        if x1 <= x0 or y1 <= y0:
            return None

        return EyeRoi(image=frame_bgr[y0:y1, x0:x1].copy(), x0=x0, y0=y0)

    @staticmethod
    def _preprocess_steps(eye_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray_blur)
        return gray, gray_blur, gray_clahe

    def detect_threshold(self, eye_bgr: np.ndarray) -> Optional[PupilDetection]:
        _, _, gray = self._preprocess_steps(eye_bgr)

        # Isolate the darkest 15% of the image
        t = int(np.percentile(gray, 15))
        binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)[1]

        # Use a slightly smaller kernel so we don't erase thin, squinting pupils
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = gray.shape[:2]
        image_area = float(h * w)
        best: Optional[PupilDetection] = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Lowered the minimum area requirement to catch thin pupil slivers
            if area < image_area * 0.001 or area > image_area * 0.30:
                continue

            # IMPROVEMENT: Use Image Moments to find the Center of Mass of the blob.
            # This works flawlessly on half-moons, ovals, and occluded pupils.
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # We still generate a radius for the UI drawing
            _, radius = cv2.minEnclosingCircle(cnt)
            radius = float(radius)

            # Prefer darker interiors
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_intensity = float(cv2.mean(gray, mask=mask)[0])
            darkness = 1.0 - np.clip(mean_intensity / 255.0, 0.0, 1.0)

            # Prefer blobs that are relatively central to the eye socket
            dist_from_center = np.hypot(cx - w/2, cy - h/2) / (np.hypot(w, h) + 1e-6)
            center_score = 1.0 - np.clip(dist_from_center * 2.5, 0.0, 1.0)

            # Score is now based purely on how dark and centered it is, ignoring circularity
            score = 0.7 * darkness + 0.3 * center_score
            
            if score > best_score:
                best_score = score
                best = PupilDetection(
                    center=(cx, cy),
                    radius=int(radius),
                    confidence=float(np.clip(score, 0.0, 1.0)),
                    method="threshold:moments",
                )

        return best

    def detect_hough(self, eye_bgr: np.ndarray) -> Optional[PupilDetection]:
        _, _, gray = self._preprocess_steps(eye_bgr)
        h, w = gray.shape[:2]

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(8, int(min(h, w) * 0.2)),
            param1=80,
            param2=16,
            minRadius=self.min_radius,
            maxRadius=min(self.max_radius, int(min(h, w) * 0.45)),
        )
        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype(np.int32)
        best: Optional[PupilDetection] = None
        best_score = -1.0

        cx_img, cy_img = w * 0.5, h * 0.5
        diag = float(np.hypot(w, h)) + 1e-6
        for c in circles:
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            if r < self.min_radius or r > self.max_radius:
                continue

            # Favor circles near center and with dark interior.
            dist = float(np.hypot(x - cx_img, y - cy_img)) / diag
            center_score = 1.0 - np.clip(dist * 2.2, 0.0, 1.0)

            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r, 255, -1)
            mean_intensity = float(cv2.mean(gray, mask=mask)[0])
            darkness = 1.0 - np.clip(mean_intensity / 255.0, 0.0, 1.0)

            score = 0.5 * center_score + 0.5 * darkness
            if score > best_score:
                best_score = score
                best = PupilDetection(
                    center=(x, y),
                    radius=r,
                    confidence=float(np.clip(score, 0.0, 1.0)),
                    method="hough",
                )

        return best

    def detect(self, eye_bgr: np.ndarray, method: PupilMethod = "hybrid") -> Optional[PupilDetection]:
        if method == "threshold":
            return self.detect_threshold(eye_bgr)
        if method == "hough":
            return self.detect_hough(eye_bgr)

        thr = self.detect_threshold(eye_bgr)
        if thr is not None and thr.confidence >= 0.35:
            return PupilDetection(
                center=thr.center,
                radius=thr.radius,
                confidence=thr.confidence,
                method="hybrid:threshold",
            )

        hough = self.detect_hough(eye_bgr)
        if hough is not None:
            return PupilDetection(
                center=hough.center,
                radius=hough.radius,
                confidence=hough.confidence,
                method="hybrid:hough",
            )

        return None

    def detect_pupils_on_frame(
        self,
        frame_bgr: np.ndarray,
        landmarks: Iterable,
        method: PupilMethod = "hybrid",
    ) -> dict[str, Optional[PupilDetection]]:
        result: dict[str, Optional[PupilDetection]] = {"left": None, "right": None}

        landmarks_seq = list(landmarks)

        left_roi = self._extract_eye_roi(frame_bgr, landmarks_seq, self._LEFT_EYE_IDX)
        right_roi = self._extract_eye_roi(frame_bgr, landmarks_seq, self._RIGHT_EYE_IDX)

        if left_roi is not None:
            left_det = self.detect(left_roi.image, method=method)
            if left_det is not None:
                result["left"] = PupilDetection(
                    center=(left_det.center[0] + left_roi.x0, left_det.center[1] + left_roi.y0),
                    radius=left_det.radius,
                    confidence=left_det.confidence,
                    method=left_det.method,
                )

        if right_roi is not None:
            right_det = self.detect(right_roi.image, method=method)
            if right_det is not None:
                result["right"] = PupilDetection(
                    center=(right_det.center[0] + right_roi.x0, right_det.center[1] + right_roi.y0),
                    radius=right_det.radius,
                    confidence=right_det.confidence,
                    method=right_det.method,
                )

        return result


def draw_pupil(frame_bgr: np.ndarray, det: PupilDetection, eye_label: str) -> None:
    cv2.circle(frame_bgr, det.center, max(1, det.radius), (0, 255, 255), 2)
    cv2.circle(frame_bgr, det.center, 2, (0, 0, 255), -1)
    cv2.putText(
        frame_bgr,
        f"{eye_label}: {det.method} ({det.confidence:.2f})",
        (det.center[0] - 45, det.center[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _make_labeled_tile(image: np.ndarray, label: str, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    if image.ndim == 2:
        tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        tile = image.copy()

    tile = cv2.resize(tile, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    cv2.rectangle(tile, (0, 0), (target_w - 1, target_h - 1), (80, 80, 80), 1)
    cv2.putText(
        tile,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return tile


def _build_eye_steps_row(detector: PupilDetector, eye_roi: Optional[EyeRoi], eye_name: str) -> np.ndarray:
    tile_size = (300, 170)
    if eye_roi is None or eye_roi.image.size == 0:
        empty = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
        empty[:] = (30, 30, 30)
        return cv2.hconcat(
            [
                _make_labeled_tile(empty, f"{eye_name} ROI (no eye)", tile_size),
                _make_labeled_tile(empty, f"{eye_name} Grayscale", tile_size),
                _make_labeled_tile(empty, f"{eye_name} CLAHE", tile_size),
            ]
        )

    gray, _, gray_clahe = detector._preprocess_steps(eye_roi.image)
    return cv2.hconcat(
        [
            _make_labeled_tile(eye_roi.image, f"{eye_name} ROI", tile_size),
            _make_labeled_tile(gray, f"{eye_name} Grayscale", tile_size),
            _make_labeled_tile(gray_clahe, f"{eye_name} CLAHE", tile_size),
        ]
    )


def _build_debug_canvas(
    frame_bgr: np.ndarray,
    detector: PupilDetector,
    left_roi: Optional[EyeRoi],
    right_roi: Optional[EyeRoi],
    method: PupilMethod,
) -> np.ndarray:
    frame_h, frame_w = frame_bgr.shape[:2]
    top_h = 420
    top_w = 900
    top = cv2.resize(frame_bgr, (top_w, top_h), interpolation=cv2.INTER_AREA)
    cv2.putText(
        top,
        f"Main Feed | Mode: {method}",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )

    left_row = _build_eye_steps_row(detector, left_roi, "Left Eye")
    right_row = _build_eye_steps_row(detector, right_roi, "Right Eye")
    bottom = cv2.vconcat([left_row, right_row])

    if bottom.shape[1] != top.shape[1]:
        bottom = cv2.resize(bottom, (top.shape[1], bottom.shape[0]), interpolation=cv2.INTER_LINEAR)

    canvas = cv2.vconcat([top, bottom])
    footer = np.full((40, canvas.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(
        footer,
        "Steps: ROI crop -> Grayscale -> CLAHE",
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return cv2.vconcat([canvas, footer])


def run_demo(camera_index: int = 0) -> int:
    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        print(f"Could not open webcam at index {camera_index}")
        return 1

    provider = FaceLandmarksProvider()
    detector = PupilDetector()
    method: PupilMethod = "hybrid"

    print("Pupil localization demo started.")
    print("Controls: 1=threshold, 2=hough, 3=hybrid, q=quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = provider.get_primary_face_landmarks(rgb)
            left_roi: Optional[EyeRoi] = None
            right_roi: Optional[EyeRoi] = None
            if landmarks is not None:
                landmarks_seq = list(landmarks)
                left_roi = detector._extract_eye_roi(frame, landmarks_seq, detector._LEFT_EYE_IDX)
                right_roi = detector._extract_eye_roi(frame, landmarks_seq, detector._RIGHT_EYE_IDX)

                detections = detector.detect_pupils_on_frame(frame, landmarks, method=method)
                if detections["left"] is not None:
                    draw_pupil(frame, detections["left"], "L")
                if detections["right"] is not None:
                    draw_pupil(frame, detections["right"], "R")
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 100, 255),
                    2,
                    cv2.LINE_AA,
                )

            canvas = _build_debug_canvas(frame, detector, left_roi, right_roi, method)
            cv2.imshow("Pupil Localization Steps", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("1"):
                method = "threshold"
            elif key == ord("2"):
                method = "hough"
            elif key == ord("3"):
                method = "hybrid"
    finally:
        cap.release()
        provider.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    return run_demo(camera_index=0)


if __name__ == "__main__":
    sys.exit(main())