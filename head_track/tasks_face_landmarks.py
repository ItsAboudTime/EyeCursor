import ssl
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import mediapipe as mp
import numpy as np


@dataclass
class FaceLandmarksObservation:
    landmarks: Iterable
    facial_transformation_matrix: Optional[list[list[float]]]


class FaceLandmarksProvider:
    """Provides primary face landmarks using MediaPipe Tasks FaceLandmarker."""

    def __init__(self, face_model_path: Optional[str] = None) -> None:
        self._tasks_timestamp_ms = 0
        self._face_model_path = face_model_path
        self._landmarker = self._initialize_landmarker()

    def _resolve_model_path(self) -> Path:
        if self._face_model_path:
            candidate = Path(self._face_model_path).expanduser().resolve()
            if not candidate.exists():
                raise RuntimeError(f"Face Landmarker model not found at: {candidate}")
            return candidate

        cache_dir = Path.home() / ".cache" / "eyecursor"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "face_landmarker.task"
        if model_path.exists():
            return model_path

        model_url = (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )
        try:
            urllib.request.urlretrieve(model_url, str(model_path))
        except Exception:
            try:
                import certifi

                ctx = ssl.create_default_context(cafile=certifi.where())
                opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
                with opener.open(model_url, timeout=60) as response, open(model_path, "wb") as out_file:
                    out_file.write(response.read())
            except Exception as exc:
                raise RuntimeError(
                    "Failed to download MediaPipe Face Landmarker model due to SSL/network issues. "
                    "On macOS with python.org builds, run 'Install Certificates.command', or provide "
                    "a local model path."
                ) from exc

        return model_path

    def _initialize_landmarker(self):
        model_path = self._resolve_model_path()

        try:
            from mediapipe.tasks import python as mp_tasks_python
            from mediapipe.tasks.python import vision as mp_vision
        except Exception as exc:
            raise RuntimeError("MediaPipe Tasks API is unavailable in this environment.") from exc

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp_vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _extract_primary_facial_matrix(result) -> Optional[list[list[float]]]:
        matrices = getattr(result, "facial_transformation_matrixes", None)
        if not matrices:
            return None

        matrix = matrices[0]
        candidates = [matrix]
        data_attr = getattr(matrix, "data", None)
        if data_attr is not None:
            candidates.append(data_attr)

        for candidate in candidates:
            try:
                arr = np.asarray(candidate, dtype=float)
            except Exception:
                continue

            if arr.size != 16:
                continue

            arr = np.reshape(arr, (4, 4))
            return [[float(arr[r, c]) for c in range(4)] for r in range(4)]

        return None

    def get_primary_face_observation(self, rgb_image) -> Optional[FaceLandmarksObservation]:
        if self._landmarker is None:
            return None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        self._tasks_timestamp_ms = max(self._tasks_timestamp_ms + 1, int(time.time() * 1000))
        result = self._landmarker.detect_for_video(mp_image, self._tasks_timestamp_ms)
        if not result.face_landmarks:
            return None

        return FaceLandmarksObservation(
            landmarks=result.face_landmarks[0],
            facial_transformation_matrix=self._extract_primary_facial_matrix(result),
        )

    def get_primary_face_landmarks(self, rgb_image):
        observation = self.get_primary_face_observation(rgb_image)
        if observation is None:
            return None
        return observation.landmarks

    def release(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
