import platform
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.core.devices.camera_model import CameraInfo


class CameraManager:
    MAX_SCAN_INDEX = 10

    def __init__(self) -> None:
        self._open_cameras: Dict[int, cv2.VideoCapture] = {}

    def discover_cameras(self) -> List[CameraInfo]:
        indices = self._candidate_indices()
        cameras = []
        for idx in indices:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                continue
            h, w = frame.shape[:2]
            cameras.append(CameraInfo(index=idx, width=w, height=h))
            cap.release()
        return cameras

    def open_camera(self, index: int) -> cv2.VideoCapture:
        if index in self._open_cameras:
            cap = self._open_cameras[index]
            if cap.isOpened():
                return cap
            cap.release()

        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open Camera {index}. "
                "Try selecting another camera or closing other apps that may be using it."
            )
        self._open_cameras[index] = cap
        return cap

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        cap = self._open_cameras.get(index)
        if cap is None or not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok:
            return None
        return frame

    def release_camera(self, index: int) -> None:
        cap = self._open_cameras.pop(index, None)
        if cap is not None:
            cap.release()

    def release_all(self) -> None:
        for cap in self._open_cameras.values():
            cap.release()
        self._open_cameras.clear()

    def is_open(self, index: int) -> bool:
        cap = self._open_cameras.get(index)
        return cap is not None and cap.isOpened()

    def _candidate_indices(self) -> List[int]:
        if platform.system() == "Linux":
            dev_indices = self._linux_device_indices()
            if dev_indices:
                return dev_indices
        return list(range(self.MAX_SCAN_INDEX))

    @staticmethod
    def _linux_device_indices() -> List[int]:
        dev_path = Path("/dev")
        indices = []
        for entry in sorted(dev_path.glob("video*")):
            name = entry.name
            if name.startswith("video"):
                try:
                    indices.append(int(name[5:]))
                except ValueError:
                    continue
        return indices
