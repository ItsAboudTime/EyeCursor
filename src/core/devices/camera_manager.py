import platform
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.core.devices.camera_model import CameraInfo
from src.core.devices.stable_camera_id import stable_id_for_index


class CameraManager:
    MAX_SCAN_INDEX = 10

    def __init__(self) -> None:
        self._open_cameras: Dict[int, cv2.VideoCapture] = {}
        # Cache the last discovered list so callers (modes, calibrations)
        # can resolve an index -> stable_id without re-scanning hardware.
        self._last_scan: List[CameraInfo] = []

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
            cameras.append(
                CameraInfo(
                    index=idx,
                    width=w,
                    height=h,
                    stable_id=stable_id_for_index(idx),
                )
            )
            cap.release()
        self._last_scan = cameras
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

    def stable_id_for_index(self, index: int) -> Optional[str]:
        """Return the stable ID for a `/dev/videoN` index without re-scanning.

        Falls back to a fresh sysfs read when the index isn't in the cached
        scan results.
        """
        for cam in self._last_scan:
            if cam.index == index:
                return cam.stable_id
        return stable_id_for_index(index)

    def index_for_stable_id(self, stable_id: Optional[str]) -> Optional[int]:
        """Find the current /dev/videoN index for a previously-seen stable ID.

        Uses the cached scan list, falling back to a fresh discovery if the
        cache is empty. Returns ``None`` if the camera with that stable ID
        is not currently connected.
        """
        if not stable_id:
            return None
        cameras = self._last_scan or self.discover_cameras()
        for cam in cameras:
            if cam.stable_id == stable_id:
                return cam.index
        return None

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
