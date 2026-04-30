from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class TrackingMode(ABC):
    id: str = ""
    display_name: str = ""
    description: str = ""
    required_camera_count: int = 1
    requires_head_pose_calibration: bool = False
    requires_eye_gesture_calibration: bool = False
    requires_stereo_calibration: bool = False
    requires_gaze_calibration: bool = False

    @abstractmethod
    def validate_requirements(
        self,
        profile_calibrations: Dict[str, Optional[dict]],
        selected_cameras: List[int],
    ) -> Tuple[bool, str]:
        ...

    @abstractmethod
    def start(
        self,
        profile_calibrations: Dict[str, Optional[dict]],
        selected_cameras: List[int],
        cursor,
        settings: Optional[dict] = None,
    ) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def pause(self) -> None:
        ...

    @abstractmethod
    def resume(self) -> None:
        ...

    def update_settings(self, settings: Dict) -> None:
        """Apply live setting changes to long-lived collaborators.

        The default implementation is a no-op. Concrete modes should override
        to push the new settings into their cursor / gesture / gaze
        controllers. Implementations must be safe to call from the GUI
        thread and tolerate being called before, during, or after the
        tracking loop is running.
        """
        return
