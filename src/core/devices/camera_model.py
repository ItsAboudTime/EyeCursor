from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CameraInfo:
    index: int
    width: int = 0
    height: int = 0
    label: str = ""
    is_available: bool = True
    # Stable identifier derived from the camera's USB device metadata. May be
    # ``None`` on platforms or devices where no stable ID can be determined;
    # in that case callers should treat the device as "no stable ID known"
    # and fall back to numeric-index matching with a warning.
    stable_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"Camera {self.index}"
