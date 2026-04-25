from dataclasses import dataclass, field


@dataclass
class CameraInfo:
    index: int
    width: int = 0
    height: int = 0
    label: str = ""
    is_available: bool = True

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"Camera {self.index}"
