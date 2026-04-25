from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional
import uuid


@dataclass
class ProfileModel:
    id: str
    display_name: str
    default_mode: str = "one_camera_head_pose"
    preferred_cameras: Dict[str, Optional[int]] = field(default_factory=lambda: {
        "one_camera": None,
        "eye_gaze": None,
        "two_camera_left": None,
        "two_camera_right": None,
    })
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "default_mode": self.default_mode,
            "preferred_cameras": self.preferred_cameras,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProfileModel":
        return cls(
            id=data["id"],
            display_name=data["display_name"],
            default_mode=data.get("default_mode", "one_camera_head_pose"),
            preferred_cameras=data.get("preferred_cameras", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())[:8]
