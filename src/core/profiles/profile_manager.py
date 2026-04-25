import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.core.profiles.profile_model import ProfileModel


class ProfileManager:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        self._profiles_dir = self._data_dir / "profiles"
        self._profiles_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> List[ProfileModel]:
        profiles = []
        if not self._profiles_dir.exists():
            return profiles
        for entry in sorted(self._profiles_dir.iterdir()):
            profile_file = entry / "profile.json"
            if entry.is_dir() and profile_file.exists():
                try:
                    profiles.append(self._read_profile(profile_file))
                except (json.JSONDecodeError, KeyError):
                    continue
        return profiles

    def create_profile(self, display_name: str) -> ProfileModel:
        profile = ProfileModel(
            id=ProfileModel.generate_id(),
            display_name=display_name,
        )
        profile_dir = self._profiles_dir / profile.id
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "calibrations").mkdir(exist_ok=True)
        (profile_dir / "stereo").mkdir(exist_ok=True)
        self._write_profile(profile)
        return profile

    def load_profile(self, profile_id: str) -> Optional[ProfileModel]:
        profile_file = self._profiles_dir / profile_id / "profile.json"
        if not profile_file.exists():
            return None
        return self._read_profile(profile_file)

    def save_profile(self, profile: ProfileModel) -> None:
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        self._write_profile(profile)

    def delete_profile(self, profile_id: str) -> None:
        profile_dir = self._profiles_dir / profile_id
        if profile_dir.exists():
            shutil.rmtree(profile_dir)

    def rename_profile(self, profile_id: str, new_name: str) -> Optional[ProfileModel]:
        profile = self.load_profile(profile_id)
        if profile is None:
            return None
        profile.display_name = new_name
        self.save_profile(profile)
        return profile

    def load_calibration(self, profile_id: str, mode_id: str) -> Optional[Dict]:
        cal_path = self._calibration_path(profile_id, mode_id)
        if not cal_path.exists():
            return None
        with open(cal_path, "r") as f:
            return json.load(f)

    def save_calibration(self, profile_id: str, mode_id: str, data: Dict) -> None:
        cal_path = self._calibration_path(profile_id, mode_id)
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cal_path, "w") as f:
            json.dump(data, f, indent=2)

    def reset_calibration(self, profile_id: str, mode_id: str) -> None:
        cal_path = self._calibration_path(profile_id, mode_id)
        if cal_path.exists():
            cal_path.unlink()

    def reset_all_calibrations(self, profile_id: str) -> None:
        cal_dir = self._profiles_dir / profile_id / "calibrations"
        if cal_dir.exists():
            shutil.rmtree(cal_dir)
            cal_dir.mkdir()
        stereo_dir = self._profiles_dir / profile_id / "stereo"
        if stereo_dir.exists():
            shutil.rmtree(stereo_dir)
            stereo_dir.mkdir()

    def load_stereo_calibration(self, profile_id: str) -> Optional[Dict]:
        stereo_path = self._profiles_dir / profile_id / "stereo" / "calibration.json"
        if not stereo_path.exists():
            return None
        with open(stereo_path, "r") as f:
            return json.load(f)

    def save_stereo_calibration(self, profile_id: str, data: Dict) -> None:
        stereo_dir = self._profiles_dir / profile_id / "stereo"
        stereo_dir.mkdir(parents=True, exist_ok=True)
        with open(stereo_dir / "calibration.json", "w") as f:
            json.dump(data, f, indent=2)

    def get_calibration_status(self, profile_id: str) -> Dict[str, bool]:
        cal_dir = self._profiles_dir / profile_id / "calibrations"
        stereo_path = self._profiles_dir / profile_id / "stereo" / "calibration.json"
        modes = [
            "one_camera_head_pose",
            "two_camera_head_pose",
            "eye_gaze",
            "eye_gestures",
        ]
        status = {}
        for mode_id in modes:
            status[mode_id] = (cal_dir / f"{mode_id}.json").exists()
        status["stereo"] = stereo_path.exists()
        return status

    def _calibration_path(self, profile_id: str, mode_id: str) -> Path:
        return self._profiles_dir / profile_id / "calibrations" / f"{mode_id}.json"

    def _write_profile(self, profile: ProfileModel) -> None:
        profile_dir = self._profiles_dir / profile.id
        profile_dir.mkdir(parents=True, exist_ok=True)
        with open(profile_dir / "profile.json", "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    @staticmethod
    def _read_profile(path: Path) -> ProfileModel:
        with open(path, "r") as f:
            return ProfileModel.from_dict(json.load(f))
