from typing import Dict, List, Type

from src.core.modes.base import TrackingMode


class ModeRegistry:
    def __init__(self) -> None:
        self._modes: Dict[str, Type[TrackingMode]] = {}

    def register(self, mode_cls: Type[TrackingMode]) -> None:
        self._modes[mode_cls.id] = mode_cls

    def get(self, mode_id: str) -> Type[TrackingMode]:
        return self._modes[mode_id]

    def all_modes(self) -> List[Type[TrackingMode]]:
        return list(self._modes.values())

    def mode_ids(self) -> List[str]:
        return list(self._modes.keys())
