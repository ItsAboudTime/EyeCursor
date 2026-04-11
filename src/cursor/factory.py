import sys
import importlib
from typing import Type
from src.cursor.base import Cursor
from src.cursor.constants import (
    DEFAULT_MOVE_PX_PER_SEC,
    DEFAULT_FRAME_RATE,
    DEFAULT_SCROLL_UNITS_PER_SEC,
)

_PLATFORM_IMPLS: dict[str, tuple[str, str]] = {
    "win": ("src.cursor.windows", "WindowsCursor"),
    "darwin": ("src.cursor.macos", "MacOSCursor"),
    "linux": ("src.cursor.linux", "LinuxCursor"),
}


def _load_impl_for_platform() -> Type[Cursor]:
    plat = sys.platform
    for prefix, (module_name, class_name) in _PLATFORM_IMPLS.items():
        if plat.startswith(prefix):
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
    raise RuntimeError(f"No cursor implementation available for OS: {plat!r}")


def create_cursor(
    move_px_per_sec: float = DEFAULT_MOVE_PX_PER_SEC,
    frame_rate: int = DEFAULT_FRAME_RATE,
    scroll_units_per_sec: float = DEFAULT_SCROLL_UNITS_PER_SEC,
) -> Cursor:
    """
    Factory function to create a platform-specific Cursor instance.
    """
    impl_cls = _load_impl_for_platform()
    return impl_cls(
        move_px_per_sec=move_px_per_sec,
        frame_rate=frame_rate,
        scroll_units_per_sec=scroll_units_per_sec,
    )
