import sys
import importlib
from typing import Type
from cursor.base import Cursor
from cursor.config import DEFAULT_SPEED_PX_PER_SEC, DEFAULT_FRAME_RATE

_PLATFORM_IMPLS: dict[str, tuple[str, str]] = {
    "win": ("cursor.windows", "WindowsCursor"),
    "darwin": ("cursor.macos", "MacOSCursor"),
    # "linux": ("cursor.linux", "LinuxCursor"),
}

def _load_impl_for_platform() -> Type[Cursor]:
    plat = sys.platform
    for prefix, (module_name, class_name) in _PLATFORM_IMPLS.items():
        if plat.startswith(prefix):
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
    raise RuntimeError(f"No cursor implementation available for OS: {plat!r}")

def create_cursor(speed_px_per_sec=DEFAULT_SPEED_PX_PER_SEC,
                  frame_rate=DEFAULT_FRAME_RATE) -> Cursor:
    impl_cls = _load_impl_for_platform()
    return impl_cls(speed_px_per_sec=speed_px_per_sec, frame_rate=frame_rate)
