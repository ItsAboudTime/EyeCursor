"""Lifecycle helpers for the capture subprocess.

Tracking modes consume frames through :class:`CaptureSupervisor`, but every
mode needs the same boilerplate around it: spawn the subprocess, translate
spawn errors to a user-readable message, watch for premature subprocess
death inside the loop, and tear it all down at the end. These helpers
collapse that into one ``with capture_session(...) as supervisor:`` block
plus an :func:`assert_capture_alive` call per loop iteration.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List

from src.capture.supervisor import CaptureSupervisor


@contextmanager
def capture_session(
    camera_indices: List[int], *, startup_timeout: float = 5.0
) -> Iterator[CaptureSupervisor]:
    supervisor = CaptureSupervisor(camera_indices=camera_indices)
    try:
        supervisor.start(timeout=startup_timeout)
    except RuntimeError as exc:
        if len(camera_indices) == 1:
            raise RuntimeError(
                f"Could not open Camera {camera_indices[0]}. "
                f"Try selecting another camera or closing other apps that may be using it. "
                f"Detail: {exc}"
            ) from exc
        raise RuntimeError(
            "Could not open one or both cameras. "
            "Try closing other apps that may be using them. "
            f"Detail: {exc}"
        ) from exc

    try:
        yield supervisor
    finally:
        supervisor.stop()


def assert_capture_alive(supervisor: CaptureSupervisor) -> None:
    if not supervisor.is_alive():
        tail = supervisor.last_stderr_lines(3)
        raise RuntimeError(
            f"Capture process exited unexpectedly. Last stderr: {tail}"
        )
