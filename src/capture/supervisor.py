"""Spawn and supervise the capture subprocess.

The supervisor:
- creates a :class:`FrameReceiver` bound to an OS-assigned UDP port,
- spawns ``python -m src.capture.frame_capture`` with the resolved camera
  indices and the receiver's port as CLI args,
- pumps the subprocess's stderr to our own stderr (with a ``[capture]``
  prefix) and parses the ``READY=1`` / ``READY=0 reason=...`` handshake,
- on stop, sends SIGTERM, waits for ``grace`` seconds, escalates to
  SIGKILL, then tears down the receiver.
"""

from __future__ import annotations

import collections
import subprocess
import sys
import threading
from typing import Deque, List, Optional

from src.capture.frame_receiver import FrameReceiver


class CaptureSupervisor:
    def __init__(self, camera_indices: List[int]) -> None:
        if len(camera_indices) not in (1, 2):
            raise ValueError(
                f"camera_indices must have 1 or 2 entries, got {len(camera_indices)}"
            )
        self._camera_indices: List[int] = [int(i) for i in camera_indices]
        self._receiver: Optional[FrameReceiver] = None
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._ready_ok = False
        self._ready_reason = ""
        self._stderr_lines: Deque[str] = collections.deque(maxlen=50)
        self._stderr_lock = threading.Lock()

    @property
    def receiver(self) -> FrameReceiver:
        if self._receiver is None:
            raise RuntimeError("CaptureSupervisor not started")
        return self._receiver

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def last_stderr_lines(self, n: int = 10) -> List[str]:
        with self._stderr_lock:
            lines = list(self._stderr_lines)
        return lines[-n:]

    def start(self, timeout: float = 5.0) -> None:
        if self._receiver is not None or self._proc is not None:
            raise RuntimeError("CaptureSupervisor already started")

        receiver = FrameReceiver(host="127.0.0.1", port=0)
        receiver.start()
        self._receiver = receiver
        port = receiver.actual_port

        argv = [
            sys.executable,
            "-m",
            "src.capture.frame_capture",
            "--cam0",
            str(self._camera_indices[0]),
            "--port",
            str(port),
        ]
        if len(self._camera_indices) == 2:
            argv += ["--cam1", str(self._camera_indices[1])]

        try:
            self._proc = subprocess.Popen(
                argv,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                start_new_session=True,
                text=True,
                bufsize=1,
            )
        except (OSError, ValueError) as e:
            receiver.stop()
            self._receiver = None
            raise RuntimeError(f"failed to spawn capture process: {e}") from e

        self._stderr_thread = threading.Thread(
            target=self._pump_stderr, daemon=True, name="capture-stderr-pump"
        )
        self._stderr_thread.start()

        if not self._ready_event.wait(timeout=timeout):
            tail = self.last_stderr_lines(5)
            self.stop()
            raise RuntimeError(
                f"capture process did not signal ready within {timeout}s. "
                f"Last stderr lines: {tail}"
            )

        if not self._ready_ok:
            reason = self._ready_reason
            self.stop()
            raise RuntimeError(reason or "capture process failed to start")

    def _pump_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw in proc.stderr:
            line = raw.rstrip()
            with self._stderr_lock:
                self._stderr_lines.append(line)
            try:
                sys.stderr.write(f"[capture] {line}\n")
                sys.stderr.flush()
            except OSError:
                pass
            if not self._ready_event.is_set() and line.startswith("READY="):
                self._handle_ready(line)

    def _handle_ready(self, line: str) -> None:
        if line.startswith("READY=1"):
            self._ready_ok = True
            self._ready_event.set()
            return
        if line.startswith("READY=0"):
            self._ready_ok = False
            after = line[len("READY=0") :].strip()
            if after.startswith("reason="):
                self._ready_reason = after[len("reason=") :]
            else:
                self._ready_reason = after
            self._ready_event.set()

    def stop(self, grace: float = 2.0) -> None:
        proc = self._proc
        if proc is not None:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
            self._proc = None

        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
            self._stderr_thread = None

        if self._receiver is not None:
            self._receiver.stop()
            self._receiver = None
