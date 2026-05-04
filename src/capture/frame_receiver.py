"""Host-side UDP receiver for the capture protocol.

A background thread reads UDP packets, hands them to the
:class:`Reassembler`, decodes completed JPEGs to BGR ndarrays, and stores
the latest frame per cam_id. The mode loop pulls frames via
:meth:`get_latest_bgr` (single) or :meth:`get_latest_pair` (stereo).

Owned by :class:`src.capture.supervisor.CaptureSupervisor` -- consumers
do not construct this directly.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.capture.protocol import Reassembler


_RCV_BUF_BYTES = 4 * 1024 * 1024
_PACKET_RECV_SIZE = 65536


class FrameReceiver:
    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RCV_BUF_BYTES)
        except OSError:
            pass
        self._sock.bind((host, port))
        self._sock.settimeout(0.2)
        self._actual_port = self._sock.getsockname()[1]

        self._reassembler = Reassembler(ttl=0.2)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        # cam_id -> (bgr, timestamp)
        self._latest: Dict[int, Tuple[np.ndarray, float]] = {}
        # cam_id -> wall-clock time of last received complete frame (diagnostics)
        self._last_seen: Dict[int, float] = {}

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def actual_port(self) -> int:
        return self._actual_port

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="frame-receiver"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Wake any waiters so they don't block past stop.
        with self._cond:
            self._cond.notify_all()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                packet, _addr = self._sock.recvfrom(_PACKET_RECV_SIZE)
            except socket.timeout:
                # Periodic prune of expired partials.
                self._reassembler.prune()
                continue
            except OSError:
                # Socket closed during stop().
                break

            now = time.monotonic()
            completed = self._reassembler.feed(packet, now=now)
            if completed is None:
                continue

            arr = np.frombuffer(completed.jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            with self._cond:
                self._latest[completed.cam_id] = (frame, completed.timestamp)
                self._last_seen[completed.cam_id] = now
                self._cond.notify_all()

    def get_latest_bgr(
        self,
        cam_id: int,
        since: float = 0.0,
        timeout: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Return the latest decoded BGR frame for ``cam_id`` whose capture
        timestamp is strictly greater than ``since``. Blocks up to
        ``timeout`` seconds for a fresh frame; returns ``None`` if it
        doesn't arrive.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while True:
                if self._stop.is_set():
                    return None
                latest = self._latest.get(cam_id)
                if latest is not None and latest[1] > since:
                    return latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)

    def get_latest_pair(
        self,
        cam_id_left: int = 1,
        cam_id_right: int = 2,
        timeout: float = 0.5,
        max_skew: float = 0.05,
        since_left: float = 0.0,
        since_right: float = 0.0,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
        """Return the most recent (left, right) frame pair whose timestamps
        are within ``max_skew`` seconds of each other and both newer than
        the respective ``since_*`` watermarks. Blocks up to ``timeout`` for
        a satisfying pair to arrive; returns ``None`` on timeout.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while True:
                if self._stop.is_set():
                    return None
                left = self._latest.get(cam_id_left)
                right = self._latest.get(cam_id_right)
                if (
                    left is not None
                    and right is not None
                    and left[1] > since_left
                    and right[1] > since_right
                    and abs(left[1] - right[1]) <= max_skew
                ):
                    return left[0], right[0], left[1], right[1]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)

    def last_seen(self, cam_id: int) -> Optional[float]:
        with self._lock:
            return self._last_seen.get(cam_id)
