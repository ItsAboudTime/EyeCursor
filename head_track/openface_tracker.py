"""
Linux-only head pose tracker using OpenFace 2.0 via a Docker TCP service.

Provides the same API as HeadPoseTracker but delegates face analysis to an
OpenFace container reachable over a plain TCP socket.  The container must be
started separately (see openface_server/).

Usage::

    # 1. Build & run the container (once):
    #    cd openface_server
    #    docker build -t openface-server .
    #    docker run --rm -p 5555:5555 openface-server
    #
    # 2. Then run the host program:
    #    python examples/openface_cursor.py
"""

import csv
import io
import math
import socket
import struct
import sys
from collections import deque
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class OpenFaceTracker:
    """
    Head pose tracker backed by an OpenFace 2.0 TCP service.

    The public interface mirrors :class:`HeadPoseTracker`:

    * :meth:`start` / :meth:`stop` manage the webcam and TCP connection.
    * :meth:`next_position` returns ``(pos, frame, angles)`` per frame.
    * :meth:`calibrate_center` re-centres the mapping.

    Additionally exposes ``last_au45`` (float) — the most recent AU45_r blink
    intensity from OpenFace, useful for click detection.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        yaw_span: float = 20.0,
        pitch_span: float = 10.0,
        smooth_len: int = 8,
        camera_index: int = 0,
        timeout_s: float = 5.0,
        debug: bool = False,
    ) -> None:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("OpenFaceTracker currently supports Linux only.")

        self.host = host
        self.port = port
        self.yaw_span = float(yaw_span)
        self.pitch_span = float(pitch_span)
        self.smooth_len = int(smooth_len)
        self.camera_index = int(camera_index)
        self.timeout_s = float(timeout_s)
        self._debug = debug

        self._cap: Optional[cv2.VideoCapture] = None
        self._sock: Optional[socket.socket] = None

        self._angles_buf: deque[Tuple[float, float]] = deque(maxlen=self.smooth_len)

        self.calib_yaw: float = 0.0
        self.calib_pitch: float = 0.0

        self.last_au45: float = 0.0
        self.last_aus: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the webcam and connect to the OpenFace TCP server."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam (index {self.camera_index})")
        self._cap = cap

        self._connect()

    def _connect(self) -> None:
        """(Re-)establish the TCP connection to the server."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_s)
        sock.connect((self.host, self.port))
        self._sock = sock
        if self._debug:
            print(f"[DEBUG] Connected to {self.host}:{self.port}")

    def stop(self) -> None:
        """Release the webcam and close the TCP connection."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # TCP helpers (length-prefixed protocol)
    # ------------------------------------------------------------------

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("server disconnected")
            buf += chunk
        return buf

    def _send_msg(self, data: bytes) -> None:
        self._sock.sendall(struct.pack(">I", len(data)) + data)

    def _recv_msg(self) -> bytes:
        raw_len = self._recv_exact(4)
        msg_len = struct.unpack(">I", raw_len)[0]
        return self._recv_exact(msg_len)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_center(self, yaw: float, pitch: float) -> None:
        """Store offsets so the current yaw/pitch maps to screen centre."""
        self.calib_yaw = -yaw
        self.calib_pitch = -pitch

    # ------------------------------------------------------------------
    # Core per-frame method
    # ------------------------------------------------------------------

    def next_position(
        self, screen_w: int, screen_h: int,
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray, Optional[Tuple[float, float]]]:
        """
        Capture a frame, send it to OpenFace, and map the result to screen
        coordinates.

        Returns ``(pos, frame, angles)`` — same contract as
        :meth:`HeadPoseTracker.next_position`.
        """
        if self._cap is None or self._sock is None:
            raise RuntimeError("Tracker not started. Call start() first.")

        ok, frame = self._cap.read()
        if not ok:
            return None, np.zeros((1, 1, 3), dtype=np.uint8), None

        # Encode frame as JPEG and send to the container.
        success, jpeg = cv2.imencode(".jpg", frame)
        if not success:
            return None, frame, None

        try:
            self._send_msg(jpeg.tobytes())
            csv_bytes = self._recv_msg()
            csv_text = csv_bytes.decode("utf-8")
        except (OSError, ConnectionError, struct.error) as e:
            if self._debug:
                print(f"[DEBUG] TCP error: {e}")
            # Try to reconnect for the next frame.
            try:
                self._connect()
            except OSError:
                pass
            return None, frame, None

        if self._debug:
            print(f"[DEBUG] Response ({len(csv_text)} chars): {csv_text[:300]}")

        if csv_text.startswith("ERROR"):
            if self._debug:
                print(f"[DEBUG] Server error: {csv_text}")
            return None, frame, None

        row = self._parse_csv(csv_text)
        if row is None:
            if self._debug:
                print("[DEBUG] CSV parse returned None")
            return None, frame, None

        if self._debug:
            print(f"[DEBUG] Parsed keys: {list(row.keys())[:15]}...")
            print(f"[DEBUG] success={row.get('success')}, "
                  f"pose_Rx={row.get('pose_Rx')}, pose_Ry={row.get('pose_Ry')}, "
                  f"confidence={row.get('confidence')}")

        # OpenFace reports success=0 when no face is detected.
        if row.get("success", 0.0) == 0.0:
            if self._debug:
                print("[DEBUG] OpenFace success=0 (no face detected)")
            return None, frame, None

        # OpenFace pose: radians  ->  degrees
        yaw_deg = math.degrees(row.get("pose_Ry", 0.0))
        pitch_deg = math.degrees(row.get("pose_Rx", 0.0))

        # Store AU data
        self.last_au45 = row.get("AU45_r", 0.0)
        self.last_aus = {k: v for k, v in row.items() if k.startswith("AU")}

        # Smoothing
        self._angles_buf.append((yaw_deg, pitch_deg))
        avg_yaw = sum(a[0] for a in self._angles_buf) / len(self._angles_buf)
        avg_pitch = sum(a[1] for a in self._angles_buf) / len(self._angles_buf)

        # Apply calibration offset
        adj_yaw = avg_yaw + self.calib_yaw
        adj_pitch = avg_pitch + self.calib_pitch

        # Map to screen coordinates:
        #   yaw  in [-yaw_span, +yaw_span]  ->  [0, screen_w]
        #   pitch in [-pitch_span, +pitch_span] ->  [0, screen_h]
        sx = int(((adj_yaw + self.yaw_span) / (2.0 * self.yaw_span)) * screen_w)
        sy = int(((adj_pitch + self.pitch_span) / (2.0 * self.pitch_span)) * screen_h)

        sx = max(0, min(screen_w - 1, sx))
        sy = max(0, min(screen_h - 1, sy))

        return (sx, sy), frame, (avg_yaw, avg_pitch)

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_csv(csv_text: str) -> Optional[Dict[str, float]]:
        """Parse the OpenFace CSV output and return the first data row as a
        dict of ``{column_name: float}``."""
        reader = csv.reader(io.StringIO(csv_text))
        try:
            header = next(reader)
        except StopIteration:
            return None

        # OpenFace pads column names with spaces — strip them.
        header = [col.strip() for col in header]

        try:
            values = next(reader)
        except StopIteration:
            return None

        row: Dict[str, float] = {}
        for col, val in zip(header, values):
            try:
                row[col] = float(val.strip())
            except (ValueError, AttributeError):
                continue
        return row
