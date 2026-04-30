from __future__ import annotations

import json
import socket
import threading
import time
from typing import Optional


class DepthClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 7345) -> None:
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._latest_depth: Optional[float] = None
        self._last_rx: float = 0.0
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._enabled = False

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.bind((self.host, self.port))
            self._sock = s
            self._enabled = True
        except OSError as e:
            print(f"[depth_client] bind failed on {self.host}:{self.port} ({e}); depth disabled")
            self._enabled = False
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="DepthClient")
        self._thread.start()

    def _run(self) -> None:
        sock = self._sock
        if sock is None:
            return
        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                obj = json.loads(data.decode("utf-8"))
                d = obj.get("depth", None)
                if d is None:
                    continue
                value = float(d)
            except (ValueError, json.JSONDecodeError, AttributeError):
                continue
            with self._lock:
                self._latest_depth = value
                self._last_rx = time.monotonic()

    def get_depth(self) -> Optional[float]:
        if not self._enabled:
            return None
        with self._lock:
            if self._latest_depth is None:
                return None
            if (time.monotonic() - self._last_rx) > 2.0:
                return None
            return self._latest_depth

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
