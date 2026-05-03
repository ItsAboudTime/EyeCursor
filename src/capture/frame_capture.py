"""Capture subprocess: reads frames from one or two cameras and streams them
over UDP to the desktop process as JPEG-encoded fragments.

Spawned by :class:`src.capture.supervisor.CaptureSupervisor`. Indices come
from CLI args; the desktop has already resolved stable IDs to /dev/videoN
indices before invoking us.

Single-camera invocation:
    python -m src.capture.frame_capture --cam0 0 --port 9123

Stereo invocation:
    python -m src.capture.frame_capture --cam0 0 --cam1 2 --port 9123

cam_id mapping in the wire protocol:
    single  --cam0  -> cam_id 0
    stereo  --cam0  -> cam_id 1 (left)
            --cam1  -> cam_id 2 (right)

Stderr handshake (parsed by the supervisor):
    READY=1 cameras=N           -- captures opened, sender threads started
    READY=0 reason=<short>      -- failed to open one or more captures
"""

from __future__ import annotations

import argparse
import signal
import socket
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2

from src.capture.protocol import pack_packets


SINGLE_CAM_ID = 0
STEREO_LEFT_CAM_ID = 1
STEREO_RIGHT_CAM_ID = 2


def _open_camera(
    index: int,
    width: int,
    height: int,
    fps: int,
    mjpeg: bool,
    buffer_size: int,
) -> cv2.VideoCapture:
    """Open a camera and apply the same configuration the rest of the project
    uses (`hardware/stereo_clahe_preview.py`): MJPEG fourcc for higher USB
    framerates, fixed resolution, requested fps, buffer size 1 to minimize
    latency.
    """
    cap = cv2.VideoCapture(index)
    if mjpeg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    return cap


def _camera_loop(
    cap: cv2.VideoCapture,
    cam_id: int,
    sock: socket.socket,
    addr: Tuple[str, int],
    jpeg_quality: int,
    stop_event: threading.Event,
) -> None:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    frame_id = 0
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok or frame is None:
            # Brief sleep avoids a busy-spin if the camera is momentarily
            # unavailable (cap.read normally blocks on the next frame).
            time.sleep(0.001)
            continue
        h, w = frame.shape[:2]
        ok2, encoded = cv2.imencode(".jpg", frame, encode_params)
        if not ok2:
            continue
        timestamp = time.monotonic()
        packets = pack_packets(
            cam_id=cam_id,
            frame_id=frame_id & 0xFFFFFFFF,
            timestamp=timestamp,
            width=w,
            height=h,
            jpeg_bytes=bytes(encoded),
        )
        for pkt in packets:
            try:
                sock.sendto(pkt, addr)
            except OSError:
                # UDP send can fail under transient network conditions; drop
                # the packet and keep streaming. Reassembler TTL handles the
                # rest.
                pass
        frame_id = (frame_id + 1) & 0xFFFFFFFF


def _print_status(line: str) -> None:
    """Stderr handshake line. The supervisor parses these; flush so it
    doesn't sit in a buffer when we exit immediately after."""
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.capture.frame_capture",
        description=(
            "Capture frames from 1 or 2 cameras and stream them over UDP "
            "as JPEG-encoded fragments."
        ),
    )
    parser.add_argument("--cam0", type=int, required=True, help="First camera index.")
    parser.add_argument(
        "--cam1",
        type=int,
        default=None,
        help="Second camera index (sets stereo mode). cam_id=1 for --cam0, cam_id=2 for --cam1.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Destination IP for UDP datagrams.")
    parser.add_argument("--port", type=int, required=True, help="Destination port for UDP datagrams.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--no-mjpeg",
        action="store_true",
        help="Disable MJPEG fourcc on the camera (default: MJPEG enabled).",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--buffer-size", type=int, default=1, help="cv2 CAP_PROP_BUFFERSIZE.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.cam1 is None:
        cam_setup: List[Tuple[int, int]] = [(SINGLE_CAM_ID, args.cam0)]
    else:
        cam_setup = [
            (STEREO_LEFT_CAM_ID, args.cam0),
            (STEREO_RIGHT_CAM_ID, args.cam1),
        ]

    captures: List[Tuple[int, cv2.VideoCapture]] = []
    for cam_id, idx in cam_setup:
        cap = _open_camera(
            index=idx,
            width=args.width,
            height=args.height,
            fps=args.fps,
            mjpeg=not args.no_mjpeg,
            buffer_size=args.buffer_size,
        )
        if not cap.isOpened():
            cap.release()
            for _, c in captures:
                c.release()
            _print_status(f"READY=0 reason=could_not_open_camera_index_{idx}")
            return 1
        captures.append((cam_id, cap))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)

    stop_event = threading.Event()

    def _handle_signal(signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    threads: List[threading.Thread] = []
    for cam_id, cap in captures:
        t = threading.Thread(
            target=_camera_loop,
            args=(cap, cam_id, sock, addr, args.jpeg_quality, stop_event),
            daemon=True,
            name=f"capture-cam{cam_id}",
        )
        threads.append(t)

    _print_status(f"READY=1 cameras={len(captures)} port={args.port}")

    for t in threads:
        t.start()

    # Polling wait so the SIGTERM/SIGINT handler reliably wakes us; on some
    # platforms a blocking Event.wait() does not interrupt on signal.
    while not stop_event.wait(timeout=0.5):
        pass

    for t in threads:
        t.join(timeout=2.0)

    try:
        sock.close()
    except OSError:
        pass
    for _, cap in captures:
        cap.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
