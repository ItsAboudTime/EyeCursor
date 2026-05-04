"""UDP framing protocol for the capture-process / desktop-process boundary.

A JPEG-encoded frame is split into one or more datagrams, each prefixed by a
fixed 28-byte header. The receiver reassembles complete frames keyed by
(cam_id, frame_id). No OpenCV / numpy dependency lives here — this module
is pure-Python so it can be unit-tested without hardware or image libs.
"""

from __future__ import annotations

import struct
import time
from typing import Dict, List, NamedTuple, Optional, Tuple


MAGIC = b"EYEC"
VERSION = 1

# Stay under a typical 1500-byte Ethernet MTU once the 28-byte header and
# IP/UDP overhead are accounted for. 1400 leaves comfortable headroom.
MAX_PAYLOAD = 1400

HEADER_FMT = "<4sBBIHHdHHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 28, f"unexpected header size {HEADER_SIZE}"


class PacketHeader(NamedTuple):
    cam_id: int
    frame_id: int
    packet_idx: int
    total_pkts: int
    timestamp: float
    width: int
    height: int
    payload_len: int


class CompletedFrame(NamedTuple):
    cam_id: int
    frame_id: int
    timestamp: float
    width: int
    height: int
    jpeg_bytes: bytes


def pack_packets(
    cam_id: int,
    frame_id: int,
    timestamp: float,
    width: int,
    height: int,
    jpeg_bytes: bytes,
) -> List[bytes]:
    """Split a JPEG byte string into header-prefixed UDP datagrams."""
    payload_len = len(jpeg_bytes)
    if payload_len == 0:
        return []
    total = (payload_len + MAX_PAYLOAD - 1) // MAX_PAYLOAD
    if total > 0xFFFF:
        raise ValueError(f"frame too large: {payload_len} bytes -> {total} packets")
    packets: List[bytes] = []
    for i in range(total):
        start = i * MAX_PAYLOAD
        end = min(start + MAX_PAYLOAD, payload_len)
        chunk = jpeg_bytes[start:end]
        header = struct.pack(
            HEADER_FMT,
            MAGIC,
            VERSION,
            cam_id & 0xFF,
            frame_id & 0xFFFFFFFF,
            i,
            total,
            timestamp,
            width & 0xFFFF,
            height & 0xFFFF,
            len(chunk),
        )
        packets.append(header + chunk)
    return packets


def parse_packet(packet: bytes) -> Tuple[PacketHeader, bytes]:
    """Decode a single datagram into (header, payload). Raises on bad input."""
    if len(packet) < HEADER_SIZE:
        raise ValueError(f"packet shorter than header: {len(packet)} bytes")
    fields = struct.unpack(HEADER_FMT, packet[:HEADER_SIZE])
    magic, version, cam_id, frame_id, packet_idx, total_pkts, timestamp, width, height, payload_len = fields
    if magic != MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version: {version}")
    payload = packet[HEADER_SIZE : HEADER_SIZE + payload_len]
    if len(payload) != payload_len:
        raise ValueError(
            f"truncated payload: header says {payload_len}, got {len(payload)}"
        )
    return (
        PacketHeader(
            cam_id=cam_id,
            frame_id=frame_id,
            packet_idx=packet_idx,
            total_pkts=total_pkts,
            timestamp=timestamp,
            width=width,
            height=height,
            payload_len=payload_len,
        ),
        payload,
    )


class _PartialFrame:
    __slots__ = ("packets", "total", "deadline", "timestamp", "width", "height")

    def __init__(self, total: int, deadline: float, timestamp: float, width: int, height: int) -> None:
        self.packets: Dict[int, bytes] = {}
        self.total = total
        self.deadline = deadline
        self.timestamp = timestamp
        self.width = width
        self.height = height

    def add(self, idx: int, payload: bytes) -> None:
        if idx not in self.packets:
            self.packets[idx] = payload

    def is_complete(self) -> bool:
        return len(self.packets) == self.total

    def assemble(self) -> bytes:
        return b"".join(self.packets[i] for i in range(self.total))


class Reassembler:
    """Buffer fragmented frames keyed by (cam_id, frame_id), drop expired ones.

    Out-of-order packets within a frame reassemble correctly. Stragglers of
    an already-delivered frame_id are dropped (by comparison against the
    latest delivered frame_id per cam_id). Incomplete frames are dropped
    after ``ttl`` seconds via :py:meth:`prune`.
    """

    def __init__(self, ttl: float = 0.2) -> None:
        self._partials: Dict[Tuple[int, int], _PartialFrame] = {}
        self._latest_frame_id: Dict[int, int] = {}
        self._ttl = ttl

    def feed(self, packet: bytes, now: Optional[float] = None) -> Optional[CompletedFrame]:
        if now is None:
            now = time.monotonic()
        try:
            header, payload = parse_packet(packet)
        except ValueError:
            return None

        latest = self._latest_frame_id.get(header.cam_id)
        if latest is not None and header.frame_id <= latest:
            # Already delivered (or older): drop straggler.
            return None

        key = (header.cam_id, header.frame_id)
        partial = self._partials.get(key)
        if partial is None:
            partial = _PartialFrame(
                total=header.total_pkts,
                deadline=now + self._ttl,
                timestamp=header.timestamp,
                width=header.width,
                height=header.height,
            )
            self._partials[key] = partial
        partial.add(header.packet_idx, payload)
        if not partial.is_complete():
            return None

        del self._partials[key]
        self._latest_frame_id[header.cam_id] = header.frame_id
        # Drop any older partials for this cam (we've moved on past them).
        stale = [k for k in self._partials if k[0] == header.cam_id and k[1] < header.frame_id]
        for k in stale:
            del self._partials[k]
        return CompletedFrame(
            cam_id=header.cam_id,
            frame_id=header.frame_id,
            timestamp=partial.timestamp,
            width=partial.width,
            height=partial.height,
            jpeg_bytes=partial.assemble(),
        )

    def prune(self, now: Optional[float] = None) -> int:
        if now is None:
            now = time.monotonic()
        expired = [k for k, p in self._partials.items() if p.deadline <= now]
        for k in expired:
            del self._partials[k]
        return len(expired)
