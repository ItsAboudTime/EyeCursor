"""Unit tests for the capture-process / desktop-process UDP framing protocol.

Pure bytes-in / bytes-out -- no sockets, no cameras, no OpenCV.

Run with:

    python -m unittest tests.test_capture_protocol -v
"""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.capture.protocol import (
    HEADER_FMT,
    HEADER_SIZE,
    MAGIC,
    MAX_PAYLOAD,
    VERSION,
    Reassembler,
    pack_packets,
    parse_packet,
)


class HeaderSizeTests(unittest.TestCase):
    def test_header_is_28_bytes(self) -> None:
        self.assertEqual(HEADER_SIZE, 28)
        self.assertEqual(struct.calcsize(HEADER_FMT), 28)


class PackParseRoundTripTests(unittest.TestCase):
    def test_single_packet_round_trip(self) -> None:
        jpeg = b"\xff\xd8" + b"x" * 100 + b"\xff\xd9"
        packets = pack_packets(
            cam_id=1,
            frame_id=42,
            timestamp=1234.5,
            width=640,
            height=480,
            jpeg_bytes=jpeg,
        )
        self.assertEqual(len(packets), 1)
        header, payload = parse_packet(packets[0])
        self.assertEqual(header.cam_id, 1)
        self.assertEqual(header.frame_id, 42)
        self.assertEqual(header.packet_idx, 0)
        self.assertEqual(header.total_pkts, 1)
        self.assertAlmostEqual(header.timestamp, 1234.5)
        self.assertEqual(header.width, 640)
        self.assertEqual(header.height, 480)
        self.assertEqual(header.payload_len, len(jpeg))
        self.assertEqual(payload, jpeg)

    def test_multi_packet_split_and_reassemble(self) -> None:
        jpeg = bytes(range(256)) * 200  # ~50 KB
        self.assertGreater(len(jpeg), MAX_PAYLOAD * 30)
        packets = pack_packets(
            cam_id=2,
            frame_id=7,
            timestamp=99.9,
            width=1280,
            height=720,
            jpeg_bytes=jpeg,
        )
        expected_total = (len(jpeg) + MAX_PAYLOAD - 1) // MAX_PAYLOAD
        self.assertEqual(len(packets), expected_total)

        for i, pkt in enumerate(packets):
            header, _ = parse_packet(pkt)
            self.assertEqual(header.packet_idx, i)
            self.assertEqual(header.total_pkts, expected_total)
            self.assertEqual(header.cam_id, 2)
            self.assertEqual(header.frame_id, 7)

        reasm = Reassembler()
        result = None
        for pkt in packets:
            result = reasm.feed(pkt, now=0.0) or result
        self.assertIsNotNone(result)
        self.assertEqual(result.cam_id, 2)
        self.assertEqual(result.frame_id, 7)
        self.assertAlmostEqual(result.timestamp, 99.9)
        self.assertEqual(result.width, 1280)
        self.assertEqual(result.height, 720)
        self.assertEqual(result.jpeg_bytes, jpeg)

    def test_empty_payload_yields_no_packets(self) -> None:
        self.assertEqual(pack_packets(0, 0, 0.0, 0, 0, b""), [])


class ParsingErrorTests(unittest.TestCase):
    def test_short_packet_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_packet(b"\x00" * (HEADER_SIZE - 1))

    def test_bad_magic_raises(self) -> None:
        bad_header = struct.pack(HEADER_FMT, b"WRNG", VERSION, 0, 0, 0, 1, 0.0, 1, 1, 1)
        with self.assertRaises(ValueError):
            parse_packet(bad_header + b"x")

    def test_bad_version_raises(self) -> None:
        bad_header = struct.pack(HEADER_FMT, MAGIC, 99, 0, 0, 0, 1, 0.0, 1, 1, 1)
        with self.assertRaises(ValueError):
            parse_packet(bad_header + b"x")

    def test_truncated_payload_raises(self) -> None:
        # Header claims 100-byte payload but only 1 byte follows.
        header = struct.pack(HEADER_FMT, MAGIC, VERSION, 0, 0, 0, 1, 0.0, 1, 1, 100)
        with self.assertRaises(ValueError):
            parse_packet(header + b"x")


class ReassemblerTests(unittest.TestCase):
    def _make_packets(self, cam_id: int, frame_id: int, payload_size: int = 5000):
        jpeg = bytes((i + frame_id) & 0xFF for i in range(payload_size))
        packets = pack_packets(cam_id, frame_id, frame_id * 0.1, 640, 480, jpeg)
        return jpeg, packets

    def test_out_of_order_packets_reassemble(self) -> None:
        jpeg, packets = self._make_packets(cam_id=1, frame_id=1, payload_size=8000)
        reasm = Reassembler()

        # Reverse-feed everything but verify only the last completion fires.
        reversed_packets = list(reversed(packets))
        result = None
        for pkt in reversed_packets:
            r = reasm.feed(pkt, now=0.0)
            if r is not None:
                self.assertIsNone(result, "completion should fire only once")
                result = r
        self.assertIsNotNone(result)
        self.assertEqual(result.jpeg_bytes, jpeg)

    def test_ttl_drops_partial_frame(self) -> None:
        _, packets = self._make_packets(cam_id=1, frame_id=1, payload_size=8000)
        reasm = Reassembler(ttl=0.2)
        # Feed only half.
        for pkt in packets[: len(packets) // 2]:
            self.assertIsNone(reasm.feed(pkt, now=0.0))
        # Before TTL: still buffered.
        self.assertEqual(reasm.prune(now=0.1), 0)
        # After TTL: dropped.
        self.assertEqual(reasm.prune(now=0.5), 1)
        # Feeding the remaining packets now starts a fresh (incomplete)
        # partial because the original was pruned.
        for pkt in packets[len(packets) // 2 :]:
            self.assertIsNone(reasm.feed(pkt, now=0.6))

    def test_straggler_of_delivered_frame_is_dropped(self) -> None:
        _, packets = self._make_packets(cam_id=1, frame_id=5, payload_size=8000)
        reasm = Reassembler()
        # Deliver all but one.
        first_n = packets[:-1]
        last = packets[-1]
        for pkt in first_n:
            self.assertIsNone(reasm.feed(pkt, now=0.0))
        # Last packet completes.
        completed = reasm.feed(last, now=0.0)
        self.assertIsNotNone(completed)
        # A re-delivery of any packet from frame 5 should not cause re-completion.
        self.assertIsNone(reasm.feed(packets[0], now=0.0))
        self.assertIsNone(reasm.feed(packets[-1], now=0.0))

    def test_older_frame_dropped_after_newer_delivered(self) -> None:
        # Frame 3 partially arrives, then frame 4 fully arrives, then frame 3
        # finishes. The frame-3 straggler should be discarded.
        _, p3 = self._make_packets(cam_id=1, frame_id=3, payload_size=8000)
        _, p4 = self._make_packets(cam_id=1, frame_id=4, payload_size=8000)
        reasm = Reassembler()

        # First half of frame 3.
        for pkt in p3[: len(p3) // 2]:
            self.assertIsNone(reasm.feed(pkt, now=0.0))
        # All of frame 4 arrives.
        result4 = None
        for pkt in p4:
            r = reasm.feed(pkt, now=0.0)
            if r is not None:
                result4 = r
        self.assertIsNotNone(result4)
        self.assertEqual(result4.frame_id, 4)
        # Now the second half of frame 3 arrives -- should be dropped.
        for pkt in p3[len(p3) // 2 :]:
            self.assertIsNone(reasm.feed(pkt, now=0.0))

    def test_multiplexed_cam_ids_do_not_collide(self) -> None:
        jpeg_a, pkts_a = self._make_packets(cam_id=1, frame_id=1, payload_size=8000)
        jpeg_b, pkts_b = self._make_packets(cam_id=2, frame_id=1, payload_size=8000)
        reasm = Reassembler()

        # Interleave the two streams by index.
        results = []
        for a, b in zip(pkts_a, pkts_b):
            for r in (reasm.feed(a, now=0.0), reasm.feed(b, now=0.0)):
                if r is not None:
                    results.append(r)
        # If the streams had differing packet counts (they don't here, but
        # be defensive), drain the leftover.
        for pkt in pkts_a[len(pkts_b):] + pkts_b[len(pkts_a):]:
            r = reasm.feed(pkt, now=0.0)
            if r is not None:
                results.append(r)

        self.assertEqual(len(results), 2)
        by_cam = {r.cam_id: r for r in results}
        self.assertEqual(by_cam[1].jpeg_bytes, jpeg_a)
        self.assertEqual(by_cam[2].jpeg_bytes, jpeg_b)

    def test_feed_with_garbage_returns_none(self) -> None:
        reasm = Reassembler()
        self.assertIsNone(reasm.feed(b""))
        self.assertIsNone(reasm.feed(b"too short"))
        self.assertIsNone(reasm.feed(b"WRNG" + b"\x00" * (HEADER_SIZE - 4) + b"x"))


class PackPacketsBoundsTests(unittest.TestCase):
    def test_chunk_boundary_payload_size(self) -> None:
        # Payload exactly MAX_PAYLOAD bytes -> 1 packet.
        jpeg = b"\x42" * MAX_PAYLOAD
        packets = pack_packets(0, 0, 0.0, 1, 1, jpeg)
        self.assertEqual(len(packets), 1)
        # Payload MAX_PAYLOAD + 1 -> 2 packets.
        jpeg2 = b"\x42" * (MAX_PAYLOAD + 1)
        packets2 = pack_packets(0, 0, 0.0, 1, 1, jpeg2)
        self.assertEqual(len(packets2), 2)

    def test_per_packet_payload_len_matches_chunk(self) -> None:
        jpeg = b"x" * (MAX_PAYLOAD * 3 + 17)
        packets = pack_packets(0, 1, 0.0, 1, 1, jpeg)
        # First three are exactly MAX_PAYLOAD; last is the remainder.
        for pkt in packets[:-1]:
            header, payload = parse_packet(pkt)
            self.assertEqual(header.payload_len, MAX_PAYLOAD)
            self.assertEqual(len(payload), MAX_PAYLOAD)
        last_header, last_payload = parse_packet(packets[-1])
        self.assertEqual(last_header.payload_len, 17)
        self.assertEqual(len(last_payload), 17)


if __name__ == "__main__":
    unittest.main()
