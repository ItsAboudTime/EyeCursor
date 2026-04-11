"""
TCP server that receives JPEG frames, runs OpenFace FeatureExtraction,
and returns the CSV output.

Runs inside the algebr/openface:latest Docker container (Python 3.4).

Protocol (length-prefixed):
  Client sends:  [4 bytes big-endian length][JPEG bytes]
  Server replies: [4 bytes big-endian length][CSV bytes]
"""

import os
import socket
import struct
import subprocess
import tempfile
import time


OPENFACE_BIN = "/home/openface-build/build/bin/FeatureExtraction"
PORT = 5555


def recv_exact(conn, n):
    """Read exactly *n* bytes from *conn*."""
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client disconnected")
        buf += chunk
    return buf


def send_msg(conn, data):
    """Send a length-prefixed message."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    conn.sendall(struct.pack(">I", len(data)) + data)


def recv_msg(conn):
    """Receive a length-prefixed message and return raw bytes."""
    raw_len = recv_exact(conn, 4)
    msg_len = struct.unpack(">I", raw_len)[0]
    return recv_exact(conn, msg_len)


def handle_frame(jpeg_bytes, frame_count):
    """Run FeatureExtraction on a single JPEG and return CSV text."""
    fd, frame_path = tempfile.mkstemp(suffix=".jpg")
    try:
        os.write(fd, jpeg_bytes)
        os.close(fd)

        out_dir = tempfile.mkdtemp()
        name = "frame"

        try:
            subprocess.call(
                [
                    OPENFACE_BIN,
                    "-f", frame_path,
                    "-out_dir", out_dir,
                    "-of", name,
                ],
                stdout=open(os.devnull, "w"),
                stderr=open(os.devnull, "w"),
            )

            csv_path = os.path.join(out_dir, name + ".csv")
            if os.path.exists(csv_path):
                with open(csv_path, "r") as f:
                    return f.read()
            else:
                return "ERROR: no CSV output"
        except Exception as e:
            return "ERROR: " + str(e)
        finally:
            for fname in os.listdir(out_dir):
                try:
                    os.unlink(os.path.join(out_dir, fname))
                except OSError:
                    pass
            try:
                os.rmdir(out_dir)
            except OSError:
                pass
    finally:
        try:
            os.unlink(frame_path)
        except OSError:
            pass


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(1)
    print("OpenFace frame server listening on tcp://0.0.0.0:%d" % PORT)

    while True:
        conn, addr = srv.accept()
        print("Client connected from %s" % str(addr))
        frame_count = 0
        try:
            while True:
                jpeg_bytes = recv_msg(conn)
                t0 = time.time()
                frame_count += 1

                csv_text = handle_frame(jpeg_bytes, frame_count)
                send_msg(conn, csv_text)

                elapsed = time.time() - t0
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                print("Frame %d: %.1fms (%.1f FPS)" % (frame_count, elapsed * 1000, fps))
        except (ConnectionError, struct.error) as e:
            print("Client disconnected: %s" % str(e))
        finally:
            conn.close()


if __name__ == "__main__":
    main()
