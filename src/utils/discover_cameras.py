"""
Discover connected cameras and preview them so you can identify which index
corresponds to which physical camera.

Usage:
    python utils/discover_cameras.py

The script scans video device indices 0-9, opens each one that exists, and
shows a live preview window labelled with its index.  Once you know which
index is left and which is right, update LEFT_CAMERA_INDEX / RIGHT_CAMERA_INDEX
in examples/two_camera_final.py.

Press q to quit all preview windows.
"""

import os
import cv2

from src.core.devices.stable_camera_id import stable_id_for_index


def discover():
    # Use V4L2 backend directly to avoid noisy fallback attempts.
    video_devices = sorted(
        int(f.replace("video", ""))
        for f in os.listdir("/dev")
        if f.startswith("video") and f[5:].isdigit()
    )

    caps = {}
    sids = {}
    for i in video_devices:
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                caps[i] = cap
                sid = stable_id_for_index(i) or "no-stable-id"
                sids[i] = sid
                print(f"  Camera index {i}: {frame.shape[1]}x{frame.shape[0]}  [{sid}]")
            else:
                cap.release()
        else:
            cap.release()

    if not caps:
        print("No cameras found.")
        return

    print(f"\nFound {len(caps)} camera(s). Showing live previews — press 'q' to quit.")
    print("Each feed has its INDEX shown in the top-left corner.\n")

    while True:
        for idx, cap in list(caps.items()):
            ret, frame = cap.read()
            if not ret:
                continue
            # Draw the index large and clear on the frame.
            label = f"INDEX {idx}"
            cv2.putText(
                frame, label, (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5,
            )
            cv2.imshow(f"Camera {idx}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Scanning for cameras...\n")
    discover()
