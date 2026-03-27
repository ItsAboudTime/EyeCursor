import sys
import threading
import queue
import time
import cv2

from cursor import create_cursor
from ui import SettingsWindow
from head_track.face_analysis_pipeline import FaceAnalysisPipeline


def parse_coords(raw: str):
    """
    Parse user input like '800 400', '800,400' or 'q'.

    Returns:
      - (x, y) as ints
      - the string "quit" if the user entered 'q' (case-insensitive)
    Raises:
      - ValueError for invalid input
    """
    s = raw.strip()
    if s.lower() == "q":
        return "quit"

    s = s.replace(",", " ")
    parts = [p for p in s.split() if p]
    if len(parts) != 2:
        raise ValueError("Please enter two numbers like: 800 400 or 800,400")

    x = int(float(parts[0]))
    y = int(float(parts[1]))
    return x, y


def run_cli_loop(cur, stop_queue):
    """
    Runs the CLI blocking input loop in a background thread.
    Signals the main thread to stop via stop_queue.
    """
    minx, miny, maxx, maxy = cur.get_virtual_bounds()
    print("Cursor Move (CLI)")
    print(f"Virtual screen bounds: x [{minx}..{maxx}], y [{miny}..{maxy}]")
    print(f"Move Speed: {cur.move_px_per_sec} px/s")
    print("Enter coordinates as 'x y' or 'x,y' (type 'q' to quit).")
    print("Commands: 'left' for left click, 'right' for right click, 'scroll <delta>' for scrolling.\n")

    while True:
        try:
            # This blocks, so it must live in a thread
            sys.stdout.write("Command > ")
            sys.stdout.flush()
            raw = sys.stdin.readline()
            if not raw:
                break
            raw = raw.strip()
        except KeyboardInterrupt:
            print("\nBye.")
            stop_queue.put("QUIT")
            break

        try:
            if raw.lower() == "q":
                print("Bye.")
                stop_queue.put("QUIT")
                break

            if raw.lower() == "left":
                print("Performing left click...")
                cur.left_click()
                continue

            if raw.lower() == "right":
                print("Performing right click...")
                cur.right_click()
                continue

            if raw.lower().startswith("scroll"):
                try:
                    parts = raw.split()
                    delta = int(parts[1])
                    print(f"Scrolling {'up' if delta > 0 else 'down'} by {delta}...")
                    cur.scroll_with_speed(delta)
                except (IndexError, ValueError):
                    print("Invalid scroll command. Use 'scroll <delta>' where delta is an integer.")
                continue

            result = parse_coords(raw)
            x, y = result
            print(f"Moving to ({x}, {y}) ...")
            cur.move_to_with_speed(x, y)
        except ValueError as e:
            print(e)


def run_tracker_loop(cur, stop_event, stop_queue):
    minx, miny, maxx, maxy = cur.get_virtual_bounds()
    screen_w = maxx - minx + 1
    screen_h = maxy - miny + 1

    camera_indices = {
        "cam4": 4,
        "cam6": 6,
    }
    latest_pos = {"cam4": None, "cam6": None}
    latest_lock = threading.Lock()

    def worker(name, camera_index):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Failed to start {name}: could not open webcam (index {camera_index})")
            stop_queue.put("QUIT")
            return

        face_analysis_pipeline = FaceAnalysisPipeline(yaw_span=20.0, pitch_span=10.0, ema_alpha=0.25)

        try:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pos = None

                face_analysis = face_analysis_pipeline.analyze(
                    rgb_frame=rgb,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    screen_width=screen_w,
                    screen_height=screen_h,
                )
                if face_analysis is not None:
                    pos = face_analysis.screen_position

                with latest_lock:
                    latest_pos[name] = pos
        except Exception as e:
            print(f"{name} tracking error: {e}")
            stop_queue.put("QUIT")
        finally:
            cap.release()
            face_analysis_pipeline.release()

    workers = [
        threading.Thread(target=worker, args=("cam4", camera_indices["cam4"]), daemon=True),
        threading.Thread(target=worker, args=("cam6", camera_indices["cam6"]), daemon=True),
    ]

    for t in workers:
        t.start()

    print("Stereo head tracking running on cameras 4 and 6.")

    while not stop_event.is_set():
        with latest_lock:
            p1 = latest_pos["cam4"]
            p2 = latest_pos["cam6"]

        valid = [p for p in (p1, p2) if p is not None]
        if valid:
            avg_x = int(sum(p[0] for p in valid) / len(valid))
            avg_y = int(sum(p[1] for p in valid) / len(valid))

            target_x = max(minx, min(maxx, avg_x + minx))
            target_y = max(miny, min(maxy, avg_y + miny))
            cur.step_towards(target_x, target_y)

        time.sleep(0.005)

    for t in workers:
        t.join(timeout=1.0)


def main():
    cur = create_cursor()
    msg_queue = queue.Queue()
    stop_event = threading.Event()

    # Initialize UI (Inject the cursor directly)
    try:
        # Call the class method directly
        root = SettingsWindow.create_app(cursor=cur)
    except Exception as e:
        print(f"Fatal Error: Could not start Tkinter: {e}")
        return

    # Define Main Thread Polling
    def check_queue():
        try:
            msg = msg_queue.get_nowait()
            if msg == "QUIT":
                stop_event.set()
                root.quit()
                root.destroy()
                return
        except queue.Empty:
            pass
        root.after(100, check_queue)

    # Start Background Thread (CLI)
    t = threading.Thread(target=run_cli_loop, args=(cur, msg_queue), daemon=True)
    t.start()

    t2 = threading.Thread(target=run_tracker_loop, args=(cur, stop_event, msg_queue), daemon=True)
    t2.start()

    root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), root.quit(), root.destroy()))

    # Start App
    check_queue()
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())