import sys
import threading
import tkinter as tk
import queue

from cursor import create_cursor
from ui import SettingsWindow


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


def main():
    cur = create_cursor()
    msg_queue = queue.Queue()

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
                root.quit()
                sys.exit(0)
        except queue.Empty:
            pass
        root.after(100, check_queue)

    # Start Background Thread (CLI)
    t = threading.Thread(target=run_cli_loop, args=(cur, msg_queue), daemon=True)
    t.start()

    # Start App
    check_queue()
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())