import sys
import os

# Ensure project root is on sys.path so 'cursor' package is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cursor import create_cursor


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


def main():
    cur = create_cursor()

    minx, miny, maxx, maxy = cur.get_virtual_bounds()
    print("Cursor Move (CLI)")
    print(f"Virtual screen bounds: x [{minx}..{maxx}], y [{miny}..{maxy}]")
    print(f"Speed: {cur.speed_px_per_sec} px/s")
    print("Enter coordinates as 'x y' or 'x,y' (type 'q' to quit).")
    print("Commands: 'left' for left click, 'right' for right click, 'scroll <delta>' for scrolling.\n")

    while True:
        try:
            raw = input("Command > ")
        except KeyboardInterrupt:
            print("\nBye.")
            break

        try:
            if raw.lower() == "q":
                print("Bye.")
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


if __name__ == "__main__":
    sys.exit(main())
