import sys

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

if not sys.platform.startswith("win"):
    sys.exit("This script is for Windows only.")

# Create a cursor instance (WindowsCursor under the hood)
cur = create_cursor()

minx, miny, maxx, maxy = cur.get_virtual_bounds()
print("Cursor Move (Windows)")
print(f"Virtual screen bounds: x [{minx}..{maxx}], y [{miny}..{maxy}]")
print(f"Speed: {cur.speed_px_per_sec} px/s")
print("Enter coordinates as 'x y' or 'x,y' (type 'q' to quit).")
print()

while True:
    try:
        raw = input("Go to (x y) > ")
    except KeyboardInterrupt:
        print("\nBye.")
        break

    try:
        result = parse_coords(raw)
        if result == "quit":
            print("Bye.")
            break

        x, y = result
        print(f"Moving to ({x}, {y}) ...")
        cur.move_to(x, y)
    except ValueError as e:
        print(e)
