import sys
import helpers.cursor as cursor

if sys.platform != "win32":
    sys.exit("This script is for Windows only.")

minx, miny, maxx, maxy = cursor.get_virtual_bounds()
print("Cursor Move (Windows)")
print(f"Virtual screen bounds: x [{minx}..{maxx}], y [{miny}..{maxy}]")
print(f"Speed: {cursor.SPEED_PX_PER_SEC} px/s")
print("Enter coordinates as 'x y' or 'x,y' (type 'q' to quit).")
print()

while True:
    try:
        raw = input("Go to (x y) > ")
    except KeyboardInterrupt:
        print("\nBye.")
        break

    try:
        result = cursor.parse_coords(raw)
        if result == "quit":
            print("Bye.")
            break
        x, y = result
        print(f"Moving to ({x}, {y}) ...")
        cursor.move_to(x, y)
    except ValueError as e:
        print(e)
