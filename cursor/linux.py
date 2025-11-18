import subprocess
from typing import Tuple
from cursor.base import Cursor

class LinuxCursor(Cursor):
    def get_pos(self) -> Tuple[int, int]:
        output = subprocess.check_output(['xdotool', 'getmouselocation', '--shell']).decode()
        pos = {}
        for line in output.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=')
                pos[key] = int(value)
        return pos['X'], pos['Y']

    def set_pos(self, x: int, y: int) -> None:
        subprocess.call(['xdotool', 'mousemove', str(int(x)), str(int(y))])

    def get_virtual_bounds(self) -> Tuple[int, int, int, int]:
        output = subprocess.check_output(['xrandr']).decode()
        for line in output.splitlines():
            if ' connected ' in line:
                parts = line.split()
                for part in parts:
                    if '+' in part and 'x' in part:
                        res = part.split('+')[0]
                        width, height = map(int, res.split('x'))
                        return 0, 0, width - 1, height - 1
        raise RuntimeError('Could not determine screen size')
