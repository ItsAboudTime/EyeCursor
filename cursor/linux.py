import importlib
from typing import Any, Optional, Tuple
from cursor.base import Cursor


class LinuxCursor(Cursor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mouse: Optional[Any] = None
        self._button: Optional[Any] = None

    def _controller(self) -> Any:
        if self._mouse is None:
            mouse_mod = importlib.import_module('pynput.mouse')
            self._mouse = mouse_mod.Controller()
            self._button = mouse_mod.Button
        return self._mouse

    def _buttons(self) -> Any:
        self._controller()
        return self._button

    def get_pos(self) -> Tuple[int, int]:
        x, y = self._controller().position
        return int(x), int(y)

    def set_pos(self, x: int, y: int) -> None:
        self._controller().position = (int(x), int(y))

    def _x11_bounds(self) -> Optional[Tuple[int, int]]:
        try:
            xdisplay = importlib.import_module('Xlib.display')
            display = xdisplay.Display()
            try:
                screen = display.screen()
                return int(screen.width_in_pixels), int(screen.height_in_pixels)
            finally:
                display.close()
        except Exception:
            return None

    def _tk_bounds(self) -> Optional[Tuple[int, int]]:
        try:
            tk = importlib.import_module('tkinter')
            root = tk.Tk()
            root.withdraw()
            try:
                return int(root.winfo_screenwidth()), int(root.winfo_screenheight())
            finally:
                root.destroy()
        except Exception:
            return None

    def get_virtual_bounds(self) -> Tuple[int, int, int, int]:
        dims = self._x11_bounds() or self._tk_bounds()
        if dims is None:
            raise RuntimeError('Could not determine screen size')
        width, height = dims
        return 0, 0, int(width) - 1, int(height) - 1

    def left_click(self) -> None:
        self.left_down()
        self.left_up()

    def left_down(self) -> None:
        self._controller().press(self._buttons().left)

    def left_up(self) -> None:
        self._controller().release(self._buttons().left)

    def right_click(self) -> None:
        self.right_down()
        self.right_up()

    def right_down(self) -> None:
        self._controller().press(self._buttons().right)

    def right_up(self) -> None:
        self._controller().release(self._buttons().right)

    def scroll(self, delta: int) -> None:
        self._controller().scroll(0, int(delta))
