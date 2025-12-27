import sys
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any

from cursor.constants import (
    DEFAULT_MOVE_PX_PER_SEC,
    DEFAULT_FRAME_RATE,
    DEFAULT_SCROLL_UNITS_PER_SEC,
)

class SettingsWindow:
    WIDTH = 400
    HEIGHT = 340

    def __init__(self, master, cursor: Any = None):
        """
        Standard initializer. 
        Attaches this UI logic to an existing Tkinter window (master).
        """
        self.master = master
        self.cursor = cursor
        
        master.title("Cursor Configuration")
        master.resizable(False, False)

        style = ttk.Style()
        try:
             style.theme_use('clam') 
        except tk.TclError:
             pass 

        style.configure("Bold.TButton", font=("TkDefaultFont", 11, "bold"))
        style.configure(
            "Modern.TEntry",
            padding=10,
            relief="flat",
            fieldbackground="#F0F0F0",
            borderwidth=0,
        )
        style.map("Modern.TEntry",
            fieldbackground=[("focus", "white")],
            lightcolor=[("focus", "#4CAF50")], 
            bordercolor=[("focus", "#4CAF50")]
        )

        container = ttk.Frame(master, padding=20)
        container.pack(fill="both", expand=True)

        # --- Move Speed ---
        ttk.Label(container, text="Move Speed (px/s):", font=("TkDefaultFont", 10)).pack(anchor="w", pady=(0, 5))
        self.move_speed_var = tk.StringVar(value=str(DEFAULT_MOVE_PX_PER_SEC))
        ttk.Entry(container, textvariable=self.move_speed_var, font=("TkDefaultFont", 11), style="Modern.TEntry").pack(fill="x", pady=(0, 15))

        # --- Frame Rate ---
        ttk.Label(container, text="Frame Rate (fps):", font=("TkDefaultFont", 10)).pack(anchor="w", pady=(0, 5))
        self.frame_rate_var = tk.StringVar(value=str(DEFAULT_FRAME_RATE))
        ttk.Entry(container, textvariable=self.frame_rate_var, font=("TkDefaultFont", 11), style="Modern.TEntry").pack(fill="x", pady=(0, 15))

        # --- Scroll Speed ---
        ttk.Label(container, text="Scroll Speed (units/s):", font=("TkDefaultFont", 10)).pack(anchor="w", pady=(0, 5))
        self.scroll_units_var = tk.StringVar(value=str(DEFAULT_SCROLL_UNITS_PER_SEC))
        ttk.Entry(container, textvariable=self.scroll_units_var, font=("TkDefaultFont", 11), style="Modern.TEntry").pack(fill="x", pady=(0, 15))

        # --- Save Button ---
        ttk.Button(
            container,
            text="Save",
            style="Bold.TButton",
            command=self.save_config,
            cursor="hand2"
        ).pack(fill="both", pady=(10, 0))

    @classmethod
    def create_app(cls, cursor: Any = None) -> tk.Tk:
        """
        Factory Method: Creates the root window, centers it, 
        and initializes the Application. Returns the root object.
        """
        root = tk.Tk()
        root.configure(bg="#ffffff") 
        cls._center_window(root, cls.WIDTH, cls.HEIGHT)
        cls(root, cursor=cursor)
        return root

    @staticmethod
    def _center_window(root: tk.Tk, width: int, height: int):
        """Internal helper to center the window."""
        root.update_idletasks()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        root.geometry(f"{width}x{height}+{x}+{y}")

    def save_config(self):
        try:
            move_speed = float(self.move_speed_var.get())
            frame_rate = int(self.frame_rate_var.get())
            scroll_units = float(self.scroll_units_var.get())

            if move_speed <= 0 or frame_rate <= 0 or scroll_units <= 0:
                 messagebox.showerror("Error", "Values must be greater than zero.")
                 return

            if self.cursor:
                self.cursor.update_config(
                    move_px_per_sec=move_speed, 
                    frame_rate=frame_rate, 
                    scroll_units_per_sec=scroll_units
                )
                messagebox.showinfo("Success", "Configuration updated successfully!")
            else:
                print(f"[Simulation] Config Saved: {move_speed}, {frame_rate}, {scroll_units}")
                messagebox.showinfo("Success", "Configuration simulated (No cursor connected)")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")

def main():
    """Standalone test."""
    class MockCursor:
        def update_config(self, **kwargs):
            print(f"Mock Cursor Received: {kwargs}")

    root = SettingsWindow.create_app(cursor=MockCursor())
    root.mainloop()

if __name__ == "__main__":
    sys.exit(main())