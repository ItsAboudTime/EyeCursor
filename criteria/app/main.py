from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from criteria.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("EyeCursor TestLab")
    app.setOrganizationName("EyeCursorTeam")
    style_path = Path(__file__).resolve().parents[1] / "resources" / "styles" / "app.qss"
    if style_path.exists():
        app.setStyleSheet(style_path.read_text(encoding="utf-8"))

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

