import sys

from PySide6.QtWidgets import QApplication

from src.app.bootstrap import initialize_app


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("EyeCursor")
    app.setOrganizationName("EyeCursorTeam")

    window = initialize_app()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
