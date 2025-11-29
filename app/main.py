"""
Application entry point.

Run with: python -m app.main
"""

import sys
from PySide6.QtWidgets import QApplication

from app.gui.main_window import MainWindow


def main() -> None:
    """Initialize and run the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Image Processing Mini-Suite")
    app.setApplicationVersion("0.1.0")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
