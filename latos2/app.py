"""Start the Latos V2 HIOKI Measurement Analyzer.

    python -m latos2 [workbook.xlsx]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from latos2.ui.main_window import MainWindow
from latos2.ui.style import apply_light_theme


def main(argv: list[str] | None = None) -> int:
    args = sys.argv if argv is None else argv
    app = QApplication(args)
    app.setApplicationName("Latos V2")
    apply_light_theme(app)
    window = MainWindow()
    window.showMaximized()
    if len(args) > 1:
        window.open_file(Path(args[1]))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
