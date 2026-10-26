"""A fixed light appearance, independent of the Windows colour scheme."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

# Restrained neutrals; colour is kept for data (heating red, cooling blue) and state.
WINDOW = "#f2f2f2"
PANEL = "#fafafa"
BASE = "#ffffff"
BORDER = "#c8c8c8"
TEXT = "#1b1b1b"
MUTED = "#5f5f5f"
ACCENT = "#1f5fa8"
SELECTION = "#cfe1f5"
WARNING_BG = "#fff7e0"
WARNING_BORDER = "#e0c068"
OK = "#1e6b3a"
CAUTION = "#8a5a00"
ERROR = "#a32b20"
HEATING = "#c0392b"
COOLING = "#1f6fb2"

STYLESHEET = f"""
QMainWindow, QWidget#workspace {{ background: {WINDOW}; }}
QToolBar {{ background: {WINDOW}; border: none; border-bottom: 1px solid {BORDER}; spacing: 2px; padding: 2px; }}
QToolBar QToolButton {{ padding: 3px 8px; }}
QStatusBar {{ background: {WINDOW}; border-top: 1px solid {BORDER}; }}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QWidget#panel {{ background: {PANEL}; }}
QLabel#panelTitle {{ font-weight: 600; padding: 5px 8px; border-bottom: 1px solid {BORDER}; background: {WINDOW}; }}
QLabel#sectionTitle {{ font-weight: 600; color: {TEXT}; padding: 6px 0 2px 0; border-bottom: 1px solid {BORDER}; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#warning {{ background: {WARNING_BG}; border: 1px solid {WARNING_BORDER}; padding: 6px 8px; color: #5a4300; }}
QTreeWidget, QTableWidget {{ background: {BASE}; border: 1px solid {BORDER}; alternate-background-color: #f7f7f7; }}
QTreeWidget::item {{ padding: 2px 0; }}
QTreeWidget::item:selected, QTableWidget::item:selected {{ background: {SELECTION}; color: {TEXT}; }}
QHeaderView::section {{ background: {WINDOW}; border: none; border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER}; padding: 3px 4px; font-weight: 600; }}
QTabBar#documentTabs::tab {{ background: {WINDOW}; border: 1px solid {BORDER}; border-bottom: none;
    padding: 5px 14px; margin-right: 2px; color: {MUTED}; }}
QTabBar#documentTabs::tab:selected {{ background: {BASE}; color: {TEXT}; font-weight: 600; }}
QWidget#plotToolbar {{ background: {BASE}; border-bottom: 1px solid #e2e2e2; }}
QWidget#plotToolbar QToolButton {{ padding: 3px 9px; border: 1px solid transparent; border-radius: 2px; }}
QWidget#plotToolbar QToolButton:hover {{ border-color: {BORDER}; background: {WINDOW}; }}
QWidget#plotToolbar QToolButton:checked {{ background: {SELECTION}; border-color: #9dbbe0; }}
QWidget#workflow {{ background: {BASE}; border-bottom: 1px solid {BORDER}; }}
QToolButton#step {{ border: none; padding: 4px 6px; color: {MUTED}; }}
QToolButton#step[state="current"] {{ color: {ACCENT}; font-weight: 600; }}
QToolButton#step[state="done"] {{ color: {OK}; }}
QToolButton#step:hover {{ text-decoration: underline; }}
QToolButton#disclosure {{ border: none; font-weight: 600; padding: 6px 0 2px 0; text-align: left; }}
QWidget#page {{ background: {BASE}; }}
QFrame#card {{ background: {BASE}; border: 1px solid {BORDER}; border-radius: 3px; }}
QFrame#card[primary="true"] {{ border: 1px solid {ACCENT}; }}
QFrame#card QLabel {{ border: none; }}
"""


def apply_light_theme(app: QApplication) -> None:
    """Fusion style with a fixed light palette, whatever the system theme."""
    hints = app.styleHints()
    if hasattr(hints, "setColorScheme"):
        hints.setColorScheme(Qt.ColorScheme.Light)
    app.setStyle("Fusion")
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: WINDOW,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: BASE,
        QPalette.ColorRole.AlternateBase: "#f7f7f7",
        QPalette.ColorRole.ToolTipBase: "#ffffe8",
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.PlaceholderText: MUTED,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: "#f4f4f4",
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.BrightText: "#ffffff",
        QPalette.ColorRole.Light: "#ffffff",
        QPalette.ColorRole.Midlight: "#e6e6e6",
        QPalette.ColorRole.Mid: "#bdbdbd",
        QPalette.ColorRole.Dark: "#9a9a9a",
        QPalette.ColorRole.Shadow: "#6e6e6e",
        QPalette.ColorRole.Highlight: SELECTION,
        QPalette.ColorRole.HighlightedText: TEXT,
        QPalette.ColorRole.Link: ACCENT,
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#9a9a9a"))
    app.setPalette(palette)
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    app.setStyleSheet(STYLESHEET)
