"""The HIOKI Measurement Analyzer window: explorer, plot workspace, properties."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStyle,
    QVBoxLayout,
    QWidget,
)

import latos2
from latos2.export import ExportError, export_analysis
from latos2.figures import draw_coefficients, draw_timeseries, draw_voltage_vs_delta_t
from latos2.hioki_reader import HiokiFormatError, LoggerTrace, read_hioki_xlsx
from latos2.thermovoltage import (
    SIGN_CONVENTION,
    STATEMENT,
    Analysis,
    ChannelRoles,
    CycleBoundaryError,
    CycleWindow,
    analyze,
    event_boundaries,
    windows_from_boundaries,
)
from latos2.ui import style
from latos2.ui.panels import DataExplorer, PlotWorkspace, PropertiesPanel, WorkflowBar
from latos2.units import UnitError, quantization_step, voltage_factor_to_mv

# The assignment in the ionic-liquid campaign's experimental record. Shown
# pre-selected and always changeable; the wiring is never pre-selected.
RECORDED_ROLES = ChannelRoles(hot="CH1-A-3", cold="CH1-A-1", voltage="CH1-A-10")

_FIGURE_NAMES = ("timeseries", "voltage_vs_dT", "coefficients")


def _inside(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


class MainWindow(QMainWindow):
    """Open a workbook, confirm channels and polarity, inspect, export."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Latos V2 — HIOKI Measurement Analyzer")
        self.resize(1440, 900)
        self.settings = QSettings("Latos", "LatosV2")
        self.trace: LoggerTrace | None = None
        self.analysis: Analysis | None = None
        self.windows: tuple[CycleWindow, ...] = ()
        self.view_cycle: int | None = None
        self.view_segment: str | None = None
        self.exported = False

        icons = self.style()
        self.open_action = QAction(icons.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "Open…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.setToolTip("Open a HIOKI workbook (.xlsx)")
        self.open_action.triggered.connect(lambda: self.open_file())
        self.export_action = QAction(
            icons.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Export results…", self
        )
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.setToolTip("Export tables, provenance record and graphs to a new folder")
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(lambda: self.export())
        quit_action = QAction("Exit", self)
        quit_action.triggered.connect(self.close)
        about_action = QAction("About Latos V2", self)
        about_action.triggered.connect(self._about)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)
        self.menuBar().addMenu("&Help").addAction(about_action)
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.export_action)

        self.workflow = WorkflowBar()
        self.explorer = DataExplorer()
        self.workspace = PlotWorkspace()
        self.properties = PropertiesPanel()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.explorer)
        splitter.addWidget(self.workspace)
        splitter.addWidget(self.properties)
        splitter.setChildrenCollapsible(False)
        self.explorer.setMinimumWidth(210)
        self.properties.setMinimumWidth(320)
        self.workspace.setMinimumWidth(480)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 800, 380])
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.workflow)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self.coordinates = QLabel("")
        self.coordinates.setMinimumWidth(190)
        self.statusBar().addPermanentWidget(self.coordinates)
        self.export_link = QLabel("")
        self.export_link.setOpenExternalLinks(False)
        self.export_link.linkActivated.connect(lambda url: QDesktopServices.openUrl(QUrl(url)))
        self.statusBar().addPermanentWidget(self.export_link)

        # The widgets the researcher works with, by the names the tests use.
        p = self.properties
        self.hot_combo, self.cold_combo, self.voltage_combo = p.hot_combo, p.cold_combo, p.voltage_combo
        self.wiring_buttons = p.wiring_buttons
        self.status_label, self.results_table, self.events_table = p.status_label, p.results_table, p.events_table

        for combo in (self.hot_combo, self.cold_combo, self.voltage_combo):
            combo.currentIndexChanged.connect(self._reanalyze)
        # The sign is read from which button is checked, never from a button
        # id: QButtonGroup reports -1 when nothing is checked, which would
        # read as "V+ on the HOT electrode".
        p.wiring_group.buttonToggled.connect(lambda _button, checked: checked and self._reanalyze())
        self.workflow.stepClicked.connect(self._on_step)
        self.explorer.openRequested.connect(lambda: self.open_file())
        self.explorer.selectionChanged.connect(self._set_view)
        self.workspace.viewChanged.connect(lambda cycle: self._set_view(cycle, None))
        self.workspace.tabChanged.connect(lambda _index: self._render_current())
        self.workspace.saveRequested.connect(self._save_figure)
        self.workspace.coordinatesChanged.connect(self.coordinates.setText)

        self._show_status("Open a HIOKI workbook (File ▸ Open…, Ctrl+O) to begin.", style.MUTED)
        self._invalidate()

    # -- actions ------------------------------------------------------------

    def open_file(self, path: Path | None = None) -> bool:
        """Read a workbook and show it. Returns True when it was read."""
        if path is None:
            start = self.settings.value("last_open_dir", "")
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Open HIOKI workbook", str(start), "Excel workbook (*.xlsx)"
            )
            if not chosen:
                return False
            path = Path(chosen)
        try:
            trace = read_hioki_xlsx(path)
        except (HiokiFormatError, OSError) as exc:
            QMessageBox.critical(self, "Cannot read the workbook", str(exc))
            return False
        self.settings.setValue("last_open_dir", str(path.parent))
        self.trace = trace
        self.analysis = None
        self.view_cycle, self.view_segment = None, None
        try:
            self.windows = windows_from_boundaries(
                event_boundaries([e.number for e in trace.events], [e.time_s for e in trace.events])
            )
        except CycleBoundaryError:
            self.windows = ()
        self.explorer.show_trace(trace, self.windows)
        self.properties.show_trace(trace)
        self.properties.set_settings_enabled(True)
        self.workspace.set_cycles(len(self.windows))
        self._fill_channel_combos(trace)
        # The wiring is declared afresh for every recording.
        p = self.properties
        p.wiring_group.setExclusive(False)
        for button in self.wiring_buttons.values():
            button.setChecked(False)
        p.wiring_group.setExclusive(True)
        self.statusBar().showMessage(f"Opened {trace.source.name}", 5000)
        self.export_link.setText("")
        self._reanalyze()
        return True

    def set_wiring(self, sign: int) -> None:
        """Declare the wiring, as clicking its button does."""
        self.wiring_buttons[sign].setChecked(True)

    def wiring(self) -> int | None:
        """The declared wiring sign, or None when none was chosen."""
        for sign, button in self.wiring_buttons.items():
            if button.isChecked():
                return sign
        return None

    def export(self, out_dir: Path | None = None) -> list[Path]:
        """Export the current analysis into a new folder under `out_dir`."""
        if self.analysis is None or self.analysis.reductions is None:
            return []
        if out_dir is None:
            default = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
            start = self.settings.value("last_export_dir", default)
            chosen = QFileDialog.getExistingDirectory(self, "Export into folder", str(start))
            if not chosen:
                return []
            out_dir = Path(chosen)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = Path(out_dir) / f"{self.analysis.trace.source.stem}_latos2_{stamp}"
        try:
            written = export_analysis(self.analysis, target)
        except (ExportError, OSError) as exc:
            QMessageBox.warning(self, "Export refused", str(exc))
            return []
        self.settings.setValue("last_export_dir", str(out_dir))
        self.exported = True
        self.statusBar().showMessage(f"Exported {len(written)} files to {target}")
        self.export_link.setText(f'<a href="{QUrl.fromLocalFile(str(target)).toString()}">Open export folder</a>')
        self._update_workflow()
        return written

    def _save_figure(self) -> None:
        if self.trace is None:
            return
        index = self.workspace.current_index()
        view = "all-cycles" if self.view_cycle is None else f"cycle{self.view_cycle}"
        default_dir = self.settings.value(
            "last_export_dir",
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation),
        )
        name = f"{self.trace.source.stem}_{_FIGURE_NAMES[index]}_{view}.png"
        chosen, chosen_filter = QFileDialog.getSaveFileName(
            self,
            "Save figure",
            str(Path(default_dir) / name),
            "PNG image, 300 dpi (*.png);;SVG vector, editable text (*.svg)",
        )
        if not chosen:
            return
        path = Path(chosen)
        if path.suffix.lower() not in (".png", ".svg"):
            path = path.with_suffix(".svg" if "SVG" in chosen_filter else ".png")
        if _inside(path.parent.resolve(), self.trace.source.parent.resolve()):
            QMessageBox.warning(
                self, "Save refused", "Choose a folder outside the one holding the original recordings."
            )
            return
        svg = path.suffix.lower() == ".svg"
        self.workspace.current_page().figure.savefig(path, dpi=300, metadata={"Date": None} if svg else None)
        self.statusBar().showMessage(f"Saved {path}", 8000)

    def _on_step(self, step: int) -> None:
        if step == 1:
            self.open_file()
        elif step == 2 and self.trace is not None:
            self.properties.focus_settings()
        elif step == 3 and self.analysis is not None:
            self.workspace.tabs.setCurrentIndex(1)
        elif step == 4:
            if self.export_action.isEnabled():
                self.export()
            else:
                self.statusBar().showMessage("Complete steps 1 and 2 before exporting.", 5000)

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About Latos V2",
            f"<b>Latos V2 {latos2.__version__}</b> — HIOKI Measurement Analyzer<br><br>"
            f"{STATEMENT}<br><br>{SIGN_CONVENTION}",
        )

    # -- state --------------------------------------------------------------

    def _fill_channel_combos(self, trace: LoggerTrace) -> None:
        defaults = {
            self.hot_combo: RECORDED_ROLES.hot,
            self.cold_combo: RECORDED_ROLES.cold,
            self.voltage_combo: RECORDED_ROLES.voltage,
        }
        for combo, default in defaults.items():
            combo.blockSignals(True)
            combo.clear()
            for channel in trace.channels:
                combo.addItem(f"{channel.name}  [{channel.unit}]", channel.name)
            combo.setCurrentIndex(combo.findData(default))
            combo.blockSignals(False)

    def _roles(self) -> ChannelRoles | None:
        names = [c.currentData() for c in (self.hot_combo, self.cold_combo, self.voltage_combo)]
        if any(name is None for name in names):
            return None
        return ChannelRoles(hot=names[0], cold=names[1], voltage=names[2])

    def _show_status(self, text: str, color: str) -> None:
        """Guidance, warnings and errors stay in the panel; a clean result goes to the status bar."""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setVisible(color != style.OK)
        if color == style.OK:
            self.statusBar().showMessage(text, 8000)

    def _reanalyze(self) -> None:
        self.analysis = None
        self.exported = False
        self.export_action.setEnabled(False)
        self.properties.clear_results()
        trace = self.trace
        if trace is None:
            return
        roles = self._roles()
        if roles is None:
            self._show_status("Assign the hot, cold and voltage channels.", style.ERROR)
            self._invalidate()
            return
        try:
            voltage = trace.channel(roles.voltage)
            factor = voltage_factor_to_mv(voltage.unit)
            step_uv = quantization_step(voltage.values) * factor * 1000.0
            self.properties.show_conversion(
                f"Voltage [{voltage.unit}] × {factor:g} → mV; reading step {step_uv:.3g} µV<br>"
                "ΔT = T<sub>hot</sub> − T<sub>cold</sub> in K; coefficients in mV/K"
            )
        except (UnitError, KeyError) as exc:
            self.properties.show_conversion(str(exc))

        sign = self.wiring()
        if sign is None:
            self._show_status(
                "Choose the voltage lead polarity in Measurement Settings to compute the coefficients.",
                "#8a5a00",
            )
            self._invalidate()
            return
        try:
            analysis = analyze(trace, roles, sign)
        except (ValueError, KeyError) as exc:
            # UnitError and CycleBoundaryError are ValueErrors.
            kind = "Event marks" if isinstance(exc, CycleBoundaryError) else "Cannot analyse"
            self._show_status(f"{kind}: {exc}", style.ERROR)
            self._invalidate()
            return

        self.analysis = analysis
        rejected = {c.window.index for c in analysis.cycles if c.problems}
        self.explorer.mark_rejected(rejected)
        self.properties.show_results(analysis)
        self.properties.show_selection(analysis, self.view_cycle, self.view_segment)
        if analysis.reductions is None:
            self._show_status("No cycle passed the checks:\n• " + "\n• ".join(analysis.problems), style.ERROR)
        elif analysis.problems:
            self._show_status(
                "Some cycles were rejected and left out of the means:\n• " + "\n• ".join(analysis.problems),
                "#8a5a00",
            )
        else:
            self._show_status(f"{len(analysis.cycles)} cycles analysed; all passed the checks.", style.OK)
        self.export_action.setEnabled(analysis.reductions is not None)
        self._invalidate()

    def _set_view(self, cycle: int | None, segment: str | None) -> None:
        self.view_cycle, self.view_segment = cycle, segment
        self.workspace.set_view(cycle)
        self.explorer.select(cycle, segment)
        self.properties.show_selection(self.analysis, cycle, segment)
        self._invalidate()

    def _invalidate(self) -> None:
        """Mark every figure out of date, redraw the visible one, update the steps."""
        for page in self.workspace.pages:
            page.dirty = True
        self._render_current()
        self._update_workflow()

    def _render_current(self) -> None:
        index = self.workspace.current_index()
        page = self.workspace.pages[index]
        if not page.dirty:
            return
        roles = self._roles()
        try:
            if self.trace is None:
                page.placeholder("Open a HIOKI workbook to begin\n(File ▸ Open…, Ctrl+O)")
            elif roles is None:
                page.placeholder("Assign the hot, cold and voltage channels.")
            elif index == 0:
                draw_timeseries(page.figure, self.trace, roles, self.view_cycle, self.view_segment)
            elif self.analysis is None:
                page.placeholder(self.status_label.text())
            elif index == 1:
                draw_voltage_vs_delta_t(page.figure, self.analysis, self.view_cycle, self.view_segment)
            else:
                draw_coefficients(page.figure, self.analysis, self.view_cycle)
        except (UnitError, KeyError, ValueError) as exc:
            page.placeholder(str(exc))
        page.refresh()
        page.dirty = False

    def _update_workflow(self) -> None:
        if self.trace is None:
            states = ["current", "pending", "pending", "pending"]
        elif self.analysis is None or self.analysis.reductions is None:
            states = ["done", "current", "pending", "pending"]
        elif not self.exported:
            states = ["done", "done", "current", "pending"]
        else:
            states = ["done", "done", "done", "done"]
        self.workflow.set_states(states)
