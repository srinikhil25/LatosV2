"""The analyzer's panels: workflow strip, Data Explorer, plot workspace, properties."""

from __future__ import annotations

import math

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPixmap, QTextOption
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from latos2.hioki_reader import LoggerTrace
from latos2.thermovoltage import KINDS, UNIT, WIRING, Analysis, CycleBoundaryError, CycleWindow, trace_windows
from latos2.ui import style

WARNING_TEXT = (
    "Reported coefficients are apparent dynamic values, not validated steady-state Seebeck "
    "coefficients."
)
_KIND_NAMES = {"combined": "Whole cycle", "heating": "Heating only", "cooling": "Cooling only"}


def _styled(widget: QWidget, name: str) -> QWidget:
    """Give a plain QWidget an object name its stylesheet background applies to."""
    widget.setObjectName(name)
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    return widget


def _label(text: str = "", name: str | None = None, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    if name:
        label.setObjectName(name)
    label.setWordWrap(wrap)
    return label


def _swatch(color: str) -> QIcon:
    pixmap = QPixmap(10, 10)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


def _fit_table_height(table: QTableWidget) -> None:
    """Size a short table to show every row without its own scroll bar."""
    height = table.horizontalHeader().sizeHint().height() + 2 * table.frameWidth()
    height += sum(table.rowHeight(r) for r in range(table.rowCount()))
    table.setFixedHeight(height)


def _fmt(value: float | None, digits: int = 4, signed: bool = True) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


class Collapsible(QWidget):
    """A titled section whose body can be shown or hidden."""

    def __init__(self, title: str, body: QWidget, expanded: bool = False) -> None:
        super().__init__()
        self.body = body
        self.button = QToolButton()
        self.button.setObjectName("disclosure")
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.toggled.connect(self._toggle)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.button)
        layout.addWidget(body)
        self.button.setChecked(expanded)
        self._toggle(expanded)

    def _toggle(self, expanded: bool) -> None:
        self.button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.body.setVisible(expanded)


class WorkflowBar(QWidget):
    """The four steps of an analysis, each showing whether it is done."""

    stepClicked = Signal(int)
    STEPS = (
        "Open workbook",
        "Confirm channels and polarity",
        "Inspect cycles and graphs",
        "Export results",
    )

    def __init__(self) -> None:
        super().__init__()
        _styled(self, "workflow")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(2)
        self.buttons: list[QToolButton] = []
        for index, text in enumerate(self.STEPS, start=1):
            button = QToolButton()
            button.setObjectName("step")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, i=index: self.stepClicked.emit(i))
            self.buttons.append(button)
            layout.addWidget(button)
            if index < len(self.STEPS):
                layout.addWidget(_label("›", "muted"))
        layout.addStretch(1)
        self.set_states(["current", "pending", "pending", "pending"])

    def set_states(self, states: list[str]) -> None:
        for index, (button, state) in enumerate(zip(self.buttons, states, strict=True), start=1):
            mark = "✓" if state == "done" else str(index)
            button.setText(f"{mark}  {self.STEPS[index - 1]}")
            button.setProperty("state", state)
            button.style().unpolish(button)
            button.style().polish(button)


class DataExplorer(QWidget):
    """The open workbook, its measurement, its cycles and their branches."""

    selectionChanged = Signal(object, object)  # cycle index or None, "heating"/"cooling"/None
    openRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        _styled(self, "panel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(_label("Data Explorer", "panelTitle"))

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.empty = QWidget()
        empty_layout = QVBoxLayout(self.empty)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_layout.addWidget(_label("No workbook open.", "muted"))
        open_button = QPushButton("Open HIOKI workbook…")
        open_button.clicked.connect(self.openRequested.emit)
        empty_layout.addWidget(open_button)
        empty_layout.addStretch(1)
        layout.addWidget(self.empty)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Item", "Time"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setStretchLastSection(False)
        self.tree.setUniformRowHeights(True)
        self.tree.itemSelectionChanged.connect(self._on_selection)
        self.tree.hide()
        layout.addWidget(self.tree, 1)

        # Plain text that wraps anywhere, so a path or SHA-256 is never clipped
        # and copies exactly, with no inserted break characters.
        self.info_text = QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFrameShape(QFrame.Shape.NoFrame)
        self.info_text.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.info_text.setStyleSheet(f"background: {style.PANEL};")
        self.info_text.setFixedHeight(self.fontMetrics().lineSpacing() * 13)
        self.info = Collapsible("File information", self.info_text)
        self.info.hide()
        layout.addWidget(self.info)
        outer.addWidget(body, 1)

    def show_trace(self, trace: LoggerTrace, windows: tuple[CycleWindow, ...]) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        root = QTreeWidgetItem([trace.source.name, ""])
        root.setIcon(0, self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        bold = QFont(root.font(0))
        bold.setBold(True)
        root.setFont(0, bold)
        root.setFlags(Qt.ItemFlag.ItemIsEnabled)
        root.setToolTip(0, str(trace.source))
        self.tree.addTopLevelItem(root)
        root.setFirstColumnSpanned(True)

        measurement = QTreeWidgetItem(root, [f"Measurement · sheet {trace.sheet}", ""])
        measurement.setFlags(Qt.ItemFlag.ItemIsEnabled)
        measurement.setToolTip(
            0, f"Recording started {trace.timestamps[0]}\n{trace.time_s.size} samples"
        )
        measurement.setFirstColumnSpanned(True)

        self._items: dict[tuple[int | None, str | None], QTreeWidgetItem] = {}
        everything = QTreeWidgetItem(
            measurement, ["All cycles", f"{trace.time_s[0]:.0f}–{trace.time_s[-1]:.0f} s"]
        )
        self._register(everything, None, None)
        for w in windows:
            cycle = QTreeWidgetItem(measurement, [f"Cycle {w.index}", f"{w.on_s:.1f}–{w.end_s:.1f} s"])
            self._register(cycle, w.index, None)
            for segment, duration, color in (
                ("heating", w.heating_s, style.HEATING),
                ("cooling", w.cooling_s, style.COOLING),
            ):
                item = QTreeWidgetItem(cycle, [segment.capitalize(), f"{duration:.1f} s"])
                item.setIcon(0, _swatch(color))
                self._register(item, w.index, segment)
        if not windows:
            missing = QTreeWidgetItem(measurement, ["No complete cycles in the event marks", ""])
            missing.setFlags(Qt.ItemFlag.ItemIsEnabled)
            missing.setForeground(0, QColor(style.ERROR))
            missing.setFirstColumnSpanned(True)
        self.tree.expandAll()
        self.tree.setCurrentItem(everything)
        self.tree.blockSignals(False)

        headers = ", ".join(["Time", *(c.header for c in trace.channels)] + (["Event"] if trace.has_event_column else []))
        self.info_text.setPlainText(
            f"Path: {trace.source}\n"
            f"SHA-256: {trace.sha256}\n"
            f"Size: {trace.size_bytes:,} bytes\n"
            f"Sheet: {trace.sheet}\n"
            f"Preamble: {' | '.join(trace.preamble) or '—'}\n"
            f"Headers: {headers}\n"
            f"Ignored columns: {', '.join(trace.ignored_columns) or 'none'}"
        )
        self.empty.hide()
        self.tree.show()
        self.info.show()

    def mark_cycles(self, states: dict[int, str]) -> None:
        """Label cycles left out of the specimen value: "excluded" or "rejected"."""
        for (cycle, segment), item in getattr(self, "_items", {}).items():
            if cycle is not None and segment is None:
                state = states.get(cycle)
                item.setText(0, f"Cycle {cycle}" + (f"  ({state})" if state else ""))
                color = {"rejected": style.ERROR, "excluded": style.MUTED}.get(state, style.TEXT)
                item.setForeground(0, QColor(color))
                font = QFont(item.font(0))
                font.setItalic(state == "excluded")
                item.setFont(0, font)

    def select(self, cycle: int | None, segment: str | None) -> None:
        item = getattr(self, "_items", {}).get((cycle, segment))
        if item is not None and item is not self.tree.currentItem():
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(item)
            self.tree.blockSignals(False)

    def _register(self, item: QTreeWidgetItem, cycle: int | None, segment: str | None) -> None:
        item.setData(0, Qt.ItemDataRole.UserRole, (cycle, segment))
        self._items[(cycle, segment)] = item

    def _on_selection(self) -> None:
        item = self.tree.currentItem()
        target = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if target is not None:
            self.selectionChanged.emit(*target)


class PlotPage(QWidget):
    """One embedded figure. Its navigation toolbar is kept hidden and driven
    from the workspace's own compact toolbar."""

    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.nav = NavigationToolbar2QT(self.canvas, self, coordinates=False)
        self.nav.hide()
        self.dirty = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def placeholder(self, text: str) -> None:
        self.figure.clear()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", color=style.MUTED, fontsize=11, wrap=True)

    def refresh(self) -> None:
        self.canvas.draw_idle()
        # A new drawing starts a new zoom history, so "Reset view" returns to it.
        self.nav.update()


class PlotWorkspace(QWidget):
    """Document tabs over one compact toolbar and the three figures."""

    viewChanged = Signal(object)  # cycle index or None
    tabChanged = Signal(int)
    saveRequested = Signal()
    coordinatesChanged = Signal(str)
    TABS = (
        "Temperature and Voltage vs Time",
        "Voltage vs Temperature Difference",
        "Coefficient Comparison",
    )

    def __init__(self) -> None:
        super().__init__()
        _styled(self, "workspace")
        self.tabs = QTabBar()
        self.tabs.setObjectName("documentTabs")
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        for title in self.TABS:
            self.tabs.addTab(title)

        toolbar = _styled(QWidget(), "plotToolbar")
        bar = QHBoxLayout(toolbar)
        bar.setContentsMargins(6, 3, 6, 3)
        bar.setSpacing(2)
        bar.addWidget(_label("View:", "muted"))
        self.view_group = QButtonGroup(self)
        self.view_group.setExclusive(True)
        self.view_group.idClicked.connect(lambda i: self.viewChanged.emit(None if i == 0 else i))
        self.view_box = QHBoxLayout()
        self.view_box.setSpacing(2)
        bar.addLayout(self.view_box)
        bar.addStretch(1)
        self.pan_button = self._tool(bar, "Pan", "Drag to move the plot (toggle)", checkable=True)
        self.zoom_button = self._tool(bar, "Zoom", "Drag a rectangle to zoom (toggle)", checkable=True)
        self.reset_button = self._tool(bar, "Reset view", "Return to the full view of this graph")
        bar.addSpacing(6)
        self.save_button = self._tool(bar, "Save figure…", "Save this graph as PNG (300 dpi) or SVG")
        self.pan_button.clicked.connect(lambda: self._navigate("pan"))
        self.zoom_button.clicked.connect(lambda: self._navigate("zoom"))
        self.reset_button.clicked.connect(lambda: self.current_page().nav.home())
        self.save_button.clicked.connect(self.saveRequested.emit)

        self.stack = QStackedWidget()
        self.pages = [PlotPage() for _ in self.TABS]
        for page in self.pages:
            self.stack.addWidget(page)
            page.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.tabs.currentChanged.connect(self._on_tab)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tabs)
        layout.addWidget(toolbar)
        layout.addWidget(self.stack, 1)
        self.set_cycles(0)

    @staticmethod
    def _tool(bar: QHBoxLayout, text: str, tip: str, checkable: bool = False) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.setCheckable(checkable)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        bar.addWidget(button)
        return button

    def set_cycles(self, count: int) -> None:
        """One view button for all cycles and one per cycle."""
        for button in self.view_group.buttons():
            self.view_group.removeButton(button)
            self.view_box.removeWidget(button)
            button.setParent(None)  # off screen now, not at the next event-loop pass
            button.deleteLater()
        for index in range(count + 1):
            button = QToolButton()
            button.setText("All cycles" if index == 0 else f"Cycle {index}")
            button.setCheckable(True)
            button.setEnabled(count > 0)
            # Never squeezed: a view button reading "…" says nothing.
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self.view_group.addButton(button, index)
            self.view_box.addWidget(button)
        self.view_group.button(0).setChecked(True)

    def set_view(self, cycle: int | None) -> None:
        button = self.view_group.button(0 if cycle is None else cycle)
        if button is not None:
            button.setChecked(True)

    def current_index(self) -> int:
        return self.tabs.currentIndex()

    def current_page(self) -> PlotPage:
        return self.pages[self.current_index()]

    def _navigate(self, mode: str) -> None:
        nav = self.current_page().nav
        if mode == "pan":
            nav.pan()
        else:
            nav.zoom()
        self._sync_mode()

    def _sync_mode(self) -> None:
        mode = self.current_page().nav.mode.name
        self.pan_button.setChecked(mode == "PAN")
        self.zoom_button.setChecked(mode == "ZOOM")

    def _on_tab(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self._sync_mode()
        self.coordinatesChanged.emit("")
        self.tabChanged.emit(index)

    def _on_motion(self, event) -> None:
        if event.inaxes is not None and event.xdata is not None and event.ydata is not None:
            self.coordinatesChanged.emit(f"x = {event.xdata:.4g}    y = {event.ydata:.4g}")
        else:
            self.coordinatesChanged.emit("")


class PropertiesPanel(QWidget):
    """Settings, results and measurement information for the current selection."""

    def __init__(self) -> None:
        super().__init__()
        _styled(self, "panel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(_label("Analysis and Properties", "panelTitle"))

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(10, 2, 10, 10)
        layout.setSpacing(4)

        # Measurement settings: what the researcher confirms.
        self.settings_title = _label("Measurement Settings", "sectionTitle")
        layout.addWidget(self.settings_title)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setVerticalSpacing(4)
        self.hot_combo = QComboBox()
        self.cold_combo = QComboBox()
        self.voltage_combo = QComboBox()
        form.addRow("Hot temperature", self.hot_combo)
        form.addRow("Cold temperature", self.cold_combo)
        form.addRow("Voltage", self.voltage_combo)
        polarity = QWidget()
        polarity_layout = QVBoxLayout(polarity)
        polarity_layout.setContentsMargins(0, 2, 0, 0)
        polarity_layout.setSpacing(2)
        self.wiring_group = QButtonGroup(self)
        self.wiring_buttons: dict[int, QRadioButton] = {}
        for sign in (+1, -1):
            button = QRadioButton(f"{WIRING[sign]}  (σ = {sign:+d})")
            self.wiring_group.addButton(button)
            self.wiring_buttons[sign] = button
            polarity_layout.addWidget(button)
        hint = _label("Set from the lab record; never inferred.", "muted")
        hint.setToolTip(
            "The polarity is declared for every recording. It is never inferred from the file "
            "name or from the data."
        )
        polarity_layout.addWidget(hint)
        form.addRow("Lead polarity", polarity)
        layout.addLayout(form)

        # Analysis results for the current selection.
        layout.addWidget(_label("Analysis Results", "sectionTitle"))
        self.status_label = _label("", wrap=True)
        layout.addWidget(self.status_label)
        self.selection_label = _label("", wrap=True)
        bold = QFont(self.selection_label.font())
        bold.setBold(True)
        self.selection_label.setFont(bold)
        layout.addWidget(self.selection_label)
        # Shown only while one cycle is selected.
        self.exclusion_box = QWidget()
        exclusion = QHBoxLayout(self.exclusion_box)
        exclusion.setContentsMargins(0, 0, 0, 0)
        self.exclude_check = QCheckBox("Exclude from specimen value")
        self.exclude_check.setToolTip(
            "Leave this cycle out of the mean, SD and SEM, for example an initial test cycle "
            "recorded in the lab notebook. The cycle stays visible and in the exports."
        )
        self.exclude_reason = QLineEdit()
        self.exclude_reason.setPlaceholderText("Reason, e.g. Initial test cycle")
        exclusion.addWidget(self.exclude_check)
        exclusion.addWidget(self.exclude_reason, 1)
        self.exclusion_box.hide()
        layout.addWidget(self.exclusion_box)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        value_font = QFont(self.font())
        value_font.setPointSizeF(10.5)
        self.headline: dict[str, tuple[QLabel, QLabel, QLabel]] = {}
        for row, kind in enumerate(KINDS):
            name = _label(_KIND_NAMES[kind])
            value = _label("—")
            value.setFont(value_font)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            extra = _label("", "muted")
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            grid.addWidget(extra, row, 2)
            self.headline[kind] = (name, value, extra)
        grid.setColumnStretch(2, 1)
        layout.addLayout(grid)
        layout.addSpacing(2)
        self.table_caption = _label(f"Per cycle ({UNIT})", "muted", True)
        layout.addWidget(self.table_caption)
        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels([_KIND_NAMES[k] for k in KINDS])
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.results_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.results_table.verticalHeader().setDefaultSectionSize(22)
        self.results_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.results_table)
        self.repeat_note = _label("", "muted", True)
        layout.addWidget(self.repeat_note)

        # Measurement information: what the file says.
        layout.addWidget(_label("Measurement Information", "sectionTitle"))
        info = QFormLayout()
        info.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        info.setHorizontalSpacing(12)
        self.info_values: dict[str, QLabel] = {}
        for key in ("Samples", "Sampling interval", "Duration", "Started", "Original units", "Conversion"):
            value = _label("—", wrap=True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            info.addRow(key, value)
            self.info_values[key] = value
        layout.addLayout(info)
        layout.addWidget(_label("Event marks"))
        self.events_table = QTableWidget(0, 4)
        self.events_table.setHorizontalHeaderLabels(["Mark", "Time (s)", "Event", "Phase (s)"])
        self.events_table.verticalHeader().setVisible(False)
        self.events_table.verticalHeader().setDefaultSectionSize(21)
        self.events_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.events_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.events_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header = self.events_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.events_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.events_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.events_table)
        self.events_note = _label("", "muted", True)
        layout.addWidget(self.events_note)
        layout.addStretch(1)

        self.scroll = QScrollArea()
        self.scroll.setWidget(body)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll, 1)

        warning = _label(WARNING_TEXT, "warning", True)
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(8, 4, 8, 6)
        holder_layout.addWidget(warning)
        outer.addWidget(holder)
        self.set_settings_enabled(False)
        self.clear_results()

    def set_settings_enabled(self, enabled: bool) -> None:
        for widget in (self.hot_combo, self.cold_combo, self.voltage_combo, *self.wiring_buttons.values()):
            widget.setEnabled(enabled)

    def focus_settings(self) -> None:
        self.scroll.ensureWidgetVisible(self.settings_title)
        unchecked = [b for b in self.wiring_buttons.values() if not b.isChecked()]
        (unchecked[0] if len(unchecked) == 2 else self.hot_combo).setFocus()

    def show_trace(self, trace: LoggerTrace) -> None:
        values = self.info_values
        values["Samples"].setText(f"{trace.time_s.size:,}")
        values["Sampling interval"].setText(f"{trace.sample_interval_s:.3g} s")
        values["Duration"].setText(f"{trace.time_s[-1]:.1f} s")
        values["Started"].setText(trace.timestamps[0])
        values["Original units"].setText("<br>".join(f"{c.name}: {c.unit}" for c in trace.channels))
        values["Conversion"].setText("—")

        events = trace.events
        n = len(events)
        recording_end = float(trace.time_s[-1])
        self.events_table.setRowCount(n)
        for row, e in enumerate(events):
            if e.number == n and n % 2 == 1:
                meaning = "End of run"
            elif e.number % 2 == 1:
                meaning = f"Heating on · cycle {(e.number + 1) // 2}"
            else:
                meaning = f"Heating off · cycle {e.number // 2}"
            if row + 1 < n:
                phase = f"{events[row + 1].time_s - e.time_s:.1f}"
            elif n % 2 == 0:
                # The final cooling period runs to the last recorded sample.
                phase = f"{recording_end - e.time_s:.1f}"
            else:
                phase = "—"
            for col, text in enumerate([str(e.number), f"{e.time_s:.1f}", meaning, phase]):
                item = QTableWidgetItem(text)
                if col != 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item.setToolTip(f"Sheet row {e.sheet_row}, logged {e.timestamp}")
                self.events_table.setItem(row, col, item)
        _fit_table_height(self.events_table)
        try:
            windows, end_of_run = trace_windows(trace)
        except CycleBoundaryError as exc:
            self.events_note.setText(f"{n} marks: {exc}.")
        else:
            self.events_note.setText(
                f"{n} marks → {len(windows)} heating/cooling cycles; the run ends at "
                f"{end_of_run}. Phase = time to the next mark or to the end of the run."
            )

    def show_conversion(self, text: str) -> None:
        self.info_values["Conversion"].setText(text)

    def clear_results(self) -> None:
        self.selection_label.setText("")
        self.selection_label.hide()
        self.exclusion_box.hide()
        for name, value, extra in self.headline.values():
            value.setText("—")
            extra.setText("")
            name.setFont(self.font())
        self.results_table.setRowCount(0)
        _fit_table_height(self.results_table)
        self.repeat_note.setText("")

    def show_results(self, a: Analysis) -> None:
        rows: list[tuple[str, list[str]]] = []
        for c in a.cycles:
            index = c.window.index
            mark = " (excl.)" if index in a.excluded else (" ✕" if c.problems else "")
            rows.append((f"Cycle {index}{mark}", [_fmt(c.fit(k).slope) for k in KINDS]))
        if a.reductions is not None:
            r = a.reductions
            rows.append(("Mean", [_fmt(r[k].mean) for k in KINDS]))
            rows.append(("SD", [_fmt(r[k].sd, signed=False) for k in KINDS]))
            rows.append(("SEM", [_fmt(r[k].sem, signed=False) for k in KINDS]))
        table = self.results_table
        table.setRowCount(len(rows))
        table.setVerticalHeaderLabels([label for label, _ in rows])
        for row, (_, cells) in enumerate(rows):
            left_out = row < len(a.cycles) and a.cycles[row].window.index not in a.accepted
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if left_out:
                    item.setForeground(QColor(style.MUTED))
                table.setItem(row, col, item)
        _fit_table_height(table)
        k = len(a.accepted)
        marks = []
        if a.excluded:
            marks.append("excl. = excluded")
        if any(c.problems for c in a.cycles):
            marks.append("✕ = rejected")
        self.table_caption.setText(f"Per cycle ({UNIT})" + (f"; {', '.join(marks)}" if marks else ""))
        excluded = "".join(f" Cycle {i} excluded: {reason}." for i, reason in sorted(a.excluded.items()))
        self.repeat_note.setText(
            f"Mean, SD and SEM over {k} accepted cycle{'' if k == 1 else 's'}: repeated measurements "
            "of one specimen (one mounting). They describe within-specimen repeatability only, not "
            f"variation between specimens.{excluded}"
        )

    def show_selection(self, a: Analysis | None, cycle: int | None, segment: str | None) -> None:
        if a is None:
            return
        self.selection_label.show()
        for kind, (name, value, extra) in self.headline.items():
            emphasised = QFont(self.font())
            emphasised.setBold(segment == kind)
            name.setFont(emphasised)
            if cycle is None:
                r = a.reductions[kind] if a.reductions else None
                value.setText(
                    f"{_fmt(r.mean)} ± {_fmt(r.sem, signed=False)} {UNIT}" if r and r.sem is not None
                    else (f"{_fmt(r.mean)} {UNIT}" if r else "—")
                )
                extra.setText("mean ± SEM" if r and r.sem is not None else "")
            else:
                fit = a.cycles[cycle - 1].fit(kind)
                value.setText(f"{_fmt(fit.slope)} {UNIT}")
                extra.setText(f"R² {fit.r_squared:.4f}" if math.isfinite(fit.r_squared) else "")
        self.exclusion_box.setVisible(cycle is not None)
        if cycle is None:
            k = len(a.accepted)
            self.selection_label.setText(f"All cycles · {k} accepted cycle{'s' if k != 1 else ''} of one specimen")
        else:
            c = a.cycles[cycle - 1]
            w = c.window
            part = f" · {segment} branch" if segment else ""
            if cycle in a.excluded:
                state = f" · excluded: {a.excluded[cycle]}"
            else:
                state = "" if not c.problems else " · rejected: " + "; ".join(c.problems)
            self.selection_label.setText(
                f"Cycle {cycle}{part} · heating {w.heating_s:.1f} s, cooling {w.cooling_s:.1f} s{state}"
            )
            for widget in (self.exclude_check, self.exclude_reason):
                widget.blockSignals(True)
            self.exclude_check.setChecked(cycle in a.excluded)
            self.exclude_reason.setText(a.excluded.get(cycle, ""))
            for widget in (self.exclude_check, self.exclude_reason):
                widget.blockSignals(False)
        table = self.results_table
        for row in range(table.rowCount()):
            selected = cycle is not None and row == cycle - 1
            for col in range(table.columnCount()):
                item = table.item(row, col)
                if item is not None:
                    item.setBackground(QColor(style.SELECTION) if selected else QColor(0, 0, 0, 0))
