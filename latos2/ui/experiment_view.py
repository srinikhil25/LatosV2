"""The Experiment workspace: definition, specimens and their measurement sessions."""

from __future__ import annotations

import html
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from latos2 import experiment as ex
from latos2.export import analysis_record
from latos2.hioki_reader import file_sha256
from latos2.project_io import SUFFIX, ProjectError, load_experiment, save_experiment
from latos2.thermovoltage import UNIT, WIRING, Analysis
from latos2.ui import style
from latos2.ui.panels import WARNING_TEXT, _fit_table_height, _fmt, _label, _styled

FILE_FILTER = f"Latos experiment (*{SUFFIX})"
NOT_RECORDED = "Not recorded"
_KIND_NAMES = {"continuous": "Continuous", "discrete": "Discrete", "categorical": "Categorical"}
_OBJECTIVE_NAMES = {"maximize": "Maximize", "minimize": "Minimize", "target": "Reach a target"}
_SOURCE_STATE = {"ok": "file unchanged", "changed": "file CHANGED since analysis", "missing": "file NOT FOUND"}
_CAPTION = (
    "Each output value is the specimen's primary session: the mean over its accepted cycles "
    "± within-specimen SEM (number of cycles). Supplementary sessions, such as reversed leads, "
    "are kept for comparison and never averaged in. Mass fractions x_i = m_i / Σ m_j are "
    "computed from the actual masses."
)
_NUMERIC = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter


def _number(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"“{text}” is not a number") from None


def _plain(value: float | None) -> str:
    """A number as the researcher typed it, without rounding."""
    return "" if value is None else f"{value:.10g}"


def _item(text: str, editable: bool = False, tip: str = "", numeric: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
    if editable:
        flags |= Qt.ItemFlag.ItemIsEditable
    item.setFlags(flags)
    if tip:
        item.setToolTip(tip)
    if numeric:
        item.setTextAlignment(_NUMERIC)
    return item


def _muted(item: QTableWidgetItem, italic: bool = True) -> QTableWidgetItem:
    item.setForeground(QColor(style.MUTED))
    font = QFont(item.font())
    font.setItalic(italic)
    item.setFont(font)
    return item


def _value_and_sem(m: ex.Measurement) -> str:
    """Analyzer values to the 0.1 µV/K it reports; hand-entered values as entered."""
    if m.method == ex.MANUAL:
        text = f"{m.value:.6g}"
        return text + (f" ± {m.sem:.3g}" if m.sem is not None else "")
    text = _fmt(m.value)
    return text + (f" ± {_fmt(m.sem, signed=False)}" if m.sem is not None else "")


def _value_text(m: ex.Measurement | None) -> str:
    if m is None or m.value is None:
        return "—"
    return _value_and_sem(m) + (f" ({m.n})" if m.n else "")


def _section(layout: QVBoxLayout, title: str, hint: str = "") -> QLabel:
    heading = _label(title, "sectionTitle")
    layout.addWidget(heading)
    if hint:
        layout.addWidget(_label(hint, "muted", True))
    return heading


def _table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(24)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    return table


def _buttons(*specs: tuple[str, Callable[[], object]]) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 2, 0, 4)
    for text, slot in specs:
        button = QPushButton(text)
        button.clicked.connect(lambda _checked=False, s=slot: s())
        row.addWidget(button)
    row.addStretch(1)
    return row


def _unique(prefix: str, taken: list[str], joiner: str = " ") -> str:
    n = 1
    while f"{prefix}{joiner}{n}" in taken:
        n += 1
    return f"{prefix}{joiner}{n}"


class AttachDialog(QDialog):
    """Confirm which specimen, output and session an analysis belongs to."""

    NEW = "\x00new"

    def __init__(self, parent: QWidget, exp: ex.Experiment, analysis: Analysis, selected_id: str | None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Attach HIOKI analysis")
        self.setMinimumWidth(480)
        self.exp = exp
        a = analysis
        layout = QVBoxLayout(self)
        layout.addWidget(_label("Analysis in the HIOKI Analyzer", "sectionTitle"))
        info = QFormLayout()
        excluded = ", ".join(f"cycle {i} ({r})" for i, r in sorted(a.excluded.items())) or "none"
        whole = a.reductions["combined"]
        for key, value in (
            ("File", a.trace.source.name),
            ("Lead polarity", f"{WIRING[a.wiring_sign]} (σ = {a.wiring_sign:+d})"),
            ("Cycles in value", ", ".join(map(str, a.accepted))),
            ("Excluded", excluded),
            ("End of run", a.end_of_run),
            ("Whole cycle", f"{_fmt(whole.mean)} ± {_fmt(whole.sem, signed=False)} {UNIT} (mean ± SEM)"),
        ):
            label = _label(value, wrap=True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            info.addRow(key, label)
        layout.addLayout(info)

        layout.addWidget(_label("Attach to", "sectionTitle"))
        form = QFormLayout()
        self.specimen_combo = QComboBox()
        for s in exp.specimens:
            self.specimen_combo.addItem(f"{s.id}   {s.label}".rstrip(), s.id)
        self.specimen_combo.addItem("New specimen…", self.NEW)
        index = self.specimen_combo.findData(selected_id) if selected_id else -1
        self.specimen_combo.setCurrentIndex(index if index >= 0 else self.specimen_combo.count() - 1)
        self.new_id = QLineEdit(a.trace.source.stem)
        self.output_combo = QComboBox()
        for o in exp.outputs:
            if o.method.startswith("hioki:"):
                self.output_combo.addItem(f"{o.name}  ({ex.METHOD_LABELS[o.method]})", o.name)
        self.label_edit = QLineEdit(WIRING[a.wiring_sign])
        self.role_combo = QComboBox()
        self.role_combo.addItem("Primary: the specimen's response", "primary")
        self.role_combo.addItem("Supplementary: kept for comparison only", "supplementary")
        form.addRow("Specimen", self.specimen_combo)
        self.new_id_label = QLabel("New specimen ID")
        form.addRow(self.new_id_label, self.new_id)
        form.addRow("Output", self.output_combo)
        form.addRow("Session name", self.label_edit)
        form.addRow("Role", self.role_combo)
        layout.addLayout(form)
        layout.addWidget(
            _label(
                "One primary session per specimen and output supplies the response; sessions are "
                "never averaged together.",
                "muted",
                True,
            )
        )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Attach")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.specimen_combo.currentIndexChanged.connect(self._update)
        self.output_combo.currentIndexChanged.connect(self._update)
        self._update()

    def _update(self) -> None:
        new = self.specimen_combo.currentData() == self.NEW
        self.new_id.setVisible(new)
        self.new_id_label.setVisible(new)
        has_primary = False
        if not new:
            specimen = self.exp.specimen(self.specimen_combo.currentData())
            has_primary = ex.primary_measurement(self.exp, specimen, self.output_combo.currentData()) is not None
        self.role_combo.setCurrentIndex(1 if has_primary else 0)

    def choice(self) -> tuple[str, bool, str, str, str]:
        """(specimen ID, whether to create it, output, session name, role)."""
        data = self.specimen_combo.currentData()
        create = data == self.NEW
        specimen_id = self.new_id.text().strip() if create else data
        return specimen_id, create, self.output_combo.currentData(), self.label_edit.text(), self.role_combo.currentData()


class ValueDialog(QDialog):
    """Enter a value measured outside Latos, for an output entered by hand."""

    def __init__(self, parent: QWidget, exp: ex.Experiment, specimen: ex.Specimen) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Enter value for specimen {specimen.id}")
        self.setMinimumWidth(420)
        self.exp, self.specimen = exp, specimen
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.output_combo = QComboBox()
        for o in exp.outputs:
            if o.method == ex.MANUAL:
                self.output_combo.addItem(f"{o.name} ({o.unit})" if o.unit else o.name, o.name)
        self.value_edit = QLineEdit()
        self.sem_edit = QLineEdit()
        self.sem_edit.setPlaceholderText("optional")
        self.n_edit = QLineEdit()
        self.n_edit.setPlaceholderText("optional")
        self.label_edit = QLineEdit("Manual entry")
        self.role_combo = QComboBox()
        self.role_combo.addItem("Primary: the specimen's response", "primary")
        self.role_combo.addItem("Supplementary: kept for comparison only", "supplementary")
        self.note_edit = QLineEdit()
        form.addRow("Output", self.output_combo)
        form.addRow("Value", self.value_edit)
        form.addRow("Standard error", self.sem_edit)
        form.addRow("Repeats (n)", self.n_edit)
        form.addRow("Session name", self.label_edit)
        form.addRow("Role", self.role_combo)
        form.addRow("Note", self.note_edit)
        layout.addLayout(form)
        self.error = _label("", wrap=True)
        self.error.setStyleSheet(f"color: {style.ERROR};")
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.output_combo.currentIndexChanged.connect(self._update)
        self._update()

    def _update(self) -> None:
        has_primary = ex.primary_measurement(self.exp, self.specimen, self.output_combo.currentData()) is not None
        self.role_combo.setCurrentIndex(1 if has_primary else 0)

    def values(self) -> dict:
        n_text = self.n_edit.text().strip()
        if n_text and not (n_text.isdigit() and int(n_text) > 0):
            raise ValueError("the number of repeats must be a whole number")
        value = _number(self.value_edit.text())
        if value is None:
            raise ValueError("enter the value")
        sem = _number(self.sem_edit.text())
        if sem is not None and sem < 0:
            raise ValueError("a standard error cannot be negative")
        return {
            "output_name": self.output_combo.currentData(),
            "value": value,
            "sem": sem,
            "n": int(n_text) if n_text else None,
            "label": self.label_edit.text(),
            "role": self.role_combo.currentData(),
            "note": self.note_edit.text().strip(),
        }

    def accept(self) -> None:
        try:
            self.values()
        except ValueError as exc:
            self.error.setText(str(exc).capitalize())
            return
        super().accept()


class ExperimentView(QWidget):
    """Define the experiment, add specimens, attach measurement sessions, save and reopen."""

    titleChanged = Signal(str)
    statusMessage = Signal(str)

    def __init__(self, current_analysis: Callable[[], Analysis | None]) -> None:
        super().__init__()
        self.current_analysis = current_analysis
        self.settings = QSettings("Latos", "LatosV2")
        self.experiment = ex.Experiment(mixture=ex.Mixture())
        self.path: Path | None = None
        self.dirty = False
        self.sources: dict[str, str] = {}
        self.selected_id: str | None = None
        self._building = False
        self.mass_edits: dict[str, QLineEdit] = {}
        self.input_editors: dict[str, QLineEdit | QComboBox] = {}
        self.session_cards: dict[str, QFrame] = {}

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_explorer())
        splitter.addWidget(self._build_centre())
        splitter.addWidget(self._build_properties())
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 760, 400])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.refresh_all()

    # -- construction -------------------------------------------------------

    def _build_explorer(self) -> QWidget:
        panel = _styled(QWidget(), "panel")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(_label("Experiment Explorer", "panelTitle"))
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        layout.addWidget(self.tree, 1)
        outer.addWidget(body, 1)
        panel.setMinimumWidth(210)
        return panel

    def _build_centre(self) -> QWidget:
        centre = _styled(QWidget(), "workspace")
        self.tabs = QTabBar()
        self.tabs.setObjectName("documentTabs")
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        self.tabs.addTab("Observations")
        self.tabs.addTab("Definition")
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_observations())
        self.stack.addWidget(self._build_definition())
        self.tabs.currentChanged.connect(self.stack.setCurrentIndex)
        layout = QVBoxLayout(centre)
        layout.setContentsMargins(6, 6, 6, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tabs)
        layout.addWidget(self.stack, 1)
        centre.setMinimumWidth(480)
        return centre

    def _tool(self, bar: QHBoxLayout, text: str, tip: str, slot: Callable[[], object]) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        button.clicked.connect(lambda _checked=False: slot())
        bar.addWidget(button)
        return button

    def _build_observations(self) -> QWidget:
        page = _styled(QWidget(), "page")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        toolbar = _styled(QWidget(), "plotToolbar")
        bar = QHBoxLayout(toolbar)
        bar.setContentsMargins(6, 3, 6, 3)
        bar.setSpacing(2)
        self._tool(bar, "Add specimen", "Add a physical specimen to the observation table", self.add_specimen)
        self.remove_button = self._tool(bar, "Remove specimen", "Remove the selected specimen", self._on_remove_specimen)
        bar.addSpacing(8)
        self._tool(
            bar,
            "Attach HIOKI analysis…",
            "Attach the analysis shown in the HIOKI Analyzer to a specimen, as a measurement session",
            self._on_attach,
        )
        self._tool(bar, "Enter value…", "Enter a value measured outside Latos for the selected specimen", self._on_value)
        bar.addStretch(1)
        outer.addWidget(toolbar)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.empty_label = _label(
            "No specimens yet. Define the mixture and outputs on the Definition tab, then add "
            "specimens here or attach the analysis open in the HIOKI Analyzer.",
            "muted",
            True,
        )
        layout.addWidget(self.empty_label)
        self.current_label = _label("", "muted", True)
        layout.addWidget(self.current_label)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        layout.addWidget(self.table, 1)
        layout.addWidget(_label(_CAPTION, "muted", True))
        self.observation_warning = _label(WARNING_TEXT, "warning", True)
        layout.addWidget(self.observation_warning)
        outer.addWidget(body, 1)
        return page

    def _build_definition(self) -> QWidget:
        self.definition_scroll = QScrollArea()
        self.definition_scroll.setWidgetResizable(True)
        self.definition_scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = _styled(QWidget(), "page")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(12, 6, 12, 12)
        layout.setSpacing(4)
        self.definition_message = _label("", wrap=True)
        self.definition_message.setStyleSheet(f"color: {style.ERROR};")
        self.definition_message.hide()
        layout.addWidget(self.definition_message)
        self.sections: dict[str, QLabel] = {}

        self.sections["experiment"] = _section(layout, "Experiment")
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self._on_name)
        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("What is varied and why")
        self.description_edit.editingFinished.connect(self._on_description)
        form.addRow("Name", self.name_edit)
        form.addRow("Description", self.description_edit)
        layout.addLayout(form)

        self.sections["mixture"] = _section(
            layout,
            "Mixture",
            "The components every specimen is mixed from, any number. Each specimen's mass "
            "fractions are computed from its actual masses: x_i = m_i / Σ m_j.",
        )
        self.components_table = _table(["Component"])
        self.components_table.itemChanged.connect(self._on_component_item)
        layout.addWidget(self.components_table)
        layout.addLayout(_buttons(("Add component", self.add_component), ("Remove component", self._on_remove_component)))
        total = QHBoxLayout()
        self.variable_total = QRadioButton("Total mass varies")
        self.fixed_total = QRadioButton("Fixed total mass")
        self.total_edit = QLineEdit()
        self.total_edit.setPlaceholderText("total")
        self.total_edit.setMaximumWidth(80)
        self.tolerance_edit = QLineEdit()
        self.tolerance_edit.setPlaceholderText("tolerance")
        self.tolerance_edit.setMaximumWidth(80)
        for widget in (self.variable_total, self.fixed_total):
            total.addWidget(widget)
        total.addSpacing(6)
        total.addWidget(self.total_edit)
        total.addWidget(_label("±"))
        total.addWidget(self.tolerance_edit)
        total.addWidget(_label("g"))
        total.addStretch(1)
        layout.addLayout(total)
        self.fixed_total.toggled.connect(self._on_total_mode)
        self.total_edit.editingFinished.connect(self._on_total_values)
        self.tolerance_edit.editingFinished.connect(self._on_total_values)

        self.sections["inputs"] = _section(
            layout,
            "Inputs",
            "Variables other than the composition that are set for each specimen, such as a "
            "temperature or a solvent. Continuous: any value between the limits. Discrete: one of "
            "the listed numbers. Categorical: one of the listed labels. Separate values with commas.",
        )
        self.inputs_table = _table(["Name", "Type", "Unit", "Minimum", "Maximum", "Allowed values"])
        self.inputs_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.inputs_table.itemChanged.connect(self._on_input_item)
        layout.addWidget(self.inputs_table)
        layout.addLayout(_buttons(("Add input", self.add_input), ("Remove input", self._on_remove_input)))

        self.sections["outputs"] = _section(
            layout,
            "Outputs",
            f"The measured properties and the goal for each. An output measured by the HIOKI "
            f"analyzer takes its values from attached analyses, in {UNIT}; any other output takes "
            "values entered by hand.",
        )
        self.outputs_table = _table(["Name", "Unit", "Measured by", "Goal", "Target"])
        for col in (2, 3):
            self.outputs_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.outputs_table.itemChanged.connect(self._on_output_item)
        layout.addWidget(self.outputs_table)
        layout.addLayout(_buttons(("Add output", self.add_output), ("Remove output", self._on_remove_output)))
        layout.addStretch(1)
        self.definition_scroll.setWidget(body)
        return self.definition_scroll

    def _build_properties(self) -> QWidget:
        panel = _styled(QWidget(), "panel")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(_label("Specimen Properties", "panelTitle"))
        self.form_scroll = QScrollArea()
        self.form_scroll.setWidgetResizable(True)
        self.form_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.form_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.form_scroll, 1)
        panel.setMinimumWidth(330)
        return panel

    # -- public operations --------------------------------------------------

    def title(self) -> str:
        return self.experiment.name + (" *" if self.dirty else "")

    def components(self) -> list[str]:
        return self.experiment.mixture.components if self.experiment.mixture else []

    def refresh_all(self) -> None:
        self._fill_definition()
        self._fill_table()
        self._fill_tree()
        self._build_specimen_form()
        self.refresh_current()
        self.titleChanged.emit(self.title())

    def refresh_current(self) -> None:
        """Show which analysis "Attach HIOKI analysis" would attach."""
        a = self.current_analysis()
        if a is None or a.reductions is None:
            self.current_label.setText("HIOKI Analyzer: no completed analysis to attach.")
        else:
            self.current_label.setText(
                f"Ready to attach from the HIOKI Analyzer: {a.trace.source.name}, {WIRING[a.wiring_sign]} "
                f"(σ = {a.wiring_sign:+d}), cycles {', '.join(map(str, a.accepted))} in the value."
            )

    def select_specimen(self, specimen_id: str | None) -> None:
        self.selected_id = specimen_id
        self._sync_selection()
        self._build_specimen_form()

    def add_specimen(self, specimen_id: str | None = None, label: str = "") -> ex.Specimen:
        ids = [s.id for s in self.experiment.specimens]
        specimen_id = (specimen_id or "").strip() or _unique("S", ids, joiner="")
        if specimen_id in ids:
            raise ValueError(f"a specimen {specimen_id} already exists")
        specimen = ex.Specimen(id=specimen_id, label=label)
        self.experiment.specimens.append(specimen)
        self.selected_id = specimen_id
        self._changed(rebuild_form=True)
        return specimen

    def remove_specimen(self, specimen_id: str) -> None:
        specimen = self.experiment.specimen(specimen_id)
        index = self.experiment.specimens.index(specimen)
        self.experiment.specimens.remove(specimen)
        remaining = self.experiment.specimens
        self.selected_id = remaining[min(index, len(remaining) - 1)].id if remaining else None
        self._changed(rebuild_form=True)

    def add_component(self, name: str | None = None) -> str:
        if self.experiment.mixture is None:
            self.experiment.mixture = ex.Mixture()
        components = self.experiment.mixture.components
        name = (name or "").strip() or _unique("Component", components)
        if name in components:
            raise ValueError(f"a component named {name} already exists")
        components.append(name)
        self._structure_changed()
        return name

    def add_input(self, name: str | None = None, kind: str = "continuous", unit: str = "",
                  lower: float | None = None, upper: float | None = None,
                  levels: list[str] | None = None) -> ex.InputVariable:
        names = [v.name for v in self.experiment.inputs]
        name = (name or "").strip() or _unique("Input", names)
        if name in names:
            raise ValueError(f"an input named {name} already exists")
        variable = ex.InputVariable(name, kind, unit, lower, upper, list(levels or []))
        self.experiment.inputs.append(variable)
        self._structure_changed()
        return variable

    def add_output(self, name: str | None = None, unit: str = "", method: str = ex.MANUAL,
                   objective: str = "maximize", target: float | None = None) -> ex.OutputDefinition:
        names = [o.name for o in self.experiment.outputs]
        name = (name or "").strip() or _unique("Output", names)
        if name in names:
            raise ValueError(f"an output named {name} already exists")
        if method.startswith("hioki:"):
            unit = UNIT
        output = ex.OutputDefinition(name, unit, method, objective, target)
        self.experiment.outputs.append(output)
        self._structure_changed()
        return output

    def attach_analysis(self, analysis: Analysis | None, specimen_id: str, output_name: str,
                        label: str = "", role: str | None = None, create: bool = False) -> ex.Measurement:
        """Attach an analyzer result to a specimen as one measurement session.

        The values are copied from the analysis exactly as shown; nothing is
        recalculated. Raises ValueError when the attachment is not allowed.
        """
        exp = self.experiment
        if analysis is None or analysis.reductions is None:
            raise ValueError("there is no completed analysis to attach")
        current = file_sha256(analysis.trace.source)
        if current != analysis.trace.sha256:
            raise ValueError(
                f"{analysis.trace.source.name} changed after it was opened; reopen it in the HIOKI Analyzer"
            )
        output = exp.output(output_name)
        if output is None:
            raise ValueError("choose an output measured by the HIOKI analyzer")
        record = analysis_record(analysis, current)
        record["source"]["path"] = str(Path(record["source"]["path"]).resolve())
        specimen_id = (specimen_id or "").strip()
        try:
            specimen = exp.specimen(specimen_id)
        except KeyError:
            if not create:
                raise ValueError(f"there is no specimen {specimen_id!r}") from None
            if not specimen_id:
                raise ValueError("the new specimen needs an ID") from None
            specimen = ex.Specimen(id=specimen_id)
        if role is None:
            role = "supplementary" if ex.primary_measurement(exp, specimen, output_name) else "primary"
        measurement = ex.hioki_measurement(record, output, label, role)
        ex.attach(exp, specimen, measurement)
        if specimen not in exp.specimens:
            exp.specimens.append(specimen)
        self.sources[measurement.id] = "ok"
        self.selected_id = specimen.id
        self._changed(rebuild_form=True)
        return measurement

    def enter_value(self, specimen_id: str, output_name: str, value: float, sem: float | None = None,
                    n: int | None = None, label: str = "", role: str | None = None, note: str = "") -> ex.Measurement:
        exp = self.experiment
        specimen = exp.specimen(specimen_id)
        output = exp.output(output_name)
        if output is None:
            raise ValueError(f"no output named {output_name!r}")
        if role is None:
            role = "supplementary" if ex.primary_measurement(exp, specimen, output_name) else "primary"
        measurement = ex.manual_measurement(output, value, label, role, sem=sem, n=n, note=note)
        ex.attach(exp, specimen, measurement)
        self.selected_id = specimen.id
        self._changed(rebuild_form=True)
        return measurement

    def make_primary(self, measurement_id: str) -> None:
        specimen = self._owner(measurement_id)
        ex.set_primary(specimen, measurement_id)
        self._changed()

    def remove_session(self, measurement_id: str) -> None:
        specimen = self._owner(measurement_id)
        specimen.measurements = [m for m in specimen.measurements if m.id != measurement_id]
        self.sources.pop(measurement_id, None)
        self._changed()

    def new_experiment(self) -> bool:
        if not self.maybe_discard():
            return False
        self.experiment = ex.Experiment(mixture=ex.Mixture())
        self.path, self.dirty, self.sources, self.selected_id = None, False, {}, None
        self.refresh_all()
        return True

    def open_experiment(self, path: Path | None = None) -> bool:
        if not self.maybe_discard():
            return False
        if path is None:
            start = self.settings.value("last_experiment_dir", self._documents())
            chosen, _ = QFileDialog.getOpenFileName(self, "Open experiment", str(start), FILE_FILTER)
            if not chosen:
                return False
            path = Path(chosen)
        try:
            experiment = load_experiment(path)
        except ProjectError as exc:
            QMessageBox.critical(self, "Cannot open the experiment", str(exc))
            return False
        self.settings.setValue("last_experiment_dir", str(Path(path).parent))
        self.experiment = experiment
        self.path, self.dirty = Path(path), False
        self.sources = ex.check_sources(experiment)
        self.selected_id = experiment.specimens[0].id if experiment.specimens else None
        self.refresh_all()
        problems = sum(state != "ok" for state in self.sources.values())
        if problems:
            self.statusMessage.emit(
                f"{problems} recording(s) missing or changed since analysis; see the specimens' checks."
            )
        else:
            self.statusMessage.emit(f"Opened {Path(path).name}")
        return True

    def save(self, path: Path | None = None) -> bool:
        path = path or self.path
        if path is None:
            return self.save_as()
        try:
            written = save_experiment(self.experiment, path)
        except (ProjectError, OSError) as exc:
            QMessageBox.warning(self, "Cannot save the experiment", str(exc))
            return False
        self.settings.setValue("last_experiment_dir", str(written.parent))
        self.path, self.dirty = written, False
        self._fill_tree()
        self.titleChanged.emit(self.title())
        self.statusMessage.emit(f"Saved {written}")
        return True

    def save_as(self) -> bool:
        start = Path(self.settings.value("last_experiment_dir", self._documents()))
        name = self.path.name if self.path else "".join(
            ch if ch.isalnum() or ch in "-_ " else "_" for ch in self.experiment.name
        ) + SUFFIX
        chosen, _ = QFileDialog.getSaveFileName(self, "Save experiment", str(start / name), FILE_FILTER)
        return bool(chosen) and self.save(Path(chosen))

    def maybe_discard(self) -> bool:
        """True when there are no unsaved changes, or the researcher saved or discarded them."""
        if not self.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved experiment",
            f"Save the changes to “{self.experiment.name}”?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    # -- button handlers that ask first ------------------------------------

    def _on_attach(self) -> None:
        analysis = self.current_analysis()
        if analysis is None or analysis.reductions is None:
            QMessageBox.information(
                self,
                "Attach HIOKI analysis",
                "Open a workbook in the HIOKI Analyzer and choose its lead polarity first. The "
                "analysis shown there, with any excluded cycles, is what gets attached.",
            )
            return
        if not any(o.method.startswith("hioki:") for o in self.experiment.outputs):
            answer = QMessageBox.question(
                self,
                "Attach HIOKI analysis",
                f"No output is measured by the HIOKI analyzer yet. Add the output “S_app” "
                f"({UNIT}, whole cycle, maximize)? It can be changed on the Definition tab.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self.add_output("S_app", UNIT, "hioki:combined", "maximize")
        dialog = AttachDialog(self, self.experiment, analysis, self.selected_id)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        specimen_id, create, output, label, role = dialog.choice()
        try:
            self.attach_analysis(analysis, specimen_id, output, label, role, create)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot attach the analysis", str(exc).capitalize())
            return
        self.statusMessage.emit(f"Attached {analysis.trace.source.name} to specimen {specimen_id} as a {role} session")

    def _on_value(self) -> None:
        specimen = self._selected()
        if specimen is None or not any(o.method == ex.MANUAL for o in self.experiment.outputs):
            QMessageBox.information(
                self,
                "Enter value",
                "Select a specimen, and define an output measured by manual entry on the Definition tab.",
            )
            return
        dialog = ValueDialog(self, self.experiment, specimen)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.enter_value(specimen.id, **dialog.values())
            except ValueError as exc:
                QMessageBox.warning(self, "Cannot enter the value", str(exc).capitalize())

    def _on_remove_specimen(self) -> None:
        specimen = self._selected()
        if specimen is None:
            return
        answer = QMessageBox.question(
            self,
            "Remove specimen",
            f"Remove specimen {specimen.id} and its {len(specimen.measurements)} measurement session(s) "
            "from the experiment? The recordings themselves are not touched.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.remove_specimen(specimen.id)

    def _on_remove_session(self, measurement_id: str) -> None:
        m = next(m for m in self._owner(measurement_id).measurements if m.id == measurement_id)
        answer = QMessageBox.question(
            self, "Remove session", f"Remove the session “{m.label}”? The recording itself is not touched."
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.remove_session(measurement_id)

    def _confirm_definition_removal(self, what: str, used: int) -> bool:
        if not used:
            return True
        answer = QMessageBox.question(
            self, "Remove", f"{used} specimen(s) have values for {what}. Remove it and those values?"
        )
        return answer == QMessageBox.StandardButton.Yes

    def _on_remove_component(self) -> None:
        row = self.components_table.currentRow()
        if row < 0:
            return
        name = self.components()[row]
        used = sum(name in s.masses_g or s.pure_component == name for s in self.experiment.specimens)
        if self._confirm_definition_removal(name, used):
            ex.remove_component(self.experiment, name)
            self._structure_changed()

    def _on_remove_input(self) -> None:
        row = self.inputs_table.currentRow()
        if row < 0:
            return
        name = self.experiment.inputs[row].name
        used = sum(name in s.inputs for s in self.experiment.specimens)
        if self._confirm_definition_removal(name, used):
            ex.remove_input(self.experiment, name)
            self._structure_changed()

    def _on_remove_output(self) -> None:
        row = self.outputs_table.currentRow()
        if row < 0:
            return
        name = self.experiment.outputs[row].name
        used = sum(any(m.output == name for m in s.measurements) for s in self.experiment.specimens)
        if used:
            answer = QMessageBox.question(
                self,
                "Remove output",
                f"{used} specimen(s) have measurement sessions for {name}. Remove the output and "
                "those sessions? The recordings themselves are not touched.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        ex.remove_output(self.experiment, name)
        self._structure_changed()

    # -- change propagation -------------------------------------------------

    def _changed(self, rebuild_form: bool = False) -> None:
        """After an edit of specimen data: mark unsaved and refresh what depends on it."""
        self.dirty = True
        self.titleChanged.emit(self.title())
        if self._building:
            return
        self._fill_table()
        self._fill_tree()
        if rebuild_form:
            self._build_specimen_form()
        else:
            self._refresh_form()

    def _definition_changed(self, refill: bool = False) -> None:
        """After a definition edit: the table columns and the specimen form follow."""
        self.definition_message.hide()
        if refill:
            # Deferred: the edited table must not be rebuilt inside its own signal.
            QTimer.singleShot(0, self._fill_definition)
        self._changed(rebuild_form=True)

    def _structure_changed(self) -> None:
        self.definition_message.hide()
        self._fill_definition()
        self._changed(rebuild_form=True)

    def _definition_error(self, text: str) -> None:
        self.definition_message.setText(text[:1].upper() + text[1:])
        self.definition_message.show()
        QTimer.singleShot(0, self._fill_definition)

    # -- definition tab -----------------------------------------------------

    def _fill_definition(self) -> None:
        exp = self.experiment
        self.name_edit.setText(exp.name)
        self.description_edit.setText(exp.description)

        table = self.components_table
        table.blockSignals(True)
        table.setRowCount(0)
        for row, name in enumerate(self.components()):
            table.insertRow(row)
            table.setItem(row, 0, _item(name, editable=True))
        table.blockSignals(False)
        _fit_table_height(table)
        mixture = exp.mixture
        fixed = mixture is not None and mixture.total_mode == "fixed"
        for widget in (self.variable_total, self.fixed_total):
            widget.blockSignals(True)
            widget.setEnabled(bool(self.components()))
        self.fixed_total.setChecked(fixed)
        self.variable_total.setChecked(not fixed)
        for widget in (self.variable_total, self.fixed_total):
            widget.blockSignals(False)
        self.total_edit.setText(_plain(mixture.total_g) if mixture else "")
        self.tolerance_edit.setText(_plain(mixture.tolerance_g) if mixture else "")
        self.total_edit.setEnabled(fixed)
        self.tolerance_edit.setEnabled(fixed)

        table = self.inputs_table
        table.blockSignals(True)
        table.setRowCount(0)
        for row, v in enumerate(exp.inputs):
            table.insertRow(row)
            categorical = v.kind == "categorical"
            table.setItem(row, 0, _item(v.name, editable=True))
            combo = QComboBox()
            for kind, text in _KIND_NAMES.items():
                combo.addItem(text, kind)
            combo.setCurrentIndex(combo.findData(v.kind))
            combo.currentIndexChanged.connect(lambda _i, r=row, c=combo: self._on_input_kind(r, c.currentData()))
            table.setCellWidget(row, 1, combo)
            table.setItem(row, 2, _item(v.unit, editable=not categorical))
            for col, value in ((3, v.lower), (4, v.upper)):
                item = _item(_plain(value), editable=v.kind == "continuous", numeric=True)
                if v.kind != "continuous":
                    item = _muted(_item("—", numeric=True), italic=False)
                table.setItem(row, col, item)
            levels = _item(", ".join(v.levels), editable=v.kind != "continuous")
            table.setItem(row, 5, levels if v.kind != "continuous" else _muted(_item("—"), italic=False))
        table.blockSignals(False)
        _fit_table_height(table)

        table = self.outputs_table
        table.blockSignals(True)
        table.setRowCount(0)
        for row, o in enumerate(exp.outputs):
            table.insertRow(row)
            hioki = o.method.startswith("hioki:")
            table.setItem(row, 0, _item(o.name, editable=True))
            table.setItem(row, 1, _item(o.unit, editable=not hioki, tip="Fixed by the HIOKI analyzer" if hioki else ""))
            method = QComboBox()
            for key, text in ex.METHOD_LABELS.items():
                method.addItem(text, key)
            method.setCurrentIndex(method.findData(o.method))
            method.currentIndexChanged.connect(lambda _i, r=row, c=method: self._on_output_method(r, c.currentData()))
            table.setCellWidget(row, 2, method)
            goal = QComboBox()
            for key, text in _OBJECTIVE_NAMES.items():
                goal.addItem(text, key)
            goal.setCurrentIndex(goal.findData(o.objective))
            goal.currentIndexChanged.connect(lambda _i, r=row, c=goal: self._on_output_goal(r, c.currentData()))
            table.setCellWidget(row, 3, goal)
            target = o.objective == "target"
            item = _item(_plain(o.target), editable=True, numeric=True) if target else _muted(_item("—", numeric=True), False)
            table.setItem(row, 4, item)
        table.blockSignals(False)
        _fit_table_height(table)

    def _on_name(self) -> None:
        name = self.name_edit.text().strip() or "Untitled experiment"
        if name != self.experiment.name:
            self.experiment.name = name
            self._changed()

    def _on_description(self) -> None:
        text = self.description_edit.text().strip()
        if text != self.experiment.description:
            self.experiment.description = text
            self._changed()

    def _on_component_item(self, item: QTableWidgetItem) -> None:
        old = self.components()[item.row()]
        new = item.text().strip()
        if new == old:
            return
        try:
            if not new:
                raise ValueError("a component needs a name")
            ex.rename_component(self.experiment, old, new)
        except ValueError as exc:
            self._definition_error(str(exc))
            return
        self._definition_changed()

    def _on_total_mode(self, fixed: bool) -> None:
        if self.experiment.mixture is None:
            return
        self.experiment.mixture.total_mode = "fixed" if fixed else "variable"
        self.total_edit.setEnabled(fixed)
        self.tolerance_edit.setEnabled(fixed)
        self._definition_changed()

    def _on_total_values(self) -> None:
        mixture = self.experiment.mixture
        if mixture is None:
            return
        try:
            total, tolerance = _number(self.total_edit.text()), _number(self.tolerance_edit.text())
            if (total is not None and total <= 0) or (tolerance is not None and tolerance < 0):
                raise ValueError("the total mass must be positive and the tolerance not negative")
        except ValueError as exc:
            self._definition_error(f"Fixed total mass: {exc}")
            return
        if (total, tolerance) != (mixture.total_g, mixture.tolerance_g):
            mixture.total_g, mixture.tolerance_g = total, tolerance
            self._definition_changed()

    def _on_input_item(self, item: QTableWidgetItem) -> None:
        v = self.experiment.inputs[item.row()]
        col, text = item.column(), item.text().strip()
        try:
            if col == 0:
                if not text:
                    raise ValueError("an input needs a name")
                ex.rename_input(self.experiment, v.name, text)
            elif col == 2:
                v.unit = text
            elif col in (3, 4):
                value = _number(text)
                lower, upper = (value, v.upper) if col == 3 else (v.lower, value)
                if lower is not None and upper is not None and lower > upper:
                    raise ValueError(f"{v.name}: the minimum is larger than the maximum")
                v.lower, v.upper = lower, upper
            elif col == 5:
                levels = [part.strip() for part in text.split(",") if part.strip()]
                if v.kind == "discrete":
                    for level in levels:
                        _number(level)
                v.levels = levels
        except ValueError as exc:
            self._definition_error(str(exc))
            return
        self._definition_changed()

    def _on_input_kind(self, row: int, kind: str) -> None:
        v = self.experiment.inputs[row]
        v.kind = kind
        if kind != "continuous":
            v.lower = v.upper = None
        if kind == "categorical":
            v.unit = ""
        if kind == "discrete":
            try:
                for level in v.levels:
                    _number(level)
            except ValueError:
                v.levels = []
        if kind == "continuous":
            v.levels = []
        self._definition_changed(refill=True)

    def _on_output_item(self, item: QTableWidgetItem) -> None:
        o = self.experiment.outputs[item.row()]
        col, text = item.column(), item.text().strip()
        try:
            if col == 0:
                if not text:
                    raise ValueError("an output needs a name")
                ex.rename_output(self.experiment, o.name, text)
            elif col == 1:
                o.unit = text
            elif col == 4:
                o.target = _number(text)
        except ValueError as exc:
            self._definition_error(str(exc))
            return
        self._definition_changed()

    def _on_output_method(self, row: int, method: str) -> None:
        o = self.experiment.outputs[row]
        o.method = method
        if method.startswith("hioki:"):
            o.unit = UNIT
        self._definition_changed(refill=True)

    def _on_output_goal(self, row: int, objective: str) -> None:
        o = self.experiment.outputs[row]
        o.objective = objective
        if objective != "target":
            o.target = None
        self._definition_changed(refill=True)

    # -- observation table and explorer ------------------------------------

    def _messages(self, specimen: ex.Specimen) -> list[ex.Message]:
        return ex.specimen_messages(self.experiment, specimen, self.sources)

    def _fill_table(self) -> None:
        # Composition, inputs and responses first; the raw masses they come from after.
        exp = self.experiment
        components = self.components()
        headers = ["Specimen", "Label", *(f"{c}\nmass fraction" for c in components)]
        headers += [f"{v.name}\n({v.unit})" if v.unit else v.name for v in exp.inputs]
        headers += [f"{o.name} ({o.unit})\nmean ± SEM (n)" if o.unit else f"{o.name}\nmean ± SEM (n)" for o in exp.outputs]
        if components:
            headers += [*(f"{c}\nmass (g)" for c in components), "Total\nmass (g)"]
        headers += ["Sessions", "Primary source", "Checks"]

        table = self.table
        table.blockSignals(True)
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(exp.specimens))
        selected_row = -1
        for row, s in enumerate(exp.specimens):
            if s.id == self.selected_id:
                selected_row = row
            pure = s.pure_component
            cells: list[QTableWidgetItem] = [_item(s.id), _item(s.label, tip=s.notes)]
            x = ex.fractions(exp, s)
            for c in components:
                if x is None:
                    cells.append(_muted(_item("—", tip="Needs every mass", numeric=True), italic=False))
                elif pure:
                    tip = f"Pure {pure}: the fraction is 1 or 0 by identity"
                    cells.append(_item(f"{x[c]:.0f} (pure)", tip=tip, numeric=True))
                else:
                    cells.append(_item(f"{x[c]:.4f}", numeric=True))
            for v in exp.inputs:
                value = s.inputs.get(v.name)
                text = "—" if value is None else (value if isinstance(value, str) else f"{value:g}")
                cells.append(_item(text, numeric=not isinstance(value, str)))
            sources = []
            for o in exp.outputs:
                primary = ex.primary_measurement(exp, s, o.name)
                tip = ""
                if primary is not None:
                    tip = primary.label
                    if primary.record:
                        tip += f"\n{primary.wiring}\n{primary.source_path.name}"
                        sources.append(primary.source_path.name)
                cells.append(_item(_value_text(primary), tip=tip, numeric=True))
            if components:
                for c in components:
                    if pure and c != pure:
                        cells.append(_muted(_item("—", tip=f"Pure {pure}", numeric=True), italic=False))
                    elif c in s.masses_g:
                        cells.append(_item(f"{s.masses_g[c]:.4f}", numeric=True))
                    else:
                        cells.append(_muted(_item(NOT_RECORDED, numeric=True)))
                total = ex.total_mass(exp, s)
                cells.append(
                    _muted(_item(NOT_RECORDED, numeric=True)) if total is None else _item(f"{total:.4f}", numeric=True)
                )
            sessions_tip = "\n".join(f"{m.label}: {m.role}, {m.output} {_value_text(m)}" for m in s.measurements)
            cells.append(_item(str(len(s.measurements)), tip=sessions_tip, numeric=True))
            cells.append(_item(", ".join(dict.fromkeys(sources)) or "—"))
            messages = self._messages(s)
            warnings = sum(m.level == "warning" for m in messages)
            check = _item("OK" if not warnings else f"{warnings} warning{'s' if warnings > 1 else ''}")
            check.setForeground(QColor(style.OK if not warnings else style.CAUTION))
            check.setToolTip("\n".join(("⚠ " if m.level == "warning" else "• ") + m.text for m in messages))
            cells.append(check)
            for col, item in enumerate(cells):
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, s.id)
                table.setItem(row, col, item)
        if selected_row >= 0:
            table.selectRow(selected_row)
        table.blockSignals(False)
        self.empty_label.setVisible(not exp.specimens)
        self.remove_button.setEnabled(self.selected_id is not None)
        self.observation_warning.setVisible(any(o.method.startswith("hioki:") for o in exp.outputs))

    def _fill_tree(self) -> None:
        exp = self.experiment
        tree = self.tree
        tree.blockSignals(True)
        tree.clear()
        root = QTreeWidgetItem([self.title()])
        bold = QFont(root.font(0))
        bold.setBold(True)
        root.setFont(0, bold)
        root.setToolTip(0, str(self.path) if self.path else "Not saved yet")
        root.setData(0, Qt.ItemDataRole.UserRole, ("definition", "experiment"))
        tree.addTopLevelItem(root)
        definition = QTreeWidgetItem(root, ["Definition"])
        definition.setData(0, Qt.ItemDataRole.UserRole, ("definition", "experiment"))
        components = self.components()
        for key, text in (
            ("mixture", f"Mixture: {len(components)} component{'s' if len(components) != 1 else ''}"),
            ("inputs", f"Inputs: {len(exp.inputs)}"),
            ("outputs", f"Outputs: {len(exp.outputs)}"),
        ):
            item = QTreeWidgetItem(definition, [text])
            item.setData(0, Qt.ItemDataRole.UserRole, ("definition", key))
        specimens = QTreeWidgetItem(root, [f"Specimens: {len(exp.specimens)}"])
        specimens.setData(0, Qt.ItemDataRole.UserRole, ("observations", None))
        current = None
        self._tree_items: dict[str, QTreeWidgetItem] = {}
        for s in exp.specimens:
            warnings = sum(m.level == "warning" for m in self._messages(s))
            item = QTreeWidgetItem(specimens, [s.id + (f"   ⚠ {warnings}" if warnings else "")])
            if warnings:
                item.setForeground(0, QColor(style.CAUTION))
            item.setToolTip(0, (s.label or s.id) + (f"\n{warnings} warning(s): see Checks" if warnings else ""))
            item.setData(0, Qt.ItemDataRole.UserRole, ("specimen", s.id))
            for m in s.measurements:
                child = QTreeWidgetItem(item, [f"{m.label} ({m.role})"])
                child.setForeground(0, QColor(style.MUTED))
                child.setToolTip(0, f"{m.output}: {_value_text(m)}" + (f"\n{m.source_path.name}" if m.record else ""))
                child.setData(0, Qt.ItemDataRole.UserRole, ("specimen", s.id))
            self._tree_items[s.id] = item
            if s.id == self.selected_id:
                current = item
        tree.expandAll()
        if current is not None:
            tree.setCurrentItem(current)
        tree.blockSignals(False)

    def _sync_selection(self) -> None:
        """Highlight the selected specimen in the table and the explorer."""
        self.table.blockSignals(True)
        self.table.clearSelection()
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == self.selected_id:
                self.table.selectRow(row)
                break
        self.table.blockSignals(False)
        item = self._tree_items.get(self.selected_id)
        if item is not None and item is not self.tree.currentItem():
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(item)
            self.tree.blockSignals(False)
        self.remove_button.setEnabled(self.selected_id is not None)

    def _on_table_selection(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            specimen_id = self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
            if specimen_id != self.selected_id:
                self.select_specimen(specimen_id)

    def _on_tree_selection(self) -> None:
        item = self.tree.currentItem()
        target = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if target is None:
            return
        kind, key = target
        if kind == "definition":
            self.tabs.setCurrentIndex(1)
            self.definition_scroll.ensureWidgetVisible(self.sections[key], 0, 0)
        else:
            self.tabs.setCurrentIndex(0)
            if kind == "specimen" and key != self.selected_id:
                self.select_specimen(key)

    # -- specimen properties -----------------------------------------------

    def _selected(self) -> ex.Specimen | None:
        if self.selected_id is None:
            return None
        try:
            return self.experiment.specimen(self.selected_id)
        except KeyError:
            return None

    def _owner(self, measurement_id: str) -> ex.Specimen:
        for s in self.experiment.specimens:
            if any(m.id == measurement_id for m in s.measurements):
                return s
        raise KeyError(f"no session {measurement_id!r}")

    def _build_specimen_form(self) -> None:
        self._building = True
        try:
            body = _styled(QWidget(), "panel")
            layout = QVBoxLayout(body)
            layout.setContentsMargins(10, 2, 10, 10)
            layout.setSpacing(4)
            self.mass_edits, self.input_editors, self.session_cards = {}, {}, {}
            self.form_message = _label("", wrap=True)
            self.form_message.setStyleSheet(f"color: {style.ERROR};")
            self.form_message.hide()
            specimen = self._selected()
            if specimen is None:
                layout.addWidget(_label("Select a specimen in the table, or add one with Add specimen.", "muted", True))
            else:
                self._specimen_sections(layout, specimen)
            layout.addStretch(1)
            old = self.form_scroll.takeWidget()
            self.form_scroll.setWidget(body)
            if old is not None:
                old.deleteLater()
        finally:
            self._building = False
        self._refresh_form()

    def _specimen_sections(self, layout: QVBoxLayout, s: ex.Specimen) -> None:
        exp = self.experiment
        _section(layout, "Specimen")
        layout.addWidget(self.form_message)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.id_edit = QLineEdit(s.id)
        self.id_edit.editingFinished.connect(self._on_id)
        self.label_edit = QLineEdit(s.label)
        self.label_edit.setPlaceholderText("Nominal label from the lab record")
        self.label_edit.editingFinished.connect(self._on_label)
        self.notes_edit = QLineEdit(s.notes)
        self.notes_edit.editingFinished.connect(self._on_notes)
        form.addRow("ID", self.id_edit)
        form.addRow("Label", self.label_edit)
        form.addRow("Notes", self.notes_edit)
        layout.addLayout(form)

        components = self.components()
        if components:
            _section(layout, "Composition")
            form = QFormLayout()
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            self.type_combo = QComboBox()
            self.type_combo.addItem("Mixture", None)
            for c in components:
                self.type_combo.addItem(f"Pure {c}", c)
            if s.pure_component and s.pure_component not in components:
                self.type_combo.addItem(f"Pure {s.pure_component} (not a component)", s.pure_component)
            self.type_combo.setCurrentIndex(max(0, self.type_combo.findData(s.pure_component)))
            self.type_combo.currentIndexChanged.connect(self._on_type)
            form.addRow("Specimen type", self.type_combo)
            for c in components:
                edit = QLineEdit(_plain(s.masses_g.get(c)))
                edit.setPlaceholderText(NOT_RECORDED)
                edit.editingFinished.connect(lambda c=c: self._on_mass(c))
                row = QHBoxLayout()
                row.addWidget(edit, 1)
                row.addWidget(_label("g", "muted"))
                form.addRow(f"Mass of {c}", row)
                self.mass_edits[c] = edit
            self.total_label = _label("")
            form.addRow("Total mass", self.total_label)
            self.fractions_label = _label("", wrap=True)
            form.addRow("Mass fractions", self.fractions_label)
            layout.addLayout(form)

        if exp.inputs:
            _section(layout, "Inputs")
            form = QFormLayout()
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for v in exp.inputs:
                value = s.inputs.get(v.name)
                if v.levels:
                    editor = QComboBox()
                    editor.addItem("", None)
                    for level in v.levels:
                        editor.addItem(level, level)
                    index = 0
                    if value is not None:
                        index = -1
                        for i, level in enumerate(v.levels, start=1):
                            if level == value or (not isinstance(value, str) and _number(level) == value):
                                index = i
                        if index < 0:
                            editor.addItem(f"{value:g} (not allowed)" if not isinstance(value, str)
                                           else f"{value} (not allowed)", str(value))
                            index = editor.count() - 1
                    editor.setCurrentIndex(index)
                    editor.currentIndexChanged.connect(lambda _i, n=v.name: self._on_input_value(n))
                else:
                    text = "" if value is None else (value if isinstance(value, str) else _plain(value))
                    editor = QLineEdit(text)
                    if v.lower is not None or v.upper is not None:
                        low = "" if v.lower is None else f"{v.lower:g}"
                        high = "" if v.upper is None else f"{v.upper:g}"
                        editor.setPlaceholderText(f"{low} – {high}")
                    editor.editingFinished.connect(lambda n=v.name: self._on_input_value(n))
                row = QHBoxLayout()
                row.addWidget(editor, 1)
                if v.unit:
                    row.addWidget(_label(v.unit, "muted"))
                form.addRow(v.name, row)
                self.input_editors[v.name] = editor
            layout.addLayout(form)

        _section(layout, "Measurement Sessions")
        self.sessions_box = QVBoxLayout()
        self.sessions_box.setSpacing(6)
        layout.addLayout(self.sessions_box)
        layout.addLayout(_buttons(("Attach HIOKI analysis…", self._on_attach), ("Enter value…", self._on_value)))
        _section(layout, "Checks")
        self.messages_label = _label("", wrap=True)
        self.messages_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.messages_label)

    def _refresh_form(self) -> None:
        """Update the derived parts of the specimen form after an edit."""
        if self._building:
            return
        exp, s = self.experiment, self._selected()
        if s is None or not hasattr(self, "messages_label"):
            return
        if self.mass_edits:
            for c, edit in self.mass_edits.items():
                edit.setEnabled(not s.pure_component or c == s.pure_component)
            total = ex.total_mass(exp, s)
            self.total_label.setText(NOT_RECORDED if total is None else f"{total:.4f} g")
            x = ex.fractions(exp, s)
            if x is None:
                self.fractions_label.setText("— (needs every mass)")
            else:
                suffix = "  (pure, by identity)" if s.pure_component else ""
                self.fractions_label.setText("\n".join(f"{c}: {value:.4f}" for c, value in x.items()) + suffix)

        while self.sessions_box.count():
            widget = self.sessions_box.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.session_cards = {}
        if not s.measurements:
            self.sessions_box.addWidget(_label("No sessions yet.", "muted"))
        for m in s.measurements:
            card = self._session_card(s, m)
            self.session_cards[m.id] = card
            self.sessions_box.addWidget(card)

        lines = []
        for message in self._messages(s):
            text = html.escape(message.text)
            if message.level == "warning":
                lines.append(f'<span style="color:{style.CAUTION}">⚠ {text}</span>')
            else:
                lines.append(f'<span style="color:{style.MUTED}">• {text}</span>')
        self.messages_label.setText(
            "<br>".join(lines) if lines else f'<span style="color:{style.OK}">No problems found.</span>'
        )

    def _session_card(self, s: ex.Specimen, m: ex.Measurement) -> QFrame:
        exp = self.experiment
        used = ex.primary_measurement(exp, s, m.output) is m
        card = QFrame()
        card.setObjectName("card")
        card.setProperty("primary", used)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)
        # Unwrapped lines: wrapped labels in a scrolled card are given too little height.
        if used:
            role = f"primary · response for {m.output}"
        elif m.role == "primary":
            role = "primary · not usable for this output"
        else:
            role = "supplementary · not used"
        title = _label(f"<b>{html.escape(m.label)}</b> &nbsp;·&nbsp; {html.escape(role)}")
        title.setToolTip(f"{m.label}: {role}")
        layout.addWidget(title)
        unit = f" {m.unit}" if m.unit else ""
        value = f"{m.output} = {_value_and_sem(m)}{unit}"
        if m.record:
            value += f", {m.n} cycle{'s' if m.n != 1 else ''}"
        elif m.n:
            value += f", n = {m.n}"
        value_label = _label(value)
        value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(value_label)
        details = []
        if m.record:
            sign = m.record["declarations"]["wiring_sign"]
            details.append(f"{m.wiring} (σ = {sign:+d})")
            excluded = m.record.get("excluded_cycles") or {}
            if excluded:
                details.append("Excluded: " + ", ".join(f"cycle {i} ({r})" for i, r in excluded.items()))
            state = _SOURCE_STATE.get(self.sources.get(m.id, ""), "file not checked")
            details.append(f"{m.source_path.name} · {state}")
        else:
            details.append(ex.METHOD_LABELS.get(m.method, m.method))
        if m.note:
            details.append(m.note)
        for line in details:
            detail = _label(line, "muted")
            detail.setToolTip(str(m.source_path) if m.record else line)
            layout.addWidget(detail)
        row = QHBoxLayout()
        row.setContentsMargins(0, 2, 0, 0)
        primary = QPushButton("Make primary")
        primary.setEnabled(not used and exp.output(m.output) is not None)
        primary.clicked.connect(lambda _checked=False, i=m.id: self.make_primary(i))
        remove = QPushButton("Remove")
        remove.clicked.connect(lambda _checked=False, i=m.id: self._on_remove_session(i))
        row.addWidget(primary)
        row.addWidget(remove)
        row.addStretch(1)
        layout.addLayout(row)
        return card

    def _form_error(self, text: str) -> None:
        self.form_message.setText(text)
        self.form_message.show()

    def _on_id(self) -> None:
        s = self._selected()
        new = self.id_edit.text().strip()
        if s is None or new == s.id:
            return
        if not new or any(o.id == new for o in self.experiment.specimens):
            self._form_error("The specimen ID must be unique and not empty.")
            self.id_edit.setText(s.id)
            return
        s.id = new
        self.selected_id = new
        self.form_message.hide()
        self._changed()

    def _on_label(self) -> None:
        s = self._selected()
        if s is not None and self.label_edit.text().strip() != s.label:
            s.label = self.label_edit.text().strip()
            self._changed()

    def _on_notes(self) -> None:
        s = self._selected()
        if s is not None and self.notes_edit.text().strip() != s.notes:
            s.notes = self.notes_edit.text().strip()
            self._changed()

    def _on_type(self) -> None:
        s = self._selected()
        if s is not None:
            s.pure_component = self.type_combo.currentData()
            self._changed()

    def _on_mass(self, component: str) -> None:
        s = self._selected()
        edit = self.mass_edits.get(component)
        if s is None or edit is None:
            return
        try:
            value = _number(edit.text())
            if value is not None and value < 0:
                raise ValueError("a mass cannot be negative")
        except ValueError as exc:
            self._form_error(f"Mass of {component}: {exc}.")
            edit.setText(_plain(s.masses_g.get(component)))
            return
        self.form_message.hide()
        if value == s.masses_g.get(component):
            return
        if value is None:
            s.masses_g.pop(component, None)
        else:
            s.masses_g[component] = value
        self._changed()

    def _on_input_value(self, name: str) -> None:
        s = self._selected()
        editor = self.input_editors.get(name)
        variable = next((v for v in self.experiment.inputs if v.name == name), None)
        if s is None or editor is None or variable is None:
            return
        try:
            if isinstance(editor, QComboBox):
                value = editor.currentData()
                if value is not None and variable.kind != "categorical":
                    value = float(value)
            else:
                value = ex.parse_input(variable, editor.text())
        except ValueError as exc:
            self._form_error(f"{exc}.")
            return
        self.form_message.hide()
        if value == s.inputs.get(name):
            return
        if value is None:
            s.inputs.pop(name, None)
        else:
            s.inputs[name] = value
        self._changed()

    @staticmethod
    def _documents() -> str:
        return QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
