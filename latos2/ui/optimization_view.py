"""The Optimization workspace: fit a Gaussian process and review a provisional next condition."""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import QStandardPaths, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
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
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QTableWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from latos2 import experiment as ex
from latos2.bo_figures import draw_acquisition, draw_surrogate
from latos2.gp import RESTARTS, Z95
from latos2.optimization import (
    STATUS,
    Candidate,
    Problem,
    ProblemError,
    Result,
    build_problem,
    default_settings,
    prepare,
    recommend,
)
from latos2.ui import style
from latos2.ui.experiment_view import _item, _muted, _number, _plain, _section
from latos2.ui.panels import PlotPage, _fit_table_height, _label, _styled

_GOALS = {"maximize": "Maximize", "minimize": "Minimize", "target": "Reach a target"}
_ACQUISITIONS = {"ei": "Expected improvement", "explore": "Largest model uncertainty (explore)"}
_TABS = ("Observations", "Surrogate Model", "Acquisition and Recommendation")


def _value(output: ex.OutputDefinition, value: float | None, signed: bool = True) -> str:
    """Analyzer values to four decimals; other outputs to six significant digits."""
    if value is None:
        return "—"
    if output.method.startswith("hioki:"):
        return f"{value:+.4f}" if signed else f"{value:.4f}"
    return f"{value:.6g}"


def _condition(problem: Problem, c: Candidate) -> str:
    parts = []
    if c.fractions:
        parts += [f"{name}: {x:.4f}" for name, x in c.fractions.items()]
    for v, value in zip(problem.space.variables, c.values):
        if v.kind != "fraction":
            parts.append(f"{v.label}: {value:.6g}" + (f" {v.unit}" if v.unit else ""))
    return "; ".join(parts)


class RecordDialog(QDialog):
    """The full model record, as JSON, for review or for keeping with lab notes."""

    def __init__(self, parent: QWidget, record: dict) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model record")
        self.resize(640, 560)
        layout = QVBoxLayout(self)
        layout.addWidget(_label(STATUS + ". Nothing is frozen or authorised by this record.", "muted", True))
        self.text = QPlainTextEdit(json.dumps(record, indent=2, ensure_ascii=False))
        self.text.setReadOnly(True)
        layout.addWidget(self.text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy = buttons.addButton("Copy", QDialogButtonBox.ButtonRole.ActionRole)
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.text.toPlainText()))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class OptimizationView(QWidget):
    """Set up the search, fit the model, and inspect the provisional recommendation."""

    changed = Signal()  # the researcher edited settings stored in the experiment
    statusMessage = Signal(str)

    def __init__(self, experiment: Callable[[], ex.Experiment]) -> None:
        super().__init__()
        self.experiment = experiment
        self.problem: Problem | None = None
        self.result: Result | None = None
        self._fingerprint: str | None = None
        self._loading = False
        self.bound_edits: dict[str, tuple[QLineEdit, QLineEdit]] = {}

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_explorer())
        splitter.addWidget(self._build_centre())
        splitter.addWidget(self._build_panel())
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([230, 760, 410])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    # -- construction -------------------------------------------------------

    def _build_explorer(self) -> QWidget:
        panel = _styled(QWidget(), "panel")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(_label("Optimization Explorer", "panelTitle"))
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemSelectionChanged.connect(self._on_tree)
        layout.addWidget(self.tree)
        outer.addWidget(body, 1)
        panel.setMinimumWidth(200)
        return panel

    def _plot_tab(self) -> tuple[QWidget, PlotPage, QVBoxLayout]:
        page = _styled(QWidget(), "page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        toolbar = _styled(QWidget(), "plotToolbar")
        bar = QHBoxLayout(toolbar)
        bar.setContentsMargins(6, 3, 6, 3)
        bar.addStretch(1)
        save = QToolButton()
        save.setText("Save figure…")
        save.setToolTip("Save this graph as PNG (300 dpi) or SVG")
        save.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        bar.addWidget(save)
        layout.addWidget(toolbar)
        plot = PlotPage()
        save.clicked.connect(lambda _checked=False, p=plot: self._save_figure(p))
        layout.addWidget(plot, 1)
        return page, plot, layout

    def _build_centre(self) -> QWidget:
        centre = _styled(QWidget(), "workspace")
        self.tabs = QTabBar()
        self.tabs.setObjectName("documentTabs")
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        for title in _TABS:
            self.tabs.addTab(title)
        self.stack = QStackedWidget()

        page = _styled(QWidget(), "page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.observations = QTableWidget()
        self.observations.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.observations.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.observations.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.observations.setAlternatingRowColors(True)
        self.observations.setWordWrap(False)
        self.observations.verticalHeader().setVisible(False)
        self.observations.verticalHeader().setDefaultSectionSize(24)
        self.observations.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.observations.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.observations, 1)
        self.observations_caption = _label("", "muted", True)
        layout.addWidget(self.observations_caption)
        self.stack.addWidget(page)

        page, self.surrogate_plot, layout = self._plot_tab()
        self.surrogate_caption = _label("", "muted", True)
        self.surrogate_caption.setContentsMargins(8, 4, 8, 6)
        layout.addWidget(self.surrogate_caption)
        self.stack.addWidget(page)

        page, self.acquisition_plot, layout = self._plot_tab()
        holder = QWidget()
        lower = QVBoxLayout(holder)
        lower.setContentsMargins(8, 4, 8, 8)
        lower.setSpacing(4)
        lower.addWidget(_label("Noise sensitivity: the same data and settings under other observation-noise assumptions", "sectionTitle"))
        self.sensitivity = QTableWidget(0, 7)
        self.sensitivity.setHorizontalHeaderLabels(
            ["σ_add", "σ_add / SD(y)", "Candidate", "Predicted ± SD", "Length-scale", "LOO RMSE", "LOO in 95 %"]
        )
        self.sensitivity.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sensitivity.verticalHeader().setVisible(False)
        self.sensitivity.verticalHeader().setDefaultSectionSize(22)
        self.sensitivity.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sensitivity.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.sensitivity.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sensitivity.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sensitivity.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lower.addWidget(self.sensitivity)
        self.sensitivity_note = _label("", "muted", True)
        lower.addWidget(self.sensitivity_note)
        layout.addWidget(holder)
        self.stack.addWidget(page)

        self.tabs.currentChanged.connect(self.stack.setCurrentIndex)
        layout = QVBoxLayout(centre)
        layout.setContentsMargins(6, 6, 6, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tabs)
        layout.addWidget(self.stack, 1)
        centre.setMinimumWidth(480)
        return centre

    def _build_panel(self) -> QWidget:
        panel = _styled(QWidget(), "panel")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(_label("Model and Recommendation", "panelTitle"))
        body = _styled(QWidget(), "panel")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(10, 2, 10, 10)
        layout.setSpacing(4)

        _section(layout, "Setup")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.output_combo = QComboBox()
        self.output_combo.currentIndexChanged.connect(self._on_output)
        self.goal_combo = QComboBox()
        for key, text in _GOALS.items():
            self.goal_combo.addItem(text, key)
        self.goal_combo.currentIndexChanged.connect(self._on_goal)
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("target value")
        self.target_edit.editingFinished.connect(self._on_target)
        goal = QHBoxLayout()
        goal.addWidget(self.goal_combo, 1)
        goal.addWidget(self.target_edit, 1)
        form.addRow("Output", self.output_combo)
        form.addRow("Goal", goal)
        layout.addLayout(form)
        layout.addWidget(_label("Search domain (mass fractions)", "muted"))
        self.domain_box = QFormLayout()
        self.domain_box.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addLayout(self.domain_box)
        self.domain_note = _label("", "muted", True)
        layout.addWidget(self.domain_note)
        self.outside_check = QCheckBox("Also fit specimens outside the search domain")
        self.outside_check.toggled.connect(self._on_outside)
        layout.addWidget(self.outside_check)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.noise_edit = QLineEdit()
        self.noise_edit.setToolTip(
            "Added in quadrature to every specimen's measured SEM. The SEM of repeated cycles measures "
            "within-specimen repeatability only, so 0 is a lower bound on the observation noise."
        )
        self.noise_edit.editingFinished.connect(self._on_noise)
        self.noise_unit = _label("", "muted")
        noise = QHBoxLayout()
        noise.addWidget(self.noise_edit, 1)
        noise.addWidget(self.noise_unit)
        self.acquisition_combo = QComboBox()
        for key, text in _ACQUISITIONS.items():
            self.acquisition_combo.addItem(text, key)
        self.acquisition_combo.currentIndexChanged.connect(self._on_acquisition)
        self.separation_edit = QLineEdit()
        self.separation_edit.setToolTip("Smallest distance from any measured condition, as a fraction of each input's search range")
        self.separation_edit.editingFinished.connect(self._on_separation)
        self.replicates_check = QCheckBox("Allow replicates")
        self.replicates_check.setToolTip("Allow recommending a condition that has already been measured")
        self.replicates_check.toggled.connect(self._on_replicates)
        separation = QHBoxLayout()
        separation.addWidget(self.separation_edit, 1)
        separation.addWidget(_label("× range", "muted"))
        separation.addWidget(self.replicates_check)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.valueChanged.connect(self._on_seed)
        form.addRow("Additional noise", noise)
        form.addRow("Acquisition", self.acquisition_combo)
        form.addRow("Min. separation", separation)
        form.addRow("Random seed", self.seed_spin)
        layout.addLayout(form)
        self.fit_button = QPushButton("Fit model and recommend")
        self.fit_button.setDefault(True)
        self.fit_button.clicked.connect(lambda: self.fit())
        layout.addWidget(self.fit_button)
        self.setup_message = _label("", wrap=True)
        self.setup_message.setStyleSheet(f"color: {style.ERROR};")
        self.setup_message.hide()
        layout.addWidget(self.setup_message)

        self.result_box = QWidget()
        box = QVBoxLayout(self.result_box)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        _section(box, "Provisional Recommendation")
        box.addWidget(_label(STATUS + ".", "warning", True))
        self.candidate_label = _label("", wrap=True)
        self.candidate_label.setTextFormat(Qt.TextFormat.RichText)
        self.candidate_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(self.candidate_label)
        self.explanation_label = _label("", "muted", True)
        box.addWidget(self.explanation_label)

        _section(box, "Preparation")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.total_edit = QLineEdit()
        self.total_edit.setPlaceholderText("enter the total to prepare")
        self.total_edit.editingFinished.connect(self._on_preparation)
        self.resolution_edit = QLineEdit()
        self.resolution_edit.editingFinished.connect(self._on_preparation)
        total = QHBoxLayout()
        total.addWidget(self.total_edit, 1)
        total.addWidget(_label("g", "muted"))
        resolution = QHBoxLayout()
        resolution.addWidget(self.resolution_edit, 1)
        resolution.addWidget(_label("g", "muted"))
        form.addRow("Total mass", total)
        form.addRow("Balance resolution", resolution)
        box.addLayout(form)
        self.masses_table = QTableWidget(0, 3)
        self.masses_table.setHorizontalHeaderLabels(["Component", "Mass to weigh (g)", "Resulting fraction"])
        self.masses_table.verticalHeader().setVisible(False)
        self.masses_table.verticalHeader().setDefaultSectionSize(22)
        self.masses_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.masses_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.masses_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.masses_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.masses_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        box.addWidget(self.masses_table)
        self.preparation_note = _label("", "muted", True)
        box.addWidget(self.preparation_note)

        _section(box, "Baseline Comparison")
        self.baseline_label = _label("", wrap=True)
        self.baseline_label.setTextFormat(Qt.TextFormat.RichText)
        box.addWidget(self.baseline_label)

        _section(box, "Model")
        self.model_label = _label("", wrap=True)
        self.model_label.setTextFormat(Qt.TextFormat.RichText)
        box.addWidget(self.model_label)

        _section(box, "Small-Data Evaluation")
        self.evaluation_label = _label("", wrap=True)
        self.evaluation_label.setTextFormat(Qt.TextFormat.RichText)
        box.addWidget(self.evaluation_label)

        _section(box, "Warnings")
        self.warnings_label = _label("", wrap=True)
        self.warnings_label.setTextFormat(Qt.TextFormat.RichText)
        box.addWidget(self.warnings_label)
        record = QPushButton("Show model record…")
        record.clicked.connect(lambda: self.result and RecordDialog(self, self.result.record).exec())
        box.addWidget(record, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.result_box)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)
        panel.setMinimumWidth(340)
        return panel

    # -- public operations --------------------------------------------------

    @property
    def settings(self) -> ex.OptimizationSettings:
        return self.experiment().optimization

    def refresh(self) -> None:
        """Show the current experiment; a result survives only if the experiment is unchanged."""
        exp = self.experiment()
        if exp.optimization is None:
            exp.optimization = default_settings(exp)
        components = exp.mixture.components if exp.mixture else []
        bounds = exp.optimization.fraction_bounds
        for c in [c for c in bounds if c not in components]:
            del bounds[c]
        for c in components:
            bounds.setdefault(c, [0.0, 1.0])
        # No output chosen yet (the workspace was opened before any was defined) or
        # the chosen one no longer exists: take the default, which the Output box shows.
        if exp.output(exp.optimization.output or "") is None:
            exp.optimization.output = default_settings(exp).output
        fingerprint = self._fingerprint_of(exp)
        if fingerprint != self._fingerprint:
            self.result = None
            self._fingerprint = fingerprint
        self._load_settings()
        self._update_problem()
        self._show_result()

    def set_fraction_bounds(self, component: str, lower: float, upper: float) -> None:
        """Set a component's search domain, as typing it does."""
        lo, hi = self.bound_edits[component]
        lo.setText(_plain(lower))
        hi.setText(_plain(upper))
        self._on_bounds(component)

    def fit(self) -> bool:
        """Fit the model and select the next condition. True when a recommendation was made."""
        if self.problem is None:
            return False
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.result = recommend(self.problem, ex.check_sources(self.experiment()))
        except ProblemError as exc:
            self.result = None
            self._setup_error(str(exc))
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self._show_result()
        self.tabs.setCurrentIndex(1)
        self.statusMessage.emit(f"Model fitted to {len(self.problem.used)} specimens; {STATUS.lower()}.")
        return True

    @staticmethod
    def _fingerprint_of(exp: ex.Experiment) -> str:
        data = asdict(exp)
        data.pop("optimization", None)
        return f"{id(exp)}:{json.dumps(data, sort_keys=True, default=str)}"

    # -- settings -----------------------------------------------------------

    def _load_settings(self) -> None:
        exp, s = self.experiment(), self.settings
        self._loading = True
        try:
            self.output_combo.clear()
            for o in exp.outputs:
                self.output_combo.addItem(f"{o.name} ({o.unit})" if o.unit else o.name, o.name)
            self.output_combo.setCurrentIndex(max(0, self.output_combo.findData(s.output)))
            output = exp.output(s.output) if s.output else None
            self.goal_combo.setEnabled(output is not None)
            self.goal_combo.setCurrentIndex(self.goal_combo.findData(output.objective) if output else 0)
            self.target_edit.setVisible(output is not None and output.objective == "target")
            self.target_edit.setText(_plain(output.target) if output else "")
            self.noise_unit.setText(output.unit if output else "")

            while self.domain_box.rowCount():
                self.domain_box.removeRow(0)
            self.bound_edits = {}
            components = exp.mixture.components if exp.mixture else []
            for c in components:
                lo, hi = (QLineEdit(_plain(b)) for b in s.fraction_bounds[c])
                for edit in (lo, hi):
                    edit.editingFinished.connect(lambda c=c: self._on_bounds(c))
                row = QHBoxLayout()
                row.addWidget(lo, 1)
                row.addWidget(_label("to", "muted"))
                row.addWidget(hi, 1)
                self.domain_box.addRow(c, row)
                self.bound_edits[c] = (lo, hi)
            notes = []
            if len(components) > 1:
                free = ", ".join(components[:-1])
                notes.append(f"Optimized: {free}; {components[-1]} is the balance, 1 − Σ x.")
            elif not components:
                notes.append("No mixture is defined.")
            for v in exp.inputs:
                if v.kind == "categorical":
                    notes.append(f"{v.name}: categorical inputs are not yet supported.")
                elif v.kind == "discrete":
                    notes.append(f"{v.name}: one of {', '.join(v.levels) or '—'} (Experiment ▸ Definition).")
                else:
                    unit = f" {v.unit}" if v.unit else ""
                    notes.append(f"{v.name}: {_plain(v.lower) or '?'} to {_plain(v.upper) or '?'}{unit} (Experiment ▸ Definition).")
            self.domain_note.setText("\n".join(notes))
            self.outside_check.setChecked(s.include_outside)
            self.noise_edit.setText(_plain(s.additional_noise))
            self.acquisition_combo.setCurrentIndex(max(0, self.acquisition_combo.findData(s.acquisition)))
            self.separation_edit.setText(_plain(s.min_separation))
            self.replicates_check.setChecked(s.allow_replicates)
            self.separation_edit.setEnabled(not s.allow_replicates)
            self.seed_spin.setValue(s.seed)
            self.total_edit.setText(_plain(s.preparation_total_g))
            self.resolution_edit.setText(_plain(s.balance_resolution_g))
        finally:
            self._loading = False

    def _settings_changed(self, refit_needed: bool = True) -> None:
        if refit_needed:
            self.result = None
        self.changed.emit()
        self._fingerprint = self._fingerprint_of(self.experiment())
        self._load_settings()
        self._update_problem()
        self._show_result()

    def _setup_error(self, text: str) -> None:
        self.setup_message.setText(text[:1].upper() + text[1:] + ("" if text.endswith(".") else "."))
        self.setup_message.show()

    def _on_output(self) -> None:
        if not self._loading:
            self.settings.output = self.output_combo.currentData()
            self._settings_changed()

    def _on_goal(self) -> None:
        output = self.experiment().output(self.settings.output or "")
        if self._loading or output is None:
            return
        output.objective = self.goal_combo.currentData()
        if output.objective != "target":
            output.target = None
        self._settings_changed()

    def _on_target(self) -> None:
        output = self.experiment().output(self.settings.output or "")
        if self._loading or output is None:
            return
        try:
            value = _number(self.target_edit.text())
        except ValueError as exc:
            self._setup_error(str(exc))
            return
        if value != output.target:
            output.target = value
            self._settings_changed()

    def _on_bounds(self, component: str) -> None:
        if self._loading:
            return
        lo_edit, hi_edit = self.bound_edits[component]
        try:
            lo, hi = _number(lo_edit.text()), _number(hi_edit.text())
            if lo is None or hi is None or not 0 <= lo <= hi <= 1:
                raise ValueError(f"the bounds of {component} must satisfy 0 ≤ minimum ≤ maximum ≤ 1")
        except ValueError as exc:
            self._setup_error(str(exc))
            lo_edit.setText(_plain(self.settings.fraction_bounds[component][0]))
            hi_edit.setText(_plain(self.settings.fraction_bounds[component][1]))
            return
        if [lo, hi] != self.settings.fraction_bounds[component]:
            self.settings.fraction_bounds[component] = [lo, hi]
            self._settings_changed()

    def _on_outside(self, checked: bool) -> None:
        if not self._loading:
            self.settings.include_outside = checked
            self._settings_changed()

    def _on_noise(self) -> None:
        if self._loading:
            return
        try:
            value = _number(self.noise_edit.text()) or 0.0
            if value < 0:
                raise ValueError("the additional noise cannot be negative")
        except ValueError as exc:
            self._setup_error(str(exc))
            self.noise_edit.setText(_plain(self.settings.additional_noise))
            return
        if value != self.settings.additional_noise:
            self.settings.additional_noise = value
            self._settings_changed()

    def _on_acquisition(self) -> None:
        if not self._loading:
            self.settings.acquisition = self.acquisition_combo.currentData()
            self._settings_changed()

    def _on_separation(self) -> None:
        if self._loading:
            return
        try:
            value = _number(self.separation_edit.text()) or 0.0
            if not 0 <= value < 1:
                raise ValueError("the minimum separation must be at least 0 and below 1 (a fraction of the range)")
        except ValueError as exc:
            self._setup_error(str(exc))
            self.separation_edit.setText(_plain(self.settings.min_separation))
            return
        if value != self.settings.min_separation:
            self.settings.min_separation = value
            self._settings_changed()

    def _on_replicates(self, checked: bool) -> None:
        if not self._loading:
            self.settings.allow_replicates = checked
            self._settings_changed()

    def _on_seed(self, value: int) -> None:
        if not self._loading and value != self.settings.seed:
            self.settings.seed = value
            self._settings_changed()

    def _on_preparation(self) -> None:
        if self._loading:
            return
        try:
            total = _number(self.total_edit.text())
            resolution = _number(self.resolution_edit.text())
            if (total is not None and total <= 0) or resolution is None or resolution <= 0:
                raise ValueError("the total mass and the balance resolution must be positive")
        except ValueError as exc:
            self.preparation_note.setText(str(exc).capitalize() + ".")
            return
        s = self.settings
        if (total, resolution) != (s.preparation_total_g, s.balance_resolution_g):
            s.preparation_total_g, s.balance_resolution_g = total, resolution
            self.changed.emit()
        self._show_preparation()

    # -- display ------------------------------------------------------------

    def _update_problem(self) -> None:
        self.setup_message.hide()
        try:
            self.problem = build_problem(self.experiment(), self.settings)
        except ProblemError as exc:
            self.problem = None
            self._setup_error(str(exc))
        self.fit_button.setEnabled(self.problem is not None)
        if self.problem is not None and len(self.problem.used) < self.problem.minimum_needed:
            self._setup_error(
                f"{len(self.problem.used)} independent specimen(s) can be used; at least "
                f"{self.problem.minimum_needed} are needed to fit a model"
            )
            self.fit_button.setEnabled(False)
        self._fill_observations()
        self._fill_tree()

    def _fill_observations(self) -> None:
        table, p, r = self.observations, self.problem, self.result
        table.clear()
        if p is None:
            table.setRowCount(0)
            table.setColumnCount(0)
            self.observations_caption.setText("")
            return
        output = p.output
        components = list(p.space.components)
        inputs = [v for v in p.space.variables if v.kind != "fraction"]
        unit = f" ({output.unit})" if output.unit else ""
        headers = ["Specimen", "Nominal label", *(f"{c}\nmass fraction" for c in components)]
        headers += [f"{v.label}\n({v.unit})" if v.unit else v.label for v in inputs]
        headers += [f"{output.name}{unit}\nmean ± SEM", "Cycles", "Session", "In the fit",
                    f"Leave-one-out\nprediction ± SD", "LOO error", "z"]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(p.observations))
        loo_index = {o.specimen_id: i for i, o in enumerate(p.used)}
        for row, o in enumerate(p.observations):
            cells = [_item(o.specimen_id), _item(o.label)]
            for c in components:
                cells.append(_item("—" if o.fractions is None else f"{o.fractions[c]:.4f}", numeric=True))
            for k, v in enumerate(inputs, start=len(p.space.free)):
                cells.append(_item("—" if o.values is None else f"{o.values[k]:.6g}", numeric=True))
            response = _value(output, o.response)
            if o.response is not None and o.sem is not None:
                response += f" ± {_value(output, o.sem, signed=False)}"
            cells.append(_item(response, numeric=True))
            cells.append(_item("—" if o.n is None else str(o.n), numeric=True))
            cells.append(_item(o.session, tip=o.source))
            fit = _item("Yes" if o.used else "No", tip=o.reason)
            fit.setForeground(QColor(style.OK if o.used else style.CAUTION))
            cells.append(fit)
            if r is not None and r.loo is not None and o.specimen_id in loo_index:
                i = loo_index[o.specimen_id]
                cells.append(_item(f"{_value(output, r.loo.predicted[i])} ± {_value(output, r.loo.predictive_sd[i], False)}", numeric=True))
                cells.append(_item(_value(output, r.loo.errors[i]), numeric=True))
                z = _item(f"{r.loo.z[i]:+.2f}", numeric=True)
                if not r.loo.inside[i]:
                    z.setForeground(QColor(style.CAUTION))
                cells.append(z)
            else:
                cells += [_muted(_item("—", numeric=True), False) for _ in range(3)]
            for col, item in enumerate(cells):
                if not o.used and col != headers.index("In the fit"):
                    _muted(item, italic=False)
                table.setItem(row, col, item)
        text = (
            f"{len(p.used)} of {len(p.observations)} specimens are in the fit; each specimen is one observation, "
            "the mean of its accepted cycles in its primary session. Mass fractions are measured (from the weighed "
            "masses); the nominal label is the lab record's name."
        )
        left_out = [o for o in p.observations if not o.used]
        if left_out:
            text += " Not in the fit: " + "; ".join(f"{o.specimen_id}, {o.reason}" for o in left_out) + "."
        if r is not None and r.loo is not None:
            text += (
                " Leave-one-out: each specimen predicted by a model refitted without it; z is the error over the "
                "predictive SD (model SD and the specimen's assumed noise). Predictions at the ends of the range "
                "are extrapolations."
            )
        self.observations_caption.setText(text)

    def _fill_tree(self) -> None:
        tree, p, exp = self.tree, self.problem, self.experiment()
        tree.blockSignals(True)
        tree.clear()
        root = QTreeWidgetItem([exp.name])
        bold = QFont(root.font(0))
        bold.setBold(True)
        root.setFont(0, bold)
        tree.addTopLevelItem(root)
        if p is not None:
            o = p.output
            target = f" {o.target:g}" if o.objective == "target" and o.target is not None else ""
            QTreeWidgetItem(root, [f"Output: {o.name} ({o.unit}), {_GOALS[o.objective].lower()}{target}"])
            variables = QTreeWidgetItem(root, [f"Optimization variables: {p.space.d}"])
            for v in p.space.variables:
                QTreeWidgetItem(variables, [f"{v.label}: {v.lower:.6g} to {v.upper:.6g}"])
            if len(p.space.components) > 1:
                QTreeWidgetItem(variables, [f"Balance: {p.space.components[-1]}"])
            used = QTreeWidgetItem(root, [f"In the fit: {len(p.used)}"])
            for obs in p.used:
                item = QTreeWidgetItem(used, [obs.specimen_id])
                item.setData(0, Qt.ItemDataRole.UserRole, obs.specimen_id)
            left_out = [obs for obs in p.observations if not obs.used]
            out = QTreeWidgetItem(root, [f"Not in the fit: {len(left_out)}"])
            for obs in left_out:
                item = QTreeWidgetItem(out, [obs.specimen_id])
                item.setToolTip(0, obs.reason)
                item.setForeground(0, QColor(style.MUTED))
                item.setData(0, Qt.ItemDataRole.UserRole, obs.specimen_id)
            if self.result is not None:
                c = self.result.choice
                item = QTreeWidgetItem(root, ["Provisional candidate: " + _condition(p, c).split(";")[0]])
                item.setForeground(0, QColor(style.HEATING))
        items = [root]
        while items:
            item = items.pop()
            item.setToolTip(0, item.toolTip(0) or item.text(0))
            items += [item.child(i) for i in range(item.childCount())]
        tree.expandAll()
        tree.blockSignals(False)

    def _on_tree(self) -> None:
        item = self.tree.currentItem()
        specimen = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if specimen is None:
            return
        self.tabs.setCurrentIndex(0)
        for row in range(self.observations.rowCount()):
            if self.observations.item(row, 0).text() == specimen:
                self.observations.selectRow(row)

    def _show_result(self) -> None:
        r, p = self.result, self.problem
        self.result_box.setVisible(r is not None)
        self._fill_observations()
        self._fill_tree()
        if r is None:
            message = (
                "Fit the model to see the GP prediction"
                if p is not None and self.fit_button.isEnabled() else (self.setup_message.text() or "Set up the search")
            )
            for plot in (self.surrogate_plot, self.acquisition_plot):
                plot.placeholder(message)
                plot.refresh()
            self.surrogate_caption.setText("")
            self.sensitivity.setRowCount(0)
            _fit_table_height(self.sensitivity)
            self.sensitivity_note.setText("")
            return
        draw_surrogate(self.surrogate_plot.figure, r)
        draw_acquisition(self.acquisition_plot.figure, r)
        for plot in (self.surrogate_plot, self.acquisition_plot):
            plot.refresh()
        output = p.output
        unit = f" {output.unit}" if output.unit else ""
        self.surrogate_caption.setText(
            f"Gaussian process: amplitude² × Matérn (ν = 5/2). Observation noise per specimen: SEM² + σ_add², "
            f"σ_add = {r.gp.additional_noise:g}{unit}. The band is the model's own 95 % interval for the mean "
            "response, mean ± 1.96 SD; it is illustrative, and its coverage has not been demonstrated."
        )

        c, b = r.choice, r.baseline
        lo, hi = c.mean - Z95 * c.sd, c.mean + Z95 * c.sd
        self.candidate_label.setText(
            f"<b>{html.escape(_condition(p, c))}</b><br>"
            f"Predicted {html.escape(output.name)}: <b>{_value(output, c.mean)}</b> ± {_value(output, c.sd, False)}"
            f"{html.escape(unit)} (model SD)<br>"
            f"Model-based 95 % interval: {_value(output, lo)} to {_value(output, hi)}{html.escape(unit)} (illustrative)<br>"
            f"SD for a new specimen's measurement: {_value(output, c.predictive_sd, False)}{html.escape(unit)} "
            "(model SD with σ_add and the median measured SEM)<br>"
            f"{html.escape(c.principle)}: {c.score:.4g}"
        )
        self.explanation_label.setText(r.explanation)
        self._show_preparation()
        self.baseline_label.setText(
            "<table cellspacing='0' cellpadding='2'>"
            "<tr><th align='left'>Strategy</th><th align='left'>Condition</th><th align='left'>Predicted</th></tr>"
            f"<tr><td>GP / {html.escape(c.principle)}</td><td>{html.escape(_condition(p, c).split(';')[0])}</td>"
            f"<td>{_value(output, c.mean)} ± {_value(output, c.sd, False)}</td></tr>"
            f"<tr><td>Space-filling baseline</td><td>{html.escape(_condition(p, b).split(';')[0])}</td>"
            f"<td>{_value(output, b.mean)} ± {_value(output, b.sd, False)}</td></tr></table>"
            f"<span style='color:{style.MUTED}'>The GP candidate is chosen by {html.escape(c.principle.lower())} under the "
            "model; the baseline is the allowed condition farthest from every measured one and ignores the "
            "responses. This compares proposed decisions; it is not evidence that either performs better.</span>"
        )

        variables = p.space.variables
        scales = ", ".join(
            f"{ls:.3g} × range (= {ls * v.span:.3g} in {html.escape(v.label)})" for ls, v in zip(r.gp.length_scales, variables)
        )
        bounds = "".join(f"<br><span style='color:{style.CAUTION}'>⚠ {html.escape(b)}</span>" for b in r.gp.at_bounds)
        self.model_label.setText(
            "Kernel: amplitude² × Matérn (ν = 5/2), one length-scale per input<br>"
            "Inputs scaled to [0, 1] over the search range; responses centred and scaled<br>"
            f"Fitted length-scale: {scales}{bounds}<br>"
            f"Fitted amplitude: {r.gp.amplitude:.3g}{html.escape(unit)}<br>"
            f"Observation noise: SEM² + σ_add², σ_add = {r.gp.additional_noise:g}{html.escape(unit)} (fixed, not fitted)<br>"
            f"Hyperparameters by maximum marginal likelihood, {RESTARTS} seeded restarts, seed {p.settings.seed}<br>"
            f"Candidates: {len(r.candidates)}, of which {int(r.excluded.sum())} lie within the minimum separation"
        )

        n = len(r.y)
        rows = [
            f"Independent physical specimens: {n} (repeated cycles are not counted)",
            f"Optimization variables: {p.space.d}",
        ]
        if p.space.d == 1:
            v = variables[0]
            rows.append(f"{v.label} values fitted: " + ", ".join(f"{x:.4g}" for x in sorted(r.x[:, 0])))
            rows.append(f"Farthest any allowed value lies from a fitted one: {r.largest_gap * v.span:.3g}")
        else:
            rows.append(f"Largest unsampled region: radius {r.largest_gap:.3g} × range")
        if r.loo is not None:
            rows.append(f"Leave-one-out RMSE: {_value(output, r.loo.rmse, False)}{unit}")
            rows.append(
                f"Inside the nominal 95 % predictive interval: {int(r.loo.inside.sum())} of {n}. With {n} specimens "
                "this count cannot confirm or refute the interval's calibration."
            )
        if p.space.d == 1:
            xs = [row.values[0] for row in r.sensitivity]
            rows.append(
                f"Noise sensitivity: across σ_add = {r.sensitivity[0].additional_noise:.3g} to "
                f"{r.sensitivity[-1].additional_noise:.3g}{unit} the candidate lies between "
                f"{min(xs):.4f} and {max(xs):.4f}."
            )
        rows.append("Training-fit accuracy is not shown: it is not evidence of predictive accuracy.")
        self.evaluation_label.setText("<br>".join(html.escape(t) for t in rows))
        self.warnings_label.setText(
            "<br>".join(f"<span style='color:{style.CAUTION}'>⚠</span> {html.escape(w)}" for w in r.warnings)
        )
        self._fill_sensitivity()

    def _fill_sensitivity(self) -> None:
        r, table = self.result, self.sensitivity
        output = r.problem.output
        unit = f" {output.unit}" if output.unit else ""
        table.setRowCount(len(r.sensitivity))
        for row, s in enumerate(r.sensitivity):
            condition = ", ".join(f"{v:.4f}" for v in s.values)
            cells = [
                _item(f"{s.additional_noise:.4g}{unit}" + ("  (selected)" if s.selected else ""), numeric=True),
                _item("—" if s.fraction_of_sd is None else f"{s.fraction_of_sd:g}", numeric=True),
                _item(condition, numeric=True),
                _item(f"{_value(output, s.mean)} ± {_value(output, s.sd, False)}", numeric=True),
                _item(", ".join(f"{v:.3g}" for v in s.length_scales), numeric=True),
                _item("—" if s.loo_rmse is None else _value(output, s.loo_rmse, False), numeric=True),
                _item("—" if s.loo_inside is None else f"{s.loo_inside} of {s.n}", numeric=True),
            ]
            for col, item in enumerate(cells):
                if s.selected:
                    font = QFont(item.font())
                    font.setBold(True)
                    item.setFont(font)
                table.setItem(row, col, item)
        _fit_table_height(table)
        self.sensitivity_note.setText(
            "σ_add levels are fractions of the SD of the observed responses, a scale reference rather than a noise "
            "estimate. The SEM of each specimen is always included."
        )

    def _show_preparation(self) -> None:
        r, s = self.result, self.settings
        table = self.masses_table
        table.setRowCount(0)
        if r is None:
            return
        fractions = r.choice.fractions
        if not fractions:
            self.preparation_note.setText("No mixture: there is nothing to weigh.")
            _fit_table_height(table)
            return
        if s.preparation_total_g is None:
            self.preparation_note.setText(
                "Enter the total mass to prepare. The recommendation is a composition; Latos does not choose "
                "the total, so no masses are given yet."
            )
            _fit_table_height(table)
            return
        try:
            prep = prepare(self.problem.space, fractions, s.preparation_total_g, s.balance_resolution_g)
        except ProblemError as exc:
            self.preparation_note.setText(str(exc).capitalize() + ".")
            _fit_table_height(table)
            return
        decimals = max(0, -int(f"{prep.resolution_g:e}".split("e")[1]))
        table.setRowCount(len(prep.masses_g))
        for row, (c, mass) in enumerate(prep.masses_g.items()):
            table.setItem(row, 0, _item(c))
            table.setItem(row, 1, _item(f"{mass:.{decimals}f}", numeric=True))
            table.setItem(row, 2, _item(f"{prep.fractions[c]:.4f}", numeric=True))
        _fit_table_height(table)
        status = "" if prep.feasible else "⚠ "
        self.preparation_note.setText(
            status + f"Total {prep.total_g:.{decimals}f} g at a resolution of {prep.resolution_g:g} g. "
            + " ".join(prep.notes)
        )

    def _save_figure(self, plot: PlotPage) -> None:
        if self.result is None:
            return
        start = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        name = "surrogate_model.png" if plot is self.surrogate_plot else "acquisition.png"
        chosen, chosen_filter = QFileDialog.getSaveFileName(
            self, "Save figure", str(Path(start) / name), "PNG image, 300 dpi (*.png);;SVG vector, editable text (*.svg)"
        )
        if not chosen:
            return
        path = Path(chosen)
        if path.suffix.lower() not in (".png", ".svg"):
            path = path.with_suffix(".svg" if "SVG" in chosen_filter else ".png")
        svg = path.suffix.lower() == ".svg"
        plot.figure.savefig(path, dpi=300, metadata={"Date": None} if svg else None)
        self.statusMessage.emit(f"Saved {path}")
