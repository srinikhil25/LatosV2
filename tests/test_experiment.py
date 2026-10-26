"""Milestone 2 acceptance tests: experiment definition, specimens, sessions, save and reopen.

Reference values for the five IL specimens: forward leads (σ = +1), whole-cycle
S_app in mV/K from the Milestone 1 analyzer, with cycle 1 of each pure liquid
excluded as an initial test cycle per the lab record. Masses are the recorded
specimen masses (IL1 = Aliquat 336TG, IL2 = [HMIM][TFSI]); the pure liquids have
no recorded mass.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict
from functools import cache
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from latos2 import experiment as ex  # noqa: E402
from latos2.export import analysis_record  # noqa: E402
from latos2.hioki_reader import LoggerChannel, LoggerEvent, LoggerTrace, read_hioki_xlsx  # noqa: E402
from latos2.project_io import ProjectError, load_experiment, save_experiment  # noqa: E402
from latos2.thermovoltage import UNIT, ChannelRoles, CycleBoundaryError, analyze, trace_windows  # noqa: E402

DATA = Path(os.environ.get("LATOS2_IL_DATA", r"D:\Latos\data\ionic-liquid-campaign\Measurement files"))
ROLES = ChannelRoles(hot="CH1-A-3", cold="CH1-A-1", voltage="CH1-A-10")
INITIAL = {1: "Initial test cycle"}
needs_data = pytest.mark.skipif(not DATA.is_dir(), reason="authentic IL workbooks not available")

# stem: (pure component, masses in g, excluded cycles, mean, SEM, reversed-lead mean)
SPECIMENS = {
    "IL1_0-IL2_100": ("IL2", {}, INITIAL, -2.6434, 0.3703, -1.6778),
    "IL1_25-IL2_75": (None, {"IL1": 0.0609, "IL2": 0.1933}, {}, 2.0256, 0.0069, None),
    "IL1_50-IL2_50": (None, {"IL1": 0.0709, "IL2": 0.0764}, {}, 2.0187, 0.0769, 1.9305),
    "IL1_75-IL2_25": (None, {"IL1": 0.1940, "IL2": 0.0617}, {}, 2.1900, 0.0155, 2.2733),
    "IL1_100-IL2_0": ("IL1", {}, INITIAL, 1.5642, 0.0279, 1.8005),
}
X_IL1 = {"IL1_0-IL2_100": 0.0, "IL1_25-IL2_75": 0.239575, "IL1_50-IL2_50": 0.481331,
         "IL1_75-IL2_25": 0.758702, "IL1_100-IL2_0": 1.0}
TOL = 1e-4


@cache
def _trace(stem: str) -> LoggerTrace:
    return read_hioki_xlsx(DATA / f"{stem}.xlsx")


def _record(stem: str, sign: int, excluded: dict | None = None) -> dict:
    trace = _trace(stem)
    return analysis_record(analyze(trace, ROLES, sign, excluded), trace.sha256)


def _il_experiment() -> ex.Experiment:
    exp = ex.Experiment(
        name="IL binary mixtures",
        mixture=ex.Mixture(components=["IL1", "IL2"]),
        outputs=[ex.OutputDefinition("S_app", UNIT, "hioki:combined", "maximize")],
    )
    output = exp.outputs[0]
    for stem, (pure, masses, excluded, *_values) in SPECIMENS.items():
        specimen = ex.Specimen(stem, pure_component=pure, masses_g=dict(masses))
        exp.specimens.append(specimen)
        ex.attach(exp, specimen, ex.hioki_measurement(_record(stem, +1, excluded), output, "Forward leads", "primary"))
        if (DATA / f"{stem}-R.xlsx").exists():
            reversed_leads = ex.hioki_measurement(_record(f"{stem}-R", -1), output, "Reversed leads", "supplementary")
            ex.attach(exp, specimen, reversed_leads)
    return exp


def test_mass_fractions_for_any_number_of_components():
    for masses in ([0.1940, 0.0617], [1.0, 2.0, 1.0], [0.5, 0.25, 0.125, 0.125]):
        names = [f"C{i}" for i in range(len(masses))]
        exp = ex.Experiment(mixture=ex.Mixture(components=names))
        specimen = ex.Specimen("S1", masses_g=dict(zip(names, masses)))
        x = ex.fractions(exp, specimen)
        assert x == pytest.approx({n: m / sum(masses) for n, m in zip(names, masses)})
        assert sum(x.values()) == pytest.approx(1.0)
        assert ex.total_mass(exp, specimen) == pytest.approx(sum(masses))

    exp = ex.Experiment(mixture=ex.Mixture(components=["IL1", "IL2"]))
    # A pure liquid: fractions 1 and 0 by identity; its mass is not invented.
    pure = ex.Specimen("P", pure_component="IL2")
    assert ex.fractions(exp, pure) == {"IL1": 0.0, "IL2": 1.0}
    assert ex.total_mass(exp, pure) is None and pure.masses_g == {}
    assert ex.Message("note", "Pure IL2; mass not recorded") in ex.specimen_messages(exp, pure)
    pure.masses_g["IL2"] = 0.25  # entered later
    assert ex.total_mass(exp, pure) == 0.25 and ex.fractions(exp, pure) == {"IL1": 0.0, "IL2": 1.0}
    # A mixture with a missing mass has no fractions, and says why.
    partial = ex.Specimen("M", masses_g={"IL1": 0.07})
    assert ex.fractions(exp, partial) is None and ex.total_mass(exp, partial) is None
    assert ex.Message("warning", "Mass of IL2 not recorded") in ex.specimen_messages(exp, partial)
    # A fixed total mass: a deviation beyond the tolerance is flagged.
    exp.mixture.total_mode, exp.mixture.total_g, exp.mixture.tolerance_g = "fixed", 0.25, 0.005
    off = ex.Specimen("F1", masses_g={"IL1": 0.1940, "IL2": 0.0617})  # 0.2557 g
    within = ex.Specimen("F2", masses_g={"IL1": 0.125, "IL2": 0.127})  # 0.252 g
    assert any("differs from the fixed total" in m.text for m in ex.specimen_messages(exp, off))
    assert not ex.specimen_messages(exp, within)


def test_inputs_types_ranges_and_levels():
    temperature = ex.InputVariable("Annealing temperature", "continuous", "°C", 20.0, 150.0)
    speed = ex.InputVariable("Stirring speed", "discrete", "rpm", levels=["200", "400", "800"])
    solvent = ex.InputVariable("Solvent", "categorical", levels=["water", "ethanol"])
    assert ex.input_problem(temperature, 80.0) is None
    assert "above the allowed maximum" in ex.input_problem(temperature, 200.0)
    assert ex.input_problem(speed, 400.0) is None and "not one of" in ex.input_problem(speed, 300.0)
    assert ex.input_problem(solvent, "ethanol") is None and "not one of" in ex.input_problem(solvent, "acetone")
    assert ex.parse_input(temperature, " 85.5 ") == 85.5 and ex.parse_input(solvent, "water") == "water"
    assert ex.parse_input(temperature, "") is None
    with pytest.raises(ValueError):
        ex.parse_input(temperature, "hot")

    specimen = ex.Specimen("S1", inputs={"Annealing temperature": 200.0, "Solvent": "water"})
    exp = ex.Experiment(inputs=[temperature, speed, solvent], specimens=[specimen])
    warnings = [m.text for m in ex.specimen_messages(exp, specimen) if m.level == "warning"]
    assert "Stirring speed not entered" in warnings
    assert any("above the allowed maximum" in w for w in warnings)
    ex.rename_input(exp, "Solvent", "Medium")
    assert specimen.inputs == {"Annealing temperature": 200.0, "Medium": "water"}


def test_final_cooling_ends_at_last_recorded_sample():
    """D2 on a synthetic recording: an even number of marks ends at the final sample."""
    trace = _synthetic([10.0, 70.0, 130.0, 190.0], end_s=250.0)
    windows, end_of_run = trace_windows(trace)
    assert [w.index for w in windows] == [1, 2]
    assert windows[-1].end_s == trace.time_s[-1] == 250.0
    assert "final recorded sample" in end_of_run
    a = analyze(trace, SYNTHETIC_ROLES, +1)
    assert a.accepted == (1, 2) and a.reductions["combined"].mean == pytest.approx(2.0, abs=0.02)
    assert "final recorded sample" in analysis_record(a, trace.sha256)["end_of_run"]
    # A closing mark still ends the run, as before.
    windows, end_of_run = trace_windows(_synthetic([10.0, 70.0, 130.0, 190.0, 240.0], end_s=250.0))
    assert windows[-1].end_s == 240.0 and end_of_run == "event mark 5"
    # A last mark on the final sample leaves no cooling period: refused, not invented.
    with pytest.raises(CycleBoundaryError):
        trace_windows(_synthetic([10.0, 70.0, 130.0, 250.0], end_s=250.0))
    # D1 mechanics: an excluded cycle stays in the analysis but leaves the mean.
    a = analyze(trace, SYNTHETIC_ROLES, +1, {1: "Initial test cycle"})
    assert len(a.cycles) == 2 and a.accepted == (2,) and a.excluded == {1: "Initial test cycle"}
    assert a.reductions["combined"].k == 1 and a.reductions["combined"].sem is None
    with pytest.raises(ValueError, match="cannot be excluded"):
        analyze(trace, SYNTHETIC_ROLES, +1, {3: "no such cycle"})


@needs_data
def test_five_il_specimens():
    exp = _il_experiment()
    assert [s.id for s in exp.specimens] == list(SPECIMENS)
    for specimen in exp.specimens:
        pure, masses, excluded, mean, sem, reversed_mean = SPECIMENS[specimen.id]
        x = ex.fractions(exp, specimen)
        assert x["IL1"] == pytest.approx(X_IL1[specimen.id], abs=1e-6)
        assert x["IL1"] + x["IL2"] == pytest.approx(1.0)
        primary = ex.primary_measurement(exp, specimen, "S_app")
        # The response is this specimen's own forward recording.
        assert primary.source_path.stem == specimen.id and primary.label == "Forward leads"
        assert primary.record["declarations"]["wiring_sign"] == +1
        assert primary.value == pytest.approx(mean, abs=TOL) and primary.sem == pytest.approx(sem, abs=TOL)
        assert primary.n == 3
        # D1: the pure liquids leave out their initial test cycle, which stays in the record.
        assert primary.record["excluded_cycles"] == {str(k): v for k, v in excluded.items()}
        if excluded:
            assert len(primary.record["cycles"]) == 4
            assert primary.record["cycles"][0]["excluded_reason"] == "Initial test cycle"
            assert ex.Message("note", "Cycle 1 excluded: Initial test cycle") in ex.specimen_messages(exp, specimen)
        # D3: no mass is invented for a pure liquid.
        assert ex.total_mass(exp, specimen) == (None if pure else pytest.approx(sum(masses.values())))
        # D4: reversed leads are a separate, supplementary session of the same specimen.
        others = [m for m in specimen.measurements if m is not primary]
        if reversed_mean is None:
            assert others == []
        else:
            (reverse,) = others
            assert reverse.role == "supplementary" and reverse.record["declarations"]["wiring_sign"] == -1
            assert reverse.value == pytest.approx(reversed_mean, abs=TOL)
        assert not [m for m in ex.specimen_messages(exp, specimen) if m.level == "warning"]

    # D2: 25/75 is analysed with the researcher's event marks, ending at its closing mark.
    record = ex.primary_measurement(exp, exp.specimen("IL1_25-IL2_75"), "S_app").record
    assert [mark["number"] for mark in record["event_marks"]] == list(range(1, 8))
    assert record["end_of_run"] == "event mark 7"

    # One explicitly chosen primary supplies the response; sessions are never averaged.
    specimen = exp.specimen("IL1_75-IL2_25")
    forward, reverse = specimen.measurements
    ex.set_primary(specimen, reverse.id)
    assert ex.primary_measurement(exp, specimen, "S_app") is reverse and forward.role == "supplementary"
    ex.set_primary(specimen, forward.id)

    # A recording belongs to one specimen; an output not measured by HIOKI refuses it.
    with pytest.raises(ValueError, match="already attached"):
        ex.attach(exp, exp.specimens[0], ex.hioki_measurement(_record("IL1_75-IL2_25", +1), exp.outputs[0], "", "primary"))
    with pytest.raises(ValueError, match="cannot be attached"):
        ex.hioki_measurement(_record("IL1_75-IL2_25", +1), ex.OutputDefinition("Viscosity", "mPa s"), "", "primary")
    with pytest.raises(ValueError, match="mV/K"):
        ex.hioki_measurement(_record("IL1_75-IL2_25", +1), ex.OutputDefinition("S", "uV/K", "hioki:combined"), "", "primary")


@needs_data
def test_save_and_reopen_keeps_everything(tmp_path):
    exp = _il_experiment()
    exp.inputs.append(ex.InputVariable("Drying time", "continuous", "h", 0.0, 48.0))
    exp.specimens[1].inputs["Drying time"] = 12.0
    exp.specimens[1].notes = "dried overnight"
    path = save_experiment(exp, tmp_path / "il")
    assert path.name == "il.latos.json" and not list(tmp_path.glob("*.tmp"))
    again = load_experiment(path)
    assert asdict(again) == asdict(exp)
    assert set(ex.check_sources(again).values()) == {"ok"}

    # On a copy of a recording: refusal to save beside it, and change detection.
    raw = tmp_path / "raw"
    raw.mkdir()
    copy = raw / "IL1_75-IL2_25.xlsx"
    shutil.copy2(DATA / "IL1_75-IL2_25.xlsx", copy)
    trace = read_hioki_xlsx(copy)
    small = ex.Experiment(outputs=[ex.OutputDefinition("S_app", UNIT, "hioki:combined")])
    small.specimens.append(ex.Specimen("S1"))
    record = analysis_record(analyze(trace, ROLES, +1), trace.sha256)
    ex.attach(small, small.specimens[0], ex.hioki_measurement(record, small.outputs[0], "", "primary"))
    with pytest.raises(ProjectError, match="original recordings"):
        save_experiment(small, raw / "small")
    path = save_experiment(small, tmp_path / "small.latos.json")
    with copy.open("ab") as handle:
        handle.write(b"\0")
    reopened = load_experiment(path)
    sources = ex.check_sources(reopened)
    assert list(sources.values()) == ["changed"]
    assert reopened.specimens[0].measurements[0].value == small.specimens[0].measurements[0].value
    texts = [m.text for m in ex.specimen_messages(reopened, reopened.specimens[0], sources)]
    assert any("changed since it was analysed" in t for t in texts)
    copy.unlink()
    assert list(ex.check_sources(reopened).values()) == ["missing"]
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ProjectError):
        load_experiment(tmp_path / "other.json")


@needs_data
def test_experiment_workspace_gui(tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from latos2.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.settings = window.experiment_view.settings = QSettings(
        str(tmp_path / "settings.ini"), QSettings.Format.IniFormat
    )
    view = window.experiment_view

    # Analyzer: exclude the pure liquid's initial test cycle with the checkbox.
    assert window.open_file(DATA / "IL1_0-IL2_100.xlsx")
    window.set_wiring(+1)
    window._set_view(1, None)
    window.properties.exclude_reason.setText("Initial test cycle")
    window.properties.exclude_check.setChecked(True)
    assert window.analysis.excluded == {1: "Initial test cycle"} and window.analysis.accepted == (2, 3, 4)
    assert window.results_table.verticalHeaderItem(0).text() == "Cycle 1 (excl.)"
    assert window.results_table.item(4, 0).text() == "-2.6434"

    # Experiment: three components, a HIOKI output and a hand-entered output.
    window.set_mode("experiment")
    assert window.pages.currentWidget() is view and "Experiment" in window.windowTitle()
    for name in ("IL1", "IL2", "Water"):
        view.add_component(name)
    view.add_output("S_app", UNIT, "hioki:combined", "maximize")
    view.add_output("Viscosity", "mPa s", objective="target", target=100.0)
    view.attach_analysis(window.analysis, "IL1_0-IL2_100", "S_app", "Forward leads", create=True)
    view.type_combo.setCurrentIndex(view.type_combo.findData("IL2"))
    view.add_specimen("M3", "1 : 2 : 1")
    for component, mass in (("IL1", "0.1"), ("IL2", "0.2"), ("Water", "0.1")):
        view.mass_edits[component].setText(mass)
        view.mass_edits[component].editingFinished.emit()
    view.enter_value("M3", "Viscosity", 98.5, sem=1.5, n=3)

    def table():
        t = view.table
        headers = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
        return [dict(zip(headers, (t.item(r, c).text() for c in range(t.columnCount())))) for r in range(t.rowCount())]

    pure, mixture = table()
    assert pure["Specimen"] == "IL1_0-IL2_100" and pure["S_app (mV/K)\nmean ± SEM (n)"] == "-2.6434 ± 0.3703 (3)"
    assert [pure[f"{c}\nmass fraction"] for c in ("IL1", "IL2", "Water")] == ["0 (pure)", "1 (pure)", "0 (pure)"]
    assert pure["IL2\nmass (g)"] == pure["Total\nmass (g)"] == "Not recorded"
    assert [mixture[f"{c}\nmass fraction"] for c in ("IL1", "IL2", "Water")] == ["0.2500", "0.5000", "0.2500"]
    assert mixture["Total\nmass (g)"] == "0.4000" and mixture["S_app (mV/K)\nmean ± SEM (n)"] == "—"
    assert mixture["Viscosity (mPa s)\nmean ± SEM (n)"] == "98.5 ± 1.5 (3)"
    assert mixture["Checks"] == "1 warning"  # no S_app measured yet

    path = tmp_path / "workspace.latos.json"
    assert view.save(path) and not view.dirty
    before = table()
    assert view.new_experiment() and view.table.rowCount() == 0
    assert view.open_experiment(path) and table() == before
    assert set(view.sources.values()) == {"ok"}
    window.close()
    app.processEvents()


SYNTHETIC_ROLES = ChannelRoles(hot="CH1-A-3", cold="CH1-A-1", voltage="CH1-A-10")


def _synthetic(marks: list[float], end_s: float, s_true_mv: float = 2.0) -> LoggerTrace:
    """A recording whose ΔT ramps up while heating and decays while cooling; V = S·ΔT."""
    time = np.round(np.arange(0.0, end_s + 1e-9, 0.2), 1)
    heating = np.zeros(time.size, dtype=bool)
    for on, off in zip(marks[0::2], marks[1::2]):
        heating |= (time >= on) & (time < off)
    delta_t = np.zeros(time.size)
    for i in range(1, time.size):
        rate = 5.0 / 60.0 if heating[i - 1] else -delta_t[i - 1] / 20.0
        delta_t[i] = delta_t[i - 1] + 0.2 * rate
    volts = np.round(s_true_mv * 1e-3 * delta_t / 5e-6) * 5e-6
    channels = (
        LoggerChannel("CH1-A-1", "°C", "CH1-A-1[°C]", np.full(time.size, 25.0)),
        LoggerChannel("CH1-A-3", "°C", "CH1-A-3[°C]", 25.0 + delta_t),
        LoggerChannel("CH1-A-10", "V", "CH1-A-10[V]", volts),
    )
    events = tuple(
        LoggerEvent(number=k, time_s=t, row=int(t / 0.2), sheet_row=int(t / 0.2) + 3, timestamp="")
        for k, t in enumerate(marks, start=1)
    )
    return LoggerTrace(
        source=Path("synthetic.xlsx"), sha256="0" * 64, size_bytes=0, sheet="Sheet1", preamble=(),
        timestamps=tuple("" for _ in time), time_s=time, channels=channels, has_event_column=True,
        events=events, sample_interval_s=0.2,
    )
