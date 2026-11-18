"""Milestone 4 acceptance tests: frozen predictions, prospective scoring, preparation sheet."""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from latos2 import experiment as ex  # noqa: E402
from latos2 import predictions as preds  # noqa: E402
from latos2.gp import Z95  # noqa: E402
from latos2.optimization import build_problem, recommend  # noqa: E402
from latos2.project_io import ProjectError, load_experiment, save_experiment  # noqa: E402
from latos2.thermovoltage import UNIT  # noqa: E402
from test_experiment import needs_data  # noqa: E402
from test_optimization import IL_DOMAIN, _temperature_experiment  # noqa: E402

JST = timezone(timedelta(hours=9))
FROZEN = datetime(2026, 11, 18, 10, 0, tzinfo=JST)


def _mixture_experiment() -> ex.Experiment:
    """Two components; the response peaks at x_A = 0.62. Measured values entered by hand."""
    exp = ex.Experiment(name="Two-component test", mixture=ex.Mixture(["A", "B"]),
                        outputs=[ex.OutputDefinition("P", "u")])
    for i, xa in enumerate((0.1, 0.3, 0.5, 0.7, 0.9)):
        specimen = ex.Specimen(f"S{i}", masses_g={"A": xa, "B": round(1.0 - xa, 10)})
        exp.specimens.append(specimen)
        value = 10.0 - 20.0 * (xa - 0.62) ** 2
        ex.attach(exp, specimen, ex.manual_measurement(exp.outputs[0], value, "", "primary", sem=0.05, n=3))
    return exp


def _frozen(exp: ex.Experiment, total: float | None = None) -> ex.Prediction:
    settings = ex.OptimizationSettings(output="P", fraction_bounds={"A": [0.0, 1.0], "B": [0.0, 1.0]},
                                       preparation_total_g=total)
    return preds.freeze(exp, recommend(build_problem(exp, settings)), now=FROZEN)


@needs_data
def test_freeze_il_prediction_save_and_reopen(tmp_path):
    from test_experiment import _il_experiment

    exp = _il_experiment()
    settings = ex.OptimizationSettings(output="S_app", fraction_bounds=dict(IL_DOMAIN), preparation_total_g=0.25)
    result = recommend(build_problem(exp, settings))
    p = preds.freeze(exp, result, "IL demonstration", now=FROZEN)
    assert p.id == "P1" and p.frozen_at == "2026-11-18T10:00:00+09:00" and p.output == "S_app" and p.unit == UNIT
    assert p.fractions["IL1"] == pytest.approx(0.6719, abs=0.002) and sum(p.fractions.values()) == pytest.approx(1.0)
    assert p.predicted_mean == pytest.approx(2.2043, abs=1e-4) and p.latent_sd == pytest.approx(0.1250, abs=1e-4)
    assert p.predictive_sd == pytest.approx(0.1268, abs=1e-4)
    assert p.interval_95 == pytest.approx([p.predicted_mean - Z95 * p.predictive_sd, p.predicted_mean + Z95 * p.predictive_sd])
    assert p.best_before == pytest.approx(2.1900, abs=1e-4) and p.best_before_specimen == "IL1_75-IL2_25"
    assert p.preparation["masses_g"] == pytest.approx({"IL1": 0.1680, "IL2": 0.0820})
    assert p.record == json.loads(json.dumps(result.record)) and preds.intact(p)
    with pytest.raises(ValueError, match="already frozen as P1"):
        preds.freeze(exp, result)

    path = save_experiment(exp, tmp_path / "il")
    reopened = load_experiment(path)
    assert reopened.predictions == [p] and preds.intact(reopened.predictions[0])
    assert preds.status(reopened, reopened.predictions[0]) == "Open: awaiting a specimen"

    # An edit made to the file outside Latos is reported, not trusted.
    data = json.loads(path.read_text(encoding="utf-8"))
    data["experiment"]["predictions"][0]["predicted_mean"] = 2.30
    path.write_text(json.dumps(data), encoding="utf-8")
    altered = load_experiment(path)
    assert not preds.intact(altered.predictions[0])
    assert preds.status(altered, altered.predictions[0]) == "Altered outside Latos"


def test_frozen_prediction_unchanged_by_later_changes():
    exp = _temperature_experiment()
    settings = ex.OptimizationSettings(output="Yield")
    p1 = preds.freeze(exp, recommend(build_problem(exp, settings)), now=FROZEN)
    snapshot = copy.deepcopy(p1)
    # Later data and settings: a corrected value, a new specimen, another noise assumption, a refit.
    exp.specimens[2].measurements[0].value += 3.0
    new = ex.Specimen("S9", inputs={"Temperature": 65.0})
    exp.specimens.append(new)
    ex.attach(exp, new, ex.manual_measurement(exp.outputs[0], 49.0, "", "primary", sem=0.1, n=3))
    settings.additional_noise = 0.5
    p2 = preds.freeze(exp, recommend(build_problem(exp, settings)), now=FROZEN + timedelta(days=1))
    assert p1 == snapshot and preds.intact(p1)
    assert p2.id == "P2" and p2.record != p1.record and p2.inputs != p1.inputs
    # Names a frozen prediction refers to cannot be changed under it.
    with pytest.raises(ValueError, match="P1 refers to the output"):
        ex.rename_output(exp, "Yield", "Yield (%)")
    with pytest.raises(ValueError, match="P1 refers to the input"):
        ex.rename_input(exp, "Temperature", "T")
    assert exp.outputs[0].name == "Yield" and exp.inputs[0].name == "Temperature"


def test_scoring_a_linked_specimen():
    exp = _mixture_experiment()
    p = _frozen(exp)
    with pytest.raises(ValueError, match="was in the data P1 was predicted from"):
        preds.link(exp, exp.specimens[0], "P1")

    x_a = round(p.fractions["A"], 3)
    specimen = ex.Specimen("N1", masses_g={"A": x_a, "B": round(1.0 - x_a, 10)})
    exp.specimens.append(specimen)
    preds.link(exp, specimen, "P1")
    (pending,) = preds.tests(exp)
    assert not pending.scored and pending.problem == "No primary measurement of P yet."
    assert preds.summary([pending]) == "No prospective test has been measured yet."
    assert preds.status(exp, p) == "Awaiting measurement: N1"
    assert any("Prospective test of frozen prediction P1" in m.text for m in ex.specimen_messages(exp, specimen))

    measured = p.predicted_mean + 0.1
    ex.attach(exp, specimen, ex.manual_measurement(exp.outputs[0], measured, "", "primary", sem=0.04, n=3))
    (test,) = preds.tests(exp)
    assert test.scored and test.measured == measured and test.sem == 0.04
    assert test.error == pytest.approx(0.1) and test.z == pytest.approx(0.1 / p.predictive_sd)
    assert test.inside == (0.1 <= Z95 * p.predictive_sd)
    assert p.best_before_specimen == "S3" and test.gain == pytest.approx(measured - p.best_before)
    assert test.deviations["A mass fraction"] == pytest.approx(x_a - p.fractions["A"])
    assert test.timing.state == "unknown" and "Entered by hand" in test.timing.text
    assert preds.summary([test]).startswith("1 prospective test measured: ")
    assert "One test neither validates nor refutes the model." in preds.summary([test])
    assert preds.status(exp, p) == "Tested by N1"

    # A prediction being tested cannot be withdrawn; a withdrawn one cannot be tested.
    with pytest.raises(ValueError, match="being tested"):
        preds.withdraw(exp, "P1", "changed plan")
    preds.link(exp, specimen, None)
    with pytest.raises(ValueError, match="reason"):
        preds.withdraw(exp, "P1", "  ")
    preds.withdraw(exp, "P1", "Batch spoiled before preparation", now=FROZEN + timedelta(hours=2))
    assert preds.status(exp, p) == "Withdrawn" and p in exp.predictions and preds.intact(p)
    with pytest.raises(ValueError, match="withdrawn"):
        preds.link(exp, specimen, "P1")


def test_prospective_timing():
    p = ex.Prediction(
        id="P1", frozen_at=FROZEN.isoformat(), output="S_app", unit=UNIT, objective="maximize", target=None,
        fractions=None, inputs={}, predicted_mean=2.2, latent_sd=0.1, predictive_sd=0.1, interval_95=[2.0, 2.4],
        best_before=None, best_before_specimen=None, preparation=None, record={},
    )

    def recorded(stamp: str) -> ex.Measurement:
        return ex.Measurement("S_app", "Forward leads", "primary", "hioki:combined", UNIT, 2.2,
                              record={"source": {"first_timestamp": stamp}})

    state = lambda stamp: preds.timing(p, recorded(stamp)).state  # noqa: E731
    assert state("11/20/2026 09:00:00.0s") == "after"  # only month/day is a valid reading
    assert state("11/18/2026 10:30:00.0s") == "after" and state("11/18/2026 09:59:59.0s") == "before"
    assert state("11/10/2026 09:00:00.0s") == "before"  # 10 November or 11 October: both before
    assert state("12/12/2026 09:00:00.0s") == "after"
    assert state("2026/11/20 09:00:00s") == "after"
    ambiguous = preds.timing(p, recorded("12/1/2026 09:00:00.0s"))  # 1 December or 12 January
    assert ambiguous.state == "unknown" and "does not choose" in ambiguous.text
    assert state("") == "unknown" and state("10/5/26 20:02:25.0s") == "unknown"
    manual = ex.Measurement("S_app", "Manual", "primary", ex.MANUAL, UNIT, 2.2)
    assert preds.timing(p, manual).state == "unknown" and preds.timing(p, None).text == "Not measured yet."


def test_preparation_sheet_pdf(tmp_path):
    from PySide6.QtWidgets import QApplication

    from latos2.prep_sheet import sheet_html, write_preparation_sheet

    app = QApplication.instance() or QApplication([])  # noqa: F841 - fonts need a GUI application
    exp = _mixture_experiment()
    p = _frozen(exp, total=0.5)
    masses = p.preparation["masses_g"]
    assert sum(masses.values()) == pytest.approx(0.5)
    text = sheet_html(exp, p)
    assert "frozen prediction P1" in text and f"{masses['A']:.4f}" in text and p.sha256 in text
    (tmp_path / "out").mkdir()
    written = write_preparation_sheet(exp, p, tmp_path / "out" / "sheet")
    assert written.name == "sheet.pdf" and written.read_bytes().startswith(b"%PDF")

    # Never into the folder of an original recording.
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    exp.specimens[0].measurements.append(ex.Measurement(
        "P", "Recording", "supplementary", "hioki:combined", UNIT, 1.0,
        record={"source": {"path": str(recordings / "run.xlsx"), "sha256_at_read": "0" * 64}},
    ))
    with pytest.raises(ProjectError, match="original recordings"):
        write_preparation_sheet(exp, p, recordings / "sheet.pdf")
    preds.withdraw(exp, "P1", "not prepared")
    with pytest.raises(ValueError, match="withdrawn"):
        write_preparation_sheet(exp, p, tmp_path / "out" / "again.pdf")


def test_predictions_workspace_gui(tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from latos2.ui.main_window import MainWindow

    path = save_experiment(_mixture_experiment(), tmp_path / "mix")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.settings = window.experiment_view.settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    window._replace_experiment(lambda: window.experiment_view.open_experiment(path))
    window.set_mode("optimization")
    view, experiment_view = window.optimization_view, window.experiment_view
    assert view.fit() and view.freeze_button.isEnabled()
    view.total_edit.setText("0.5")
    view.total_edit.editingFinished.emit()

    p = view.freeze("GUI walk-through")
    assert not experiment_view.dirty  # D1: freezing saved the file
    assert [q.id for q in load_experiment(path).predictions] == ["P1"]
    assert not view.freeze_button.isEnabled() and "Frozen as P1" in view.frozen_label.text()
    assert view.predictions_table.rowCount() == 1 and view.predictions_table.item(0, 5).text() == "Open: awaiting a specimen"
    sheet = view.export_sheet("P1", tmp_path / "P1_sheet.pdf")
    assert sheet.read_bytes().startswith(b"%PDF")

    window.set_mode("experiment")
    specimen = experiment_view.add_specimen("N1")
    specimen.masses_g = {"A": round(p.fractions["A"], 3), "B": round(1.0 - round(p.fractions["A"], 3), 10)}
    experiment_view.enter_value("N1", "P", p.predicted_mean - 0.05, sem=0.03, n=3)
    experiment_view.link_prediction("N1", "P1")
    assert specimen.prediction == "P1"
    window.set_mode("optimization")
    assert view.tests_table.rowCount() == 1
    row = {view.tests_table.horizontalHeaderItem(c).text(): view.tests_table.item(0, c).text()
           for c in range(view.tests_table.columnCount())}
    assert row["Prediction"] == "P1" and row["Specimen"] == "N1" and row["In frozen 95 %"] in ("Yes", "No")
    assert row["Error"] == f"{-0.05:.6g}" and row["Recorded"] == "Not confirmed"
    assert view.predictions_table.item(0, 5).text() == "Tested by N1"
    assert "1 prospective test measured" in view.tests_caption.text()
    assert experiment_view.save()
    assert load_experiment(path).specimens[-1].prediction == "P1"
    window.close()
    app.processEvents()
