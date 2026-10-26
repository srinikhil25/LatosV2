"""Milestone 3 acceptance tests: Gaussian process, expected improvement, candidates, the IL demonstration."""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from latos2 import experiment as ex  # noqa: E402
from latos2.gp import expected_improvement, expected_target_improvement, fit_gp, leave_one_out  # noqa: E402
from latos2.optimization import ProblemError, build_problem, prepare, recommend  # noqa: E402
from latos2.thermovoltage import UNIT  # noqa: E402
from test_experiment import DATA, needs_data  # noqa: E402

IL_DOMAIN = {"IL1": [0.239575, 1.0], "IL2": [0.0, 1.0]}


def _temperature_experiment(objective="maximize", target=None, temperatures=(0, 20, 40, 60, 80, 100), sem=0.1):
    """Yield peaking at 70 °C, measured once per specimen."""
    exp = ex.Experiment(
        inputs=[ex.InputVariable("Temperature", "continuous", "°C", 0.0, 100.0)],
        outputs=[ex.OutputDefinition("Yield", "%", objective=objective, target=target)],
    )
    output = exp.outputs[0]
    for i, t in enumerate(temperatures):
        specimen = ex.Specimen(f"S{i}", inputs={"Temperature": float(t)})
        exp.specimens.append(specimen)
        value = 50.0 - 100.0 * (t / 100.0 - 0.7) ** 2
        ex.attach(exp, specimen, ex.manual_measurement(output, value, "", "primary", sem=sem, n=3))
    return exp


def test_gp_posterior_and_noise():
    x = np.linspace(0.0, 1.0, 7).reshape(-1, 1)
    y = np.sin(3.0 * x[:, 0])
    sem = np.full(7, 0.01)
    gp = fit_gp(x, y, sem, additional_noise=0.0, seed=0)
    # The prediction is the textbook posterior under the fitted kernel and the stated noise.
    kernel, mean_y, scale = gp.model.kernel_, y.mean(), y.std()
    k = kernel(x) + np.diag((sem / scale) ** 2 + 1e-10)
    xs = np.array([[0.33], [0.77]])
    ks = kernel(xs, x)
    expected_mean = mean_y + scale * ks @ np.linalg.solve(k, (y - mean_y) / scale)
    expected_sd = scale * np.sqrt(kernel.diag(xs) - np.einsum("ij,ji->i", ks, np.linalg.solve(k, ks.T)))
    mean, sd = gp.predict(xs)
    assert mean == pytest.approx(expected_mean, rel=1e-6) and sd == pytest.approx(expected_sd, rel=1e-5)
    assert mean == pytest.approx(np.sin(3.0 * xs[:, 0]), abs=0.01)
    # Larger assumed observation noise: the model trusts each point less.
    noisy = fit_gp(x, y, sem, additional_noise=0.2, seed=0)
    assert noisy.predict(x)[1].mean() > gp.predict(x)[1].mean()
    assert noisy.noise_sd == pytest.approx(np.sqrt(0.01**2 + 0.2**2))
    # Leave-one-out refits every fold and predicts the held-out point.
    loo = leave_one_out(x, y, sem, 0.0, 0)
    assert len(loo.observed) == 7 and loo.rmse < 0.05
    assert leave_one_out(x[:2], y[:2], sem[:2], 0.0, 0) is None


def test_expected_improvement_closed_forms():
    rng = np.random.default_rng(0)
    draws = rng.normal(0.4, 0.3, 1_000_000)
    ei = expected_improvement(np.array([0.4]), np.array([0.3]), best=0.5, xi=0.0)[0]
    assert ei == pytest.approx(np.mean(np.maximum(0.0, draws - 0.5)), rel=5e-3)
    gain = expected_target_improvement(np.array([0.4]), np.array([0.3]), target=0.0, best_distance=0.5)[0]
    assert gain == pytest.approx(np.mean(np.maximum(0.0, 0.5 - np.abs(draws))), rel=5e-3)
    assert expected_improvement(np.array([np.nan, 1.0]), np.array([0.1, np.nan]), 0.0).tolist()[0] == 0.0
    assert expected_target_improvement(np.array([1.0]), np.array([0.1]), 0.0, 0.0)[0] == 0.0


def test_objectives_maximize_minimize_target():
    settings = ex.OptimizationSettings(output="Yield")
    best = recommend(build_problem(_temperature_experiment("maximize"), settings))
    assert best.choice.values[0] == pytest.approx(70.0, abs=2.0)
    assert best.explanation.startswith("Seeks improvement")
    low = recommend(build_problem(_temperature_experiment("minimize"), settings))
    assert low.choice.values[0] < 15.0  # the low end, outside the 5 °C separation around 0 °C
    near = recommend(build_problem(_temperature_experiment("target", 46.0), settings))
    assert min(abs(near.choice.values[0] - 50.0), abs(near.choice.values[0] - 90.0)) < 2.0
    assert near.choice.mean == pytest.approx(46.0, abs=0.5)
    # Explore: the most uncertain allowed point, whatever the objective.
    explore = recommend(build_problem(_temperature_experiment("maximize"), replace(settings, acquisition="explore")))
    assert explore.choice.sd == pytest.approx(explore.sd[~explore.excluded].max())
    # Units and scaling: the same problem in other units gives the same decision.
    scaled = _temperature_experiment("maximize")
    for s in scaled.specimens:
        s.measurements[0].value *= 1000.0
        s.measurements[0].sem *= 1000.0
    rescaled = recommend(build_problem(scaled, settings))
    assert rescaled.choice.values == best.choice.values
    assert rescaled.choice.mean == pytest.approx(1000.0 * best.choice.mean, rel=1e-6)


def test_mixture_candidates_and_preparation():
    two = ex.Experiment(mixture=ex.Mixture(["A", "B"]), outputs=[ex.OutputDefinition("P", "u")])
    settings = ex.OptimizationSettings(output="P", fraction_bounds={"A": [0.0, 1.0], "B": [0.4, 1.0]})
    space = build_problem(two, settings).space
    assert space.d == 1 and space.variables[0].upper == pytest.approx(0.6)  # x_B ≥ 0.4 caps x_A
    rows = space.candidates(0)
    assert rows.min() >= 0.0 and rows.max() <= 0.6 + 1e-12

    three = ex.Experiment(mixture=ex.Mixture(["A", "B", "C"], total_mode="fixed", total_g=0.5, tolerance_g=0.001),
                          outputs=[ex.OutputDefinition("P", "u")])
    bounds = {"A": [0.1, 0.6], "B": [0.2, 0.7], "C": [0.0, 0.5]}
    space = build_problem(three, ex.OptimizationSettings(output="P", fraction_bounds=bounds)).space
    assert space.d == 2 and space.free == ("A", "B")  # N − 1 composition variables; fixed total adds none
    rows = space.candidates(0)
    assert len(rows) > 100
    for row in rows:
        x = space.fractions(row)
        assert sum(x.values()) == pytest.approx(1.0)
        assert all(bounds[c][0] - 1e-9 <= x[c] <= bounds[c][1] + 1e-9 for c in x)

    prep = prepare(space, {"A": 0.333333, "B": 0.444444, "C": 0.222223}, 0.5, 0.0001)
    assert sum(prep.masses_g.values()) == pytest.approx(0.5)
    assert all(abs(m / 0.0001 - round(m / 0.0001)) < 1e-6 for m in prep.masses_g.values())
    assert prep.feasible and prep.fractions["A"] == pytest.approx(0.3333, abs=1e-4)
    # At a bound, rounding goes the way that keeps the composition feasible.
    edge = prepare(space, {"A": 0.10001, "B": 0.4, "C": 0.49999}, 0.25, 0.001)
    assert edge.feasible and edge.fractions["A"] >= 0.1 and edge.fractions["C"] <= 0.5
    with pytest.raises(ProblemError, match="total mass"):
        prepare(space, {"A": 0.3, "B": 0.4, "C": 0.3}, None, 0.0001)


def test_separation_replicates_and_reproducibility():
    exp = _temperature_experiment("maximize", temperatures=(0, 25, 50, 75, 100))
    settings = ex.OptimizationSettings(output="Yield", min_separation=0.1)
    result = recommend(build_problem(exp, settings))
    measured = result.problem.measured_points[:, 0]
    allowed = result.candidates[~result.excluded, 0]
    assert np.min(np.abs(allowed[:, None] - measured[None, :])) > 10.0 - 1e-6
    assert np.min(np.abs(result.choice.values[0] - measured)) > 10.0 - 1e-6
    # Same data, settings and seed: the same decision and the same record.
    again = recommend(build_problem(exp, settings))
    assert again.choice == result.choice and again.record == result.record
    # Deliberate replication may return a measured condition; otherwise even 0 separation excludes it.
    replicate = recommend(build_problem(exp, replace(settings, allow_replicates=True)))
    assert not replicate.excluded.any()
    strict = recommend(build_problem(exp, replace(settings, min_separation=0.0)))
    assert not np.isin(strict.candidates[~strict.excluded, 0], measured).any()
    # Noise sensitivity: SEM only, three fractions of the response SD, and the selected level.
    noisy = recommend(build_problem(exp, replace(settings, additional_noise=1.0)))
    levels = [row.additional_noise for row in noisy.sensitivity]
    sd = np.std([o.response for o in noisy.problem.used], ddof=1)
    assert levels == pytest.approx(sorted([0.0, 0.1 * sd, 0.25 * sd, 0.5 * sd, 1.0]))
    assert [row.selected for row in noisy.sensitivity].count(True) == 1
    assert noisy.gp.additional_noise == 1.0 and noisy.gp.noise_sd == pytest.approx(np.sqrt(0.1**2 + 1.0))


def test_independent_specimens_and_unusable_observations():
    exp = _temperature_experiment("maximize", temperatures=(10, 40, 70))
    problem = build_problem(exp, ex.OptimizationSettings(output="Yield"))
    assert len(problem.used) == 3 and [o.n for o in problem.used] == [3, 3, 3]  # cycles are not observations
    with pytest.raises(ProblemError, match="at least 3"):
        recommend(build_problem(_temperature_experiment(temperatures=(10, 40)), ex.OptimizationSettings(output="Yield")))
    # A replicate specimen is a second observation, and is reported as such.
    replicated = _temperature_experiment(temperatures=(10, 40, 40, 70, 90))
    result = recommend(build_problem(replicated, ex.OptimizationSettings(output="Yield")))
    assert len(result.problem.used) == 5 and any("replicates" in w for w in result.warnings)
    # A session whose unit no longer matches the output is not a usable observation.
    exp.specimens[0].measurements[0].unit = "g/L"
    problem = build_problem(exp, ex.OptimizationSettings(output="Yield"))
    assert not problem.observations[0].used and "no primary measurement" in problem.observations[0].reason
    # A missing input value or mass is excluded with its reason, never filled in.
    exp.specimens[1].inputs.clear()
    assert build_problem(exp, ex.OptimizationSettings(output="Yield")).observations[1].reason == "Temperature not entered"
    # Unsupported set-ups are refused with an explanation.
    exp.inputs.append(ex.InputVariable("Solvent", "categorical", levels=["water", "ethanol"]))
    with pytest.raises(ProblemError, match="categorical"):
        build_problem(exp, ex.OptimizationSettings(output="Yield"))
    with pytest.raises(ProblemError, match="target value"):
        build_problem(_temperature_experiment("target", None), ex.OptimizationSettings(output="Yield"))


@needs_data
def test_ionic_liquid_demonstration():
    from test_experiment import _il_experiment

    exp = _il_experiment()
    settings = ex.OptimizationSettings(output="S_app", fraction_bounds=dict(IL_DOMAIN))
    problem = build_problem(exp, settings)
    used = {o.specimen_id: o for o in problem.used}
    assert list(used) == ["IL1_25-IL2_75", "IL1_50-IL2_50", "IL1_75-IL2_25", "IL1_100-IL2_0"]
    (pure_il2,) = [o for o in problem.observations if not o.used]
    assert pure_il2.specimen_id == "IL1_0-IL2_100" and pure_il2.reason.startswith("outside the search domain")
    assert len(exp.specimens) == 5 and exp.specimens[0].measurements  # still in the experimental record
    responses = {k: o.response for k, o in used.items()}
    assert responses == pytest.approx(
        {"IL1_25-IL2_75": 2.0256, "IL1_50-IL2_50": 2.0187, "IL1_75-IL2_25": 2.1900, "IL1_100-IL2_0": 1.5642}, abs=1e-4
    )
    assert problem.output.unit == UNIT and problem.output.objective == "maximize"

    result = recommend(problem)
    x = result.choice.values[0]
    assert 0.239575 <= x <= 1.0 and np.min(np.abs(x - result.problem.measured_points[:, 0])) > 0.05 * (1 - 0.239575)
    assert result.choice.fractions["IL1"] + result.choice.fractions["IL2"] == pytest.approx(1.0)
    assert x == pytest.approx(0.6719, abs=0.002)  # the recorded demonstration result
    assert result.baseline.values[0] == pytest.approx((0.481331 + 0.758702) / 2, abs=0.001)  # widest-gap midpoint
    assert recommend(build_problem(exp, settings)).choice == result.choice
    assert len(result.loo.observed) == 4 and len(result.sensitivity) == 4
    assert result.record["status"] == "Provisional recommendation, awaiting researcher review"

    prep = prepare(problem.space, result.choice.fractions, 0.25, 0.0001)
    assert sum(prep.masses_g.values()) == pytest.approx(0.25) and prep.feasible
    assert prep.fractions["IL1"] == pytest.approx(x, abs=0.0004)


@needs_data
def test_optimization_workspace_gui(tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from test_experiment import _il_experiment

    from latos2.project_io import load_experiment, save_experiment
    from latos2.ui.main_window import MainWindow

    path = save_experiment(_il_experiment(), tmp_path / "il")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.settings = window.experiment_view.settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    assert window.experiment_view.open_experiment(path)
    window.set_mode("optimization")
    view = window.optimization_view
    assert "Optimization" in window.windowTitle() and view.problem is not None
    view.set_fraction_bounds("IL1", 0.239575, 1.0)
    assert view.fit() and len(view.result.problem.used) == 4
    assert view.result_box.isVisibleTo(window) and view.masses_table.rowCount() == 0  # no total, no masses
    view.total_edit.setText("0.25")
    view.total_edit.editingFinished.emit()
    masses = {view.masses_table.item(r, 0).text(): view.masses_table.item(r, 1).text() for r in range(2)}
    assert set(masses) == {"IL1", "IL2"} and sum(float(m) for m in masses.values()) == pytest.approx(0.25)
    assert view.surrogate_plot.figure.axes and view.acquisition_plot.figure.axes  # genuine model output drawn
    assert window.experiment_view.dirty
    assert window.experiment_view.save(path)
    saved = load_experiment(path).optimization
    assert saved.fraction_bounds["IL1"] == [0.239575, 1.0] and saved.preparation_total_g == 0.25
    window.close()
    app.processEvents()


def test_output_defined_after_visiting_optimization(tmp_path):
    """Regression: opening Optimization before any output existed stored no output, and the
    Output box later showed one while fitting stayed disabled ("Choose the output to optimize")."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from latos2.project_io import save_experiment
    from latos2.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.settings = window.experiment_view.settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    view = window.optimization_view
    window.set_mode("optimization")  # a new experiment: no output defined yet
    assert view.settings.output is None and not view.fit_button.isEnabled()
    window.set_mode("experiment")
    defined = _temperature_experiment()
    exp = window.experiment_view.experiment
    exp.inputs, exp.outputs, exp.specimens = defined.inputs, defined.outputs, defined.specimens
    window.set_mode("optimization")
    assert view.output_combo.currentData() == view.settings.output == "Yield"
    assert view.problem is not None and view.fit_button.isEnabled() and view.setup_message.isHidden()

    # A file saved in that state, with no output chosen, opens ready to fit.
    path = save_experiment(replace(defined, optimization=ex.OptimizationSettings()), tmp_path / "saved")
    window.experiment_view.dirty = False
    window._replace_experiment(lambda: window.experiment_view.open_experiment(path))  # as File ▸ Open does
    assert window.experiment_view.experiment.optimization is view.settings
    assert view.output_combo.currentData() == view.settings.output == "Yield" and view.fit_button.isEnabled()
    window.experiment_view.dirty = False
    window.close()
    app.processEvents()
