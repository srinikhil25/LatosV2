"""From an experiment to a provisional next experiment.

The design space comes from the experiment definition:

* an N-component mixture contributes its first N−1 mass fractions; the last
  component is the balance, x_N = 1 − Σ x_i, so no redundant fraction is fitted;
* a continuous input contributes its [minimum, maximum] range;
* a discrete input contributes its listed values, and candidates take only those;
* categorical inputs, and total mass as a design variable, are not yet
  supported; they are reported, never silently dropped.

Each physical specimen is one observation: the value of its primary measurement
session for the output, with that session's SEM. Repeated cycles live inside
that value and never count as observations. A specimen that cannot be used is
listed with the reason, and one outside the researcher's search domain stays in
the experiment.

Every input is scaled to [0, 1] over its search range before modelling;
distances (minimum separation, the space-filling baseline) are measured in
those units.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from itertools import product

import numpy as np
import scipy
import sklearn
from scipy.stats import qmc

from latos2 import __version__
from latos2 import experiment as ex
from latos2.gp import (
    LENGTH_SCALE_BOUNDS,
    RESTARTS,
    FittedGP,
    LooResult,
    expected_improvement,
    expected_target_improvement,
    fit_gp,
    leave_one_out,
)

__all__ = [
    "GRID_POINTS",
    "SENSITIVITY_FRACTIONS",
    "STATUS",
    "Candidate",
    "DesignSpace",
    "Observation",
    "Preparation",
    "Problem",
    "ProblemError",
    "Result",
    "SensitivityRow",
    "Variable",
    "build_problem",
    "default_settings",
    "prepare",
    "recommend",
]

STATUS = "Provisional recommendation, awaiting researcher review"
GRID_POINTS = 2001  # candidates along a single continuous input
SOBOL_POWER = 12  # 4096 quasi-random candidates over several inputs
XI = 0.01  # EI margin, as a fraction of the SD of the observed responses
SENSITIVITY_FRACTIONS = (0.0, 0.1, 0.25, 0.5)  # σ_add as fractions of that SD
_TOL = 1e-9


class ProblemError(ValueError):
    """The experiment cannot be optimized as set up; the message says why."""


@dataclass(frozen=True)
class Variable:
    """One model input: a free mass fraction, or a continuous or discrete input."""

    key: str
    label: str
    kind: str  # "fraction", "continuous" or "discrete"
    lower: float
    upper: float
    unit: str = ""
    levels: tuple[float, ...] = ()

    @property
    def span(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class DesignSpace:
    """The model inputs, their search ranges, and the mixture they belong to."""

    variables: tuple[Variable, ...]
    components: tuple[str, ...]
    bounds: dict[str, tuple[float, float]]  # mass-fraction bounds of every component

    @property
    def d(self) -> int:
        return len(self.variables)

    @property
    def free(self) -> tuple[str, ...]:
        return tuple(v.key for v in self.variables if v.kind == "fraction")

    def to_unit(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=float).reshape(-1, self.d)
        return np.column_stack([(rows[:, i] - v.lower) / v.span for i, v in enumerate(self.variables)])

    def from_unit(self, unit: np.ndarray) -> np.ndarray:
        unit = np.asarray(unit, dtype=float).reshape(-1, self.d)
        return np.column_stack([v.lower + unit[:, i] * v.span for i, v in enumerate(self.variables)])

    def fractions(self, row) -> dict[str, float] | None:
        """Every component's mass fraction at a point, the balance included."""
        if not self.components:
            return None
        free = {v.key: float(row[i]) for i, v in enumerate(self.variables) if v.kind == "fraction"}
        balance = self.components[-1]
        return {c: free[c] if c != balance else 1.0 - math.fsum(free.values()) for c in self.components}

    def violations(self, row) -> list[str]:
        """What puts a point outside the search domain; empty when it is inside."""
        out = []
        for i, v in enumerate(self.variables):
            value = float(row[i])
            if value < v.lower - _TOL or value > v.upper + _TOL:
                out.append(f"{v.label} = {value:.6g} is outside [{v.lower:.6g}, {v.upper:.6g}]")
            elif v.kind == "discrete" and not any(abs(value - level) <= _TOL for level in v.levels):
                out.append(f"{v.label} = {value:.6g} is not one of the listed values")
        x = self.fractions(row)
        if x is not None and len(self.components) > 1:
            balance = self.components[-1]
            lo, hi = self.bounds[balance]
            if x[balance] < lo - _TOL or x[balance] > hi + _TOL:
                out.append(f"{balance} mass fraction = {x[balance]:.6g} is outside [{lo:.6g}, {hi:.6g}]")
        return out

    def feasible(self, rows: np.ndarray) -> np.ndarray:
        return np.array([not self.violations(row) for row in np.asarray(rows, dtype=float).reshape(-1, self.d)])

    def candidates(self, seed: int) -> np.ndarray:
        """Feasible candidate points: a dense grid along one input, quasi-random over several."""
        if self.d == 1:
            v = self.variables[0]
            grid = np.array(v.levels) if v.kind == "discrete" else np.linspace(v.lower, v.upper, GRID_POINTS)
            rows = grid.reshape(-1, 1)
        else:
            unit = qmc.Sobol(d=self.d, scramble=True, seed=seed).random_base2(SOBOL_POWER)
            columns = []
            for i, v in enumerate(self.variables):
                if v.kind == "discrete":
                    index = np.minimum((unit[:, i] * len(v.levels)).astype(int), len(v.levels) - 1)
                    columns.append(np.asarray(v.levels)[index])
                else:
                    columns.append(v.lower + unit[:, i] * v.span)
            rows = np.column_stack(columns)
        rows = rows[self.feasible(rows)]
        if len(rows) == 0:
            raise ProblemError("the search domain contains no feasible point; widen the composition bounds")
        return rows


@dataclass(frozen=True)
class Observation:
    """One physical specimen, as the optimizer sees it."""

    specimen_id: str
    label: str
    values: tuple[float, ...] | None  # model inputs, in their own units
    fractions: dict[str, float] | None
    response: float | None
    sem: float | None
    n: int | None
    session: str
    source: str
    sha256: str | None
    measurement_id: str | None
    used: bool
    reason: str = ""


@dataclass(frozen=True)
class Problem:
    """An experiment set up for optimization: output, design space and observations."""

    experiment: str
    output: ex.OutputDefinition
    space: DesignSpace
    observations: tuple[Observation, ...]
    settings: ex.OptimizationSettings

    @property
    def used(self) -> list[Observation]:
        return [o for o in self.observations if o.used]

    @property
    def minimum_needed(self) -> int:
        return max(3, self.space.d + 2)

    @property
    def measured_points(self) -> np.ndarray:
        """Every specimen's condition, used in the fit or not."""
        rows = [o.values for o in self.observations if o.values is not None]
        return np.array(rows, dtype=float).reshape(-1, self.space.d)


def default_settings(exp: ex.Experiment) -> ex.OptimizationSettings:
    """Starting settings: the first output with measured values, composition free in [0, 1]."""
    measured = [o.name for o in exp.outputs if any(ex.primary_measurement(exp, s, o.name) for s in exp.specimens)]
    output = (measured or [o.name for o in exp.outputs] or [None])[0]
    components = exp.mixture.components if exp.mixture else []
    total = exp.mixture.total_g if exp.mixture and exp.mixture.total_mode == "fixed" else None
    return ex.OptimizationSettings(
        output=output, fraction_bounds={c: [0.0, 1.0] for c in components}, preparation_total_g=total
    )


def _space(exp: ex.Experiment, settings: ex.OptimizationSettings) -> DesignSpace:
    components = tuple(exp.mixture.components) if exp.mixture else ()
    bounds: dict[str, tuple[float, float]] = {}
    for c in components:
        lo, hi = (float(b) for b in settings.fraction_bounds.get(c, [0.0, 1.0]))
        if not 0.0 <= lo <= hi <= 1.0:
            raise ProblemError(f"the mass-fraction bounds of {c} must satisfy 0 ≤ minimum ≤ maximum ≤ 1")
        bounds[c] = (lo, hi)
    variables: list[Variable] = []
    if len(components) > 1:
        lows = sum(bounds[c][0] for c in components)
        highs = sum(bounds[c][1] for c in components)
        if lows > 1.0 + _TOL or highs < 1.0 - _TOL:
            raise ProblemError("no composition satisfies the mass-fraction bounds: they cannot sum to 1")
        for c in components[:-1]:
            others_low = sum(bounds[o][0] for o in components if o != c)
            lo, hi = bounds[c]
            if len(components) == 2:
                lo = max(lo, 1.0 - bounds[components[1]][1])
            hi = min(hi, 1.0 - others_low)
            if not hi > lo:
                raise ProblemError(f"the search range of the {c} mass fraction has zero width")
            variables.append(Variable(c, f"{c} mass fraction", "fraction", lo, hi))
    for v in exp.inputs:
        if v.kind == "categorical":
            raise ProblemError(
                f"categorical inputs ({v.name}) are not yet supported by the optimizer; "
                "remove the input or keep it constant outside Latos"
            )
        if v.kind == "discrete":
            if not v.levels:
                raise ProblemError(f"list the allowed values of the discrete input {v.name}")
            levels = tuple(sorted({float(level) for level in v.levels}))
            if len(levels) < 2:
                raise ProblemError(f"the discrete input {v.name} needs at least two allowed values")
            variables.append(Variable(v.name, v.name, "discrete", levels[0], levels[-1], v.unit, levels))
        else:
            if v.lower is None or v.upper is None or not v.upper > v.lower:
                raise ProblemError(f"set a minimum and a larger maximum for the input {v.name}")
            variables.append(Variable(v.name, v.name, "continuous", float(v.lower), float(v.upper), v.unit))
    if not variables:
        raise ProblemError("the experiment has nothing to optimize: define mixture components or inputs")
    return DesignSpace(tuple(variables), components, bounds)


def build_problem(exp: ex.Experiment, settings: ex.OptimizationSettings) -> Problem:
    """Set up the experiment for optimization. Raises ProblemError when it cannot be."""
    output = exp.output(settings.output) if settings.output else None
    if output is None:
        raise ProblemError("choose the output to optimize")
    if output.objective == "target" and output.target is None:
        raise ProblemError(f"set the target value of {output.name} on the Definition tab")
    space = _space(exp, settings)
    observations = []
    for s in exp.specimens:
        x = ex.fractions(exp, s) if space.components else None
        values: list[float] | None = []
        reason = ""
        if space.components and x is None:
            values, reason = None, "composition unknown: a component mass is not recorded"
        else:
            values += [x[c] for c in space.free]
            for v in space.variables[len(space.free):]:
                value = s.inputs.get(v.key)
                if value is None or isinstance(value, str):
                    values, reason = None, f"{v.key} not entered"
                    break
                values.append(float(value))
        m = ex.primary_measurement(exp, s, output.name)
        if m is None or m.value is None or not math.isfinite(m.value):
            reason = reason or f"no primary measurement of {output.name}"
        elif not reason and not settings.include_outside:
            outside = space.violations(values)
            if outside:
                reason = "outside the search domain: " + "; ".join(outside)
        observations.append(
            Observation(
                specimen_id=s.id,
                label=s.label,
                values=tuple(values) if values is not None else None,
                fractions=x,
                response=m.value if m else None,
                sem=m.sem if m else None,
                n=m.n if m else None,
                session=m.label if m else "",
                source=m.source_path.name if m and m.record else "",
                sha256=m.source_sha256 if m else None,
                measurement_id=m.id if m else None,
                used=not reason,
                reason=reason,
            )
        )
    return Problem(exp.name, output, space, tuple(observations), settings)


@dataclass(frozen=True)
class Candidate:
    """A proposed condition and what the model predicts there."""

    values: tuple[float, ...]
    fractions: dict[str, float] | None
    mean: float
    sd: float  # latent: uncertainty of the mean response
    predictive_sd: float  # for a new specimen's measurement, under the noise assumption
    score: float
    principle: str


@dataclass(frozen=True)
class SensitivityRow:
    """The model and its recommendation under one observation-noise assumption."""

    additional_noise: float
    fraction_of_sd: float | None
    selected: bool
    values: tuple[float, ...]
    mean: float
    sd: float
    length_scales: tuple[float, ...]
    loo_rmse: float | None
    loo_inside: int | None
    n: int


@dataclass
class Result:
    """One fitted model and its provisional recommendation."""

    problem: Problem
    gp: FittedGP
    x: np.ndarray  # used observations, model inputs in their units
    y: np.ndarray
    sem: np.ndarray
    candidates: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    score: np.ndarray
    excluded: np.ndarray  # candidates within the minimum separation of a measured condition
    choice: Candidate
    baseline: Candidate
    explanation: str
    loo: LooResult | None
    sensitivity: list[SensitivityRow]
    largest_gap: float
    warnings: list[str] = field(default_factory=list)
    record: dict = field(default_factory=dict)


def _distances(points: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """Distance from each point to its nearest measured condition, in unit-box units."""
    if len(measured) == 0:
        return np.full(len(points), np.inf)
    return np.min(np.linalg.norm(points[:, None, :] - measured[None, :, :], axis=2), axis=1)


def _score(gp: FittedGP, unit_obs: np.ndarray, unit_cand: np.ndarray, output: ex.OutputDefinition, acquisition: str):
    """Acquisition values on the candidates, with what each one is measured against."""
    mean, sd = gp.predict(unit_cand)
    at_obs, _ = gp.predict(unit_obs)
    if acquisition == "explore":
        return mean, sd, sd.copy(), None
    if output.objective == "target":
        best = float(np.min(np.abs(at_obs - output.target)))
        return mean, sd, expected_target_improvement(mean, sd, output.target, best), best
    sign = 1.0 if output.objective == "maximize" else -1.0
    best = float(np.max(sign * at_obs))  # plug-in incumbent: the model's best at a measured point
    xi = XI * float(np.std(gp.y))
    return mean, sd, expected_improvement(sign * mean, sd, best, xi), best


def _pick(score: np.ndarray, sd: np.ndarray, excluded: np.ndarray) -> tuple[int, bool]:
    """Best allowed candidate; falls back to the most uncertain one when every score is zero."""
    allowed = np.where(excluded, -np.inf, score)
    index = int(np.argmax(allowed))
    if allowed[index] > 0:
        return index, False
    return int(np.argmax(np.where(excluded, -np.inf, sd))), True


def _explain(output: ex.OutputDefinition, acquisition: str, mean: float, sd: float, best: float | None,
             xi: float, fallback: bool) -> str:
    unit = f" {output.unit}" if output.unit else ""
    if acquisition == "explore" or fallback:
        lead = (
            "Expected improvement is zero at every allowed candidate, so the most uncertain one is shown. "
            if fallback else ""
        )
        return lead + f"Selected for the largest model uncertainty (SD {sd:.4g}{unit}); the predicted response is not used."
    if output.objective == "target":
        gap = abs(mean - output.target)
        if gap < best:
            return (
                f"Seeks the target: the predicted response is {gap:.4g}{unit} from {output.target:g}{unit}, closer "
                f"than the best measured composition as the model sees it ({best:.4g}{unit})."
            )
        return (
            f"Explores: the predicted response is no closer to {output.target:g}{unit} than the best measured "
            f"composition ({gap:.4g} against {best:.4g}{unit}); its value comes from the model uncertainty (SD {sd:.4g}{unit})."
        )
    sign = 1.0 if output.objective == "maximize" else -1.0
    gain = sign * mean - best - xi
    z = gain / max(sd, 1e-12)
    exploit = gain * 0.5 * math.erfc(-z / math.sqrt(2))
    explore = sd * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    better = "higher" if sign > 0 else "lower"
    incumbent = sign * best
    if gain > 0 and exploit >= explore:
        return (
            f"Seeks improvement: the predicted response {mean:+.4f}{unit} is {better} than the best predicted at a "
            f"measured composition ({incumbent:+.4f}{unit}). Of the expected improvement, {exploit:.3g} comes from "
            f"the predicted gain and {explore:.3g} from uncertainty."
        )
    return (
        f"Explores: the predicted response {mean:+.4f}{unit} is not clearly {better} than the best predicted at a "
        f"measured composition ({incumbent:+.4f}{unit}); most of the expected improvement ({explore:.3g} of "
        f"{max(exploit, 0) + explore:.3g}) comes from the model uncertainty (SD {sd:.4g}{unit})."
    )


def _largest_gap(space: DesignSpace, unit_obs: np.ndarray) -> float:
    """Radius of the largest unsampled region, in unit-box units (half the widest gap in 1-D)."""
    if space.d == 1:
        edges = np.concatenate(([0.0], np.sort(np.clip(unit_obs[:, 0], 0, 1)), [1.0]))
        gaps = np.diff(edges)
        interior = gaps[1:-1].max() / 2 if len(gaps) > 2 else 0.0
        return float(max(interior, gaps[0], gaps[-1]))
    probe = qmc.Sobol(d=space.d, scramble=True, seed=0).random_base2(12)
    return float(np.max(_distances(probe, unit_obs)))


def recommend(problem: Problem, sources: dict[str, str] | None = None) -> Result:
    """Fit the model to the usable specimens and select the next condition.

    Raises:
        ProblemError: With too few independent specimens, or when every
            candidate lies within the minimum separation of a measured condition.
    """
    settings, space, output = problem.settings, problem.space, problem.output
    used = problem.used
    n, d = len(used), space.d
    if n < problem.minimum_needed:
        raise ProblemError(
            f"{n} independent specimen(s) can be used; at least {problem.minimum_needed} are needed to fit a "
            f"model of {d} variable{'s' if d != 1 else ''}"
        )
    if settings.acquisition not in ("ei", "explore"):
        raise ProblemError(f"unknown acquisition {settings.acquisition!r}")
    x = np.array([o.values for o in used], dtype=float)
    y = np.array([o.response for o in used], dtype=float)
    sem = np.array([np.nan if o.sem is None else o.sem for o in used], dtype=float)
    unit_obs = space.to_unit(x)
    candidates = space.candidates(settings.seed)
    unit_cand = space.to_unit(candidates)
    unit_measured = space.to_unit(problem.measured_points)
    nearest = _distances(unit_cand, unit_measured)
    excluded = (
        np.zeros(len(candidates), dtype=bool)
        if settings.allow_replicates
        else nearest <= max(settings.min_separation, 0.0) + _TOL
    )
    if excluded.all():
        raise ProblemError("every candidate lies within the minimum separation of a measured condition")
    noise = float(settings.additional_noise)
    typical_sem = float(np.nanmedian(sem)) if np.isfinite(sem).any() else 0.0

    def candidate(gp, index, mean, sd, score, principle) -> Candidate:
        row = candidates[index]
        return Candidate(
            values=tuple(float(v) for v in row),
            fractions=space.fractions(row),
            mean=float(mean[index]),
            sd=float(sd[index]),
            predictive_sd=float(np.sqrt(sd[index] ** 2 + noise**2 + typical_sem**2)),
            score=float(score[index]),
            principle=principle,
        )

    gp = fit_gp(unit_obs, y, sem, noise, settings.seed)
    mean, sd, score, best = _score(gp, unit_obs, unit_cand, output, settings.acquisition)
    index, fallback = _pick(score, sd, excluded)
    xi = XI * float(np.std(y))
    principle = "Largest model uncertainty" if settings.acquisition == "explore" else (
        "Expected closeness to the target" if output.objective == "target" else "Expected improvement"
    )
    choice = candidate(gp, index, mean, sd, score, principle)
    explanation = _explain(output, settings.acquisition, choice.mean, choice.sd, best, xi, fallback)
    spread = np.where(excluded, -np.inf, nearest)
    baseline = candidate(
        gp, int(np.argmax(spread)), mean, sd, score,
        "Space-filling: the allowed condition farthest from every measured one, ignoring the responses",
    )
    loo = leave_one_out(unit_obs, y, sem, noise, settings.seed)

    reference = float(np.std(y, ddof=1)) if n > 1 else 0.0
    levels = [(f * reference, f) for f in SENSITIVITY_FRACTIONS]
    if not any(abs(level - noise) <= 1e-12 for level, _f in levels):
        levels.append((noise, None))
    sensitivity = []
    for level, fraction in sorted(levels, key=lambda item: item[0]):
        g = gp if abs(level - noise) <= 1e-12 else fit_gp(unit_obs, y, sem, level, settings.seed)
        m, s, sc, _best = _score(g, unit_obs, unit_cand, output, settings.acquisition)
        i, _ = _pick(sc, s, excluded)
        fold = loo if g is gp else leave_one_out(unit_obs, y, sem, level, settings.seed)
        sensitivity.append(
            SensitivityRow(
                additional_noise=level,
                fraction_of_sd=fraction,
                selected=g is gp,
                values=tuple(float(v) for v in candidates[i]),
                mean=float(m[i]),
                sd=float(s[i]),
                length_scales=g.length_scales,
                loo_rmse=fold.rmse if fold else None,
                loo_inside=int(fold.inside.sum()) if fold else None,
                n=n,
            )
        )

    warnings = [
        f"{n} independent specimens for {d} variable{'s' if d != 1 else ''}: the model's hyperparameters are weakly "
        "determined, and its 95 % band is model-based and illustrative; its coverage has not been validated."
    ]
    warnings += [f"Fitted {item}." for item in gp.at_bounds]
    if noise == 0:
        warnings.append(
            "Observation noise is the within-specimen SEM only, a lower bound on specimen-to-specimen "
            "variation; see the noise sensitivity."
        )
    missing = [o.specimen_id for o in used if o.sem is None]
    if missing:
        warnings.append(f"No measured SEM for {', '.join(missing)}: only the additional noise is assumed there.")
    seen: dict[tuple, str] = {}
    for o, row in zip(used, np.round(unit_obs, 9)):
        key = tuple(row)
        if key in seen:
            warnings.append(f"{seen[key]} and {o.specimen_id} share one condition; both are kept as replicates.")
        seen.setdefault(key, o.specimen_id)
    for o in used:
        state = (sources or {}).get(o.measurement_id or "")
        if state in ("changed", "missing"):
            warnings.append(f"The recording behind {o.specimen_id} is {state}; its recorded value is used as saved.")
    if d == 1 and not (x[:, 0].min() - _TOL <= choice.values[0] <= x[:, 0].max() + _TOL):
        warnings.append("The candidate lies beyond the measured range: the model extrapolates there.")
    left_out = [o for o in problem.observations if not o.used]
    if left_out:
        warnings.append(
            "Not in the fit: " + "; ".join(f"{o.specimen_id} ({o.reason})" for o in left_out) + "."
        )

    result = Result(
        problem=problem, gp=gp, x=x, y=y, sem=sem, candidates=candidates, mean=mean, sd=sd, score=score,
        excluded=excluded, choice=choice, baseline=baseline, explanation=explanation, loo=loo,
        sensitivity=sensitivity, largest_gap=_largest_gap(space, unit_obs), warnings=warnings,
    )
    result.record = _record(result)
    return result


def _record(r: Result) -> dict:
    """Everything needed to reproduce and review the recommendation."""
    p = r.problem

    def point(c: Candidate) -> dict:
        return {
            "values": dict(zip((v.label for v in p.space.variables), c.values)),
            "fractions": c.fractions,
            "predicted_mean": c.mean,
            "latent_sd": c.sd,
            "predictive_sd": c.predictive_sd,
            "acquisition_value": c.score,
            "principle": c.principle,
        }

    return {
        "status": STATUS,
        "software": {"latos": __version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
                     "numpy": np.__version__},
        "experiment": p.experiment,
        "output": {"name": p.output.name, "unit": p.output.unit, "method": p.output.method,
                   "objective": p.output.objective, "target": p.output.target},
        "settings": asdict(p.settings),
        "design_space": [
            {"variable": v.label, "kind": v.kind, "lower": v.lower, "upper": v.upper, "unit": v.unit,
             "levels": list(v.levels)} for v in p.space.variables
        ],
        "model": {
            "kernel": "amplitude² × Matérn(ν = 5/2), one length-scale per input",
            "input_scaling": "each input mapped to [0, 1] over its search range",
            "response_scaling": "centred and divided by the SD of the observed responses",
            "noise": "fixed per specimen: SEM² + additional_noise²",
            "length_scale_bounds_range_units": list(LENGTH_SCALE_BOUNDS),
            "restarts": RESTARTS,
            "seed": p.settings.seed,
            "fitted_length_scales_range_units": list(r.gp.length_scales),
            "fitted_amplitude": r.gp.amplitude,
            "log_marginal_likelihood": r.gp.log_marginal_likelihood,
            "candidates": len(r.candidates),
            "excluded_by_separation": int(r.excluded.sum()),
        },
        "observations": [
            {"specimen": o.specimen_id, "inputs": list(o.values) if o.values else None, "response": o.response,
             "sem": o.sem, "cycles": o.n, "session": o.session, "source": o.source, "sha256": o.sha256,
             "used": o.used, "reason": o.reason}
            for o in p.observations
        ],
        "recommendation": point(r.choice),
        "explanation": r.explanation,
        "baseline": point(r.baseline),
        "leave_one_out": None if r.loo is None else {
            "rmse": r.loo.rmse, "inside_nominal_95": int(r.loo.inside.sum()), "n": len(r.loo.observed),
        },
    }


@dataclass(frozen=True)
class Preparation:
    """Component masses to weigh for a recommended composition."""

    total_g: float
    resolution_g: float
    masses_g: dict[str, float]
    fractions: dict[str, float]  # from the rounded masses
    feasible: bool
    notes: list[str]


def prepare(space: DesignSpace, fractions: dict[str, float], total_g: float | None,
            resolution_g: float) -> Preparation:
    """Masses for `fractions` at a researcher-chosen total, rounded to the balance resolution.

    The last component takes the remainder, so the masses add up to the total.
    The rounded composition is checked against the search domain again.
    """
    if total_g is None or not total_g > 0:
        raise ProblemError("enter the total mass to prepare; Latos does not choose it")
    if not resolution_g > 0:
        raise ProblemError("the balance resolution must be positive")
    if not space.components:
        raise ProblemError("the experiment has no mixture to prepare")
    notes = []
    steps = round(total_g / resolution_g)
    if abs(steps * resolution_g - total_g) > resolution_g * 1e-6:
        notes.append(f"The total was rounded to {steps * resolution_g:.10g} g, a whole number of balance steps.")
    components = space.components
    free = components[:-1]
    exact = [fractions[c] * steps for c in free]

    def build(counts) -> tuple[dict[str, float], dict[str, float], bool]:
        counts = list(counts) + [steps - sum(counts)]
        masses = {c: k * resolution_g for c, k in zip(components, counts)}
        actual = {c: k / steps for c, k in zip(components, counts)}
        ok = counts[-1] >= 0 and all(
            space.bounds[c][0] - _TOL <= actual[c] <= space.bounds[c][1] + _TOL for c in components
        )
        return masses, actual, ok

    masses, actual, ok = build([round(e) for e in exact])
    if not ok and len(free) <= 10:
        options = [build(combo) for combo in product(*((math.floor(e), math.ceil(e)) for e in exact))]
        feasible = [o for o in options if o[2]]
        if feasible:
            masses, actual, ok = min(feasible, key=lambda o: max(abs(o[1][c] - fractions[c]) for c in components))
            notes.append("Rounded in the direction that keeps the composition inside the search domain.")
    if not ok:
        notes.append("No rounding to the balance resolution keeps the composition inside the search domain.")
    deviation = max(abs(actual[c] - fractions[c]) for c in components)
    notes.append(f"Largest change of a mass fraction from rounding: {deviation:.2e}.")
    return Preparation(float(steps * resolution_g), float(resolution_g), masses, actual, ok, notes)
