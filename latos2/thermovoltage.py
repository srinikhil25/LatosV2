"""Event-marked heating/cooling cycles and their apparent dynamic coefficients.

Ported from Latos V1 (`analysis/thermovoltage/cycles.py` and the closed-form
fit in `slope.py`) after review. The arithmetic is V1's; the names no longer
call the slope a Seebeck coefficient.

The measurand
-------------
For each cycle the response is the ordinary least-squares slope of the oriented
voltage against the temperature difference::

    σ·V_rec(t) = S_app · ΔT(t) + b,      ΔT = T_hot − T_cold

`V_rec` is the logger's V(+) − V(−) in mV. `σ` is the wiring sign: +1 with the
positive lead on the cold electrode, −1 with it on the hot electrode, so that
σ·V_rec is always V_cold − V_hot and a positive S_app means the cold electrode
sits at the higher potential. The series is oriented, not the slope.

S_app is fitted three ways per cycle: over the whole cycle (heating and cooling
together), over the heating branch alone and over the cooling branch alone. It
is a dynamic, operational quantity of this drive programme. The cell is never
at rest during a cycle, so S_app is **not** a steady-state Seebeck coefficient,
and a gap between the heating and cooling slopes is hysteresis.

Windows
-------
Event marks numbered 1, 2, 3, … give ``on_1, off_1, …, on_n, off_n, end``: odd
marks switch the drive on, even marks switch it off, the last mark ends the
run. Cycle k is the half-open window ``[on_k, on_{k+1})``; its heating branch
is ``[on_k, off_k)`` and its cooling branch ``[off_k, on_{k+1})``. Half-open, so
a sample at a switch belongs to exactly one branch.

Reduction
---------
The cycles of one mounting are repeated measurements of one specimen::

    mean = mean(S_1 … S_k),  SEM = sd(S_1 … S_k, ddof=1) / sqrt(k)

Signs are kept. The SEM is a within-specimen, within-session estimate only:
it says nothing about remounting, and from three cycles (two degrees of
freedom) it is itself uncertain. The least-squares standard error of a single
cycle's slope is a diagnostic, never a measurement uncertainty: samples 0.2 s
apart are not independent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from latos2.hioki_reader import LoggerTrace
from latos2.units import check_temperature_unit, quantization_step, voltage_factor_to_mv

__all__ = [
    "KINDS",
    "KIND_LABELS",
    "QUANTITY",
    "SIGN_CONVENTION",
    "STATEMENT",
    "UNIT",
    "WIRING",
    "Analysis",
    "ChannelRoles",
    "CycleBoundaryError",
    "CycleFit",
    "CycleReduction",
    "CycleWindow",
    "LineFit",
    "analyze",
    "event_boundaries",
    "fit_cycle",
    "fit_line",
    "reduce_cycles",
    "windows_from_boundaries",
]

# The words every view and export uses, kept in one place.
QUANTITY = "Apparent dynamic coefficient S_app"
UNIT = "mV/K"
KINDS = ("combined", "heating", "cooling")
KIND_LABELS = {
    "combined": "Whole cycle (heating + cooling)",
    "heating": "Heating branch only",
    "cooling": "Cooling branch only",
}
STATEMENT = (
    "S_app is the least-squares slope of the oriented cell voltage against "
    "ΔT = T_hot − T_cold over event-marked heating/cooling cycles. It is a dynamic, "
    "operational quantity of this drive programme and is not a steady-state Seebeck "
    "coefficient."
)
SIGN_CONVENTION = (
    "Oriented voltage = σ·V_rec, with V_rec = V(+) − V(−) as logged; σ = +1 with V+ on the "
    "cold electrode, −1 with V+ on the hot electrode. Positive S_app: the cold electrode "
    "is at the higher potential."
)
WIRING = {
    +1: "V+ on the COLD electrode",
    -1: "V+ on the HOT electrode",
}

# Two boundaries per cycle and one for the end of the run.
_MIN_BOUNDARIES = 3
# A branch needs enough samples for its own slope to mean anything.
_MIN_BRANCH_SAMPLES = 3
# The share of a branch, at each end, averaged to decide whether ΔT rose or fell.
_DIRECTION_SHARE = 0.2


class CycleBoundaryError(ValueError):
    """The switching times offered do not describe a run of whole cycles."""


@dataclass(frozen=True, slots=True)
class LineFit:
    """Least-squares fit of ``y = slope·x + intercept``.

    Attributes:
        slope: In units of y per unit of x.
        intercept: In units of y.
        slope_stderr: Textbook standard error of the slope. Diagnostic only.
        r_squared: Coefficient of determination, NaN when y does not vary.
        n: Number of points.
    """

    slope: float
    intercept: float
    slope_stderr: float
    r_squared: float
    n: int


def fit_line(x: np.ndarray, y: np.ndarray) -> LineFit:
    """Closed-form ordinary least squares, as V1's `fit_seebeck_slope`.

    With two points the residual variance has no degrees of freedom, so the
    standard error is NaN rather than zero.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(x.size)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    dx = x - x_mean
    s_xx = float(np.sum(dx * dx))
    if s_xx <= 0:
        return LineFit(math.nan, math.nan, math.nan, math.nan, n)
    slope = float(np.sum(dx * (y - y_mean)) / s_xx)
    intercept = y_mean - slope * x_mean
    residuals = y - (slope * x + intercept)
    ss_res = float(np.sum(residuals**2))
    dof = n - 2
    slope_stderr = math.sqrt(ss_res / dof / s_xx) if dof > 0 else math.nan
    ss_tot = float(np.sum((y - y_mean) ** 2))
    # Every y identical: there is no variance to explain, so R² is meaningless.
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return LineFit(slope, intercept, slope_stderr, r_squared, n)


@dataclass(frozen=True, slots=True)
class CycleWindow:
    """One thermal cycle: ``[on_s, end_s)``, heating until `off_s`."""

    index: int
    on_s: float
    off_s: float
    end_s: float

    @property
    def heating_s(self) -> float:
        return self.off_s - self.on_s

    @property
    def cooling_s(self) -> float:
        return self.end_s - self.off_s


@dataclass(frozen=True, slots=True)
class CycleFit:
    """The three fits of one cycle, with their diagnostics.

    Attributes:
        window: The cycle that was fitted.
        n_heating: Samples in the heating branch.
        n_cooling: Samples in the cooling branch.
        combined: Fit over the whole cycle, in mV/K and mV.
        heating: Fit over the heating branch alone.
        cooling: Fit over the cooling branch alone.
        delta_t_min_k: The smallest ΔT in the window.
        delta_t_max_k: The largest.
        mean_temperature: Mean of the two electrode temperatures, in their unit.
        problems: Why this window is not a valid cycle, if it is not. A cycle
            with problems is left out of every reduction.
    """

    window: CycleWindow
    n_heating: int
    n_cooling: int
    combined: LineFit
    heating: LineFit
    cooling: LineFit
    delta_t_min_k: float
    delta_t_max_k: float
    mean_temperature: float
    problems: tuple[str, ...] = ()

    def fit(self, kind: str) -> LineFit:
        return {"combined": self.combined, "heating": self.heating, "cooling": self.cooling}[kind]


@dataclass(frozen=True, slots=True)
class CycleReduction:
    """The accepted cycles of one specimen, reduced.

    Attributes:
        k: How many cycles were reduced.
        mean: Their signed mean.
        sd: Sample standard deviation, or None with one cycle.
        sem: `sd / sqrt(k)`, or None with one cycle. Within-specimen only.
        dof: `k - 1`.
    """

    k: int
    mean: float
    sd: float | None
    sem: float | None
    dof: int


def windows_from_boundaries(boundaries: Sequence[float]) -> tuple[CycleWindow, ...]:
    """Turn ``on_1, off_1, …, on_n, off_n, end`` into one window per cycle."""
    times = [float(b) for b in boundaries]
    count = len(times)
    if count < _MIN_BOUNDARIES or count % 2 == 0:
        raise CycleBoundaryError(
            f"{count} boundaries cannot describe whole cycles: each cycle needs a switch-on "
            "and a switch-off, and the run needs one end, so the count must be odd and at "
            f"least {_MIN_BOUNDARIES}"
        )
    if not all(math.isfinite(t) for t in times):
        raise CycleBoundaryError("a boundary is not a finite time")
    for earlier, later in pairwise(times):
        if not earlier < later:
            raise CycleBoundaryError(
                f"boundaries must increase; {later:g} s does not come after {earlier:g} s"
            )
    return tuple(
        CycleWindow(index=k + 1, on_s=times[2 * k], off_s=times[2 * k + 1], end_s=times[2 * k + 2])
        for k in range(count // 2)
    )


def event_boundaries(numbers: Sequence[int], times_s: Sequence[float]) -> list[float]:
    """The switching times given by a run's event marks.

    The marks must be numbered 1, 2, 3, … in the order they were pressed, and
    there must be an odd number of them: two per cycle and one for the end.
    """
    if len(numbers) != len(times_s):
        raise ValueError("numbers and times_s must have the same length")
    if not numbers:
        raise CycleBoundaryError("the recording has no event marks")
    expected = list(range(1, len(numbers) + 1))
    if list(numbers) != expected:
        found = ", ".join(str(n) for n in numbers)
        raise CycleBoundaryError(
            f"the event marks are numbered {found}; expected 1 to {len(numbers)} in order, "
            "so a mark is missing, repeated or out of sequence"
        )
    count = len(numbers)
    if count < _MIN_BOUNDARIES or count % 2 == 0:
        raise CycleBoundaryError(
            f"{count} event mark(s) cannot describe whole cycles: each cycle needs a "
            "switch-on mark and a switch-off mark, and the run needs one end mark, so the "
            f"count must be odd and at least {_MIN_BOUNDARIES}"
        )
    return [float(t) for t in times_s]


def _direction(delta_t: np.ndarray) -> float:
    """Mean of the last share of a branch minus the mean of its first share."""
    span = max(1, round(delta_t.size * _DIRECTION_SHARE))
    return float(np.mean(delta_t[-span:]) - np.mean(delta_t[:span]))


def fit_cycle(
    time_s: np.ndarray,
    cold: np.ndarray,
    hot: np.ndarray,
    voltage_mv: np.ndarray,
    window: CycleWindow,
    *,
    wiring_sign: int,
) -> CycleFit:
    """Fit the whole cycle and each branch of one window.

    Args:
        time_s: Sample times.
        cold: Cold-electrode temperature, one value per sample.
        hot: Hot-electrode temperature, same unit.
        voltage_mv: Recorded cell voltage in mV, exactly as wired.
        window: The cycle to fit.
        wiring_sign: +1 with V+ on the cold electrode, −1 with it on the hot one.
    """
    time = np.asarray(time_s, dtype=np.float64)
    if window.on_s < time[0] or window.end_s > time[-1]:
        raise CycleBoundaryError(
            f"cycle {window.index} runs from {window.on_s:g} s to {window.end_s:g} s, outside "
            f"the recording ({time[0]:g} s to {time[-1]:g} s)"
        )
    cold_t = np.asarray(cold, dtype=np.float64)
    hot_t = np.asarray(hot, dtype=np.float64)
    delta_t = hot_t - cold_t
    oriented = np.asarray(voltage_mv, dtype=np.float64) * wiring_sign

    heating = (time >= window.on_s) & (time < window.off_s)
    cooling = (time >= window.off_s) & (time < window.end_s)
    for name, branch in (("heating", heating), ("cooling", cooling)):
        held = int(np.count_nonzero(branch))
        if held < _MIN_BRANCH_SAMPLES:
            raise CycleBoundaryError(
                f"the {name} branch of cycle {window.index} holds {held} sample(s); "
                f"at least {_MIN_BRANCH_SAMPLES} are needed"
            )
    whole = heating | cooling

    combined = fit_line(delta_t[whole], oriented[whole])
    problems: list[str] = []
    if not _direction(delta_t[heating]) > 0:
        problems.append(
            f"ΔT does not rise while the drive is on in cycle {window.index}: either the "
            "event marks or the hot and cold channel assignment are wrong"
        )
    if not _direction(delta_t[cooling]) < 0:
        problems.append(
            f"ΔT does not fall while the drive is off in cycle {window.index}: either the "
            "event marks or the hot and cold channel assignment are wrong"
        )
    if not math.isfinite(combined.slope):
        problems.append(f"ΔT does not vary in cycle {window.index}, so no slope can be fitted")

    return CycleFit(
        window=window,
        n_heating=int(np.count_nonzero(heating)),
        n_cooling=int(np.count_nonzero(cooling)),
        combined=combined,
        heating=fit_line(delta_t[heating], oriented[heating]),
        cooling=fit_line(delta_t[cooling], oriented[cooling]),
        delta_t_min_k=float(np.min(delta_t[whole])),
        delta_t_max_k=float(np.max(delta_t[whole])),
        mean_temperature=float(np.mean((cold_t[whole] + hot_t[whole]) / 2.0)),
        problems=tuple(problems),
    )


def reduce_cycles(values: Sequence[float]) -> CycleReduction:
    """Reduce the cycle values of one specimen to a signed mean and its SEM."""
    cycles = np.asarray(list(values), dtype=np.float64)
    if cycles.size == 0:
        raise ValueError("there are no cycles to reduce")
    if not np.all(np.isfinite(cycles)):
        raise ValueError("a cycle value is not finite")
    k = int(cycles.size)
    mean = float(np.mean(cycles))
    if k == 1:
        return CycleReduction(k=1, mean=mean, sd=None, sem=None, dof=0)
    sd = float(np.std(cycles, ddof=1))
    return CycleReduction(k=k, mean=mean, sd=sd, sem=sd / math.sqrt(k), dof=k - 1)


@dataclass(frozen=True, slots=True)
class ChannelRoles:
    """Which recorded channel plays which part. Chosen by the researcher."""

    hot: str
    cold: str
    voltage: str


@dataclass(frozen=True)
class Analysis:
    """Everything one analysed recording produced, for views and export.

    Attributes:
        trace: The recording as read.
        roles: The channel assignment used.
        wiring_sign: σ, +1 or −1.
        voltage_unit: The voltage channel's header unit.
        voltage_factor_to_mv: The factor applied to reach mV.
        voltage_step: The voltage channel's reading step, in its header unit.
        temperature_unit: The thermocouples' header unit.
        temperature_step: Their reading step, in that unit.
        time_s: Sample times.
        t_hot: Hot-electrode temperature.
        t_cold: Cold-electrode temperature.
        delta_t_k: T_hot − T_cold, in K.
        voltage_recorded_mv: V(+) − V(−) as logged, in mV.
        voltage_oriented_mv: σ·V_rec, in mV.
        cycles: One fit per event-marked cycle.
        accepted: Indices (1-based) of the cycles without problems.
        reductions: Mean and SEM per kind over the accepted cycles, or None
            when no cycle was accepted.
    """

    trace: LoggerTrace
    roles: ChannelRoles
    wiring_sign: int
    voltage_unit: str
    voltage_factor_to_mv: float
    voltage_step: float
    temperature_unit: str
    temperature_step: float
    time_s: np.ndarray
    t_hot: np.ndarray
    t_cold: np.ndarray
    delta_t_k: np.ndarray
    voltage_recorded_mv: np.ndarray
    voltage_oriented_mv: np.ndarray
    cycles: tuple[CycleFit, ...]
    accepted: tuple[int, ...]
    reductions: dict[str, CycleReduction] | None

    @property
    def problems(self) -> tuple[str, ...]:
        return tuple(p for cycle in self.cycles for p in cycle.problems)


def analyze(trace: LoggerTrace, roles: ChannelRoles, wiring_sign: int) -> Analysis:
    """Analyse one recording with an explicit channel assignment and wiring.

    Raises:
        ValueError: If a channel is assigned twice or `wiring_sign` is not ±1.
        KeyError: If an assigned channel was not recorded.
        UnitError: If a channel's unit is not accepted.
        CycleBoundaryError: If the event marks do not describe whole cycles.
    """
    if wiring_sign not in WIRING:
        raise ValueError("the wiring must be declared: +1 (V+ on cold) or −1 (V+ on hot)")
    if len({roles.hot, roles.cold, roles.voltage}) != 3:
        raise ValueError("the hot, cold and voltage channels must be three different channels")
    hot = trace.channel(roles.hot)
    cold = trace.channel(roles.cold)
    voltage = trace.channel(roles.voltage)
    temperature_unit = check_temperature_unit(hot.unit)
    if check_temperature_unit(cold.unit) != temperature_unit:
        raise ValueError(
            f"the hot channel is in {hot.unit} and the cold channel in {cold.unit}; "
            "ΔT needs both in the same unit"
        )
    factor = voltage_factor_to_mv(voltage.unit)
    if not trace.has_event_column:
        raise CycleBoundaryError("the recording has no Event column")
    boundaries = event_boundaries(
        [e.number for e in trace.events], [e.time_s for e in trace.events]
    )
    windows = windows_from_boundaries(boundaries)

    recorded_mv = voltage.values * factor
    cycles = tuple(
        fit_cycle(trace.time_s, cold.values, hot.values, recorded_mv, w, wiring_sign=wiring_sign)
        for w in windows
    )
    accepted = tuple(c.window.index for c in cycles if not c.problems)
    reductions = (
        {
            kind: reduce_cycles([c.fit(kind).slope for c in cycles if not c.problems])
            for kind in KINDS
        }
        if accepted
        else None
    )
    return Analysis(
        trace=trace,
        roles=roles,
        wiring_sign=wiring_sign,
        voltage_unit=voltage.unit,
        voltage_factor_to_mv=factor,
        voltage_step=quantization_step(voltage.values),
        temperature_unit=temperature_unit,
        temperature_step=quantization_step(hot.values),
        time_s=trace.time_s,
        t_hot=hot.values,
        t_cold=cold.values,
        delta_t_k=hot.values - cold.values,
        voltage_recorded_mv=recorded_mv,
        voltage_oriented_mv=recorded_mv * wiring_sign,
        cycles=cycles,
        accepted=accepted,
        reductions=reductions,
    )
