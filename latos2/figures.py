"""The three scientific figures, drawn onto a matplotlib Figure.

The same functions draw the on-screen plots and the exported files, so what a
researcher sees is what is saved. A view (all cycles or one cycle, optionally
one branch emphasised) changes only limits and emphasis, never the data.
"""

from __future__ import annotations

import math

import matplotlib
import numpy as np
from matplotlib import patheffects
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from latos2.hioki_reader import LoggerTrace
from latos2.thermovoltage import (
    KIND_LABELS,
    KINDS,
    UNIT,
    Analysis,
    ChannelRoles,
    CycleBoundaryError,
    CycleWindow,
    event_boundaries,
    windows_from_boundaries,
)
from latos2.units import voltage_factor_to_mv

PLOT_STYLE = {
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": "#e4e4e4",
    "grid.linewidth": 0.6,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
}
matplotlib.rcParams.update(PLOT_STYLE)

HEATING_COLOR = "#c0392b"
COOLING_COLOR = "#1f6fb2"
COMBINED_COLOR = "#222222"
MUTED = "#666666"
KIND_COLORS = {"combined": COMBINED_COLOR, "heating": HEATING_COLOR, "cooling": COOLING_COLOR}
# Fit lines are darker than their points and dashed differently, with a white
# halo, so each stays distinguishable on top of the data.
FIT_STYLE = {
    "combined": {"color": COMBINED_COLOR, "ls": "-", "lw": 1.7},
    "heating": {"color": "#7d1a12", "ls": (0, (6, 3)), "lw": 1.7},
    "cooling": {"color": "#0b3f73", "ls": (0, (7, 2, 1.5, 2)), "lw": 1.7},
}
FIT_LABELS = {
    "combined": "Whole-cycle fit",
    "heating": "Heating-branch fit",
    "cooling": "Cooling-branch fit",
}
FOOTNOTE = "Dynamic apparent coefficient of this drive programme; not a steady-state Seebeck coefficient."

_HALO = [patheffects.withStroke(linewidth=3.4, foreground="white")]
_DELTA_T_LABEL = r"$\Delta T = T_\mathrm{hot} - T_\mathrm{cold}$ (K)"


def _windows(trace: LoggerTrace) -> tuple[CycleWindow, ...]:
    try:
        return windows_from_boundaries(
            event_boundaries([e.number for e in trace.events], [e.time_s for e in trace.events])
        )
    except CycleBoundaryError:
        return ()


def _legend_columns(fig: Figure, entries: int) -> int:
    """One legend row on a wide figure, two on a narrow one, so it never runs off."""
    return entries if fig.get_figwidth() >= 7.8 else math.ceil(entries / 2)


def _fit_y(ax, x: np.ndarray, series: list[np.ndarray], lo: float, hi: float) -> None:
    """Scale the y axis to the samples inside ``[lo, hi]``."""
    inside = (x >= lo) & (x <= hi)
    values = np.concatenate([s[inside] for s in series])
    if values.size:
        low, high = float(values.min()), float(values.max())
        pad = 0.07 * (high - low) if high > low else 0.5
        ax.set_ylim(low - pad, high + pad)


def draw_timeseries(
    fig: Figure,
    trace: LoggerTrace,
    roles: ChannelRoles,
    cycle: int | None = None,
    segment: str | None = None,
) -> None:
    """Electrode temperatures, ΔT and the recorded voltage against time.

    Needs no wiring declaration: the voltage is drawn exactly as logged.
    `cycle` zooms the time axis to one cycle; `segment` ("heating" or
    "cooling") strengthens that branch's shading.
    """
    fig.clear()
    hot = trace.channel(roles.hot)
    cold = trace.channel(roles.cold)
    voltage = trace.channel(roles.voltage)
    v_mv = voltage.values * voltage_factor_to_mv(voltage.unit)
    t = trace.time_s
    delta_t = hot.values - cold.values
    windows = _windows(trace)

    ax_t, ax_dt, ax_v = fig.subplots(3, 1, sharex=True)
    ax_t.plot(t, hot.values, color=HEATING_COLOR, lw=1.1)
    ax_t.plot(t, cold.values, color=COOLING_COLOR, lw=1.1)
    ax_t.set_ylabel(f"Temperature ({hot.unit})")
    ax_dt.plot(t, delta_t, color=COMBINED_COLOR, lw=1.1)
    ax_dt.set_ylabel(_DELTA_T_LABEL)
    ax_v.plot(t, v_mv, color="#5b3a8c", lw=1.1)
    ax_v.set_ylabel(r"$V_\mathrm{rec}$ = V(+) $-$ V($-$) (mV)")
    ax_v.set_xlabel("Time since first sample (s)")

    for ax in (ax_t, ax_dt, ax_v):
        for w in windows:
            selected = cycle == w.index
            ax.axvspan(
                w.on_s,
                w.off_s,
                color=HEATING_COLOR,
                alpha=0.2 if selected and segment == "heating" else 0.07,
                lw=0,
            )
            ax.axvspan(
                w.off_s,
                w.end_s,
                color=COOLING_COLOR,
                alpha=0.2 if selected and segment == "cooling" else 0.07,
                lw=0,
            )
        for e in trace.events:
            ax.axvline(e.time_s, color="#777777", lw=0.8, ls="--")
    marks = ax_t.secondary_xaxis("top")
    marks.set_xticks([e.time_s for e in trace.events], labels=[str(e.number) for e in trace.events])
    marks.tick_params(length=0, labelsize=8, labelcolor=MUTED)
    marks.set_xlabel("Event mark", fontsize=8, color=MUTED)

    if cycle is not None and windows:
        w = windows[cycle - 1]
        pad = 0.03 * (w.end_s - w.on_s)
        lo, hi = w.on_s - pad, w.end_s + pad
        ax_v.set_xlim(lo, hi)
        _fit_y(ax_t, t, [hot.values, cold.values], lo, hi)
        _fit_y(ax_dt, t, [delta_t], lo, hi)
        _fit_y(ax_v, t, [v_mv], lo, hi)
    else:
        ax_v.set_xlim(float(t[0]), float(t[-1]))

    handles = [
        Line2D([], [], color=HEATING_COLOR, lw=1.4, label=rf"$T_\mathrm{{hot}}$ ({hot.name})"),
        Line2D([], [], color=COOLING_COLOR, lw=1.4, label=rf"$T_\mathrm{{cold}}$ ({cold.name})"),
        Patch(color=HEATING_COLOR, alpha=0.2, label="Heating (drive on)"),
        Patch(color=COOLING_COLOR, alpha=0.2, label="Cooling (drive off)"),
        Line2D([], [], color="#777777", lw=0.9, ls="--", label="Event mark"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=_legend_columns(fig, len(handles)))


def draw_voltage_vs_delta_t(
    fig: Figure,
    analysis: Analysis,
    cycle: int | None = None,
    segment: str | None = None,
) -> None:
    """Oriented voltage against ΔT, with the whole-cycle and branch fits.

    One panel per cycle, or one large panel for `cycle`. `segment` fades the
    other branch.
    """
    fig.clear()
    a = analysis
    cycles = a.cycles if cycle is None else (a.cycles[cycle - 1],)
    axes = fig.subplots(1, len(cycles), sharey=True, squeeze=False)[0]
    size = 4 if len(cycles) > 1 else 9
    for ax, fitted in zip(axes, cycles, strict=True):
        w = fitted.window
        heating = (a.time_s >= w.on_s) & (a.time_s < w.off_s)
        cooling = (a.time_s >= w.off_s) & (a.time_s < w.end_s)
        x, y = a.delta_t_k, a.voltage_oriented_mv
        faded = {"heating": segment == "cooling", "cooling": segment == "heating"}
        ax.scatter(
            x[heating], y[heating], s=size, color=HEATING_COLOR, lw=0,
            alpha=0.15 if faded["heating"] else 0.75,
        )
        ax.scatter(
            x[cooling], y[cooling], s=size, color=COOLING_COLOR, lw=0,
            alpha=0.15 if faded["cooling"] else 0.75,
        )
        for kind, mask in (("combined", heating | cooling), ("heating", heating), ("cooling", cooling)):
            fit = fitted.fit(kind)
            if not math.isfinite(fit.slope):
                continue
            # Each fit is drawn over the ΔT its own samples cover.
            span = np.array([x[mask].min(), x[mask].max()])
            ax.plot(
                span,
                fit.slope * span + fit.intercept,
                **FIT_STYLE[kind],
                alpha=0.35 if faded.get(kind) else 1.0,
                path_effects=_HALO,
                zorder=3,
            )
        title = f"Cycle {w.index}" + ("  (rejected)" if fitted.problems else "")
        ax.set_title(title)
        ax.set_xlabel(_DELTA_T_LABEL)
    side = "cold" if a.wiring_sign > 0 else "hot"
    axes[0].set_ylabel(
        rf"$\sigma V_\mathrm{{rec}}$ (mV),  $\sigma$ = {a.wiring_sign:+d} (V+ on {side} electrode)"
    )
    handles = [
        Line2D([], [], color=HEATING_COLOR, marker="o", ls="none", ms=5, label="Heating (drive on)"),
        Line2D([], [], color=COOLING_COLOR, marker="o", ls="none", ms=5, label="Cooling (drive off)"),
    ] + [
        Line2D([], [], **FIT_STYLE[kind], path_effects=_HALO, label=FIT_LABELS[kind]) for kind in KINDS
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=_legend_columns(fig, len(handles)))


def draw_coefficients(fig: Figure, analysis: Analysis, cycle: int | None = None) -> None:
    """Whole-cycle, heating-only and cooling-only S_app per cycle, and their means.

    `cycle` emphasises one cycle; the others stay visible for comparison.
    """
    fig.clear()
    a = analysis
    ax = fig.subplots()
    offsets = {"combined": -0.2, "heating": 0.0, "cooling": 0.2}
    n = len(a.cycles)
    if cycle is not None:
        ax.axvspan(cycle - 0.42, cycle + 0.42, color="#e9f0f9", lw=0, zorder=0)
    for kind in KINDS:
        for c in a.cycles:
            ax.plot(
                c.window.index + offsets[kind],
                c.fit(kind).slope,
                "o",
                ms=7,
                color=KIND_COLORS[kind],
                mfc="none" if c.problems else KIND_COLORS[kind],
                alpha=0.35 if cycle is not None and c.window.index != cycle else 1.0,
                zorder=3,
            )
        if a.reductions is not None:
            r = a.reductions[kind]
            ax.errorbar(
                n + 1.5 + offsets[kind],
                r.mean,
                yerr=r.sem if r.sem is not None else None,
                fmt="s",
                ms=7,
                color=KIND_COLORS[kind],
                capsize=4,
                zorder=3,
            )
    ticks = [c.window.index for c in a.cycles] + [n + 1.5]
    ax.set_xticks(ticks, [f"Cycle {i}" for i in ticks[:-1]] + ["Mean ± SEM"])
    ax.set_xlim(0.45, n + 2.05)
    ax.grid(False, axis="x")
    ax.set_ylabel(rf"Apparent dynamic coefficient $S_\mathrm{{app}}$ ({UNIT})")
    handles = [
        Line2D([], [], color=KIND_COLORS[kind], marker="o", ls="none", ms=7, label=KIND_LABELS[kind])
        for kind in KINDS
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=3)
    fig.supxlabel(FOOTNOTE, fontsize=7.5, color=MUTED)
