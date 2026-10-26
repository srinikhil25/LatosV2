"""Figures of the optimization workspace: the surrogate model and the acquisition.

Drawn for one optimization variable. With more, the figure says so rather than
presenting a slice as if it were the whole model.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from latos2.figures import COMBINED_COLOR, COOLING_COLOR, FOOTNOTE, MUTED
from latos2.gp import Z95
from latos2.optimization import Result

BAND_LABEL = "Model-based 95 % interval (illustrative; coverage not validated)"
CANDIDATE_COLOR = "#c0392b"
BASELINE_COLOR = "#7a7a7a"
_ACQUISITION_LABELS = {
    "Expected improvement": "Expected improvement",
    "Expected closeness to the target": "Expected gain in closeness to the target",
    "Largest model uncertainty": "Model standard deviation",
}


def _unit(text: str) -> str:
    return f" ({text})" if text else ""


def _not_one_dimensional(fig: Figure, r: Result) -> None:
    fig.text(
        0.5, 0.5,
        f"This problem has {r.problem.space.d} optimization variables.\n"
        "Graphs are drawn for one variable; the candidate, its prediction and the evaluation\n"
        "are listed in the panels and tables.",
        ha="center", va="center", color=MUTED, fontsize=10,
    )


def _limits(ax, v) -> None:
    """The search range with a small margin, so specimens on its edges stay whole."""
    pad = 0.02 * v.span
    ax.set_xlim(v.lower - pad, v.upper + pad)


def _footnote(fig: Figure, r: Result) -> None:
    lines = []
    left_out = [o for o in r.problem.observations if not o.used]
    if left_out:
        lines.append("Not in the fit: " + "; ".join(f"{o.specimen_id} ({o.reason})" for o in left_out))
    if r.problem.output.method.startswith("hioki:"):
        lines.append(FOOTNOTE)
    if lines:
        fig.supxlabel("\n".join(lines), fontsize=7.5, color=MUTED)


def draw_surrogate(fig: Figure, r: Result) -> None:
    """Measured specimens, the GP mean and its model-based band over the search range."""
    fig.clear()
    if r.problem.space.d != 1:
        _not_one_dimensional(fig, r)
        return
    v, output = r.problem.space.variables[0], r.problem.output
    x = r.candidates[:, 0]
    order = np.argsort(x)
    x, mean, sd = x[order], r.mean[order], r.sd[order]
    ax = fig.add_subplot()
    ax.fill_between(x, mean - Z95 * sd, mean + Z95 * sd, color=COOLING_COLOR, alpha=0.16, lw=0, label=BAND_LABEL)
    ax.plot(x, mean, color=COOLING_COLOR, lw=1.6, label="GP predictive mean")
    ax.errorbar(
        r.x[:, 0], r.y, yerr=np.nan_to_num(r.sem), fmt="o", color=COMBINED_COLOR, ms=5, capsize=3, lw=1,
        label="Specimens in the fit (mean ± SEM of cycles)", zorder=5,
    )
    c, b = r.choice, r.baseline
    ax.axvline(b.values[0], color=BASELINE_COLOR, ls=":", lw=1.2, label="Space-filling baseline")
    ax.axvline(c.values[0], color=CANDIDATE_COLOR, ls="--", lw=1.2, label="Provisional candidate")
    ax.errorbar(
        [c.values[0]], [c.mean], yerr=[Z95 * c.sd], fmt="D", color=CANDIDATE_COLOR, ms=6, capsize=3, zorder=6
    )
    _limits(ax, v)
    ax.set_xlabel(v.label + _unit(v.unit))
    ax.set_ylabel(output.name + _unit(output.unit))
    ax.grid(True, alpha=0.3)
    fig.legend(loc="outside upper center", ncols=2, frameon=False)
    _footnote(fig, r)


def draw_acquisition(fig: Figure, r: Result) -> None:
    """The acquisition over the search range, the excluded neighbourhoods and the selection."""
    fig.clear()
    if r.problem.space.d != 1:
        _not_one_dimensional(fig, r)
        return
    v, output = r.problem.space.variables[0], r.problem.output
    x = r.candidates[:, 0]
    order = np.argsort(x)
    x, score, excluded = x[order], r.score[order], r.excluded[order]
    ax = fig.add_subplot()
    label = _ACQUISITION_LABELS.get(r.choice.principle, r.choice.principle)
    ax.plot(x, score, color=COMBINED_COLOR, lw=1.5, label=label)
    if excluded.any():
        ax.fill_between(
            x, 0, 1, where=excluded, transform=ax.get_xaxis_transform(), color=BASELINE_COLOR, alpha=0.18, lw=0,
            label="Within the minimum separation of a measured specimen",
        )
    measured = r.problem.measured_points[:, 0]
    measured = measured[(measured >= v.lower - 1e-9) & (measured <= v.upper + 1e-9)]
    ax.plot(measured, np.zeros_like(measured), "|", color=COMBINED_COLOR, ms=14, mew=1.5,
            label="Measured conditions")
    c, b = r.choice, r.baseline
    ax.axvline(b.values[0], color=BASELINE_COLOR, ls=":", lw=1.2, label="Space-filling baseline")
    ax.plot([c.values[0]], [c.score], "D", color=CANDIDATE_COLOR, ms=7, zorder=6, label="Provisional candidate")
    _limits(ax, v)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(v.label + _unit(v.unit))
    ax.set_ylabel(label + _unit(output.unit))
    ax.grid(True, alpha=0.3)
    fig.legend(loc="outside upper center", ncols=3, frameon=False)
    _footnote(fig, r)
