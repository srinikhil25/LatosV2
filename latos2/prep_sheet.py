"""A printable preparation sheet for a frozen prediction, as PDF.

The sheet carries what the laboratory needs to prepare and measure the
specimen: the composition, the masses to weigh, the other conditions, the
prediction it tests and blank lines for what is actually done. It is written
from the frozen prediction only, so it shows what was frozen, not the current
model.
"""

from __future__ import annotations

import html
import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QMarginsF
from PySide6.QtGui import QFont, QPageLayout, QPageSize, QPdfWriter, QTextDocument

from latos2 import __version__
from latos2 import experiment as ex
from latos2.predictions import intact
from latos2.project_io import check_not_recordings_folder

__all__ = ["sheet_html", "write_preparation_sheet"]

_GOALS = {"maximize": "maximize", "minimize": "minimize", "target": "reach the target"}


def _number(p: ex.Prediction, value: float | None, signed: bool = True) -> str:
    if value is None:
        return "—"
    if str(p.record.get("output", {}).get("method", "")).startswith("hioki:"):
        return f"{value:+.4f}" if signed else f"{value:.4f}"
    return f"{value:.6g}"


def _when(stamp: str) -> str:
    moment = datetime.fromisoformat(stamp)
    offset = f" (UTC{stamp[-6:]})" if moment.tzinfo is not None else ""
    return moment.strftime("%Y-%m-%d %H:%M") + offset


def sheet_html(exp: ex.Experiment, p: ex.Prediction) -> str:
    """The sheet's content as HTML for QTextDocument."""
    e = html.escape
    unit = f" {p.unit}" if p.unit else ""
    method = p.record.get("output", {}).get("method", "")
    rows = []
    prep = p.preparation or {}
    masses = prep.get("masses_g") or {}
    decimals = max(0, -int(f"{prep['resolution_g']:e}".split("e")[1])) if prep else 4
    for c, x in (p.fractions or {}).items():
        mass = f"{masses[c]:.{decimals}f}" if c in masses else "—"
        rows.append(
            f"<tr><td>{e(c)}</td><td align='right'>{x:.4f}</td><td align='right'>{100 * x:.2f}</td>"
            f"<td align='right'><b>{mass}</b></td><td></td></tr>"
        )
    parts = [
        f"<h2>Preparation sheet: frozen prediction {e(p.id)}</h2>",
        f"<p>Experiment: <b>{e(exp.name)}</b><br>Frozen: {e(_when(p.frozen_at))}<br>"
        f"<span style='color:#555'>Frozen content SHA-256: {e(p.sha256)}</span></p>",
    ]
    if rows:
        parts.append("<h3>Prepare</h3>")
        parts.append(
            "<table border='1' cellspacing='0' cellpadding='4' width='100%'>"
            "<tr><th align='left'>Component</th><th align='right'>Mass fraction</th><th align='right'>wt %</th>"
            "<th align='right'>Mass to weigh (g)</th><th align='left'>Actual mass weighed (g)</th></tr>"
            + "".join(rows)
            + "</table>"
        )
        if prep:
            text = f"Total {prep['total_g']:.{decimals}f} g, rounded to a balance resolution of {prep['resolution_g']:g} g. "
            text += " ".join(prep.get("notes") or [])
            if not prep.get("feasible", True):
                text = "⚠ " + text
        else:
            text = (
                "No total mass was entered when this prediction was frozen. Mass of each component = its mass "
                "fraction × the total you prepare."
            )
        parts.append(f"<p style='color:#555'>{e(text)} Record the masses actually weighed; Latos computes the "
                     "composition from them.</p>")
    if p.inputs:
        units = {d.get("variable"): d.get("unit", "") for d in p.record.get("design_space", [])}
        items = "".join(
            f"<li>{e(name)}: <b>{value:.6g}</b>{(' ' + e(units.get(name) or '')) if units.get(name) else ''}</li>"
            for name, value in p.inputs.items()
        )
        parts.append(f"<h3>Other conditions</h3><ul>{items}</ul>")

    goal = _GOALS.get(p.objective, p.objective)
    if p.objective == "target" and p.target is not None:
        goal += f" {p.target:g}{unit}"
    lo, hi = p.interval_95
    lines = [
        f"Output: <b>{e(p.output)}</b>{e(unit and ' (' + p.unit + ')')}, "
        f"{e(ex.METHOD_LABELS.get(method, method))}; goal: {e(goal)}",
        f"Predicted mean: <b>{_number(p, p.predicted_mean)}</b> ± {_number(p, p.latent_sd, False)}{e(unit)} (model SD)",
        f"SD for a new specimen's measurement: {_number(p, p.predictive_sd, False)}{e(unit)}",
        f"95 % interval for the new measurement: <b>{_number(p, lo)} to {_number(p, hi)}{e(unit)}</b>",
    ]
    if p.best_before is not None:
        what = "Closest to the target before freezing" if p.objective == "target" else "Best measured before freezing"
        signed = p.objective != "target"
        lines.append(f"{what}: {_number(p, p.best_before, signed)}{e(unit)} ({e(p.best_before_specimen or '')})")
    if p.note:
        lines.append(f"Note at freezing: {e(p.note)}")
    parts.append("<h3>Prediction to test</h3><p>" + "<br>".join(lines) + "</p>")
    if method.startswith("hioki:"):
        parts.append(
            "<p style='color:#555'>S_app is an apparent dynamic coefficient from the heating/cooling protocol, "
            "not a steady-state Seebeck coefficient. Measure with the same protocol, cycles and lead polarity as "
            "the specimens the model was fitted to.</p>"
        )

    blank = "".join("<tr><td>" + "&nbsp;" * 4 + "</td>" + "<td></td>" * 5 + "</tr>" for _ in range(2))
    parts.append(
        "<h3>Lab record</h3>"
        "<table border='1' cellspacing='0' cellpadding='8' width='100%'>"
        "<tr><th align='left'>Specimen ID</th><th align='left'>Date prepared</th><th align='left'>Date measured</th>"
        "<th align='left'>Operator</th><th align='left'>Recording file</th><th align='left'>Notes</th></tr>"
        + blank
        + "</table>"
    )
    parts.append(
        "<h3>Bring the result into Latos</h3><ol>"
        "<li>HIOKI Analyzer: open the recording; confirm the channels and the lead polarity from the lab record.</li>"
        "<li>Experiment: <b>Add specimen</b>; enter the actual masses; <b>Attach HIOKI analysis…</b> "
        "(or <b>Enter value…</b>).</li>"
        f"<li>In the specimen form, set <b>Tests prediction</b> to {e(p.id)}; save the experiment.</li>"
        "<li>Optimization ▸ Predictions shows the measured value against this prediction.</li></ol>"
    )
    parts.append(
        f"<p style='color:#555'>Latos V2 {e(__version__)}. This prediction was frozen before the specimen was "
        "prepared and cannot be edited. One test neither validates nor refutes the model.</p>"
    )
    return "".join(parts)


def write_preparation_sheet(exp: ex.Experiment, p: ex.Prediction, path: str | Path) -> Path:
    """Write the sheet as PDF, never into a folder of original recordings.

    Raises:
        ValueError: If the prediction was withdrawn or altered.
        ProjectError: If `path` is in the folder of an attached recording.
    """
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        path = path.with_name(path.name + ".pdf")
    if p.withdrawn_at:
        raise ValueError(f"{p.id} was withdrawn; there is nothing to prepare")
    if not intact(p):
        raise ValueError(f"{p.id} no longer holds what was frozen; no sheet is written from it")
    check_not_recordings_folder(exp, path, "the preparation sheet")
    tmp = path.with_name(path.name + ".tmp")
    writer = QPdfWriter(str(tmp))
    writer.setTitle(f"Preparation sheet {p.id}")
    writer.setCreator(f"Latos V2 {__version__}")
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setPageMargins(QMarginsF(15, 15, 15, 15), QPageLayout.Unit.Millimeter)
    document = QTextDocument()
    font = QFont()
    font.setPointSizeF(9.5)
    document.setDefaultFont(font)
    document.setHtml(sheet_html(exp, p))
    document.print_(writer)
    del writer  # closes the file
    os.replace(tmp, path)
    return path
