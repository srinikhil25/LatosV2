"""Write an analysis to disk: tables, a provenance record and the figures.

Nothing is ever written next to the original recording, and the export is
refused if the original's bytes changed since it was read.
"""

from __future__ import annotations

import csv
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import latos2
from latos2.figures import draw_coefficients, draw_timeseries, draw_voltage_vs_delta_t
from latos2.hioki_reader import file_sha256
from latos2.thermovoltage import (
    KIND_LABELS,
    KINDS,
    QUANTITY,
    SIGN_CONVENTION,
    STATEMENT,
    UNIT,
    WIRING,
    Analysis,
)

__all__ = ["ExportError", "export_analysis"]

_PNG_DPI = 300


class ExportError(RuntimeError):
    """The analysis cannot be exported as asked."""


def _number(value: float | None) -> float | None:
    """A JSON-safe number: NaN and None become null."""
    return None if value is None or not math.isfinite(value) else float(value)


def _cell(value: float | None) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.10g}"


def _inside(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def export_analysis(analysis: Analysis, out_dir: Path) -> list[Path]:
    """Write the cycle table, summary, provenance JSON and figures to `out_dir`.

    Raises:
        ExportError: If `out_dir` is the original's folder or inside it, or if
            the original's SHA-256 no longer matches the one read.
    """
    a = analysis
    source = a.trace.source.resolve()
    out_dir = Path(out_dir).resolve()
    if _inside(out_dir, source.parent):
        raise ExportError(
            f"{out_dir} is inside the folder holding the original recording; choose a folder "
            "outside it so the original measurement files stay untouched"
        )
    sha_now = file_sha256(source)
    if sha_now != a.trace.sha256:
        raise ExportError(
            f"{source.name} changed since it was read (SHA-256 {a.trace.sha256[:12]}… → "
            f"{sha_now[:12]}…); reopen it before exporting"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    written = [
        _write_cycles_csv(a, out_dir / f"{stem}_cycles.csv"),
        _write_summary_csv(a, out_dir / f"{stem}_summary.csv"),
        _write_json(a, out_dir / f"{stem}_analysis.json", sha_now),
    ]
    figures = {
        "timeseries": (lambda fig: draw_timeseries(fig, a.trace, a.roles), (8.0, 7.5)),
        "voltage_vs_dT": (lambda fig: draw_voltage_vs_delta_t(fig, a), (4.2 * len(a.cycles), 4.6)),
        "coefficients": (lambda fig: draw_coefficients(fig, a), (7.0, 4.8)),
    }
    # SVG text stays text, so labels remain editable in a vector editor.
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        for name, (draw, size) in figures.items():
            fig = Figure(figsize=size, layout="constrained")
            FigureCanvasAgg(fig)
            draw(fig)
            for suffix in ("png", "svg"):
                path = out_dir / f"{stem}_{name}.{suffix}"
                fig.savefig(path, dpi=_PNG_DPI, metadata={"Date": None} if suffix == "svg" else None)
                written.append(path)
    return written


def _write_cycles_csv(a: Analysis, path: Path) -> Path:
    header = [
        "cycle",
        "on_s",
        "off_s",
        "end_s",
        "heating_duration_s",
        "cooling_duration_s",
        "n_heating",
        "n_cooling",
        "dT_min_K",
        "dT_max_K",
        f"mean_electrode_T_{a.temperature_unit}",
    ]
    for kind in KINDS:
        header += [
            f"S_app_{kind}_mV_per_K",
            f"intercept_{kind}_mV",
            f"R2_{kind}",
            f"slope_stderr_{kind}_mV_per_K_diagnostic",
        ]
    header += ["accepted", "problems"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for c in a.cycles:
            w = c.window
            row = [
                w.index,
                _cell(w.on_s),
                _cell(w.off_s),
                _cell(w.end_s),
                _cell(w.heating_s),
                _cell(w.cooling_s),
                c.n_heating,
                c.n_cooling,
                _cell(c.delta_t_min_k),
                _cell(c.delta_t_max_k),
                _cell(c.mean_temperature),
            ]
            for kind in KINDS:
                fit = c.fit(kind)
                row += [_cell(fit.slope), _cell(fit.intercept), _cell(fit.r_squared), _cell(fit.slope_stderr)]
            row += ["yes" if not c.problems else "no", " | ".join(c.problems)]
            writer.writerow(row)
    return path


def _write_summary_csv(a: Analysis, path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "kind", "unit", "k_cycles", "mean", "sd", "sem_within_specimen", "dof"])
        for kind in KINDS:
            r = a.reductions[kind] if a.reductions else None
            writer.writerow(
                [
                    QUANTITY,
                    KIND_LABELS[kind],
                    UNIT,
                    r.k if r else 0,
                    _cell(r.mean) if r else "",
                    _cell(r.sd) if r else "",
                    _cell(r.sem) if r else "",
                    r.dof if r else "",
                ]
            )
        writer.writerow([])
        writer.writerow(["note", STATEMENT])
        writer.writerow(["sign convention", SIGN_CONVENTION])
        writer.writerow(["wiring declared", WIRING[a.wiring_sign]])
        writer.writerow(["source", str(a.trace.source)])
        writer.writerow(["source SHA-256", a.trace.sha256])
    return path


def _write_json(a: Analysis, path: Path, sha_at_export: str) -> Path:
    t = a.trace
    record = {
        "software": {"name": "latos2", "version": latos2.__version__, "python": platform.python_version()},
        "analysed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "path": str(t.source.resolve()),
            "sha256_at_read": t.sha256,
            "sha256_at_export": sha_at_export,
            "size_bytes": t.size_bytes,
            "sheet": t.sheet,
            "preamble": list(t.preamble),
            "samples": int(t.time_s.size),
            "sample_interval_s": round(t.sample_interval_s, 6),
            "first_timestamp": t.timestamps[0],
            "last_timestamp": t.timestamps[-1],
            "channel_headers": [c.header for c in t.channels],
            "ignored_columns": list(t.ignored_columns),
        },
        "event_marks": [
            {"number": e.number, "time_s": e.time_s, "sheet_row": e.sheet_row, "timestamp": e.timestamp}
            for e in t.events
        ],
        "declarations": {
            "hot_channel": a.roles.hot,
            "cold_channel": a.roles.cold,
            "voltage_channel": a.roles.voltage,
            "wiring": WIRING[a.wiring_sign],
            "wiring_sign": a.wiring_sign,
        },
        "units": {
            "voltage_header_unit": a.voltage_unit,
            "voltage_factor_to_mV": a.voltage_factor_to_mv,
            "voltage_reading_step_in_header_unit": a.voltage_step,
            "voltage_reading_step_uV": a.voltage_step * a.voltage_factor_to_mv * 1000.0,
            "temperature_header_unit": a.temperature_unit,
            "temperature_reading_step": a.temperature_step,
            "delta_T": "T_hot − T_cold, in K",
            "coefficient_unit": UNIT,
        },
        "definition": {
            "quantity": QUANTITY,
            "statement": STATEMENT,
            "sign_convention": SIGN_CONVENTION,
            "windows": "odd event marks switch the drive on, even marks switch it off, the last "
            "mark ends the run; cycle k = [on_k, on_k+1), heating [on_k, off_k), cooling [off_k, on_k+1)",
            "fit": "ordinary least squares of σ·V_rec (mV) on ΔT (K), free intercept",
            "uncertainty": "SEM over accepted cycles: within-specimen, within-session only; "
            "per-cycle slope standard errors are diagnostics, not uncertainties",
        },
        "cycles": [
            {
                "cycle": c.window.index,
                "on_s": c.window.on_s,
                "off_s": c.window.off_s,
                "end_s": c.window.end_s,
                "heating_duration_s": round(c.window.heating_s, 6),
                "cooling_duration_s": round(c.window.cooling_s, 6),
                "n_heating": c.n_heating,
                "n_cooling": c.n_cooling,
                "dT_min_K": c.delta_t_min_k,
                "dT_max_K": c.delta_t_max_k,
                "mean_electrode_temperature": c.mean_temperature,
                "accepted": not c.problems,
                "problems": list(c.problems),
                **{
                    kind: {
                        "S_app_mV_per_K": _number(c.fit(kind).slope),
                        "intercept_mV": _number(c.fit(kind).intercept),
                        "r_squared": _number(c.fit(kind).r_squared),
                        "slope_stderr_mV_per_K_diagnostic": _number(c.fit(kind).slope_stderr),
                        "n": c.fit(kind).n,
                    }
                    for kind in KINDS
                },
            }
            for c in a.cycles
        ],
        "summary": {
            kind: (
                {
                    "label": KIND_LABELS[kind],
                    "k_cycles": a.reductions[kind].k,
                    "mean_mV_per_K": a.reductions[kind].mean,
                    "sd_mV_per_K": _number(a.reductions[kind].sd),
                    "sem_within_specimen_mV_per_K": _number(a.reductions[kind].sem),
                    "dof": a.reductions[kind].dof,
                }
                if a.reductions
                else None
            )
            for kind in KINDS
        },
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
