"""Milestone 1 acceptance tests, run on the authentic IL1 75 % / IL2 25 % workbooks in place.

Reference values:
* Whole-cycle S_app, forward and reversed: published by Latos V1 for these
  recordings (docs/LATOS_CURRENT_STATE_AND_RESEARCH_MAP.md §13.2-13.3).
* Heating-only and cooling-only S_app: computed here; cycle 1 agrees with the
  researcher's own Origin fits (slopes 0.00283 and 0.00272 V/K).
* Units: the voltage header is [V] and readings step by 5e-6 V (5 µV), so the
  coefficients are in mV/K (≈2.19 mV/K, not µV/K).
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from latos2.export import ExportError, export_analysis  # noqa: E402
from latos2.figures import FOOTNOTE  # noqa: E402
from latos2.hioki_reader import file_sha256, read_hioki_xlsx  # noqa: E402
from latos2.thermovoltage import (  # noqa: E402
    SIGN_CONVENTION,
    STATEMENT,
    ChannelRoles,
    CycleBoundaryError,
    analyze,
    event_boundaries,
)

DATA = Path(os.environ.get("LATOS2_IL_DATA", r"D:\Latos\data\ionic-liquid-campaign\Measurement files"))
FORWARD = DATA / "IL1_75-IL2_25.xlsx"
REVERSED = DATA / "IL1_75-IL2_25-R.xlsx"
FORWARD_SHA256 = "0316474901238fabe42548f9271932f19d320ec7837719fe912cdb4222bc4830"
REVERSED_SHA256 = "02ceea1ce3c0d9048f9d22da4e0952d315208e0bb23e62779e7d570b01e5a383"
ROLES = ChannelRoles(hot="CH1-A-3", cold="CH1-A-1", voltage="CH1-A-10")

COMBINED = [2.1696, 2.1801, 2.2204]
HEATING = [2.8269, 2.6492, 2.7629]
COOLING = [2.7177, 2.6513, 2.4756]
REVERSED_COMBINED = [2.2537, 2.2658, 2.3005]
TOL = 1e-4  # mV/K, the precision V1 published

pytestmark = pytest.mark.skipif(not FORWARD.exists(), reason="authentic IL workbooks not available")


@pytest.fixture(scope="module")
def forward():
    return analyze(read_hioki_xlsx(FORWARD), ROLES, +1)


def _slopes(analysis, kind):
    return [c.fit(kind).slope for c in analysis.cycles]


# 1. The original is read in place and never changes.
def test_original_files_unchanged(tmp_path):
    before = {p: (file_sha256(p), p.stat().st_mtime_ns) for p in (FORWARD, REVERSED)}
    assert before[FORWARD][0] == FORWARD_SHA256
    assert before[REVERSED][0] == REVERSED_SHA256

    analysis = analyze(read_hioki_xlsx(FORWARD), ROLES, +1)
    export_analysis(analysis, tmp_path / "out")
    analyze(read_hioki_xlsx(REVERSED), ROLES, -1)
    target = FORWARD.parent / "latos2_must_not_exist"
    with pytest.raises(ExportError, match="inside the folder"):
        export_analysis(analysis, target)

    assert not target.exists()
    assert {p: (file_sha256(p), p.stat().st_mtime_ns) for p in (FORWARD, REVERSED)} == before


# 2. Event marks are found and give three heating/cooling cycles.
def test_event_marks_give_three_cycles(forward):
    events = forward.trace.events
    assert [e.number for e in events] == [1, 2, 3, 4, 5, 6, 7]
    assert [e.time_s for e in events] == pytest.approx([5.6, 187.8, 377.2, 562.6, 742.8, 924.2, 1109.4])
    assert [e.sheet_row for e in events] == [31, 942, 1889, 2816, 3717, 4624, 5550]
    assert len(forward.cycles) == 3
    assert [c.window.heating_s for c in forward.cycles] == pytest.approx([182.2, 185.4, 181.4])
    assert [c.window.cooling_s for c in forward.cycles] == pytest.approx([189.4, 180.2, 185.2])
    assert forward.accepted == (1, 2, 3)

    with pytest.raises(CycleBoundaryError, match="out of sequence"):
        event_boundaries([1, 2, 4, 5, 6], [1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(CycleBoundaryError, match="odd"):
        event_boundaries([1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0])


# 3. Units come from the headers and are traced to the raw readings.
def test_units_traced_to_raw_readings(forward):
    trace = forward.trace
    assert [c.header for c in trace.channels] == ["CH1-A-1[°C]", "CH1-A-3[°C]", "CH1-A-10[V]"]
    assert forward.voltage_unit == "V"
    assert forward.voltage_factor_to_mv == 1000.0
    assert forward.voltage_step == pytest.approx(5e-6)  # 5 µV: the logger's finest step
    assert forward.temperature_step == pytest.approx(0.1)
    assert trace.sample_interval_s == pytest.approx(0.2)
    # First sample: 0.0016 V in the sheet is 1.6 mV.
    assert trace.channel("CH1-A-10").values[0] == 0.0016
    assert forward.voltage_recorded_mv[0] == pytest.approx(1.6)
    # About 41 mV across about 16 K: mV/K, not µV/K.
    assert float(np.max(forward.voltage_recorded_mv)) == pytest.approx(41.405)
    assert max(c.delta_t_max_k for c in forward.cycles) == pytest.approx(16.1)


# 4. Whole-cycle, heating-only and cooling-only S_app in mV/K.
def test_coefficients_forward(forward):
    assert _slopes(forward, "combined") == pytest.approx(COMBINED, abs=TOL)
    assert _slopes(forward, "heating") == pytest.approx(HEATING, abs=TOL)
    assert _slopes(forward, "cooling") == pytest.approx(COOLING, abs=TOL)
    # The researcher's Origin fits of cycle 1, in V/K: 0.00283 heating, 0.00272 cooling.
    assert round(forward.cycles[0].heating.slope / 1000, 5) == 0.00283
    assert round(forward.cycles[0].cooling.slope / 1000, 5) == 0.00272
    combined = forward.reductions["combined"]
    assert combined.k == 3 and combined.dof == 2
    assert combined.mean == pytest.approx(2.1900, abs=TOL)
    assert combined.sem == pytest.approx(0.0155, abs=TOL)


# 5. Polarity is declared, never inferred, and a wrong assignment is caught.
def test_polarity_and_channel_assignment(forward):
    reversed_trace = read_hioki_xlsx(REVERSED)
    as_wired = analyze(reversed_trace, ROLES, +1)
    assert all(s < 0 for s in _slopes(as_wired, "combined"))  # raw sign of the swapped leads
    declared = analyze(reversed_trace, ROLES, -1)
    assert _slopes(declared, "combined") == pytest.approx(REVERSED_COMBINED, abs=TOL)
    assert declared.reductions["combined"].mean == pytest.approx(2.2733, abs=TOL)

    flipped = analyze(forward.trace, ROLES, -1)
    assert _slopes(flipped, "combined") == pytest.approx([-s for s in COMBINED], abs=TOL)

    swapped = analyze(forward.trace, ChannelRoles(hot="CH1-A-1", cold="CH1-A-3", voltage="CH1-A-10"), +1)
    assert swapped.reductions is None
    assert all("does not rise" in " ".join(c.problems) for c in swapped.cycles)

    with pytest.raises(ValueError, match="wiring must be declared"):
        analyze(forward.trace, ROLES, 0)


# 6. Export writes tables, provenance and figures, reproducibly.
def test_export_files_and_reproducibility(forward, tmp_path):
    first = export_analysis(forward, tmp_path / "a")
    second = export_analysis(forward, tmp_path / "b")
    names = sorted(p.name for p in first)
    stem = "IL1_75-IL2_25"
    assert names == sorted(
        [f"{stem}_cycles.csv", f"{stem}_summary.csv", f"{stem}_analysis.json"]
        + [f"{stem}_{fig}.{ext}" for fig in ("timeseries", "voltage_vs_dT", "coefficients") for ext in ("png", "svg")]
    )
    assert all(p.stat().st_size > 0 for p in first)

    record = json.loads((tmp_path / "a" / f"{stem}_analysis.json").read_text(encoding="utf-8"))
    assert record["source"]["sha256_at_read"] == record["source"]["sha256_at_export"] == FORWARD_SHA256
    assert record["declarations"] == {
        "hot_channel": "CH1-A-3",
        "cold_channel": "CH1-A-1",
        "voltage_channel": "CH1-A-10",
        "wiring": "V+ on the COLD electrode",
        "wiring_sign": 1,
    }
    assert [m["number"] for m in record["event_marks"]] == [1, 2, 3, 4, 5, 6, 7]
    assert record["units"]["voltage_reading_step_uV"] == pytest.approx(5.0)
    assert record["summary"]["combined"]["mean_mV_per_K"] == pytest.approx(2.1900, abs=TOL)

    with (tmp_path / "a" / f"{stem}_cycles.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(r["S_app_combined_mV_per_K"]) for r in rows] == pytest.approx(COMBINED, abs=TOL)
    assert [float(r["S_app_heating_mV_per_K"]) for r in rows] == pytest.approx(HEATING, abs=TOL)
    assert [float(r["S_app_cooling_mV_per_K"]) for r in rows] == pytest.approx(COOLING, abs=TOL)

    for name in (f"{stem}_cycles.csv", f"{stem}_summary.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    second_record = json.loads((tmp_path / "b" / f"{stem}_analysis.json").read_text(encoding="utf-8"))
    record.pop("analysed_at_utc")
    second_record.pop("analysed_at_utc")
    assert record == second_record
    assert len(second) == len(first)


def _unqualified_seebeck(text: str) -> list[str]:
    """Every 'Seebeck coefficient' not written as 'not a (validated) steady-state Seebeck coefficient'."""
    bad = []
    for match in re.finditer(r"seebeck coefficient", text, re.IGNORECASE):
        before = text[max(0, match.start() - 40) : match.start()].lower()
        if not re.search(r"not (?:a|validated) steady-state\s+$", before):
            bad.append(text[max(0, match.start() - 40) : match.end()])
    return bad


# 7. The window works by clicks alone, and no wording claims a Seebeck coefficient.
def test_gui_click_path_and_wording(forward, tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QLabel, QRadioButton

    from latos2.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)

    assert window.open_file(FORWARD)
    assert window.events_table.rowCount() == 7
    assert window.hot_combo.currentData() == "CH1-A-3"
    assert window.cold_combo.currentData() == "CH1-A-1"
    assert window.voltage_combo.currentData() == "CH1-A-10"
    # No wiring is pre-selected, so nothing is computed yet.
    assert window.wiring() is None
    assert window.analysis is None and not window.export_action.isEnabled()
    assert "polarity" in window.status_label.text()

    window.set_wiring(+1)
    app.processEvents()
    assert window.analysis is not None and window.export_action.isEnabled()
    table = window.results_table
    assert table.rowCount() == 6  # three cycles, then mean, SD, SEM
    # Column 0 is the whole-cycle coefficient.
    assert [table.item(r, 0).text() for r in range(3)] == ["+2.1696", "+2.1801", "+2.2204"]
    assert table.item(3, 0).text() == "+2.1900"

    written = window.export(tmp_path)
    assert len(written) == 9 and all(p.exists() for p in written)

    texts = [STATEMENT, SIGN_CONVENTION, FOOTNOTE, window.windowTitle()]
    texts += [w.text() for w in window.findChildren(QLabel) + window.findChildren(QRadioButton)]
    texts += [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    texts += [p.read_text(encoding="utf-8-sig") for p in written if p.suffix in (".csv", ".json", ".svg")]
    assert not _unqualified_seebeck("\n".join(texts))
    window.close()
