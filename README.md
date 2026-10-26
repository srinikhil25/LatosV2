# Latos V2

**A materials-informatics workbench for authentic laboratory measurements.**

Latos is a desktop application for analysing laboratory instrument data and, in
later releases, for planning experiments with Bayesian optimization and
validating its recommendations in the laboratory. The current release provides
the **HIOKI Measurement Analyzer**: a scientific workbench for event-marked
heating/cooling thermovoltage recordings of ionic-liquid specimens.

Latos runs entirely on the researcher's computer. It uses no server, no cloud
service and no language model, and it never modifies an original instrument file.

---

## Capabilities

- **Direct instrument import.** Reads HIOKI Memory HiLogger Excel exports as
  recorded, with channel names and units taken verbatim from the instrument header.
- **Cycle identification.** Detects the operator's event marks and divides each
  recording into heating and cooling segments, cycle by cycle.
- **Scientific visualisation.** Temperature, temperature difference and voltage
  against time; voltage against temperature difference with whole-cycle and
  branch fits; a comparison of coefficients across cycles. Every view can show
  all cycles together or a single cycle in detail.
- **Apparent coefficients.** Whole-cycle, heating-only and cooling-only apparent
  dynamic coefficients for every cycle, with mean, standard deviation and
  within-specimen standard error.
- **Explicit declarations.** The researcher assigns the hot, cold and voltage
  channels and declares the voltage-lead polarity. Polarity is never inferred.
- **Reproducible exports.** Tables, a provenance record and publication-quality
  figures for every analysis.

## Workflow

The workbench guides each analysis through four steps:

**Open workbook → Confirm channels and polarity → Inspect cycles and graphs → Export results**

| Area | Purpose |
|---|---|
| **Data Explorer** | The open workbook, its measurement, and each cycle with its heating and cooling segments. Selecting an item sets the view across the workbench. |
| **Plot workspace** | Three graphs, each with All cycles and single-cycle views, plus pan, zoom and figure export. |
| **Analysis and Properties** | Channel assignment and lead polarity, results for the current selection, a per-cycle comparison table, and measurement information. |

## Scientific conventions

| Term | Definition |
|---|---|
| ΔT | T<sub>hot</sub> − T<sub>cold</sub>, in K |
| V<sub>rec</sub> | Cell voltage as logged, V(+) − V(−), converted from the header unit to mV |
| σ | Wiring sign: +1 with the positive lead on the cold electrode, −1 with it on the hot electrode |
| S<sub>app</sub> | Least-squares slope of σ·V<sub>rec</sub> against ΔT with a free intercept, in mV/K. A positive value means the cold electrode is at the higher potential. |
| Cycle | Odd event marks switch heating on, even marks switch it off, and the final mark ends the run. Cycle *k* spans [on<sub>k</sub>, on<sub>k+1</sub>), with heating [on<sub>k</sub>, off<sub>k</sub>) and cooling [off<sub>k</sub>, on<sub>k+1</sub>). |
| Uncertainty | Standard error of the mean over the cycles of one specimen. It describes within-specimen repeatability only; repeated cycles are not independent specimens. Per-cycle slope standard errors are reported as diagnostics. |

> **S<sub>app</sub> is an apparent dynamic coefficient** of the three-minute
> heating / three-minute cooling drive programme. It is not a validated
> steady-state Seebeck coefficient, and a difference between the heating and
> cooling slopes indicates hysteresis.

## Data integrity

- Instrument files are opened read-only and identified by their SHA-256 digest.
- An export is refused if the source file has changed since it was read, and
  nothing is ever written to the folder that holds the original recordings.
- Units come from the instrument header. A unit that cannot be interpreted is
  refused, never assumed.
- A recording with an unexpected layout is refused with the offending row; it is
  never silently repaired.

## Exports

Each export creates a new, time-stamped folder:

| File | Content |
|---|---|
| `*_cycles.csv` | Per-cycle windows, phase durations and fits (slope, intercept, R²) for the whole cycle, heating and cooling |
| `*_summary.csv` | Mean, standard deviation and within-specimen standard error for each coefficient type, with the definitions applied |
| `*_analysis.json` | Provenance record: source path and SHA-256, sheet, raw headers, event marks, channel assignment, polarity, unit conversions and results |
| `*_timeseries`, `*_voltage_vs_dT`, `*_coefficients` | Figures as PNG (300 dpi) and SVG with editable text |

## Validation

The analyzer is validated against the authentic IL1 75 % / IL2 25 % recording.
It reproduces the reference values established with Latos V1: whole-cycle
S<sub>app</sub> of +2.1696, +2.1801 and +2.2204 mV/K, mean +2.1900 ± 0.0155 mV/K
(SEM, three cycles). The reversed-lead recording, declared with the positive lead
on the hot electrode, gives +2.2733 mV/K. The voltage unit was traced to the raw
instrument values: a 5 µV reading step in a channel recorded in volts.

## Roadmap

| Area | Status |
|---|---|
| HIOKI measurement analysis | Available |
| Experiment definition and observation management | Next |
| Gaussian-process Bayesian optimization | Planned |
| Laboratory recommendations and prospective validation | Planned |
| MXene instrument analysis | Planned |

## Technology

Python, PySide6, NumPy, openpyxl and Matplotlib.
