# Latos V2

**A scientific workbench for experimental materials informatics.**

Latos is a local desktop application, under development, that helps researchers
work with authentic laboratory measurements, organize experimental observations
and make traceable decisions about what to measure or synthesize next.

Its research goal is one usable workflow that connects:

- **Authentic laboratory measurement analysis.** Raw instrument files, with their
  units, settings and sample relationships preserved.
- **Researcher-defined experiments.** Inputs, outputs, constraints and objectives
  set by the researcher, not fixed by the software.
- **Bayesian optimization.** Established methods for choosing the next
  experiments.
- **Laboratory-ready recommendations.** Feasible, concrete instructions for the
  next experiment.
- **Prospective validation.** Predictions recorded before synthesis and checked
  against the measurements that follow.
- **Small-data scientific reliability.** An honest account of how models and
  uncertainty estimates behave in small, noisy experimental campaigns.

Latos is not specific to ionic liquids, MXenes, thermoelectrics or any fixed set
of input variables.

**Current status:** Milestone 1 (HIOKI Measurement Analyzer) and Milestone 2
(experiment definition and observation management) are complete. Gaussian-process
modeling and Bayesian optimization are next; recommendations, prospective
validation and other instrument modules are planned. None of these is available
yet. Latos V2 is a research prototype, not
a validated general-purpose optimization product.

## Applications

The initial research applications are:

- **Ionic-liquid-soaked cotton:** time-resolved thermovoltage analysis, followed
  by planned composition optimization and prospective measurement. This is the
  first implementation and optimization demonstration.
- **MXene materials:** planned quantitative analysis of XRD, XPS, Raman, PPMS and
  microscopy measurements, followed by cross-technique comparisons where
  scientifically justified.

These are application datasets, not permanent restrictions on the
experiment-definition or optimization architecture.

## Available now

### HIOKI Measurement Analyzer (M1)

- Opens supported HIOKI Memory HiLogger Excel exports read-only, retaining the
  original channel labels and units.
- Uses Event marks to identify heating and cooling cycles.
- Plots temperature, temperature difference and voltage against time; voltage
  against temperature difference; and a comparison of coefficients across cycles.
- Computes whole-cycle, heating-only and cooling-only **apparent dynamic**
  coefficients, with cycle mean, SD and within-specimen SEM.
- Requires explicit channel assignments and voltage-lead polarity. Polarity is
  never inferred from file names.
- Lets the researcher exclude a cycle from the specimen value, such as an initial
  test cycle recorded in the lab notebook, with the reason recorded. Excluded
  cycles remain visible and are marked in every view and export.
- Exports per-cycle and summary CSV files, a provenance JSON record and PNG/SVG
  figures.

**Workflow:** Open workbook → Confirm channels and polarity → Inspect cycles and
graphs → Export results.

### Experiment workspace (M2)

- **Definition:** inputs (continuous, discrete or categorical, with units and
  allowed ranges or values), mixtures of any number of components with a fixed
  or variable total mass, and outputs with units and a goal (maximize, minimize
  or reach a target).
- **Observations:** one table row per physical specimen, with mass fractions,
  input values, each output's response, the raw masses and the source recording.
  Missing or out-of-range entries are flagged against the specimen.
- **Measurement sessions:** a HIOKI analysis is attached to a specimen as a
  session, copying the analyzer's values and provenance without recalculation.
  Several sessions of one specimen, such as forward and reversed leads, are kept
  side by side. One explicitly chosen **primary** session supplies the
  specimen's response, and sessions are never averaged together. Values
  measured elsewhere can be entered by hand.
- **Persistence:** an experiment saves to, and reopens from, a single
  `.latos.json` file.

Constraints are currently limited to input ranges, allowed levels and a fixed
total mass. Constraints that relate several inputs are not yet supported.

## Getting started

On Windows, from the Latos V2 project directory with the project's virtual
environment and dependencies installed:

```powershell
.venv\Scripts\python -m latos2
```

1. In **HIOKI Analyzer**, choose **File ▸ Open workbook…**, confirm the channel
   roles and voltage-lead polarity from the laboratory record, then inspect and
   export the results.
2. In **Experiment**, define the mixture, inputs and outputs on the Definition
   tab, add specimens, and use **Attach HIOKI analysis…** to attach the analysis
   shown in the analyzer to a specimen.

For dependency details, see `pyproject.toml`. Latos is not yet packaged as a
standalone installer.

## Scientific conventions

| Quantity | Current implementation |
|---|---|
| Temperature difference | ΔT = T_hot − T_cold, in K |
| Recorded voltage | V_rec = V(+) − V(−), converted from the instrument's recorded unit to mV |
| Polarity | σ = +1 if V+ is on the cold electrode, −1 if V+ is on the hot electrode |
| Apparent coefficient | S_app = least-squares slope of σ × V_rec against ΔT with a free intercept, in mV/K. A positive value means the cold electrode is at the higher potential. |
| Cycles | Odd Event marks switch heating on, even marks switch it off, and the final mark ends the run. Cycle k spans [on_k, on_k+1), with heating [on_k, off_k) and cooling [off_k, on_k+1). If the final cooling period has no closing mark, it ends at the last recorded sample; no mark is ever added to the file. |
| Reported variability | Within-specimen statistics of repeated cycles; not independent specimen-to-specimen uncertainty |
| Mass fraction | x_i = m_i / Σ m_j from the weighed masses. A pure-component specimen has fractions 1 and 0 by identity; an unrecorded mass is shown as *Not recorded*, never estimated. |

**Interpretation limit:** the heating/cooling slopes describe a
protocol-dependent apparent dynamic response. They are **not validated
steady-state Seebeck coefficients**. Heating/cooling hysteresis and the choice of
fitting window matter.

## Reference measurements

For the verified IL1 75 % / IL2 25 % recording, the three whole-cycle apparent
coefficients are **+2.1696, +2.1801 and +2.2204 mV/K**, giving
**+2.1900 ± 0.0155 mV/K** (mean ± within-specimen SEM). The separately analyzed
reversed-lead recording gives **+2.2733 mV/K** under its declared polarity.

The five specimens of the ionic-liquid campaign (IL1 = Aliquat 336TG,
IL2 = [HMIM][TFSI]), with forward leads as the primary session:

| Specimen | x_IL1 | S_app, mean ± SEM (mV/K) | Note |
|---|---|---|---|
| IL1 0 / IL2 100 | 0 (pure) | −2.6434 ± 0.3703 | Initial test cycle excluded |
| IL1 25 / IL2 75 | 0.2396 | +2.0256 ± 0.0069 | |
| IL1 50 / IL2 50 | 0.4813 | +2.0187 ± 0.0769 | |
| IL1 75 / IL2 25 | 0.7587 | +2.1900 ± 0.0155 | |
| IL1 100 / IL2 0 | 1 (pure) | +1.5642 ± 0.0279 | Initial test cycle excluded |

These are reproduced analysis results under specific recording conditions, not
evidence of universal model accuracy or steady-state material performance.

## Data integrity and exports

- Original instrument files are opened read-only and identified by SHA-256.
- Latos checks for source-file changes before exporting, and refuses to write
  exports or experiment files into the folder that holds the original recordings.
- Unknown units and unexpected file layouts are reported rather than guessed or
  silently repaired.
- An experiment file records the SHA-256 of every attached recording. When it is
  reopened, a changed or missing recording is reported; values are never
  silently recalculated.
- Results export as per-cycle and summary CSV files, provenance JSON, and figures
  in PNG (300 dpi) and SVG.

## Development roadmap

| Milestone | Purpose | Status |
|---|---|---|
| M1 | HIOKI measurement analysis and scientific desktop UI | **Complete** |
| M2 | Researcher-defined experiments, mixtures, specimen/measurement relationships, observation management | **Complete** |
| M3 | Gaussian-process modeling, uncertainty display and Bayesian acquisition | Next |
| M4 | Laboratory-ready next-experiment instructions, frozen predictions and prospective scoring | Planned |
| M5 | MXene instrument analysis and evidence-linked multitechnique comparisons | Planned |

Each milestone is scoped to a usable scientific deliverable. A milestone is
described as available only after it has been implemented and tested.

## Technology and development approach

Python, PySide6, NumPy, openpyxl and Matplotlib. Future numerical components will
use established scientific libraries rather than new implementations of
foundational algorithms. Latos runs locally, without a cloud service, a separate
backend server or a language model.

The project prioritizes scientific usefulness, transparent limitations and a
manageable development scope over production-grade infrastructure.
