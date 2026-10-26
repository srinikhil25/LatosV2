"""An experiment: its inputs, mixture, outputs, and the physical specimens measured.

Three entities stay distinct throughout:

* a **specimen** is one physically prepared sample: one row of the observation
  table;
* a **measurement session** is one recording of one specimen, such as a HIOKI
  run with forward or with reversed voltage leads;
* a **cycle** is one heating/cooling period inside a session. Cycles live only
  inside the session's record and are never counted as specimens.

Composition fractions are computed from the masses actually entered,
``x_i = m_i / Σ m_j``, and never stored. A pure-component specimen has the
fractions 1 and 0 by identity, whether or not its mass was recorded. A mass
that was not recorded is absent; it is never filled in.

A specimen's response for an output is its one primary measurement session.
Sessions are never averaged together, and a session whose method or unit does
not match its output's definition is not used.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from latos2.hioki_reader import file_sha256
from latos2.thermovoltage import KINDS, QUANTITY, UNIT

__all__ = [
    "INPUT_KINDS",
    "MANUAL",
    "METHOD_LABELS",
    "OBJECTIVES",
    "ROLES",
    "Experiment",
    "InputVariable",
    "Measurement",
    "Message",
    "Mixture",
    "OutputDefinition",
    "Specimen",
    "attach",
    "check_sources",
    "fractions",
    "hioki_measurement",
    "input_problem",
    "manual_measurement",
    "parse_input",
    "primary_measurement",
    "remove_component",
    "remove_input",
    "remove_output",
    "rename_component",
    "rename_input",
    "rename_output",
    "set_primary",
    "specimen_messages",
    "total_mass",
]

INPUT_KINDS = ("continuous", "discrete", "categorical")
OBJECTIVES = ("maximize", "minimize", "target")
ROLES = ("primary", "supplementary")
MANUAL = "manual"
METHOD_LABELS = {
    "hioki:combined": "HIOKI S_app, whole cycle",
    "hioki:heating": "HIOKI S_app, heating only",
    "hioki:cooling": "HIOKI S_app, cooling only",
    MANUAL: "Manual entry",
}
assert {f"hioki:{kind}" for kind in KINDS} <= set(METHOD_LABELS)


@dataclass
class InputVariable:
    """One experimental input other than mixture composition.

    Attributes:
        name: Unique name, such as ``Annealing temperature``.
        kind: ``continuous``, ``discrete`` or ``categorical``.
        unit: Unit text; empty for categorical inputs.
        lower: Smallest allowed value (continuous), or None for no limit.
        upper: Largest allowed value (continuous), or None for no limit.
        levels: Allowed values: numbers for discrete inputs, labels for
            categorical ones. Empty means any value.
    """

    name: str
    kind: str = "continuous"
    unit: str = ""
    lower: float | None = None
    upper: float | None = None
    levels: list[str] = field(default_factory=list)


@dataclass
class Mixture:
    """The components every specimen is mixed from.

    Attributes:
        components: Component names, any number.
        total_mode: ``variable``, or ``fixed`` when every specimen should have
            the same total mass.
        total_g: The fixed total mass in grams.
        tolerance_g: Allowed deviation from the fixed total, in grams.
    """

    components: list[str] = field(default_factory=list)
    total_mode: str = "variable"
    total_g: float | None = None
    tolerance_g: float | None = None


@dataclass
class OutputDefinition:
    """One measured property and how it is to be optimized.

    Attributes:
        name: Unique name, such as ``S_app``.
        unit: The unit every value of this output is in.
        method: How it is measured: a key of `METHOD_LABELS`.
        objective: ``maximize``, ``minimize`` or ``target``.
        target: The target value when `objective` is ``target``.
    """

    name: str
    unit: str
    method: str = MANUAL
    objective: str = "maximize"
    target: float | None = None


@dataclass
class Measurement:
    """One measurement session of one specimen, for one output.

    Attributes:
        output: The output this session measures.
        label: The researcher's name for the session, such as
            ``Forward leads``.
        role: ``primary`` (the specimen's response) or ``supplementary``
            (kept for comparison, never used as the response).
        method: How it was measured: a key of `METHOD_LABELS`.
        unit: The unit of `value`.
        value: The session's value: for HIOKI, the mean over its accepted
            cycles, exactly as the analyzer reported it.
        sd: Standard deviation over its cycles, if any.
        sem: Within-specimen standard error of the mean, if any.
        n: Number of cycles (or repeat readings) behind `value`.
        note: Free text.
        record: For HIOKI sessions, the analyzer's provenance record: source
            file and SHA-256, channels, polarity, event marks, cycles,
            exclusions and summary.
        id: Stable identifier within the experiment.
    """

    output: str
    label: str
    role: str
    method: str
    unit: str
    value: float | None
    sd: float | None = None
    sem: float | None = None
    n: int | None = None
    note: str = ""
    record: dict | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def source_path(self) -> Path | None:
        return Path(self.record["source"]["path"]) if self.record else None

    @property
    def source_sha256(self) -> str | None:
        return self.record["source"]["sha256_at_read"] if self.record else None

    @property
    def wiring(self) -> str | None:
        return self.record["declarations"]["wiring"] if self.record else None


@dataclass
class Specimen:
    """One physically prepared specimen.

    Attributes:
        id: Unique experiment ID.
        label: Nominal label as written in the lab notebook, such as
            ``IL1 75 / IL2 25``. Kept apart from the actual masses.
        pure_component: The component this specimen consists of alone, or
            None for a mixture.
        masses_g: Actual measured mass of each component, in grams. A
            component with no recorded mass is absent.
        inputs: Value of each input variable as entered: a number, or a label
            for categorical inputs.
        notes: Free text.
        measurements: Its measurement sessions.
    """

    id: str
    label: str = ""
    pure_component: str | None = None
    masses_g: dict[str, float] = field(default_factory=dict)
    inputs: dict[str, float | str] = field(default_factory=dict)
    notes: str = ""
    measurements: list[Measurement] = field(default_factory=list)


@dataclass
class Experiment:
    """An optimization problem and the observations made for it."""

    name: str = "Untitled experiment"
    description: str = ""
    inputs: list[InputVariable] = field(default_factory=list)
    mixture: Mixture | None = None
    outputs: list[OutputDefinition] = field(default_factory=list)
    specimens: list[Specimen] = field(default_factory=list)

    def output(self, name: str) -> OutputDefinition | None:
        return next((o for o in self.outputs if o.name == name), None)

    def specimen(self, specimen_id: str) -> Specimen:
        for specimen in self.specimens:
            if specimen.id == specimen_id:
                return specimen
        raise KeyError(f"no specimen {specimen_id!r}")


@dataclass(frozen=True)
class Message:
    """A finding about a specimen: ``warning`` needs attention, ``note`` informs."""

    level: str
    text: str


# -- composition ----------------------------------------------------------------


def total_mass(exp: Experiment, specimen: Specimen) -> float | None:
    """The specimen's total mass in grams, or None when it was not recorded."""
    if exp.mixture is None or not exp.mixture.components:
        return None
    if specimen.pure_component:
        return specimen.masses_g.get(specimen.pure_component)
    masses = [specimen.masses_g.get(c) for c in exp.mixture.components]
    if any(m is None for m in masses):
        return None
    return math.fsum(masses)


def fractions(exp: Experiment, specimen: Specimen) -> dict[str, float] | None:
    """Mass fractions ``x_i = m_i / Σ m_j``, or 1 and 0 for a pure component.

    None when a mass needed for the calculation is missing or invalid.
    """
    if exp.mixture is None or not exp.mixture.components:
        return None
    components = exp.mixture.components
    if specimen.pure_component in components:
        return {c: 1.0 if c == specimen.pure_component else 0.0 for c in components}
    if specimen.pure_component:
        return None
    masses = [specimen.masses_g.get(c) for c in components]
    if any(m is None or m < 0 for m in masses):
        return None
    total = math.fsum(masses)
    if total <= 0:
        return None
    return {c: m / total for c, m in zip(components, masses, strict=True)}


# -- inputs ---------------------------------------------------------------------


def parse_input(variable: InputVariable, text: str) -> float | str | None:
    """Turn text as typed into an input value; None for an empty entry."""
    text = text.strip()
    if not text:
        return None
    if variable.kind == "categorical":
        return text
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{variable.name} must be a number") from None


def input_problem(variable: InputVariable, value: float | str | None) -> str | None:
    """Why `value` is not allowed for `variable`, or None when it is."""
    if value is None:
        return f"{variable.name} not entered"
    unit = f" {variable.unit}" if variable.unit else ""
    if variable.kind == "categorical":
        if variable.levels and value not in variable.levels:
            return f"{variable.name} = {value} is not one of: {', '.join(variable.levels)}"
        return None
    if isinstance(value, str):
        return f"{variable.name} must be a number"
    if variable.kind == "discrete" and variable.levels:
        allowed = [float(level) for level in variable.levels]
        if not any(math.isclose(value, a, rel_tol=1e-9, abs_tol=1e-12) for a in allowed):
            return f"{variable.name} = {value:g}{unit} is not one of: {', '.join(variable.levels)}"
    if variable.lower is not None and value < variable.lower:
        return f"{variable.name} = {value:g}{unit} is below the allowed minimum {variable.lower:g}{unit}"
    if variable.upper is not None and value > variable.upper:
        return f"{variable.name} = {value:g}{unit} is above the allowed maximum {variable.upper:g}{unit}"
    return None


# -- measurements ---------------------------------------------------------------


def _compatible(output: OutputDefinition | None, m: Measurement) -> bool:
    return output is not None and m.method == output.method and m.unit == output.unit


def primary_measurement(exp: Experiment, specimen: Specimen, output_name: str) -> Measurement | None:
    """The session used as the specimen's response for an output, if any."""
    output = exp.output(output_name)
    for m in specimen.measurements:
        if m.output == output_name and m.role == "primary" and _compatible(output, m):
            return m
    return None


def hioki_measurement(record: dict, output: OutputDefinition, label: str, role: str) -> Measurement:
    """A session built from the HIOKI analyzer's record, without recalculation.

    Raises:
        ValueError: If the output is not measured by the HIOKI analyzer, its
            unit or quantity differs, or the analysis has no accepted cycle.
    """
    if not output.method.startswith("hioki:"):
        raise ValueError(
            f"{output.name} is defined as {METHOD_LABELS.get(output.method, output.method)}; "
            "a HIOKI analysis cannot be attached to it"
        )
    if output.unit != UNIT or record["units"]["coefficient_unit"] != UNIT:
        raise ValueError(f"{output.name} is in {output.unit}; HIOKI coefficients are in {UNIT}")
    if record["definition"]["quantity"] != QUANTITY:
        raise ValueError("the analysis measures a different quantity than this output")
    kind = output.method.split(":", 1)[1]
    summary = record["summary"].get(kind)
    if summary is None:
        raise ValueError("the analysis has no accepted cycle, so it has no value to attach")
    return Measurement(
        output=output.name,
        label=label.strip() or record["declarations"]["wiring"],
        role=role,
        method=output.method,
        unit=UNIT,
        value=summary["mean_mV_per_K"],
        sd=summary["sd_mV_per_K"],
        sem=summary["sem_within_specimen_mV_per_K"],
        n=summary["k_cycles"],
        record=record,
    )


def manual_measurement(
    output: OutputDefinition,
    value: float,
    label: str,
    role: str,
    sem: float | None = None,
    n: int | None = None,
    note: str = "",
) -> Measurement:
    """A session whose value the researcher enters, in the output's unit."""
    if output.method != MANUAL:
        raise ValueError(
            f"{output.name} is measured by the HIOKI analyzer; attach the analysis instead of typing a value"
        )
    if not math.isfinite(value):
        raise ValueError("the value must be a finite number")
    return Measurement(
        output=output.name, label=label.strip() or "Manual entry", role=role, method=MANUAL,
        unit=output.unit, value=value, sem=sem, n=n, note=note,
    )


def attach(exp: Experiment, specimen: Specimen, measurement: Measurement) -> None:
    """Add a session to a specimen.

    A recording belongs to one specimen only. Attaching a session as primary
    makes the specimen's earlier primary for that output supplementary.

    Raises:
        ValueError: If the same recording is already attached anywhere in the
            experiment, or the output is not defined.
    """
    if exp.output(measurement.output) is None:
        raise ValueError(f"no output named {measurement.output!r}")
    sha = measurement.source_sha256
    if sha:
        for other in exp.specimens:
            for m in other.measurements:
                if m.source_sha256 == sha:
                    raise ValueError(
                        f"{measurement.source_path.name} is already attached to specimen {other.id} "
                        f"({m.label}); a recording belongs to one specimen"
                    )
    specimen.measurements.append(measurement)
    if measurement.role == "primary":
        set_primary(specimen, measurement.id)


def set_primary(specimen: Specimen, measurement_id: str) -> None:
    """Make one session the response for its output; the others become supplementary."""
    chosen = next(m for m in specimen.measurements if m.id == measurement_id)
    for m in specimen.measurements:
        if m.output == chosen.output:
            m.role = "primary" if m is chosen else "supplementary"


def check_sources(exp: Experiment) -> dict[str, str]:
    """Re-check each HIOKI session's source file: ``ok``, ``missing`` or ``changed``.

    Values are never recalculated; a changed or missing file is only reported.
    """
    digests: dict[Path, str | None] = {}
    status: dict[str, str] = {}
    for specimen in exp.specimens:
        for m in specimen.measurements:
            path = m.source_path
            if path is None:
                continue
            if path not in digests:
                digests[path] = file_sha256(path) if path.is_file() else None
            digest = digests[path]
            status[m.id] = "missing" if digest is None else ("ok" if digest == m.source_sha256 else "changed")
    return status


# -- findings -------------------------------------------------------------------


def specimen_messages(
    exp: Experiment, specimen: Specimen, sources: dict[str, str] | None = None
) -> list[Message]:
    """Warnings and notes for one specimen, recomputed whenever asked."""
    out: list[Message] = []
    warn = lambda text: out.append(Message("warning", text))  # noqa: E731
    note = lambda text: out.append(Message("note", text))  # noqa: E731

    mixture = exp.mixture
    if mixture is not None and mixture.components:
        pure = specimen.pure_component
        if pure:
            if pure not in mixture.components:
                warn(f"Pure component {pure} is not a component of this mixture")
            elif pure not in specimen.masses_g:
                note(f"Pure {pure}; mass not recorded")
            for c in mixture.components:
                if c != pure and specimen.masses_g.get(c, 0.0) > 0:
                    warn(f"Pure {pure} specimen has a recorded mass of {c}")
        else:
            for c in mixture.components:
                mass = specimen.masses_g.get(c)
                if mass is None:
                    warn(f"Mass of {c} not recorded")
                elif mass < 0:
                    warn(f"Mass of {c} is negative")
            total = total_mass(exp, specimen)
            if total is not None and total <= 0:
                warn("Total mass is zero")
        total = total_mass(exp, specimen)
        if mixture.total_mode == "fixed" and mixture.total_g is not None and total is not None:
            tolerance = mixture.tolerance_g or 0.0
            if abs(total - mixture.total_g) > tolerance + 1e-12:
                warn(
                    f"Total mass {total:.4f} g differs from the fixed total {mixture.total_g:.4f} g "
                    f"(± {tolerance:.4f} g)"
                )

    for variable in exp.inputs:
        problem = input_problem(variable, specimen.inputs.get(variable.name))
        if problem:
            warn(problem)

    sources = sources or {}
    for output in exp.outputs:
        sessions = [m for m in specimen.measurements if m.output == output.name]
        if not sessions:
            warn(f"No measured {output.name}")
            continue
        primary = primary_measurement(exp, specimen, output.name)
        if primary is None:
            warn(f"No primary measurement selected for {output.name}")
        supplementary = [m for m in sessions if m is not primary and _compatible(output, m)]
        if supplementary:
            note(f"{len(supplementary)} supplementary session(s) for {output.name}, not used in the response")
        for m in sessions:
            if not _compatible(output, m):
                warn(
                    f"Session '{m.label}' is {METHOD_LABELS.get(m.method, m.method)} in {m.unit}, but "
                    f"{output.name} is defined as {METHOD_LABELS.get(output.method, output.method)} "
                    f"in {output.unit}; it is not used"
                )
            state = sources.get(m.id)
            if state == "missing":
                warn(f"Source file of '{m.label}' not found: {m.source_path}")
            elif state == "changed":
                warn(f"Source file of '{m.label}' changed since it was analysed: {m.source_path.name}")
        if primary is not None and primary.record is not None:
            record = primary.record
            for index, reason in record.get("excluded_cycles", {}).items():
                note(f"Cycle {index} excluded: {reason}")
            rejected = [str(c["cycle"]) for c in record["cycles"] if c["problems"]]
            if rejected:
                warn(f"Cycle(s) {', '.join(rejected)} rejected by the analyzer checks")
            if "final recorded sample" in record.get("end_of_run", ""):
                note("Last cycle ends at the final recorded sample")
        if primary is not None and (primary.n or 0) < 2 and primary.sem is None:
            note(f"{output.name} rests on a single value: no SEM")

    for m in specimen.measurements:
        if exp.output(m.output) is None:
            warn(f"Session '{m.label}' belongs to output '{m.output}', which is no longer defined")
    return out


# -- definition edits that keep specimens consistent --------------------------


def rename_component(exp: Experiment, old: str, new: str) -> None:
    mixture = exp.mixture
    if mixture is None or old == new:
        return
    if new in mixture.components:
        raise ValueError(f"a component named {new!r} already exists")
    mixture.components[mixture.components.index(old)] = new
    for s in exp.specimens:
        if old in s.masses_g:
            s.masses_g[new] = s.masses_g.pop(old)
        if s.pure_component == old:
            s.pure_component = new


def remove_component(exp: Experiment, name: str) -> None:
    if exp.mixture is None:
        return
    exp.mixture.components.remove(name)
    for s in exp.specimens:
        s.masses_g.pop(name, None)
        if s.pure_component == name:
            s.pure_component = None


def rename_input(exp: Experiment, old: str, new: str) -> None:
    if old == new:
        return
    if any(v.name == new for v in exp.inputs):
        raise ValueError(f"an input named {new!r} already exists")
    next(v for v in exp.inputs if v.name == old).name = new
    for s in exp.specimens:
        if old in s.inputs:
            s.inputs[new] = s.inputs.pop(old)


def remove_input(exp: Experiment, name: str) -> None:
    exp.inputs = [v for v in exp.inputs if v.name != name]
    for s in exp.specimens:
        s.inputs.pop(name, None)


def rename_output(exp: Experiment, old: str, new: str) -> None:
    if old == new:
        return
    if exp.output(new) is not None:
        raise ValueError(f"an output named {new!r} already exists")
    exp.output(old).name = new
    for s in exp.specimens:
        for m in s.measurements:
            if m.output == old:
                m.output = new


def remove_output(exp: Experiment, name: str) -> None:
    exp.outputs = [o for o in exp.outputs if o.name != name]
    for s in exp.specimens:
        s.measurements = [m for m in s.measurements if m.output != name]
