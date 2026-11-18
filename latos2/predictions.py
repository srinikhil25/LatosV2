"""Frozen predictions and their prospective tests.

Freezing turns the current provisional recommendation into a *frozen
prediction*. Its condition, predicted response, uncertainty and the full model
record are fixed before the specimen is prepared or measured, together with a
SHA-256 digest of that content. A frozen prediction is never edited and never
deleted. It can be withdrawn, with a reason, as long as no specimen tests it.

A specimen prepared for a prediction is linked to it. Once the specimen has a
primary measurement of the predicted output, that measurement is scored
against what was frozen:

* error = measured − predicted mean;
* z = error / predictive SD, the SD frozen for a new specimen's measurement;
* whether the measured value lies inside the frozen 95 % interval;
* the gain over the best response measured before freezing;
* how far the specimen's actual composition and inputs lie from the frozen
  condition: reported, never used to adjust the score;
* whether the recording was made after the freeze, from the recording's own
  first time stamp.

The interval scored against is the frozen one; it is not recomputed with the
new specimen's own SEM. A single test is reported as one test: it neither
validates nor refutes the model.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, time

from latos2 import experiment as ex
from latos2.gp import Z95
from latos2.optimization import Result, prepare

__all__ = [
    "Test",
    "Timing",
    "describe",
    "digest",
    "freeze",
    "frozen_as",
    "intact",
    "link",
    "readings",
    "score",
    "status",
    "summary",
    "testers",
    "tests",
    "timing",
    "withdraw",
]

# Set after freezing, so outside the digest.
_UNSIGNED = ("sha256", "withdrawn_at", "withdrawn_reason")
# Preparation settings may change without a refit; they do not make a new recommendation.
_PREPARATION_SETTINGS = ("preparation_total_g", "balance_resolution_g")
_STAMP = re.compile(
    r"^\s*(?P<date>\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4})\s+(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})(?:\.\d+)?s?\s*$"
)


def _now() -> datetime:
    return datetime.now().astimezone().replace(microsecond=0)


def _wall_clock(stamp: str) -> datetime:
    """Local wall-clock time of an ISO time stamp, as a naive datetime."""
    return datetime.fromisoformat(stamp).replace(tzinfo=None)


def _short(stamp: str) -> str:
    return _wall_clock(stamp).strftime("%Y-%m-%d %H:%M")


# -- freezing ---------------------------------------------------------------------


def digest(p: ex.Prediction) -> str:
    """SHA-256 of everything fixed at freezing, in a canonical JSON form."""
    data = {k: v for k, v in asdict(p).items() if k not in _UNSIGNED}
    text = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def intact(p: ex.Prediction) -> bool:
    """Whether the prediction still holds exactly what was frozen."""
    return bool(p.sha256) and digest(p) == p.sha256


def _recommendation_key(record: dict) -> str:
    settings = {k: v for k, v in record.get("settings", {}).items() if k not in _PREPARATION_SETTINGS}
    return json.dumps({**record, "settings": settings}, sort_keys=True)


def frozen_as(exp: ex.Experiment, record: dict) -> ex.Prediction | None:
    """The prediction this model record's recommendation is already frozen as, if any."""
    key = _recommendation_key(json.loads(json.dumps(record)))
    return next((p for p in exp.predictions if _recommendation_key(p.record) == key), None)


def _best_before(result: Result) -> tuple[float | None, str | None]:
    used = result.problem.used
    output = result.problem.output
    if not used:
        return None, None
    if output.objective == "target":
        best = min(used, key=lambda o: abs(o.response - output.target))
        return abs(best.response - output.target), best.specimen_id
    sign = 1.0 if output.objective == "maximize" else -1.0
    best = max(used, key=lambda o: sign * o.response)
    return best.response, best.specimen_id


def freeze(exp: ex.Experiment, result: Result, note: str = "", now: datetime | None = None) -> ex.Prediction:
    """Freeze the recommendation of `result` as the experiment's next prediction.

    The caller makes sure `result` was fitted to the experiment as it is now.

    Raises:
        ValueError: If this recommendation is already frozen, or the masses
            for the entered total cannot be prepared.
    """
    record = json.loads(json.dumps(result.record))  # plain JSON values, exactly as saved
    existing = frozen_as(exp, record)
    if existing is not None:
        raise ValueError(f"this recommendation is already frozen as {existing.id}")
    problem, c = result.problem, result.choice
    space, settings, output = problem.space, problem.settings, problem.output
    preparation = None
    if c.fractions and settings.preparation_total_g:
        prep = prepare(space, c.fractions, settings.preparation_total_g, settings.balance_resolution_g)
        preparation = {
            "total_g": prep.total_g,
            "resolution_g": prep.resolution_g,
            "masses_g": dict(prep.masses_g),
            "fractions": dict(prep.fractions),
            "feasible": prep.feasible,
            "notes": list(prep.notes),
        }
    best, best_id = _best_before(result)
    numbers = [int(m.group(1)) for p in exp.predictions if (m := re.fullmatch(r"P(\d+)", p.id))]
    prediction = ex.Prediction(
        id=f"P{max(numbers, default=0) + 1}",
        frozen_at=(now or _now()).isoformat(timespec="seconds"),
        output=output.name,
        unit=output.unit,
        objective=output.objective,
        target=output.target,
        fractions={k: float(v) for k, v in c.fractions.items()} if c.fractions else None,
        inputs={v.key: float(value) for v, value in zip(space.variables, c.values) if v.kind != "fraction"},
        predicted_mean=float(c.mean),
        latent_sd=float(c.sd),
        predictive_sd=float(c.predictive_sd),
        interval_95=[float(c.mean - Z95 * c.predictive_sd), float(c.mean + Z95 * c.predictive_sd)],
        best_before=None if best is None else float(best),
        best_before_specimen=best_id,
        preparation=preparation,
        record=record,
        note=note.strip(),
    )
    prediction.sha256 = digest(prediction)
    exp.predictions.append(prediction)
    return prediction


def testers(exp: ex.Experiment, prediction_id: str) -> list[str]:
    """IDs of the specimens linked to a prediction."""
    return [s.id for s in exp.specimens if s.prediction == prediction_id]


def withdraw(exp: ex.Experiment, prediction_id: str, reason: str, now: datetime | None = None) -> None:
    """Withdraw a prediction no specimen tests. It is kept, with the reason.

    Raises:
        ValueError: If it does not exist, is already withdrawn, has no reason
            given, or a specimen tests it.
    """
    p = exp.prediction(prediction_id)
    if p is None:
        raise ValueError(f"there is no prediction {prediction_id}")
    if p.withdrawn_at:
        raise ValueError(f"{p.id} is already withdrawn")
    if not reason.strip():
        raise ValueError("give the reason for withdrawing the prediction")
    linked = testers(exp, p.id)
    if linked:
        raise ValueError(f"{p.id} is tested by {', '.join(linked)}; a prediction that is being tested cannot be withdrawn")
    p.withdrawn_at = (now or _now()).isoformat(timespec="seconds")
    p.withdrawn_reason = reason.strip()


def link(exp: ex.Experiment, specimen: ex.Specimen, prediction_id: str | None) -> None:
    """Record that `specimen` was prepared to test a prediction; None removes the link.

    Raises:
        ValueError: If the prediction does not exist or was withdrawn, or the
            specimen was in the data the prediction was made from.
    """
    if prediction_id is None:
        specimen.prediction = None
        return
    p = exp.prediction(prediction_id)
    if p is None:
        raise ValueError(f"there is no prediction {prediction_id}")
    if p.withdrawn_at:
        raise ValueError(f"{p.id} was withdrawn; it cannot be tested")
    fitted = {o["specimen"] for o in p.record.get("observations", []) if o.get("used")}
    if specimen.id in fitted:
        raise ValueError(f"{specimen.id} was in the data {p.id} was predicted from; it cannot test it")
    specimen.prediction = p.id


def describe(p: ex.Prediction) -> str:
    """The frozen condition in one line."""
    parts = [f"{c} {x:.4f}" for c, x in (p.fractions or {}).items()]
    parts += [f"{name} {value:.6g}" for name, value in p.inputs.items()]
    return ", ".join(parts)


# -- timing -----------------------------------------------------------------------


@dataclass(frozen=True)
class Timing:
    """Whether a recording was made after the prediction was frozen.

    `state` is ``after``, ``before`` or ``unknown``.
    """

    state: str
    text: str


def readings(stamp: str) -> list[datetime]:
    """Every calendar reading of a logger time stamp such as ``11/20/2026 10:02:25.0s``.

    The logger does not say whether its dates are month/day or day/month, so
    both readings are returned when both are valid dates. A four-digit leading
    year is read as year/month/day. A two-digit year is not read.
    """
    match = _STAMP.match(stamp or "")
    if not match:
        return []
    parts = re.split(r"[/.\-]", match.group("date"))
    first, second, third = (int(part) for part in parts)
    if len(parts[0]) == 4:
        orders = [(first, second, third)]
    elif len(parts[2]) == 4:
        orders = [(third, first, second), (third, second, first)]
    else:
        return []
    clock = time(int(match.group("h")), int(match.group("m")), int(match.group("s")))
    out: list[datetime] = []
    for year, month, day in orders:
        try:
            moment = datetime(year, month, day, clock.hour, clock.minute, clock.second)
        except ValueError:
            continue
        if moment not in out:
            out.append(moment)
    return out


def timing(p: ex.Prediction, m: ex.Measurement | None) -> Timing:
    """Compare the recording's first time stamp with the time of freezing.

    Both are wall-clock times: the logger's clock and the computer's.
    """
    if m is None:
        return Timing("unknown", "Not measured yet.")
    if m.record is None:
        return Timing("unknown", "Entered by hand: there is no recording time to compare with the freeze.")
    stamp = (m.record.get("source") or {}).get("first_timestamp") or ""
    frozen = _wall_clock(p.frozen_at)
    options = readings(stamp)
    if not options:
        return Timing("unknown", f"The recording's first time stamp ({stamp or 'none'}) cannot be read as a date.")
    when = " or ".join(t.strftime("%Y-%m-%d %H:%M") for t in options)
    later = [t > frozen for t in options]
    if all(later):
        return Timing("after", f"Recorded {when}, after the freeze ({frozen:%Y-%m-%d %H:%M}).")
    if not any(later):
        return Timing("before", f"Recorded {when}, before the freeze ({frozen:%Y-%m-%d %H:%M}): not a prospective test.")
    date_text = match.group("date") if (match := _STAMP.match(stamp)) else stamp
    return Timing(
        "unknown",
        f"The recording's date {date_text} reads as {when}: one reading is before the freeze "
        f"({frozen:%Y-%m-%d %H:%M}) and one after. Latos does not choose between month/day and day/month.",
    )


# -- scoring ----------------------------------------------------------------------


@dataclass(frozen=True)
class Test:
    """One specimen tested against one frozen prediction."""

    prediction: ex.Prediction
    specimen_id: str
    measured: float | None
    sem: float | None
    n: int | None
    error: float | None
    z: float | None
    inside: bool | None
    gain: float | None  # over the best measured before freezing; positive is better
    deviations: dict[str, float] = field(default_factory=dict)  # actual − frozen
    timing: Timing = Timing("unknown", "")
    problem: str = ""  # why the test cannot be scored yet

    @property
    def scored(self) -> bool:
        return self.measured is not None


def score(exp: ex.Experiment, p: ex.Prediction, specimen: ex.Specimen) -> Test:
    """Score one specimen's primary measurement against the frozen prediction."""
    deviations: dict[str, float] = {}
    if p.fractions:
        x = ex.fractions(exp, specimen)
        for c, frozen in p.fractions.items():
            if x is not None and c in x:
                deviations[f"{c} mass fraction"] = x[c] - frozen
    for name, frozen in p.inputs.items():
        value = specimen.inputs.get(name)
        if isinstance(value, (int, float)):
            deviations[name] = float(value) - frozen
    output = exp.output(p.output)
    m = ex.primary_measurement(exp, specimen, p.output) if output is not None else None
    when = timing(p, m)
    problem = ""
    if output is None:
        problem = f"The output {p.output} is no longer defined."
    elif m is None or m.value is None:
        problem = f"No primary measurement of {p.output} yet."
    elif m.unit != p.unit:
        problem = f"Measured in {m.unit}; the prediction is in {p.unit}."
    if problem:
        return Test(p, specimen.id, None, None, None, None, None, None, None, deviations, when, problem)
    error = m.value - p.predicted_mean
    lo, hi = p.interval_95
    gain = None
    if p.best_before is not None:
        if p.objective == "target":
            gain = p.best_before - abs(m.value - p.target)
        else:
            gain = (1.0 if p.objective == "maximize" else -1.0) * (m.value - p.best_before)
    return Test(
        prediction=p,
        specimen_id=specimen.id,
        measured=m.value,
        sem=m.sem,
        n=m.n,
        error=error,
        z=error / p.predictive_sd if p.predictive_sd > 0 else None,
        inside=lo <= m.value <= hi,
        gain=gain,
        deviations=deviations,
        timing=when,
    )


def tests(exp: ex.Experiment) -> list[Test]:
    """Every specimen linked to a prediction, scored when it has been measured."""
    return [score(exp, p, s) for p in exp.predictions for s in exp.specimens if s.prediction == p.id]


def status(exp: ex.Experiment, p: ex.Prediction) -> str:
    if not intact(p):
        return "Altered outside Latos"
    if p.withdrawn_at:
        return "Withdrawn"
    linked = testers(exp, p.id)
    if not linked:
        return "Open: awaiting a specimen"
    measured = [t.specimen_id for t in tests(exp) if t.prediction is p and t.scored]
    if measured:
        return "Tested by " + ", ".join(measured)
    return "Awaiting measurement: " + ", ".join(linked)


def summary(all_tests: list[Test]) -> str:
    """What the measured tests show, and no more."""
    scored = [t for t in all_tests if t.scored]
    n = len(scored)
    if n == 0:
        return "No prospective test has been measured yet."
    inside = sum(bool(t.inside) for t in scored)
    after = sum(t.timing.state == "after" for t in scored)
    text = (
        f"{n} prospective test{'s' if n != 1 else ''} measured: {inside} of {n} inside the frozen 95 % interval; "
        f"{after} of {n} confirmed by the recording's time stamp as recorded after the freeze."
    )
    if n == 1:
        text += " One test neither validates nor refutes the model."
    elif n < 10:
        text += f" With {n} tests, the count inside the interval cannot establish its calibration."
    return text
