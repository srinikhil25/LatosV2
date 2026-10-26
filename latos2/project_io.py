"""Save and reopen an experiment as a single JSON file.

The file holds the definition, every specimen and every measurement session,
including the analyzer's provenance record of each HIOKI session. Writing goes
to a temporary file first and replaces the target only when complete, so an
interrupted save never leaves a half-written project.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from latos2 import __version__
from latos2.experiment import (
    Experiment,
    InputVariable,
    Measurement,
    Mixture,
    OutputDefinition,
    Specimen,
)

__all__ = ["SCHEMA", "SUFFIX", "ProjectError", "load_experiment", "save_experiment"]

SCHEMA = "latos2-experiment/1"
SUFFIX = ".latos.json"


class ProjectError(ValueError):
    """The experiment file cannot be written or read."""


def save_experiment(exp: Experiment, path: str | Path) -> Path:
    """Write the experiment to `path`, never into a folder of original recordings.

    Raises:
        ProjectError: If `path` is in the folder of an attached recording.
    """
    path = Path(path)
    if not path.name.endswith(SUFFIX):
        stem = path.name[: -len(".json")] if path.name.endswith(".json") else path.name
        path = path.with_name(stem + SUFFIX)
    folder = path.parent.resolve()
    for specimen in exp.specimens:
        for m in specimen.measurements:
            if m.source_path is not None and m.source_path.parent.resolve() == folder:
                raise ProjectError(
                    "the folder holds original recordings; save the experiment in another folder"
                )
    data = {
        "schema": SCHEMA,
        "software": f"Latos V2 {__version__}",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": asdict(exp),
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_experiment(path: str | Path) -> Experiment:
    """Read an experiment written by `save_experiment`.

    Raises:
        ProjectError: If the file is not a Latos V2 experiment or is damaged.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ProjectError(f"{path.name} is not a Latos V2 experiment file")
    try:
        e = data["experiment"]
        mixture = e["mixture"]
        return Experiment(
            name=e["name"],
            description=e["description"],
            inputs=[InputVariable(**v) for v in e["inputs"]],
            mixture=Mixture(**mixture) if mixture is not None else None,
            outputs=[OutputDefinition(**o) for o in e["outputs"]],
            specimens=[
                Specimen(**{**s, "measurements": [Measurement(**m) for m in s["measurements"]]})
                for s in e["specimens"]
            ],
        )
    except (KeyError, TypeError) as exc:
        raise ProjectError(f"{path.name} is damaged: {exc}") from exc
