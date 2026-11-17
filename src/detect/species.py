"""The species the GUI offers, read from configs/gui_species.json.

The GUI ships with the lobster detectors of this repository; other species (for example a
teammate's model) can be added as entries. One JSON file lists, per species, the models that can
be chosen and the image sources that can be browsed, so a model or an image folder is added on
any machine by editing that file, without touching code.

A model entry is either `{"id": "E027"}` (one of the built-in lobster models in models.py) or a
full entry with `label`, `weights` and `description`. A model whose settings are not in
experiments/records may declare them (`declared_conf`, `declared_imgsz`, `declared_source`); the
GUI then says that they are declared by the model's owner, not read from a project record
(CLAUDE.md rule 8: no number is invented here).

An image source is one of:
  {"kind": "raw", "name": "zExtra"}            a configured raw source (needs the raw dataset)
  {"kind": "split", "name": "val"}             a built YOLO split (needs data/yolo on this machine)
  {"kind": "folder", "path": "samples/..."}    any folder of images; labels are read when it
                                               follows the images/ <-> labels/ convention

Standard library only, so the CPU tests can import it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .enhanced import RAW_SOURCE
from .models import MODELS, ModelChoice
from .predict import HELD_OUT_SPLITS, LIMITATIONS, PROJECT_ROOT, SPLIT_STATUS, DetectError

SPECIES_CONFIG = PROJECT_ROOT / "configs" / "gui_species.json"
SOURCE_KINDS = ("raw", "split", "folder")


@dataclass(frozen=True)
class Source:
    """One browsable image source of a species."""

    label: str
    kind: str  # one of SOURCE_KINDS
    name: str  # raw source name, split name, or folder path (relative to the project root)
    note: str = ""  # what this source can honestly tell you; shown above the image picker
    held_out: bool = False  # True: thresholds must not be tuned while looking at it (rule 7)


@dataclass(frozen=True)
class Species:
    key: str
    label: str  # shown in the selector, e.g. "Lobster"
    animal: str  # used in sentences, e.g. "lobster", "sea slug"
    icon: str
    models: tuple[ModelChoice, ...]
    sources: tuple[Source, ...]
    limitations: tuple[str, ...]


def _fail(where: str, message: str) -> DetectError:
    return DetectError(f"configs/gui_species.json, {where}: {message}")


def _model(entry: dict, where: str) -> ModelChoice:
    if not isinstance(entry, dict) or not entry.get("id"):
        raise _fail(where, "every model needs an \"id\".")
    builtin = next((m for m in MODELS if m.experiment_id == entry["id"]), None)
    if builtin is None:
        missing = [k for k in ("label", "weights", "description") if not entry.get(k)]
        if missing:
            raise _fail(where, f"model {entry['id']!r} is not a built-in model, so it needs {', '.join(missing)}.")
        base = ModelChoice(str(entry["id"]), "", Path("."), RAW_SOURCE, "")
    else:
        base = builtin
    conf, imgsz = entry.get("declared_conf"), entry.get("declared_imgsz")
    if conf is not None and not (isinstance(conf, (int, float)) and 0.0 < conf < 1.0):
        raise _fail(where, f"declared_conf must be a number between 0 and 1, got {conf!r}.")
    if imgsz is not None and not (isinstance(imgsz, int) and imgsz > 0):
        raise _fail(where, f"declared_imgsz must be a positive whole number, got {imgsz!r}.")
    return replace(
        base,
        label=str(entry.get("label", base.label)),
        weights=Path(entry["weights"]) if entry.get("weights") else base.weights,
        image_source=str(entry.get("image_source", base.image_source)),
        description=str(entry.get("description", base.description)),
        declared_conf=float(conf) if conf is not None else base.declared_conf,
        declared_imgsz=imgsz if imgsz is not None else base.declared_imgsz,
        declared_source=str(entry.get("declared_source", base.declared_source)),
        scores_note=str(entry.get("scores_note", base.scores_note)),
    )


def _source(entry: dict, where: str) -> Source:
    kind = entry.get("kind") if isinstance(entry, dict) else None
    if kind not in SOURCE_KINDS:
        raise _fail(where, f"source kind must be one of {', '.join(SOURCE_KINDS)}, got {kind!r}.")
    if not entry.get("label"):
        raise _fail(where, "every source needs a \"label\".")
    if kind == "folder":
        if not entry.get("path"):
            raise _fail(where, "a folder source needs a \"path\".")
        return Source(str(entry["label"]), kind, str(entry["path"]), str(entry.get("note", "")),
                      bool(entry.get("held_out", False)))
    if not entry.get("name"):
        raise _fail(where, f"a {kind} source needs a \"name\".")
    name = str(entry["name"])
    if kind == "split":
        if name not in SPLIT_STATUS:
            raise _fail(where, f"unknown split {name!r}; known splits: {', '.join(sorted(SPLIT_STATUS))}.")
        return Source(str(entry["label"]), kind, name, str(entry.get("note", SPLIT_STATUS[name])),
                      bool(entry.get("held_out", name in HELD_OUT_SPLITS)))
    return Source(str(entry["label"]), kind, name, str(entry.get("note", "")), bool(entry.get("held_out", False)))


def parse_species(data: object) -> tuple[Species, ...]:
    """Validate the parsed JSON and return the species in file order. Pure function."""
    entries = data.get("species") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise _fail("top level", "expected {\"species\": [ ... ]} with at least one species.")
    out: list[Species] = []
    for i, entry in enumerate(entries):
        where = f"species #{i + 1}"
        if not isinstance(entry, dict) or not entry.get("key") or not entry.get("label"):
            raise _fail(where, "every species needs a \"key\" and a \"label\".")
        key = str(entry["key"])
        where = f"species {key!r}"
        if any(s.key == key for s in out):
            raise _fail(where, "this key is used twice.")
        models = tuple(_model(m, where) for m in entry.get("models") or [])
        if not models:
            raise _fail(where, "needs at least one model.")
        ids = [m.experiment_id for m in models]
        if len(set(ids)) != len(ids):
            raise _fail(where, "two models share an id.")
        limitations = entry.get("limitations")
        if limitations is None:
            limitations = LIMITATIONS if key == "lobster" else ()
        out.append(Species(
            key=key, label=str(entry["label"]), animal=str(entry.get("animal", entry["label"])).lower(),
            icon=str(entry.get("icon", "")), models=models,
            sources=tuple(_source(s, where) for s in entry.get("sources") or []),
            limitations=tuple(str(x) for x in limitations),
        ))
    return tuple(out)


def load_species(path: Path = SPECIES_CONFIG) -> tuple[Species, ...]:
    """Read the species file. Raises DetectError with an actionable message."""
    path = Path(path)
    if not path.is_file():
        raise DetectError(f"Species file not found: {path}\nIt lists the species, models and image sources "
                          "the GUI offers.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DetectError(f"{path} is not valid JSON: {exc}") from exc
    return parse_species(data)
