"""The validated setting ("sweet spot") of every GUI slider, for the checkpoint that is loaded.

Every value comes from a committed record or a recorded decision, never from looking at images
in the GUI (CLAUDE.md rule 7: thresholds are chosen on val, never on test):
- confidence: the run's operating confidence, chosen on val by max F1 (val_eval_record.json,
  else final_test_record.json), found by matching the checkpoint's sha256 to a run record;
- NMS IoU: 0.7, the value every reported val and test number used (Ultralytics' default);
- inference size: the run's training imgsz (run_record.json);
- match IoU: 0.5, fixed in D-020 before any result existed.

Standard library only, so the CPU tests can import it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .predict import DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_IOU, PROJECT_ROOT

DEFAULT_MATCH_IOU = 0.5
RECORDS_ROOT = PROJECT_ROOT / "experiments" / "records"


@dataclass(frozen=True)
class SweetSpot:
    """Validated slider values for one checkpoint, with where each came from."""

    conf: float
    iou: float
    imgsz: int
    match_iou: float
    experiment_id: str | None  # None: checkpoint not found in the committed records
    conf_source: str
    imgsz_source: str
    conf_declared: bool = False  # True: conf comes from configs/gui_species.json, not a record

    @property
    def known(self) -> bool:
        return self.experiment_id is not None


def file_sha256(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find_experiment_for_checkpoint(weights_sha256: str, records_root: Path = RECORDS_ROOT) -> str | None:
    """The experiment whose run_record.json lists this sha256 as its best.pt; None if none does."""
    root = Path(records_root)
    if not root.is_dir():
        return None
    for run_record in sorted(root.glob("*/run_record.json")):
        rec = _read_json(run_record) or {}
        best = ((rec.get("outputs") or {}).get("best") or {}).get("sha256")
        if best and best == weights_sha256:
            return (rec.get("experiment") or {}).get("id") or run_record.parent.name
    return None


def sweet_spot_for_sha256(weights_sha256: str, records_root: Path = RECORDS_ROOT,
                          declared_conf: float | None = None, declared_imgsz: int | None = None,
                          declared_source: str = "") -> SweetSpot:
    """Look up the validated settings for a checkpoint by its sha256. Unknown -> project defaults.

    `declared_*` are the settings a model's owner wrote in configs/gui_species.json. They are used
    only where the committed records have no value, and the GUI says that they are declared.
    """
    exp = find_experiment_for_checkpoint(weights_sha256, records_root)
    declared_text = (f"declared for this model in configs/gui_species.json "
                     f"({declared_source or 'no source given'}); not read from a project record.")
    if exp is None:
        if declared_conf is not None:
            imgsz = declared_imgsz or DEFAULT_IMGSZ
            return SweetSpot(
                conf=round(float(declared_conf), 3), iou=DEFAULT_IOU, imgsz=imgsz, match_iou=DEFAULT_MATCH_IOU,
                experiment_id=None, conf_source=declared_text,
                imgsz_source=declared_text if declared_imgsz else f"not declared: project default {DEFAULT_IMGSZ} px.",
                conf_declared=True,
            )
        return SweetSpot(
            conf=DEFAULT_CONF, iou=DEFAULT_IOU, imgsz=DEFAULT_IMGSZ, match_iou=DEFAULT_MATCH_IOU,
            experiment_id=None,
            conf_source=f"unknown checkpoint: not in experiments/records, so E001's {DEFAULT_CONF} is used. "
                        "It was not validated for this model.",
            imgsz_source=f"unknown checkpoint: project default {DEFAULT_IMGSZ} px.",
        )

    folder = Path(records_root) / exp
    conf, conf_source = DEFAULT_CONF, f"{exp} has no val-chosen confidence on record; E001's {DEFAULT_CONF} is used."
    conf_declared = False
    for name in ("val_eval_record.json", "final_test_record.json"):
        rec = _read_json(folder / name) or {}
        value = rec.get("operating_conf")
        if isinstance(value, dict):  # final_test_record.json: {"value": ..., "chosen_on": "val", ...}
            value = value.get("value") if value.get("chosen_on", "val") == "val" else None
        if isinstance(value, (int, float)):
            conf = round(float(value), 3)
            conf_source = f"{exp}'s operating confidence, chosen on val by max F1 ({name})."
            break
    else:
        if declared_conf is not None:
            conf, conf_source, conf_declared = round(float(declared_conf), 3), declared_text, True

    run = _read_json(folder / "run_record.json") or {}
    imgsz = (run.get("train_args") or {}).get("imgsz")
    if isinstance(imgsz, int) and imgsz > 0:
        imgsz_source = f"{exp} was trained at {imgsz} px (run_record.json)."
    else:
        imgsz, imgsz_source = DEFAULT_IMGSZ, f"{exp}'s training size is not on record; default {DEFAULT_IMGSZ} px."

    return SweetSpot(conf=conf, iou=DEFAULT_IOU, imgsz=imgsz, match_iou=DEFAULT_MATCH_IOU,
                     experiment_id=exp, conf_source=conf_source, imgsz_source=imgsz_source,
                     conf_declared=conf_declared)


def sweet_spot_for(weights: str | os.PathLike[str], records_root: Path = RECORDS_ROOT,
                   declared_conf: float | None = None, declared_imgsz: int | None = None,
                   declared_source: str = "") -> SweetSpot:
    """Hash the checkpoint file and return its validated settings."""
    return sweet_spot_for_sha256(file_sha256(weights), records_root, declared_conf, declared_imgsz,
                                 declared_source)


IOU_SOURCE = (f"{DEFAULT_IOU}: the NMS IoU used for every reported val and test number "
              "(Ultralytics' default). Other values are untested.")
MATCH_IOU_SOURCE = (f"{DEFAULT_MATCH_IOU}: the matching rule fixed in D-020 before any result existed "
                    "(the same as AP50 and the report's confusion matrices).")
