"""The models the GUI offers, and the scores each one has on record.

Every number shown for a model is read from its committed records (experiments/records/<id>/),
never typed in here (CLAUDE.md rule 8). A model without a record for a split simply has no score
for it: E001 and E004 were never evaluated on Zenodo `sk`, so nothing is shown for them there.

Standard library only, so the CPU tests can import it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .enhanced import ENHANCED_SOURCE, RAW_SOURCE
from .predict import PROJECT_ROOT

RECORDS_ROOT = PROJECT_ROOT / "experiments" / "records"


@dataclass(frozen=True)
class ModelChoice:
    """One selectable model: its default checkpoint and the image type it is evaluated on."""

    experiment_id: str
    label: str
    weights: Path  # relative to the repository root; weights are never committed
    image_source: str  # RAW_SOURCE or ENHANCED_SOURCE: what this model is shown
    description: str


# First entry is the default. Order: final model, then the earlier ones it is compared with.
MODELS: tuple[ModelChoice, ...] = (
    ModelChoice(
        "E027", "E027 — final model (YOLO26l, Model A data v4)", Path("weights") / "E027_best.pt", RAW_SOURCE,
        "Final lobster model (D-043): YOLO26l trained on dataset v4 (our originals + Zenodo mpa + "
        "colour-change, copy-paste and diffusion copies), tuned in two rounds on val.",
    ),
    ModelChoice(
        "E001", "E001 — first baseline (YOLO26s, v1)", Path("weights") / "E001_best.pt", RAW_SOURCE,
        "First baseline and the GUI's earlier default: YOLO26s trained on the v1 raw images only.",
    ),
    ModelChoice(
        "E004", "E004 — enhanced images (YOLO26s, v2)", Path("weights") / "E004_best.pt", ENHANCED_SOURCE,
        "YOLO26s trained on Harsh's enhanced copies of the v1 images (D-017); shown enhanced images.",
    ),
)


def model_by_id(experiment_id: str) -> ModelChoice:
    for model in MODELS:
        if model.experiment_id == experiment_id:
            return model
    known = ", ".join(m.experiment_id for m in MODELS)
    raise KeyError(f"Unknown model {experiment_id!r}. Known models: {known}.")


@dataclass(frozen=True)
class RecordedScores:
    """mAP50-95 per split as recorded; None where the model has no record for that split."""

    val: float | None
    test: float | None
    zenodo_sk: float | None

    def caption(self) -> str:
        parts = [f"{name} {value:.3f}" for name, value in
                 (("val", self.val), ("v1 test", self.test), ("Zenodo sk", self.zenodo_sk)) if value is not None]
        return "mAP50-95 on record: " + (" · ".join(parts) if parts else "none")


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _map50_95(record: dict, *keys: str) -> float | None:
    node: object = record
    for key in keys:
        node = node.get(key) if isinstance(node, dict) else None
    return float(node) if isinstance(node, (int, float)) else None


def recorded_scores(experiment_id: str, records_root: Path = RECORDS_ROOT) -> RecordedScores:
    """Val (best.pt, run_record.json) and final-test mAP50-95 (final_test*_record.json)."""
    folder = Path(records_root) / experiment_id
    return RecordedScores(
        val=_map50_95(_read_json(folder / "run_record.json"), "val_metrics", "map50_95"),
        test=_map50_95(_read_json(folder / "final_test_record.json"), "metrics", "ultralytics_reported", "map50_95"),
        zenodo_sk=_map50_95(_read_json(folder / "final_test_zenodo_sk_record.json"),
                            "metrics", "ultralytics_reported", "map50_95"),
    )
