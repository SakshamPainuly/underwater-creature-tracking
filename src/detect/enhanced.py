"""Raw vs enhanced images in the GUI: find the enhanced copy of an image, pair each with its model.

Harsh's enhanced images are read-only raw sources: `data/raw/enhanced_2026-09/` (the 2,105 v1
train/val/test images, D-017) and `data/raw/enhanced_zextra_2026-10/` (the 436 zExtra images,
D-024, optional). They are matched by filename: the delivered folders are never used as a split.
Uploaded images have no enhanced copy, and none is computed here (the enhancement code is not in
this repository).

Each model is shown the image type it was evaluated on (E001 and E027 raw, E004 enhanced; see
models.py), so the GUI never shows an unevaluated model/image combination without saying so.
Standard library only.
"""

from __future__ import annotations

import json
from pathlib import Path

from .predict import IMAGE_SUFFIXES, PROJECT_ROOT, DetectError

ENHANCED_SOURCE = "enhanced_2026-09"
ENHANCED_ZEXTRA_SOURCE = "enhanced_zextra_2026-10"  # same profile, zExtra only (D-024); optional
RAW_SOURCE = "biology"
# v4 (Model A, D-031): our raw originals plus derived copies. Its val and test are the raw v1
# frames, so a v4 model is evaluated on raw images.
MIXED_SOURCE = "mixed"


def enhanced_root() -> Path:
    from dataset_audit.config import IMAGE_SOURCES  # local import: keeps this module standalone

    return Path(IMAGE_SOURCES[ENHANCED_SOURCE])


def index_enhanced(root: Path | None = None) -> dict[str, Path]:
    """Filename -> path for every enhanced image under <root>/images/<folder>/. Read-only."""
    root = Path(root) if root is not None else enhanced_root()
    images = root / "images"
    if not images.is_dir():
        raise DetectError(
            f"Enhanced images not found at {images}\n"
            "They are Harsh's delivery (D-017): unpack it to data/raw/enhanced_2026-09/ or set "
            "UWCT_ENHANCED_ROOT to its location."
        )
    index: dict[str, Path] = {}
    for path in sorted(images.glob("*/*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.name in index:
            raise DetectError(f"Two enhanced images share the name {path.name}: {index[path.name]} and {path}")
        index[path.name] = path
    return index


def index_all_enhanced(roots: dict[str, Path] | None = None) -> dict[str, Path]:
    """Filename -> enhanced image across every enhanced delivery.

    The v1 delivery (enhanced_2026-09) is required; the zExtra delivery (D-024) is used when present
    and skipped when absent, so a machine without it still works for the labelled splits.
    """
    if roots is None:
        from dataset_audit.config import IMAGE_SOURCES

        roots = {s: Path(IMAGE_SOURCES[s]) for s in (ENHANCED_SOURCE, ENHANCED_ZEXTRA_SOURCE) if s in IMAGE_SOURCES}
    index: dict[str, Path] = {}
    for source, root in roots.items():
        if source != ENHANCED_SOURCE and not (Path(root) / "images").is_dir():
            continue  # optional delivery not on this machine
        for name, path in index_enhanced(root).items():
            if name in index:
                raise DetectError(f"{name} is in two enhanced deliveries: {index[name]} and {path}")
            index[name] = path
    return index


def checkpoint_image_source(experiment_id: str | None, records_root: Path | None = None) -> str | None:
    """Image type a recorded run is evaluated on ('biology' or 'enhanced_2026-09'); None if unknown.

    A v4 run (image_source 'mixed') counts as raw: its val and test splits are the raw v1 frames.
    """
    if experiment_id is None:
        return None
    path = Path(records_root or PROJECT_ROOT / "experiments" / "records") / experiment_id / "run_record.json"
    try:
        dataset = json.loads(path.read_text(encoding="utf-8")).get("dataset") or {}
    except (OSError, ValueError):
        return None
    source = dataset.get("image_source") or RAW_SOURCE  # v1 records predate the field: raw images
    return RAW_SOURCE if source == MIXED_SOURCE else source
