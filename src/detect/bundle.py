"""Assemble a portable copy of the GUI for a machine without this repository's data.

The bundle is a self-contained folder: the GUI code (scripts/gui_app.py, src/detect/), the
portable species file, the lobster checkpoints with the records that hold their validated
settings, a few sample images per source, the handoff documents in docs/gui_bundle/ and the
lobster-only report pack (docs/reports/lobster_only/, built with make_final_report.py --lobster-only).
The bundle carries lobster material only: teammates send their own species separately, and the
export stops if any copied text file still names another species' work (OTHER_SPECIES_WORDS).
It runs with `streamlit run scripts/gui_app.py` from the bundle folder on Windows, macOS or Linux.

Rules kept (CLAUDE.md): the raw dataset and the built splits are only read; nothing is written
inside them, and the output folder may not lie inside a raw root. The sample images are a small
demo subset for looking at, not a dataset version: no split is rebuilt from them.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .predict import (
    PROJECT_ROOT,
    DetectError,
    dataset_for_split,
    label_path_for,
    list_source_images,
    list_split_images,
    refuse_raw_output,
)
from .species import load_species

BUNDLE_SPECIES = Path("configs") / "gui_species_bundle.json"
HANDOFF_DOCS = Path("docs") / "gui_bundle"  # copied to the bundle root
CODE_FILES = ("scripts/gui_app.py", "scripts/gui_check.py")
RECORD_FILES = ("run_record.json", "val_eval_record.json", "final_test_record.json",
                "final_test_zenodo_sk_record.json")
LOBSTER_PACK = Path("docs") / "reports" / "lobster_only"
REFERENCE_DOCS = ("final_report_data.md", "final_report_experiments.csv")  # from LOBSTER_PACK
DATA_CARD = Path("docs") / "DATASET.md"  # copied without the lines that point to other species' data
OTHER_SPECIES_WORDS = ("tritonia", "hermissenda")
TEXT_SUFFIXES = {".md", ".txt", ".json", ".csv", ".py"}
SAMPLE_SPLITS = ("val", "test", "test_zenodo_sk")  # labelled samples; zExtra is added unlabelled
SAMPLES_PER_SOURCE = 10
MANIFEST = "MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evenly_spaced(items: list, n: int) -> list:
    """`n` items spread over the whole sorted list (first and last included). Deterministic."""
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[0]]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _git_state(root: Path) -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                                  check=False).stdout.strip()
        except OSError:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def export_bundle(out_dir: Path, root: Path = PROJECT_ROOT, samples: int = SAMPLES_PER_SOURCE) -> dict:
    """Write the bundle to `out_dir` (must be empty or absent) and return its manifest."""
    out_dir = refuse_raw_output(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise DetectError(f"{out_dir} is not empty. Choose a new folder: nothing is overwritten.")
    species = load_species(root / BUNDLE_SPECIES)

    # 1. code, species file, handoff documents
    for rel in CODE_FILES:
        _copy(root / rel, out_dir / rel)
    for path in sorted((root / "src" / "detect").glob("*.py")):
        if path.name != "bundle.py":  # the exporter itself needs the repository
            _copy(path, out_dir / "src" / "detect" / path.name)
    _copy(root / BUNDLE_SPECIES, out_dir / "configs" / "gui_species.json")
    handoff = root / HANDOFF_DOCS
    if not handoff.is_dir():
        raise DetectError(f"Handoff documents not found: {handoff}")
    for path in sorted(p for p in handoff.rglob("*") if p.is_file()):
        _copy(path, out_dir / path.relative_to(handoff))
    pack = root / LOBSTER_PACK
    for name in REFERENCE_DOCS:
        if not (pack / name).is_file():
            raise DetectError(f"{pack / name} not found. Build the lobster-only pack first: .venv/Scripts/python "
                              f"scripts/make_final_report.py --lobster-only --out {LOBSTER_PACK.as_posix()}")
        _copy(pack / name, out_dir / "docs" / name)
    card = (root / DATA_CARD).read_text(encoding="utf-8").splitlines(keepends=True)
    (out_dir / "docs" / DATA_CARD.name).write_text(
        "".join(line for line in card if not any(w in line.lower() for w in OTHER_SPECIES_WORDS)), encoding="utf-8")
    figures = pack / "figures"
    for path in sorted(figures.glob("final_*.png")):
        _copy(path, out_dir / "docs" / "figures" / path.name)

    # 2. checkpoints that exist here (checked against their run record) and the records of every model
    missing_weights = []
    for s in species:
        for m in s.models:
            record_dir = root / "experiments" / "records" / m.experiment_id
            for name in RECORD_FILES:
                if (record_dir / name).is_file():
                    _copy(record_dir / name, out_dir / "experiments" / "records" / m.experiment_id / name)
            weights = root / m.weights
            if not weights.is_file():
                missing_weights.append(f"{m.experiment_id}: {m.weights.as_posix()}")
                (out_dir / m.weights).parent.mkdir(parents=True, exist_ok=True)
                continue
            run_record = record_dir / "run_record.json"
            if run_record.is_file():
                expected = json.loads(run_record.read_text(encoding="utf-8"))["outputs"]["best"]["sha256"]
                if sha256(weights) != expected:
                    raise DetectError(f"{weights} is not {m.experiment_id}'s registered best.pt "
                                      f"(sha256 differs from {run_record}).")
            _copy(weights, out_dir / m.weights)

    # 3. lobster sample images (read-only copies) with their label files
    sample_root = out_dir / "samples" / "lobster"
    for split in SAMPLE_SPLITS:
        for image in evenly_spaced(list_split_images(split, root / dataset_for_split(split)), samples):
            _copy(image, sample_root / "images" / split / image.name)
            label = label_path_for(image)
            if label.is_file():
                _copy(label, sample_root / "labels" / split / label.name)
    for image in evenly_spaced(list_source_images("zExtra"), samples):
        _copy(image, sample_root / "zextra" / image.name)
    for s in species:  # every configured folder exists, so the GUI explains an empty one instead of a missing one
        for src in s.sources:
            (out_dir / src.name).mkdir(parents=True, exist_ok=True)

    # 4. lobster material only: no copied text may name another species' work
    named = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
                   and any(w in p.read_text(encoding="utf-8", errors="replace").lower() for w in OTHER_SPECIES_WORDS))
    if named:
        raise DetectError("The bundle is lobster-only, but these copied files name another species: "
                          + ", ".join(named) + f". The folder {out_dir} is incomplete: delete it, fix the source "
                          "files, export again.")

    # 5. manifest: every file with its size and sha256
    files = [{"path": p.relative_to(out_dir).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(out_dir.rglob("*")) if p.is_file()]
    manifest = {
        "schema": "uwct.gui_bundle/1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_git": _git_state(root),
        "weights_not_included": missing_weights,
        "samples_per_source": samples,
        "files": files,
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest
