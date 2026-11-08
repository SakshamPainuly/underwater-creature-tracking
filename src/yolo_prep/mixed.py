"""YOLO view of a mixed-manifest dataset version (v4, Model A: D-031, D-032).

Input: the version's mixed_manifest.csv (schema: dataset_audit/model_a.py). Its `path` and
`label_path` columns start with a raw folder name ("biology/...", "harsh_model_a_2026-11/..."),
resolved through the read-only roots in dataset_audit/config.py (BIOLOGY_ROOT, MODEL_A_ROOTS; env
vars UWCT_RAW_ROOT, UWCT_HARSH_MODEL_A_ROOT, UWCT_HARSH_V1_EXTRA_ROOT, UWCT_HARSH_V1_COLOUR_ROOT).

Layout produced:
    <out>/images/<split>/<origin>__<kind>__<file name>   copy of the raw image
    <out>/labels/<split>/<same stem>.txt                 copy of the raw label, or an empty file
    <out>/data.yaml                    train / val / test (v1 test sessions)
    <out>/data_test_zenodo_sk.yaml     the held-out Zenodo `sk` site as its `test` split
    <out>/build_manifest.csv, <out>/build_summary.json

Splits: manifest "train" and "val" as they are; manifest "test" is divided into "test" (our v1 test
sessions) and "test_zenodo_sk" (the external held-out site, D-031 point 6), so a v1 test score never
mixes the two. Rows marked "excluded" are never built. Val and both test splits must hold
unmodified originals only (D-031 point 4).

Images are always copied, never linked: Zenodo and Harsh's copies are JPGs, which Ultralytics
8.4.155 may re-save in place (CLAUDE.md rule 2), and the derived names would collide with the
originals' (a diffusion copy keeps its source's file name), hence the origin__kind__ prefix.
The sha256 of every image is recorded in build_manifest.csv, so the dataset fingerprint covers the
image bytes as well as the labels.
"""

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from dataset_audit.config import BIOLOGY_ROOT, MODEL_A_ROOTS, RAW_ROOTS
from dataset_audit.versioning import PROCESSED_ROOT, VersionError, sha256_file, verify_frozen, version_dir
from yolo_prep.build import (CLASS_NAMES, MARKER_FILE, TARGET_ULTRALYTICS_VERSION, BuildError, _git_state,
                             _is_within, dataset_fingerprint)

DATASET_KIND = "lobster_mixed"  # VERSION.json "dataset_kind" of v4
MANIFEST_FILE = "mixed_manifest.csv"
TRAIN_SPLITS = ("train", "val")
BUILT_SPLITS = ("train", "val", "test")  # in data.yaml
EVAL_ONLY_SPLITS = {"test_zenodo_sk": "data_test_zenodo_sk.yaml"}  # own data file, never trained on
VIEW_SPLITS = (*BUILT_SPLITS, *EVAL_ONLY_SPLITS)
COUNT_FIELDS = ("images", "positive", "negative", "instances")
BUILD_FIELDS = ["split", "filename", "image_id", "origin", "kind", "group_id", "label_status",
                "n_instances", "source_path", "image_sha256"]


def _read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def default_raw_roots() -> dict:
    """Raw folder name (first part of a manifest path) -> its read-only root on this machine."""
    return {"biology": Path(BIOLOGY_ROOT), **{k: Path(v) for k, v in MODEL_A_ROOTS.items()}}


def view_split(row: dict) -> str | None:
    """The view split of a manifest row; None for an excluded row."""
    split = row["split"]
    if split == "excluded":
        return None
    if split == "test":
        return "test_zenodo_sk" if row["origin"] == "zenodo" else "test"
    if split in TRAIN_SPLITS:
        return split
    raise BuildError(f"{row['image_id']}: unknown split {split!r}")


def view_filename(row: dict) -> str:
    return f"{row['origin']}__{row['kind']}__{Path(row['path']).name}"


def resolve_raw(rel: str, roots: dict) -> Path:
    head, _, rest = rel.partition("/")
    if head not in roots or not rest:
        raise BuildError(f"{rel}: path does not start with a known raw folder ({sorted(roots)})")
    return roots[head] / rest


def check_label(path: Path, n_boxes: int, where: str) -> int:
    """Count and check the YOLO rows of a label file: class 0, 4 coordinates in [0, 1]."""
    n = 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            coords = []
        if len(parts) != 5 or parts[0] != "0" or not all(0.0 <= v <= 1.0 for v in coords):
            raise BuildError(f"{where}: line {i} is not a class-0 YOLO box: {line.strip()!r}")
        n += 1
    if n != n_boxes:
        raise BuildError(f"{where}: {n} boxes on disk, manifest says {n_boxes}")
    return n


def select_rows(rows: list) -> list:
    """(view split, row) for every built row, validated against D-031 point 4."""
    selected, names = [], set()
    for r in rows:
        split = view_split(r)
        if split is None:
            continue
        if split != "train" and r["kind"] != "original":
            raise BuildError(f"{r['image_id']}: {split} may hold unmodified originals only, not {r['kind']}")
        if split in ("val", "test") and r["origin"] != "ours":
            raise BuildError(f"{r['image_id']}: {split} holds UWCT v1 frames only")
        if r["exclude_reason"]:
            raise BuildError(f"{r['image_id']}: split {r['split']} but exclude_reason {r['exclude_reason']!r}")
        status, n = r["label_status"], int(r["n_boxes"])
        if (status == "positive") != (n > 0) or status not in ("positive", "negative", "negative_confirmed"):
            raise BuildError(f"{r['image_id']}: label_status {status!r} with {n} boxes")
        name = view_filename(r)
        if name in names:
            raise BuildError(f"two manifest rows map to the same view file name {name}")
        names.add(name)
        selected.append((split, r))
    if not selected:
        raise BuildError("no rows to build")
    return selected


def _prepare_output_dir(out_dir: Path, roots: dict) -> None:
    for root in (*RAW_ROOTS, *roots.values()):
        if _is_within(out_dir, Path(root)):
            raise BuildError(f"Refusing to write inside the read-only raw folder {root}: {out_dir}")
    if out_dir.exists():
        if not (out_dir / MARKER_FILE).is_file():
            raise BuildError(f"{out_dir} exists but was not created by this script (no {MARKER_FILE}); "
                             "refusing to delete it.")
        shutil.rmtree(out_dir)  # holds copies only; the raw files are untouched
    out_dir.mkdir(parents=True)
    (out_dir / MARKER_FILE).write_text("generated by scripts/prepare_yolo_dataset.py\n")


def _data_yaml(out_dir: Path, name: str, test_dir: str, note: str) -> None:
    root = out_dir.resolve().as_posix().replace("'", "''")
    lines = [f"# Generated by scripts/prepare_yolo_dataset.py -- regenerate, do not hand-edit. {note}",
             f"path: '{root}'", "train: images/train", "val: images/val", f"test: {test_dir}", "names:",
             *[f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)], ""]
    (out_dir / name).write_text("\n".join(lines), encoding="utf-8")


def build_mixed_dataset(manifest: Path, out_dir: Path, version: str, version_status: str,
                        roots: dict | None = None) -> dict:
    """Build the view; returns build_summary.json's content. Raw files are only read."""
    roots = default_raw_roots() if roots is None else roots
    rows = _read_csv(manifest)
    selected = select_rows(rows)
    for split, r in selected:  # every source exists and every label is right before writing anything
        if not resolve_raw(r["path"], roots).is_file():
            raise BuildError(f"raw image missing: {resolve_raw(r['path'], roots)}")
        if r["label_path"]:
            check_label(resolve_raw(r["label_path"], roots), int(r["n_boxes"]), r["image_id"])
        elif r["label_status"] != "negative_confirmed":
            raise BuildError(f"{r['image_id']}: no label file, but only negative_confirmed frames may lack one")

    out_dir = Path(out_dir)
    _prepare_output_dir(out_dir, roots)
    for split in VIEW_SPLITS:
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)

    counts = {s: Counter({k: 0 for k in COUNT_FIELDS}) for s in VIEW_SPLITS}
    build_rows = []
    for split, r in sorted(selected, key=lambda t: (VIEW_SPLITS.index(t[0]), view_filename(t[1]))):
        name = view_filename(r)
        data = resolve_raw(r["path"], roots).read_bytes()
        (out_dir / "images" / split / name).write_bytes(data)
        label = out_dir / "labels" / split / f"{Path(name).stem}.txt"
        if r["label_status"] == "positive":
            shutil.copyfile(resolve_raw(r["label_path"], roots), label)
        else:
            label.write_text("", encoding="utf-8")
        n = int(r["n_boxes"])
        c = counts[split]
        c["images"] += 1
        c["positive" if n else "negative"] += 1
        c["instances"] += n
        build_rows.append({"split": split, "filename": name, "image_id": r["image_id"], "origin": r["origin"],
                           "kind": r["kind"], "group_id": r["group_id"], "label_status": r["label_status"],
                           "n_instances": n, "source_path": r["path"],
                           "image_sha256": hashlib.sha256(data).hexdigest()})

    with open(out_dir / "build_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BUILD_FIELDS)
        w.writeheader()
        w.writerows(build_rows)
    _data_yaml(out_dir, "data.yaml", "images/test", "test = UWCT v1 test sessions.")
    _data_yaml(out_dir, EVAL_ONLY_SPLITS["test_zenodo_sk"], "images/test_zenodo_sk",
               "test = held-out Zenodo sk site; final evaluation only.")

    by_kind = Counter(f"{s}:{r['origin']}/{r['kind']}" for s, r in selected)
    summary = {
        "dataset_name": f"{DATASET_KIND}_{version}",
        "dataset_kind": DATASET_KIND,
        "dataset_version": version,
        "version_status": version_status,
        "dataset_fingerprint": dataset_fingerprint(out_dir),
        "image_source": "mixed",
        "inputs": {"mixed_manifest_sha256": sha256_file(manifest)},
        "git": _git_state(),
        "class_names": {i: n for i, n in enumerate(CLASS_NAMES)},
        "target_ultralytics_version": TARGET_ULTRALYTICS_VERSION,
        "link_mode": "copy",
        "splits": {s: dict(counts[s]) for s in BUILT_SPLITS},
        "eval_only_splits": {s: {"data_file": f, **counts[s]} for s, f in EVAL_ONLY_SPLITS.items()},
        "images_by_split_and_kind": dict(sorted(by_kind.items())),
        "not_included": dict(sorted(Counter(r["exclude_reason"] for r in rows if r["split"] == "excluded").items())),
    }
    with open(out_dir / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def verify_view(dataset_dir: Path, cfg: dict, root: Path = PROCESSED_ROOT) -> dict:
    """Read-only check of a built v4 view against its frozen version and the experiment config.

    Raises VersionError (frozen registry) or BuildError (anything else) with every problem found.
    """
    d = Path(dataset_dir)
    version = cfg["dataset_version"]
    summary_path = d / "build_summary.json"
    if not summary_path.is_file():
        raise BuildError(f"{summary_path} not found; run scripts/prepare_yolo_dataset.py --version {version}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    verify_frozen(version, root)
    manifest_path = version_dir(version, root) / MANIFEST_FILE
    problems = []
    if (summary.get("dataset_kind"), summary.get("dataset_version")) != (DATASET_KIND, version):
        problems.append(f"view is {summary.get('dataset_kind')} {summary.get('dataset_version')}, config says {version}")
    if summary.get("version_status") != "frozen":
        problems.append(f"built from a {summary.get('version_status')!r} version; training uses frozen versions only")
    if summary.get("inputs", {}).get("mixed_manifest_sha256") != sha256_file(manifest_path):
        problems.append("view was built from a different mixed_manifest.csv; rebuild it")
    fingerprint = dataset_fingerprint(d)
    if fingerprint != summary.get("dataset_fingerprint"):
        problems.append("labels or build_manifest.csv changed since the build (fingerprint mismatch)")
    if fingerprint != cfg["dataset_fingerprint"]:
        problems.append(f"dataset fingerprint {fingerprint} != config dataset_fingerprint {cfg['dataset_fingerprint']}")
    if summary.get("class_names") != {str(i): n for i, n in enumerate(CLASS_NAMES)}:
        problems.append(f"class names {summary.get('class_names')} != {CLASS_NAMES}")
    if summary.get("target_ultralytics_version") != cfg["ultralytics_version"]:
        problems.append("view was built for a different Ultralytics version")

    expected = {view_filename(r): (s, r) for s, r in select_rows(_read_csv(manifest_path))}
    built = _read_csv(d / "build_manifest.csv")
    counts = {s: {k: 0 for k in COUNT_FIELDS} for s in VIEW_SPLITS}
    for b in built:
        split, r = expected.get(b["filename"], (None, None))
        if r is None or b["split"] != split or b["image_id"] != r["image_id"] \
                or b["label_status"] != r["label_status"] or int(b["n_instances"]) != int(r["n_boxes"]):
            problems.append(f"{b['filename']}: not a row of {version} as built (split/label differ)")
            continue
        img = d / "images" / split / b["filename"]
        if not img.is_file() or sha256_file(img) != b["image_sha256"]:
            problems.append(f"{b['filename']}: image content differs from the build (sha256)")
        c, n = counts[split], int(r["n_boxes"])
        c["images"] += 1
        c["positive" if n else "negative"] += 1
        c["instances"] += n
    if len(built) != len(expected):
        problems.append(f"build_manifest.csv has {len(built)} rows, {version} has {len(expected)} to build")
    recorded = {**summary.get("splits", {}), **summary.get("eval_only_splits", {})}
    for split in VIEW_SPLITS:
        if {k: recorded.get(split, {}).get(k) for k in COUNT_FIELDS} != counts[split]:
            problems.append(f"{split}: build_summary counts differ from build_manifest.csv")
        names = {b["filename"] for b in built if b["split"] == split}
        img_dir, lbl_dir = d / "images" / split, d / "labels" / split
        on_disk = {p.name for p in img_dir.iterdir()} if img_dir.is_dir() else set()
        if on_disk != names:
            problems.append(f"images/{split}: {len(on_disk ^ names)} files differ from build_manifest.csv")
        if {p.name for p in lbl_dir.glob("*.txt")} != {Path(n).stem + ".txt" for n in names}:
            problems.append(f"labels/{split}: label files differ from build_manifest.csv")
    if problems:
        raise BuildError(f"dataset view {d} failed verification:\n  - " + "\n  - ".join(problems[:20]))
    return {
        "version": version, "dataset_kind": DATASET_KIND, "status": "frozen", "frozen_registry_verified": True,
        "fingerprint": fingerprint, "image_source": "mixed",
        "mixed_manifest_sha256": summary["inputs"]["mixed_manifest_sha256"],
        "dataset_dir": str(d.resolve()), "splits_used_for_training": list(TRAIN_SPLITS),
        "counts": {s: counts[s] for s in TRAIN_SPLITS},
    }
