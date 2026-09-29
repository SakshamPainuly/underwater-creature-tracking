"""Build an Ultralytics-YOLO dataset view from the raw data + frozen manifests.

Why a derived tree is required (verified against ultralytics 8.4.155 source,
ultralytics/data/utils.py `img2label_paths` and dataset.py `get_labels`):
  * Ultralytics finds a label by replacing the last "/images/" in the image path
    with "/labels/". The raw Set 2000 labels live in a different folder from
    their images, so pointing Ultralytics at raw paths would silently treat
    every Set 2000 positive as background.
  * Ultralytics writes a "<labels dir>.cache" file next to the label folder;
    pointing it at raw folders would write into data/raw/biology/.

Layout produced:
    <out>/images/{train,val,test}/<name>.png   link (or copy) to the raw PNG
    <out>/labels/{train,val,test}/<name>.txt   positive: copy of raw label
                                               confirmed negative: empty file
    <out>/data.yaml, <out>/build_manifest.csv, <out>/build_summary.json

Image sources (D-017): labels always come from the Biology sources; images come from the
version's image source. "biology" (v1) links the raw images, and its build_manifest.csv is
unchanged, so v1 fingerprints stay valid. Any other source (e.g. enhanced images) is found by
filename under its own read-only root; each image must be a PNG with the same width and height
as its raw original, and its sha256 is recorded in build_manifest.csv, so the fingerprint
identifies the image content, not only the labels.

zExtra ("negative_pending") images are never included. Linked images share their bytes with
the raw files, so they must never be written to; non-PNG sources are only built with copies
(Ultralytics 8.4.155 re-saves corrupt JPEGs in place, ultralytics/data/utils.py check_image).
"""

import csv
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from dataset_audit.config import BIOLOGY_ROOT, DATASET_SOURCES, DEFAULT_IMAGE_SOURCE, IMAGE_SOURCES, PROJECT_ROOT
from dataset_audit.versioning import sha256_file

DATASET_BASENAME = "lobster"  # built dataset name = f"{DATASET_BASENAME}_{version}"
CLASS_NAMES = ["lobster"]  # index = YOLO class id
SPLITS = ("train", "val", "test")
INCLUDED_STATUSES = ("positive", "negative_confirmed")
TARGET_ULTRALYTICS_VERSION = "8.4.155"
MARKER_FILE = ".uwct_generated"
PNG_SIGNATURE = bytes([137, 80, 78, 71, 13, 10, 26, 10])


class BuildError(RuntimeError):
    pass


def _read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _count_label_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _link_image(src: Path, dst: Path, mode: str) -> None:
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise BuildError(f"unknown link mode {mode!r}")


def _resolve_link_mode(requested: str, probe_src: Path, probe_dir: Path) -> str:
    """'auto' -> symlink if the OS allows it, else hardlink. Never silently copies."""
    if requested != "auto":
        return requested
    for mode in ("symlink", "hardlink"):
        probe = probe_dir / f".link_probe_{mode}"
        try:
            _link_image(probe_src, probe, mode)
            probe.unlink()
            return mode
        except OSError:
            continue
    raise BuildError(
        "Neither symlinks nor hardlinks are available (e.g. raw data on another drive). "
        "Re-run with --link-mode copy to duplicate the images instead."
    )


def png_size(path: Path) -> tuple:
    """(width, height) from a PNG header; BuildError if the file is not a PNG."""
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != PNG_SIGNATURE or head[12:16] != b"IHDR":
        raise BuildError(f"{path}: not a PNG file")
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def index_image_source(root: Path) -> dict:
    """filename -> path for every PNG under an image source root. Duplicate names are an error."""
    if not root.is_dir():
        raise BuildError(f"image source root not found: {root}")
    index, dupes = {}, []
    for p in sorted(root.rglob("*.png")):
        if p.name in index:
            dupes.append(p.name)
        index[p.name] = p
    if dupes:
        raise BuildError(f"{len(dupes)} filenames occur more than once under {root} (e.g. {dupes[0]})")
    return index


def _prepare_output_dir(out_dir: Path) -> None:
    for root in (BIOLOGY_ROOT, *IMAGE_SOURCES.values()):
        if _is_within(out_dir, root):
            raise BuildError(f"Refusing to write inside the raw data folder: {out_dir}")
    if out_dir.exists():
        if not (out_dir / MARKER_FILE).is_file():
            raise BuildError(
                f"{out_dir} exists but was not created by this script (no {MARKER_FILE}); "
                "refusing to delete it."
            )
        shutil.rmtree(out_dir)  # removes links only; link targets (raw files) are untouched
    out_dir.mkdir(parents=True)
    (out_dir / MARKER_FILE).write_text("generated by scripts/prepare_yolo_dataset.py\n")


def _git_state() -> dict:
    def git(*args):
        try:
            r = subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def dataset_fingerprint(out_dir: Path) -> str:
    """sha256 over build_manifest.csv + every label file (relative path + bytes), sorted.

    Machine-independent: excludes data.yaml (absolute path) and build_summary.json.
    """
    labels = sorted(p.relative_to(out_dir).as_posix() for p in (out_dir / "labels").glob("*/*.txt"))
    h = hashlib.sha256()
    for rel in ["build_manifest.csv", *labels]:
        h.update(rel.encode())
        h.update((out_dir / rel).read_bytes())
    return h.hexdigest()


def _write_data_yaml(out_dir: Path) -> Path:
    root = out_dir.resolve().as_posix().replace("'", "''")
    lines = [
        "# Generated by scripts/prepare_yolo_dataset.py -- regenerate, do not hand-edit.",
        f"path: '{root}'",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)],
        "",
    ]
    path = out_dir / "data.yaml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_yolo_dataset(image_manifest: Path, split_manifest: Path, out_dir: Path, link_mode: str,
                       version: str, version_status: str, image_source: str = DEFAULT_IMAGE_SOURCE) -> dict:
    if image_source not in IMAGE_SOURCES:
        raise BuildError(f"unknown image source {image_source!r}; known: {sorted(IMAGE_SOURCES)}")
    image_rows = _read_csv(image_manifest)
    session_split = {r["session_id"]: r["split"] for r in _read_csv(split_manifest)}
    source_dirs = {name: (Path(img), Path(lbl) if lbl else None) for name, img, lbl in DATASET_SOURCES}

    # --- validate manifests before touching the filesystem ---
    for r in image_rows:
        frozen = session_split.get(r["session_id"])
        if frozen != r["split"]:
            raise BuildError(
                f"image_manifest.csv disagrees with split_manifest.csv for {r['filename']} "
                f"({r['split']} vs {frozen}); rebuild image_manifest.csv first"
            )
        if r["label_status"] == "invalid":
            raise BuildError(f"invalid label in manifest: {r['filename']} ({r['issues']})")

    selected = [r for r in image_rows if r["split"] in SPLITS and r["label_status"] in INCLUDED_STATUSES]
    leaked = [r["filename"] for r in selected if r["source_set"] == "zExtra"]
    if leaked:
        raise BuildError(f"zExtra images selected for training data: {leaked[:5]}")

    non_png = [r["filename"] for r in selected if Path(r["filename"]).suffix.lower() != ".png"]
    if non_png and link_mode != "copy":
        raise BuildError(
            f"{len(non_png)} non-PNG images (e.g. {non_png[0]}). Ultralytics rewrites corrupt JPEGs in "
            "place, which through a hardlink/symlink would modify the raw file. Re-run with --link-mode copy."
        )

    source_root = None
    if image_source != DEFAULT_IMAGE_SOURCE:
        source_root = Path(IMAGE_SOURCES[image_source])
        source_index = index_image_source(source_root)
        missing = [r["filename"] for r in selected if r["filename"] not in source_index]
        if missing:
            raise BuildError(f"{len(missing)} images missing from image source {image_source} "
                             f"({source_root}), e.g. {missing[0]}")

    _prepare_output_dir(out_dir)
    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)

    first_src = (source_index[selected[0]["filename"]] if source_root else
                 source_dirs[selected[0]["source_set"]][0] / selected[0]["filename"])
    mode = _resolve_link_mode(link_mode, first_src, out_dir / "images" / "train")

    counts = {s: Counter() for s in SPLITS}
    build_rows = []
    for r in selected:
        image_dir, label_dir = source_dirs[r["source_set"]]
        raw_img = image_dir / r["filename"]
        src_img = source_index[r["filename"]] if source_root else raw_img
        stem = Path(r["filename"]).stem
        src_lbl = label_dir / f"{stem}.txt" if label_dir else None
        dst_img = out_dir / "images" / r["split"] / r["filename"]
        dst_lbl = out_dir / "labels" / r["split"] / f"{stem}.txt"

        if not src_img.is_file():
            raise BuildError(f"raw image missing: {src_img}")
        if source_root and png_size(src_img) != png_size(raw_img):
            raise BuildError(f"{src_img}: size {png_size(src_img)} differs from the raw image "
                             f"{png_size(raw_img)}; the raw labels would not fit it")
        _link_image(src_img, dst_img, mode)

        expected = int(r["n_instances"])
        if r["label_status"] == "positive":
            n = _count_label_rows(src_lbl)
            if n != expected or n == 0:
                raise BuildError(f"{src_lbl}: {n} boxes on disk, manifest says {expected}")
            shutil.copyfile(src_lbl, dst_lbl)
        else:  # negative_confirmed
            if src_lbl is not None and src_lbl.exists():
                raise BuildError(f"{r['filename']} marked negative but a raw label exists: {src_lbl}")
            dst_lbl.write_text("", encoding="utf-8")

        c = counts[r["split"]]
        c["images"] += 1
        c["positive" if r["label_status"] == "positive" else "negative"] += 1
        c["instances"] += expected
        row = {
            "split": r["split"],
            "filename": r["filename"],
            "session_id": r["session_id"],
            "label_status": r["label_status"],
            "n_instances": expected,
            "raw_image": src_img.resolve().relative_to((source_root or BIOLOGY_ROOT).resolve()).as_posix(),
        }
        if source_root:  # v1 rows stay exactly as before (its fingerprint must not change)
            row.update(image_source=image_source, image_sha256=sha256_file(src_img))
        build_rows.append(row)

    with open(out_dir / "build_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(build_rows[0].keys()))
        writer.writeheader()
        writer.writerows(build_rows)

    selected_names = {r["filename"] for r in selected}
    excluded = Counter(r["label_status"] for r in image_rows if r["filename"] not in selected_names)
    summary = {
        "dataset_name": f"{DATASET_BASENAME}_{version}",
        "dataset_version": version,
        "version_status": version_status,
        "dataset_fingerprint": dataset_fingerprint(out_dir),
        "image_source": image_source,
        "inputs": {
            "image_manifest_sha256": sha256_file(image_manifest),
            "split_manifest_sha256": sha256_file(split_manifest),
        },
        "git": _git_state(),
        "class_names": {i: n for i, n in enumerate(CLASS_NAMES)},
        "target_ultralytics_version": TARGET_ULTRALYTICS_VERSION,
        "link_mode": mode,
        "splits": {s: dict(counts[s]) for s in SPLITS},
        "not_included": dict(excluded),
    }
    with open(out_dir / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary["data_yaml"] = str(_write_data_yaml(out_dir))
    return summary
