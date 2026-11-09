"""Tritonia sea slug dataset (D-029, D-030): box ingest, day split, version manifests, YOLO view.

Stages, each writing files the next one reads (standard library only):
  1. ingest_boxes      makesense.ai YOLO ZIP + notes.txt + labelling-batch manifest
                       -> new read-only label source (labels/<stem>.txt byte for byte, the ZIP,
                          notes.txt, ingest_report.json). Refuses on any problem.
  2. make_version      label source + batch manifest -> data/processed/<version>/ draft:
                       image_manifest.csv, split_manifest.csv, VERSION.json (dataset_kind tritonia).
  3. build_view        frozen version + frames + label source -> data/yolo/tritonia_<version>/
                       (images copied and hash-checked, labels, data.yaml, build_manifest.csv,
                       build_summary.json), same layout and fingerprint as the lobster views.
  4. verify_view       what training checks before it reads a view (training/experiment.py).

Label meaning in a batch (D-029): every visible slug in a labelled frame is boxed, so a label file
with no lines is a confirmed negative for that frame. A frame marked `exclude` in notes.txt was not
fully labelled and never reaches a dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from .versioning import (PROCESSED_ROOT, VersionError, is_frozen, open_new_lineage, read_version_meta,
                         sha256_file, verify_frozen, version_dir)

DATASET_KIND = "tritonia"
CLASS_NAMES = ["lobster", "sea_slug"]  # ids 0, 1 (D-029); lobster id kept so ids never change
TILE_COLS, TILE_ROWS = 3, 2  # 6-camera montage (DATASET_TRITONIA.md)
BORDER_TOLERANCE_PX = 3.0
SPLITS = ("train", "val", "test")
TRAIN_SPLITS = ("train", "val")
INCLUDED = ("positive", "negative_confirmed")
MARKER_FILE = ".uwct_generated"
NOTE_KINDS = ("exclude", "note")


class TritoniaError(RuntimeError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def stem_of(image_path: str) -> str:
    return PurePosixPath(image_path).stem


def session_of(stem: str) -> str:
    """Recording day = session (CLAUDE.md rule 5): '20130519all_15spf_t0123' -> '20130519'."""
    return stem[:8]


# ---------------------------------------------------------------- labels and notes

def parse_yolo_lines(text: str, where: str) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for n, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise TritoniaError(f"{where} line {n}: expected 'class x y w h', got {line!r}")
        try:
            cls = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            raise TritoniaError(f"{where} line {n}: not numbers: {line!r}") from None
        boxes.append((cls, x, y, w, h))
    return boxes


def box_problems(boxes, width: int, height: int, tol_px: float = BORDER_TOLERANCE_PX) -> list[str]:
    """Class ids, frame bounds, positive size, and no box across a camera-tile border."""
    xs = [width * i / TILE_COLS for i in range(1, TILE_COLS)]
    ys = [height * i / TILE_ROWS for i in range(1, TILE_ROWS)]
    out = []
    for i, (cls, x, y, w, h) in enumerate(boxes, 1):
        if cls not in range(len(CLASS_NAMES)):
            out.append(f"box {i}: class {cls} is not one of {dict(enumerate(CLASS_NAMES))}")
        if w <= 0 or h <= 0:
            out.append(f"box {i}: width/height must be > 0")
            continue
        x1, x2 = (x - w / 2) * width, (x + w / 2) * width
        y1, y2 = (y - h / 2) * height, (y + h / 2) * height
        if x1 < -tol_px or y1 < -tol_px or x2 > width + tol_px or y2 > height + tol_px:
            out.append(f"box {i}: outside the {width}x{height} frame")
        for bx in xs:
            if x1 < bx - tol_px and x2 > bx + tol_px:
                out.append(f"box {i}: crosses the camera border at x={bx:g}")
        for by in ys:
            if y1 < by - tol_px and y2 > by + tol_px:
                out.append(f"box {i}: crosses the camera border at y={by:g}")
    return out


def parse_notes(text: str, known_stems: set[str]) -> dict[str, list[dict]]:
    """notes.txt: one line per issue, '<frame> | exclude|note | <reason>'; '#' lines are comments."""
    notes: dict[str, list[dict]] = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3 or parts[1] not in NOTE_KINDS or not parts[2]:
            raise TritoniaError(f"notes.txt line {n}: expected '<frame> | exclude|note | <reason>', got {line!r}")
        stem = stem_of(parts[0]) if parts[0].lower().endswith(".png") else parts[0]
        if stem not in known_stems:
            raise TritoniaError(f"notes.txt line {n}: frame {parts[0]!r} is not in the batch manifest")
        notes.setdefault(stem, []).append({"kind": parts[1], "reason": parts[2]})
    return notes


# ---------------------------------------------------------------- stage 1: ingest

def _batch(batch_dir: Path) -> tuple[list[dict], dict, int, int]:
    rows = _read_csv(batch_dir / "manifest.csv")
    meta = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    sizes = {(s["width"], s["height"]) for s in meta["source_stacks"].values()}
    if len(sizes) != 1:
        raise TritoniaError(f"batch stacks have several frame sizes: {sorted(sizes)}")
    if meta.get("classes") != CLASS_NAMES:
        raise TritoniaError(f"batch class list {meta.get('classes')} != {CLASS_NAMES}")
    (w, h), = sizes
    return rows, meta, w, h


def ingest_boxes(zip_path: Path, notes_path: Path, batch_dir: Path, out_dir: Path,
                 extra_files: tuple[Path, ...] = ()) -> dict:
    """Check a YOLO label export against the batch and store it as a new read-only source.

    extra_files (e.g. the automatic labels' suggestions.csv and report, D-033) are copied in
    unchanged as extra/<name>, with their sha256 in the report.
    """
    zip_path, notes_path, batch_dir, out_dir = (Path(p).resolve() for p in (zip_path, notes_path, batch_dir, out_dir))
    extra_files = tuple(Path(p).resolve() for p in extra_files)
    if out_dir.exists():
        raise TritoniaError(f"{out_dir} already exists; a label source is written once")
    names = [p.name for p in extra_files]
    if len(set(names)) != len(names) or any(not p.is_file() for p in extra_files):
        raise TritoniaError(f"extra files must exist and have distinct names: {names}")
    rows, meta, width, height = _batch(batch_dir)
    stems = {stem_of(r["image"]) for r in rows}
    notes = parse_notes(notes_path.read_text(encoding="utf-8-sig"), stems)  # Notepad may add a BOM
    excluded = {s for s, ns in notes.items() if any(n["kind"] == "exclude" for n in ns)}

    problems, labels = [], {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename)
            if "__MACOSX" in name.parts or name.name.startswith("._") or name.name == ".DS_Store":
                continue
            if name.suffix != ".txt":
                problems.append(f"{info.filename}: not a .txt label file")
                continue
            if name.name in ("labels.txt", "classes.txt") and name.stem not in stems:
                if z.read(info).decode("utf-8", "replace").split() != CLASS_NAMES:
                    problems.append(f"{info.filename}: class list is not {CLASS_NAMES}")
                continue
            if name.stem not in stems:
                problems.append(f"{info.filename}: no such frame in the batch manifest")
            elif name.stem in labels:
                problems.append(f"{info.filename}: second label file for {name.stem}")
            else:
                labels[name.stem] = z.read(info)

    counts = {"frames": len(stems), "label_files": len(labels), "excluded": len(excluded),
              "boxes": 0, "boxes_per_class": {c: 0 for c in CLASS_NAMES}, "empty_label_files": 0}
    for stem in sorted(stems):
        if stem not in labels:
            if stem not in excluded:
                problems.append(f"{stem}: no label file in the ZIP (label it, or mark it '| exclude |' in notes.txt)")
            continue
        try:
            text = labels[stem].decode("utf-8")
            boxes = parse_yolo_lines(text, stem)
        except (UnicodeDecodeError, TritoniaError) as e:
            problems.append(f"{stem}: {e}")
            continue
        if stem in excluded:
            continue  # stored as delivered, never used; its boxes are not checked or counted
        problems += [f"{stem}: {p}" for p in box_problems(boxes, width, height)]
        counts["boxes"] += len(boxes)
        counts["empty_label_files"] += not boxes
        for cls, *_ in boxes:
            if 0 <= cls < len(CLASS_NAMES):
                counts["boxes_per_class"][CLASS_NAMES[cls]] += 1
    if problems:
        raise TritoniaError(f"{len(problems)} problem(s) in {zip_path.name}:\n  - " + "\n  - ".join(problems[:40]))

    (out_dir / "labels").mkdir(parents=True)
    for stem, data in sorted(labels.items()):
        (out_dir / "labels" / f"{stem}.txt").write_bytes(data)
    shutil.copyfile(zip_path, out_dir / "export.zip")
    shutil.copyfile(notes_path, out_dir / "notes.txt")
    if extra_files:
        (out_dir / "extra").mkdir()
        for p in extra_files:
            shutil.copyfile(p, out_dir / "extra" / p.name)
    report = {
        "dataset_kind": DATASET_KIND,
        "class_names": CLASS_NAMES,
        "zip": {"name": zip_path.name, "sha256": sha256_file(zip_path)},
        "notes_sha256": sha256_file(notes_path),
        "extra_files": {p.name: sha256_file(p) for p in extra_files},
        "batch": {"manifest_sha256": sha256_file(batch_dir / "manifest.csv"),
                  "seed": meta["seed"], "n": meta["n"], "min_gap": meta["min_gap"]},
        "frame_size": [width, height],
        "border_tolerance_px": BORDER_TOLERANCE_PX,
        "counts": counts,
        "excluded": {s: notes[s] for s in sorted(excluded)},
        "notes": {s: ns for s, ns in sorted(notes.items()) if s not in excluded},
        "labels": {s: _sha(d) for s, d in sorted(labels.items())},
    }
    (out_dir / "ingest_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------- stage 2: version

def split_days(days: list[str], seed: int, n_val: int, n_test: int) -> dict[str, str]:
    """Shuffle the sorted days with `seed`; the first n_val go to val, the next n_test to test."""
    order = sorted(days)
    if n_val + n_test >= len(order):
        raise TritoniaError(f"{len(order)} days cannot give {n_val} val + {n_test} test and leave train")
    random.Random(seed).shuffle(order)
    return {d: ("val" if i < n_val else "test" if i < n_val + n_test else "train") for i, d in enumerate(order)}


def make_version(version: str, decision: str, boxes_dir: Path, batch_dir: Path, seed: int,
                 n_val: int, n_test: int, root: Path = PROCESSED_ROOT, decisions_path: Path | None = None) -> Path:
    boxes_dir, batch_dir = Path(boxes_dir).resolve(), Path(batch_dir).resolve()
    report = json.loads((boxes_dir / "ingest_report.json").read_text(encoding="utf-8"))
    rows, meta, _, _ = _batch(batch_dir)
    if report["batch"]["manifest_sha256"] != sha256_file(batch_dir / "manifest.csv"):
        raise TritoniaError("the label source was ingested against a different batch manifest")
    split = split_days(sorted({session_of(stem_of(r["image"])) for r in rows}), seed, n_val, n_test)

    images = []
    for r in sorted(rows, key=lambda r: r["image"]):
        stem = stem_of(r["image"])
        label = boxes_dir / "labels" / f"{stem}.txt"
        if stem in report["excluded"]:
            status, n, n_lobster, label_sha = "excluded", 0, 0, ""
        else:
            data = label.read_bytes()
            if _sha(data) != report["labels"][stem]:
                raise TritoniaError(f"{label}: content differs from ingest_report.json")
            boxes = parse_yolo_lines(data.decode("utf-8"), stem)
            n, n_lobster, label_sha = len(boxes), sum(b[0] == 0 for b in boxes), _sha(data)
            status = "positive" if boxes else "negative_confirmed"
        images.append({"filename": f"{stem}.png", "session_id": session_of(stem), "source_tif": r["source_tif"],
                       "t": r["t"], "page_index": r["page_index"], "split": split[session_of(stem)],
                       "label_status": status, "n_instances": n, "n_lobster": n_lobster,
                       "image_sha256": r["png_sha256"], "label_sha256": label_sha})
    sessions = []
    for day in sorted(split):
        mine = [i for i in images if i["session_id"] == day]
        sessions.append({"session_id": day, "split": split[day], "n_images": len(mine),
                         "n_included": sum(i["label_status"] in INCLUDED for i in mine),
                         "n_instances": sum(i["n_instances"] for i in mine)})

    extra = {"class_names": CLASS_NAMES, "label_source": "tritonia_boxes_batch1",
             "label_source_report_sha256": sha256_file(boxes_dir / "ingest_report.json"),
             "frame_batch": report["batch"], "split_rule": {"unit": "recording day", "seed": seed,
                                                            "n_val": n_val, "n_test": n_test}}
    kwargs = {"root": root} if decisions_path is None else {"root": root, "decisions_path": decisions_path}
    d = open_new_lineage(version, decision, DATASET_KIND, extra, **kwargs)
    _write_csv(d / "image_manifest.csv", images, list(images[0]))
    _write_csv(d / "split_manifest.csv", sessions, list(sessions[0]))
    return d


# ---------------------------------------------------------------- stage 3: build view

def view_dir_for(version: str, project_root: Path) -> Path:
    return Path(project_root) / "data" / "yolo" / f"{DATASET_KIND}_{version}"


def _tritonia_meta(version: str, root: Path) -> dict:
    meta = read_version_meta(version, root)
    if not meta or meta.get("dataset_kind") != DATASET_KIND:
        raise TritoniaError(f"{version} is not a {DATASET_KIND} dataset version")
    return meta


def build_view(version: str, frames_dir: Path, boxes_dir: Path, out_dir: Path, ultralytics_version: str,
               root: Path = PROCESSED_ROOT) -> dict:
    meta = _tritonia_meta(version, root)
    if meta["status"] != "frozen" or not is_frozen(version, root):
        raise TritoniaError(f"{version} is not frozen; training views are built from frozen versions only")
    verify_frozen(version, root)
    vdir = version_dir(version, root)
    frames_dir, boxes_dir, out_dir = Path(frames_dir).resolve(), Path(boxes_dir).resolve(), Path(out_dir).resolve()
    for src in (frames_dir, boxes_dir):
        if out_dir == src or src in out_dir.parents or out_dir in src.parents:
            raise TritoniaError(f"output {out_dir} overlaps the input {src}")
    if sha256_file(boxes_dir / "ingest_report.json") != meta["label_source_report_sha256"]:
        raise TritoniaError(f"{boxes_dir} is not the label source {version} was made from")
    if out_dir.exists():
        if not (out_dir / MARKER_FILE).is_file():
            raise TritoniaError(f"{out_dir} exists and was not generated by this builder; refusing to replace it")
        shutil.rmtree(out_dir)

    rows = [r for r in _read_csv(vdir / "image_manifest.csv") if r["label_status"] in INCLUDED]
    for s in SPLITS:
        (out_dir / "images" / s).mkdir(parents=True)
        (out_dir / "labels" / s).mkdir(parents=True)
    (out_dir / MARKER_FILE).write_text("generated by scripts/prepare_tritonia_yolo.py\n", encoding="utf-8")
    built, counts = [], {s: {"images": 0, "positive": 0, "negative": 0, "instances": 0} for s in SPLITS}
    for r in rows:
        img = frames_dir / r["filename"]
        if not img.is_file() or sha256_file(img) != r["image_sha256"]:
            raise TritoniaError(f"{img}: missing or not the frame the version records (sha256)")
        label = (boxes_dir / "labels" / (Path(r["filename"]).stem + ".txt")).read_bytes()
        if _sha(label) != r["label_sha256"]:
            raise TritoniaError(f"label for {r['filename']}: content differs from the version")
        shutil.copyfile(img, out_dir / "images" / r["split"] / r["filename"])
        (out_dir / "labels" / r["split"] / (Path(r["filename"]).stem + ".txt")).write_bytes(label)
        c = counts[r["split"]]
        c["images"] += 1
        c["positive" if r["label_status"] == "positive" else "negative"] += 1
        c["instances"] += int(r["n_instances"])
        built.append({k: r[k] for k in ("filename", "session_id", "split", "label_status", "n_instances",
                                        "image_sha256", "label_sha256")})
    _write_csv(out_dir / "build_manifest.csv", built, list(built[0]))
    yaml_root = out_dir.as_posix().replace("'", "''")
    (out_dir / "data.yaml").write_text("\n".join([
        "# Generated by scripts/prepare_tritonia_yolo.py -- regenerate, do not hand-edit.",
        f"path: '{yaml_root}'", "train: images/train", "val: images/val", "test: images/test", "names:",
        *[f"  {i}: {n}" for i, n in enumerate(meta["class_names"])], ""]), encoding="utf-8")
    summary = {
        "dataset_kind": DATASET_KIND, "dataset_version": version, "version_status": "frozen",
        "inputs": {"image_manifest_sha256": sha256_file(vdir / "image_manifest.csv"),
                   "split_manifest_sha256": sha256_file(vdir / "split_manifest.csv"),
                   "label_source_report_sha256": meta["label_source_report_sha256"]},
        "dataset_fingerprint": view_fingerprint(out_dir),
        "class_names": {str(i): n for i, n in enumerate(meta["class_names"])},
        "target_ultralytics_version": ultralytics_version,
        "image_source": "tritonia_label_batch1",
        "splits": counts,
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def view_fingerprint(out_dir: Path) -> str:
    """Same rule as the lobster views (yolo_prep.build.dataset_fingerprint): build_manifest.csv +
    every label file, by relative path and bytes, sorted. Machine-independent."""
    out_dir = Path(out_dir)
    labels = sorted(p.relative_to(out_dir).as_posix() for p in (out_dir / "labels").glob("*/*.txt"))
    h = hashlib.sha256()
    for rel in ["build_manifest.csv", *labels]:
        h.update(rel.encode())
        h.update((out_dir / rel).read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- stage 4: verify for training

def verify_view(dataset_dir: Path, cfg: dict, root: Path = PROCESSED_ROOT) -> dict:
    """Read-only check of a built view against its frozen version and the experiment config."""
    d = Path(dataset_dir)
    summary_path = d / "build_summary.json"
    version = cfg["dataset_version"]
    if not summary_path.is_file():
        raise TritoniaError(f"{summary_path} not found; run scripts/prepare_tritonia_yolo.py --version {version}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = _tritonia_meta(version, root)
    try:
        verify_frozen(version, root)
    except VersionError as e:
        raise TritoniaError(str(e)) from None
    vdir = version_dir(version, root)
    problems = []
    if summary.get("dataset_kind") != DATASET_KIND or summary.get("dataset_version") != version:
        problems.append(f"view is {summary.get('dataset_kind')} {summary.get('dataset_version')}, config says {version}")
    for key, path in (("image_manifest_sha256", vdir / "image_manifest.csv"),
                      ("split_manifest_sha256", vdir / "split_manifest.csv")):
        if summary.get("inputs", {}).get(key) != sha256_file(path):
            problems.append(f"{key}: view was built from different manifests; rebuild it")
    fingerprint = view_fingerprint(d)
    if fingerprint != summary.get("dataset_fingerprint"):
        problems.append("labels or build_manifest.csv changed since the build (fingerprint mismatch)")
    if fingerprint != cfg["dataset_fingerprint"]:
        problems.append(f"dataset fingerprint {fingerprint} != config dataset_fingerprint {cfg['dataset_fingerprint']}")
    if summary.get("class_names") != {str(i): n for i, n in enumerate(meta["class_names"])}:
        problems.append(f"class names {summary.get('class_names')} != {meta['class_names']}")
    if summary.get("target_ultralytics_version") != cfg["ultralytics_version"]:
        problems.append("view was built for a different Ultralytics version")

    manifest = {r["filename"]: r for r in _read_csv(vdir / "image_manifest.csv")}
    built = _read_csv(d / "build_manifest.csv")
    counts = {s: {"images": 0, "positive": 0, "negative": 0, "instances": 0} for s in SPLITS}
    for r in built:
        m = manifest.get(r["filename"])
        if m is None or m["label_status"] not in INCLUDED or (r["split"], r["label_status"]) != (m["split"], m["label_status"]):
            problems.append(f"{r['filename']}: not an included frame of {version} as built")
            continue
        img = d / "images" / r["split"] / r["filename"]
        if not img.is_file() or sha256_file(img) != m["image_sha256"]:
            problems.append(f"{r['filename']}: image content differs from the version (sha256)")
        c = counts[r["split"]]
        c["images"] += 1
        c["positive" if m["label_status"] == "positive" else "negative"] += 1
        c["instances"] += int(m["n_instances"])
    for split in SPLITS:
        if {k: summary["splits"].get(split, {}).get(k, 0) for k in counts[split]} != counts[split]:
            problems.append(f"{split}: build_summary counts differ from build_manifest.csv")
        names = {r["filename"] for r in built if r["split"] == split}
        on_disk = {p.name for p in (d / "images" / split).iterdir()} if (d / "images" / split).is_dir() else set()
        if on_disk != names:
            problems.append(f"images/{split}: {len(on_disk ^ names)} files differ from build_manifest.csv")
        if {p.name for p in (d / "labels" / split).glob("*.txt")} != {Path(n).stem + ".txt" for n in names}:
            problems.append(f"labels/{split}: label files differ from build_manifest.csv")
    if problems:
        raise TritoniaError(f"dataset view {d} failed verification:\n  - " + "\n  - ".join(problems[:20]))
    return {
        "version": version, "dataset_kind": DATASET_KIND, "status": "frozen", "frozen_registry_verified": True,
        "fingerprint": fingerprint, "image_source": summary["image_source"],
        "image_manifest_sha256": summary["inputs"]["image_manifest_sha256"],
        "split_manifest_sha256": summary["inputs"]["split_manifest_sha256"],
        "dataset_dir": str(d.resolve()), "splits_used_for_training": list(TRAIN_SPLITS),
        "counts": {s: counts[s] for s in TRAIN_SPLITS}, "class_names": list(meta["class_names"]),
    }
