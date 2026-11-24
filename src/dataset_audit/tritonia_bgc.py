"""Tritonia background-contrast preprocessing, method `bgc1` (D-045).

Each output frame is a 3-channel PNG made from one grayscale montage frame and a per-pixel background:

    channel R = the original frame, unchanged
    channel G = max(0, frame - background) * GAIN, clipped to 0..255   ("brighter than usual")
    channel B = CLAHE of the original frame (clip 1.5, 8 x 8 tiles; D-018's settings)

The background is the per-pixel median of up to 2 * N_PER_SIDE other frames of the SAME recording day,
chosen near fixed time offsets before and after the frame (MIN_OFFSET .. WINDOW frames, 15 s per frame;
see `target_offsets`). Only frames of the same day are used, and the day is the split unit, so no frame of
another split ever enters a background (CLAUDE.md rule 5). The parameters in PARAMS are fixed by D-045 and
are never tuned on val (rule 7).

Why (DATASET_TRITONIA.md, OBSERVED 2026-11-03): at a clicked point the frame is about +15 grey levels above
that pixel's median over distant frames, and the excess is mostly gone 8 frames later. The cameras are fixed,
so the median shows the static scene and channel G shows what is brighter than usual there.

Neighbour frames come from whatever frames of the day are available: on Kaggle the 75 s-spaced (or wider)
batch-2 frames for training days and the full sequences for val/test days. `pick_neighbours` takes, for each
target offset, the nearest available frame within TARGET_TOL frames, so both give neighbours near the same
offsets; the realised offsets are reported per split so the difference can be checked.

Standard library except `process_frame`, which imports numpy and OpenCV (Kaggle side, CLAUDE.md section 7).
Nothing here writes next to its inputs; `write_processed_batch` refuses an output inside an input.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

from .tritonia_dataset import (DATASET_KIND, INCLUDED, TritoniaError, _write_csv, session_of, stem_of,
                               view_fingerprint)
from .versioning import (DECISIONS_PATH, PROCESSED_ROOT, is_frozen, open_new_lineage, read_version_meta,
                         sha256_file, verify_frozen, version_dir)

METHOD = "bgc1"
PARAMS = {
    "method": METHOD,
    "min_offset_frames": 20,     # 5 min: a slug's own excess is mostly gone after 8 frames (OBSERVED)
    "window_frames": 240,        # 1 h on each side: follows slow lighting changes
    "n_per_side": 10,            # up to 20 neighbours
    "target_tol_frames": 15,     # nearest available frame within this many frames of a target offset
    "min_neighbours": 8,         # fewer -> the frame is not processed (error, never a weaker background)
    "gain": 4.0,                 # +15 grey levels -> 60
    "clahe_clip": 1.5,
    "clahe_tiles": 8,
    "channels": {"R": "original", "G": "max(0, frame - median background) * gain", "B": "CLAHE(original)"},
}
SEQUENCE_FIELDS = ("image", "t", "png_sha256")
PREPROCESS_FIELDS = ["filename", "session_id", "split", "t", "source_png_sha256", "n_neighbours",
                     "neighbour_ts", "mean_abs_offset_error", "mean_abs_diff", "bgc_png_sha256"]


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def target_offsets(params: dict = PARAMS) -> list[int]:
    """Signed frame offsets, evenly spaced from min_offset to window on each side, nearest first:
    [-20, 20, -44, 44, ...] for the D-045 parameters."""
    g, w, n = params["min_offset_frames"], params["window_frames"], params["n_per_side"]
    if not (0 < g <= w and n >= 1):
        raise ValueError(f"bad offsets: min_offset {g}, window {w}, n_per_side {n}")
    mags = [g] if n == 1 else [round(g + i * (w - g) / (n - 1)) for i in range(n)]
    return [s * m for m in mags for s in (-1, 1)]


def pick_neighbours(t: int, available: list[int], params: dict = PARAMS) -> list[tuple[int, int]]:
    """(target offset, chosen frame) pairs for frame `t` from the frame numbers `available` (same day).

    For each target offset the nearest unused available frame on the same side, at least min_offset and at
    most window frames from t and within target_tol of t + offset; ties go to the frame closer to t.
    Deterministic. Raises TritoniaError when fewer than min_neighbours are found."""
    g, w, tol = params["min_offset_frames"], params["window_frames"], params["target_tol_frames"]
    pool = sorted(set(available) - {t})
    used, picks = set(), []
    for off in target_offsets(params):
        side = 1 if off > 0 else -1
        cands = [a for a in pool if a not in used and (a - t) * side >= g and abs(a - t) <= w
                 and abs(a - (t + off)) <= tol]
        if not cands:
            continue
        best = min(cands, key=lambda a: (abs(a - (t + off)), abs(a - t), a))
        used.add(best)
        picks.append((off, best))
    if len(picks) < params["min_neighbours"]:
        raise TritoniaError(f"frame t={t}: only {len(picks)} background frames found "
                            f"(need {params['min_neighbours']}); add the day's frames as a source")
    return picks


def day_pools(batch_dir: Path, sequence_dirs: list[Path]) -> dict[str, dict[int, tuple[Path, str]]]:
    """{day: {t: (png path, sha256 from its manifest)}} from a labelling batch and full-day sequences.
    A frame present in two sources must have the same sha256 there (same TIFF page, same PNG writer)."""
    pools: dict[str, dict[int, tuple[Path, str]]] = {}

    def add(day: str, t: int, path: Path, sha: str, where: Path) -> None:
        old = pools.setdefault(day, {}).get(t)
        if old is not None and old[1] != sha:
            raise TritoniaError(f"{day} t={t}: {where} and {old[0]} disagree (sha256)")
        if old is None:
            pools[day][t] = (path, sha)

    batch_dir = Path(batch_dir)
    for r in _read_csv(batch_dir / "manifest.csv"):
        stem = stem_of(r["image"])
        add(session_of(stem), int(r["t"]), batch_dir / "images" / f"{stem}.png", r["png_sha256"], batch_dir)
    for seq in sequence_dirs:
        seq = Path(seq)
        rows = _read_csv(seq / "manifest.csv")
        if rows and not set(SEQUENCE_FIELDS) <= set(rows[0]):
            raise TritoniaError(f"{seq}/manifest.csv lacks the fields {SEQUENCE_FIELDS}")
        for r in rows:
            stem = stem_of(r["image"])
            add(session_of(stem), int(r["t"]), seq / r["image"], r["png_sha256"], seq)
    return pools


def plan(version_rows: list[dict], pools: dict, params: dict = PARAMS) -> list[dict]:
    """One plan row per frame of the version manifest: its own png and its background frames.
    Fails (listing every problem) if a frame is missing, differs from the version, or lacks neighbours."""
    rows, problems = [], []
    for r in sorted(version_rows, key=lambda r: r["filename"]):
        day, t = r["session_id"], int(r["t"])
        pool = pools.get(day, {})
        own = pool.get(t)
        if own is None:
            problems.append(f"{r['filename']}: not in any source")
            continue
        if own[1] != r["image_sha256"]:
            problems.append(f"{r['filename']}: source png differs from the version (sha256)")
            continue
        try:
            picks = pick_neighbours(t, list(pool), params)
        except TritoniaError as e:
            problems.append(f"{r['filename']}: {e}")
            continue
        rows.append({"filename": r["filename"], "session_id": day, "split": r["split"], "t": t, "path": own[0],
                     "source_png_sha256": own[1], "neighbours": [(off, a, *pool[a]) for off, a in picks]})
    if problems:
        raise TritoniaError("cannot preprocess:\n  - " + "\n  - ".join(problems[:30]))
    return rows


def process_frame(frame_png: Path, neighbour_pngs: list[Path], params: dict = PARAMS):
    """(3-channel uint8 image in OpenCV BGR order, mean |frame - background|). Kaggle side: numpy + OpenCV."""
    import cv2  # heavy imports only here (CLAUDE.md section 7)
    import numpy as np

    def gray(p: Path):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 2 or img.dtype != np.uint8:
            raise TritoniaError(f"{p}: not an 8-bit grayscale PNG")
        return img

    frame = gray(frame_png)
    stack = np.stack([gray(p) for p in neighbour_pngs])
    if stack.shape[1:] != frame.shape:
        raise TritoniaError(f"{frame_png}: neighbour frames have another size")
    background = np.median(stack, axis=0)
    diff = frame.astype(np.float64) - background
    g = np.clip(np.round(np.maximum(diff, 0.0) * params["gain"]), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=params["clahe_clip"],
                            tileGridSize=(params["clahe_tiles"], params["clahe_tiles"])).apply(frame)
    bgr = np.dstack([clahe, g, frame])  # stored as R = original, G = contrast, B = CLAHE
    return bgr, float(np.mean(np.abs(diff)))


def write_processed_batch(batch_dir: Path, version_rows: list[dict], planned: list[dict], out: Path,
                          version: str, params: dict = PARAMS, process=process_frame,
                          write_png=None, inputs: list[Path] = (), progress=None) -> dict:
    """Write a labelling-batch-shaped folder holding only the version's frames, preprocessed:
    images/, points/, labels.txt, manifest.csv (png_sha256 = processed png), batch.json (+ "preprocessing"),
    preprocess_manifest.csv and preprocess_report.json. Refuses an existing output or one inside an input."""
    batch_dir, out = Path(batch_dir).resolve(), Path(out).resolve()
    for src in (batch_dir, *[Path(p).resolve() for p in inputs]):
        if out == src or src in out.parents or out in src.parents:
            raise TritoniaError(f"output {out} overlaps the input {src}")
    if out.exists():
        raise TritoniaError(f"{out} already exists; choose a new folder")
    if write_png is None:
        def write_png(path: Path, img) -> None:
            import cv2
            if not cv2.imwrite(str(path), img):
                raise TritoniaError(f"could not write {path}")

    by_stem = {stem_of(r["image"]): r for r in _read_csv(batch_dir / "manifest.csv")}
    wanted = {Path(r["filename"]).stem for r in version_rows}
    missing = sorted(wanted - set(by_stem))
    if missing:
        raise TritoniaError(f"{len(missing)} version frames are not in the batch manifest, e.g. {missing[:3]}")
    (out / "images").mkdir(parents=True)
    (out / "points").mkdir()
    shutil.copyfile(batch_dir / "labels.txt", out / "labels.txt")

    manifest, pre = [], []
    for k, p in enumerate(planned, 1):
        stem = Path(p["filename"]).stem
        img, mad = process(p["path"], [n[2] for n in p["neighbours"]], params)
        dst = out / "images" / p["filename"]
        write_png(dst, img)
        sha = _sha_file(dst)
        shutil.copyfile(batch_dir / "points" / f"{stem}.csv", out / "points" / f"{stem}.csv")
        manifest.append({**by_stem[stem], "png_sha256": sha})
        errs = [abs(a - (p["t"] + off)) for off, a, _, _ in p["neighbours"]]
        pre.append({"filename": p["filename"], "session_id": p["session_id"], "split": p["split"], "t": p["t"],
                    "source_png_sha256": p["source_png_sha256"], "n_neighbours": len(p["neighbours"]),
                    "neighbour_ts": " ".join(str(a) for _, a, _, _ in p["neighbours"]),
                    "mean_abs_offset_error": round(sum(errs) / len(errs), 3), "mean_abs_diff": round(mad, 4),
                    "bgc_png_sha256": sha})
        if progress:
            progress(f"[{k}/{len(planned)}] {p['filename']}: {len(p['neighbours'])} neighbours")
    manifest.sort(key=lambda r: r["image"])
    for path, rows, fields in ((out / "manifest.csv", manifest, list(manifest[0])),
                               (out / "preprocess_manifest.csv", pre, PREPROCESS_FIELDS)):
        with open(path, "x", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    meta = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    report = {"method": METHOD, "params": params, "version": version, "source_batch": batch_dir.name,
              "source_batch_manifest_sha256": _sha_file(batch_dir / "manifest.csv"),
              "frames": len(pre), "per_split": split_summary(pre)}
    meta = {**meta, "preprocessing": {**report, "note": "frames limited to the version's; see preprocess_report.json"}}
    (out / "batch.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    report["preprocess_manifest_sha256"] = _sha_file(out / "preprocess_manifest.csv")
    (out / "preprocess_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def split_summary(pre_rows: list[dict]) -> dict:
    """Per split: frames, neighbours (min / mean), mean |realised - target offset| and mean |frame - background|.
    The last is the check for camera bumps or lighting jumps: compare its spread across frames."""
    out = {}
    for split in sorted({r["split"] for r in pre_rows}):
        rows = [r for r in pre_rows if r["split"] == split]
        n = [int(r["n_neighbours"]) for r in rows]
        mad = sorted(float(r["mean_abs_diff"]) for r in rows)
        out[split] = {"frames": len(rows), "neighbours_min": min(n), "neighbours_mean": round(sum(n) / len(n), 2),
                      "mean_abs_offset_error": round(sum(float(r["mean_abs_offset_error"]) for r in rows) / len(rows), 3),
                      "mean_abs_diff_median": mad[len(mad) // 2], "mean_abs_diff_max": mad[-1]}
    return out


# ---------------------------------------------------------------- the derived dataset version

VERSION_COPY_KEYS = ("class_names", "label_source", "label_source_report_sha256", "frame_batch", "split_rule")


def make_bgc_version(version: str, decision: str, parent: str, processed_dir: Path,
                     root: Path = PROCESSED_ROOT, decisions_path: Path = DECISIONS_PATH) -> Path:
    """Draft Tritonia version = frozen `parent` with every image replaced by its preprocessed copy (D-045).

    Same frames, splits, label statuses and boxes as the parent (split_manifest.csv is copied byte for byte);
    only image_sha256 changes, and only after checking that each processed frame was made from the parent's
    frame (preprocess_manifest.csv source_png_sha256 == parent image_sha256). The label source stays the
    parent's. VERSION.json records `derived_from`, the processed batch folder and the preprocessing report."""
    processed_dir = Path(processed_dir).resolve()
    pmeta = read_version_meta(parent, root)
    if not pmeta or pmeta.get("dataset_kind") != DATASET_KIND:
        raise TritoniaError(f"{parent} is not a {DATASET_KIND} dataset version")
    if pmeta.get("status") != "frozen" or not is_frozen(parent, root):
        raise TritoniaError(f"{parent} is not frozen; a derived version needs a frozen parent")
    verify_frozen(parent, root)
    pdir = version_dir(parent, root)
    parent_rows = _read_csv(pdir / "image_manifest.csv")
    pre = {r["filename"]: r for r in _read_csv(processed_dir / "preprocess_manifest.csv")}
    batch = {Path(r["image"]).name: r for r in _read_csv(processed_dir / "manifest.csv")}
    report = json.loads((processed_dir / "preprocess_report.json").read_text(encoding="utf-8"))
    if report.get("preprocess_manifest_sha256") != sha256_file(processed_dir / "preprocess_manifest.csv"):
        raise TritoniaError("preprocess_manifest.csv differs from preprocess_report.json")
    if report.get("version") != parent:
        raise TritoniaError(f"the processed batch was made for {report.get('version')}, not {parent}")

    rows, problems = [], []
    for r in parent_rows:
        p, b = pre.get(r["filename"]), batch.get(r["filename"])
        if p is None or b is None:
            problems.append(f"{r['filename']}: not in the processed batch")
        elif p["source_png_sha256"] != r["image_sha256"]:
            problems.append(f"{r['filename']}: processed from another frame (source sha256)")
        elif p["bgc_png_sha256"] != b["png_sha256"]:
            problems.append(f"{r['filename']}: preprocess_manifest and manifest disagree")
        elif (p["split"], p["session_id"]) != (r["split"], r["session_id"]):
            problems.append(f"{r['filename']}: split or day differs from {parent}")
        else:
            png = processed_dir / "images" / r["filename"]
            if not png.is_file() or sha256_file(png) != p["bgc_png_sha256"]:
                problems.append(f"{png}: missing or differs from preprocess_manifest.csv")
            else:
                rows.append({**r, "image_sha256": p["bgc_png_sha256"]})
    extra_frames = sorted(set(pre) - {r["filename"] for r in parent_rows})
    if extra_frames:
        problems.append(f"{len(extra_frames)} processed frames are not in {parent}, e.g. {extra_frames[:3]}")
    if problems:
        raise TritoniaError(f"not a preprocessed copy of {parent}:\n  - " + "\n  - ".join(problems[:20]))

    extra = {**{k: pmeta[k] for k in VERSION_COPY_KEYS if k in pmeta},
             "derived_from": parent, "frame_batch_name": processed_dir.name,
             "preprocessing": {"method": report["method"], "params": report["params"],
                               "preprocess_report_sha256": sha256_file(processed_dir / "preprocess_report.json"),
                               "preprocess_manifest_sha256": report["preprocess_manifest_sha256"]}}
    d = open_new_lineage(version, decision, DATASET_KIND, extra, root=root, decisions_path=decisions_path)
    _write_csv(d / "image_manifest.csv", rows, list(parent_rows[0]))
    shutil.copyfile(pdir / "split_manifest.csv", d / "split_manifest.csv")
    shutil.copyfile(processed_dir / "preprocess_manifest.csv", d / "preprocess_manifest.csv")
    return d


def draft_fingerprint(version: str, boxes_dir: Path, scratch: Path, root: Path = PROCESSED_ROOT) -> str:
    """The dataset fingerprint the YOLO view of `version` will have once frozen, computed from the draft.

    The fingerprint (tritonia_dataset.view_fingerprint) covers build_manifest.csv and the label files only,
    so it is written here exactly as build_view writes them, without images and without freezing anything.
    `scratch` must not exist; it is left behind for inspection."""
    vdir, boxes_dir, scratch = version_dir(version, root), Path(boxes_dir), Path(scratch)
    if scratch.exists():
        raise TritoniaError(f"{scratch} already exists")
    meta = read_version_meta(version, root) or {}
    if sha256_file(boxes_dir / "ingest_report.json") != meta.get("label_source_report_sha256"):
        raise TritoniaError(f"{boxes_dir} is not the label source {version} names")
    rows = [r for r in _read_csv(vdir / "image_manifest.csv") if r["label_status"] in INCLUDED]
    built = []
    for r in rows:
        label = (boxes_dir / "labels" / (Path(r["filename"]).stem + ".txt")).read_bytes()
        if hashlib.sha256(label).hexdigest() != r["label_sha256"]:
            raise TritoniaError(f"label for {r['filename']}: content differs from the version")
        dst = scratch / "labels" / r["split"] / (Path(r["filename"]).stem + ".txt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(label)
        built.append({k: r[k] for k in ("filename", "session_id", "split", "label_status", "n_instances",
                                        "image_sha256", "label_sha256")})
    _write_csv(scratch / "build_manifest.csv", built, list(built[0]))
    return view_fingerprint(scratch)
