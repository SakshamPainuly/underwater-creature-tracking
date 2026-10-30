"""Tracks file format (docs/PIPELINE.md, D-028): <out>/tracks.csv + <out>/tracks_meta.json.

tracks.csv has one row per (frame, track) box, sorted by (frame_idx, track_id). Coordinates are
pixels in the original frame, (x1, y1) = top-left, x2 > x1, y2 > y1. frame_idx is 0-based over
every decoded frame of the source, including frames with no box. Written by tracking.run (GPU side)
and tracking.synthetic; read by tracking.analytics.

tracks_meta.json describes where the rows came from. `fps` is null unless the frame rate is known;
analytics reports seconds only when it is a number (CLAUDE.md rule 9).
"""

import csv
import json
import math
from pathlib import Path

TRACKS_FILE = "tracks.csv"
META_FILE = "tracks_meta.json"
META_SCHEMA = "uwct.tracks_meta/1"
COLUMNS = ("video_id", "frame_idx", "track_id", "x1", "y1", "x2", "y2", "conf", "class_id")
META_REQUIRED = ("schema", "video_id", "synthetic", "frame_width", "frame_height", "n_frames", "fps",
                 "detector", "tracker")
BOUNDS_TOLERANCE_PX = 1.0  # boxes may touch the frame edge after rounding


class TracksError(ValueError):
    pass


def check_output_dir(path: Path, raw_roots) -> Path:
    """Outputs must never be written inside read-only raw data (CLAUDE.md rule 1)."""
    p = Path(path).resolve()
    roots = (raw_roots,) if isinstance(raw_roots, (str, Path)) else tuple(raw_roots)
    for root in roots:
        r = Path(root).resolve()
        if p == r or r in p.parents:
            raise TracksError(f"{p} is inside the read-only raw data ({r}); choose another output folder")
    return p


def _number(row: dict, key: str, where: str, integer: bool):
    value = row.get(key)
    try:
        v = int(value) if integer else float(value)
    except (TypeError, ValueError):
        raise TracksError(f"{where}: {key}={value!r} is not {'an integer' if integer else 'a number'}")
    if not integer and not math.isfinite(v):
        raise TracksError(f"{where}: {key}={value!r} is not finite")
    return v


def parse_row(row: dict, where: str) -> dict:
    """One CSV row (strings) -> typed row. Raises TracksError with the row's location."""
    out = {"video_id": (row.get("video_id") or "").strip()}
    if not out["video_id"]:
        raise TracksError(f"{where}: empty video_id")
    for key in ("frame_idx", "track_id", "class_id"):
        out[key] = _number(row, key, where, integer=True)
    for key in ("x1", "y1", "x2", "y2", "conf"):
        out[key] = _number(row, key, where, integer=False)
    return out


def validate_rows(rows: list, meta: dict | None = None) -> None:
    """Check typed rows (and their agreement with meta, when given). Raises on the first problem."""
    seen = set()
    video_ids = {r["video_id"] for r in rows}
    if len(video_ids) > 1:
        raise TracksError(f"one tracks file holds one video; found {sorted(video_ids)}")
    if meta is not None and video_ids and video_ids != {meta["video_id"]}:
        raise TracksError(f"rows are for {sorted(video_ids)} but meta video_id is {meta['video_id']!r}")
    for i, r in enumerate(rows):
        where = f"row {i + 1}"
        for key in ("frame_idx", "track_id", "class_id"):
            if r[key] < 0:
                raise TracksError(f"{where}: {key} is negative")
        if not (r["x2"] > r["x1"] and r["y2"] > r["y1"]):
            raise TracksError(f"{where}: box needs x2 > x1 and y2 > y1")
        if not 0.0 <= r["conf"] <= 1.0:
            raise TracksError(f"{where}: conf {r['conf']} outside [0, 1]")
        key = (r["frame_idx"], r["track_id"])
        if key in seen:
            raise TracksError(f"{where}: track {r['track_id']} appears twice in frame {r['frame_idx']}")
        seen.add(key)
        if meta is not None:
            if r["frame_idx"] >= meta["n_frames"]:
                raise TracksError(f"{where}: frame_idx {r['frame_idx']} >= n_frames {meta['n_frames']}")
            t = BOUNDS_TOLERANCE_PX
            if (r["x1"] < -t or r["y1"] < -t or r["x2"] > meta["frame_width"] + t
                    or r["y2"] > meta["frame_height"] + t):
                raise TracksError(f"{where}: box outside the {meta['frame_width']}x{meta['frame_height']} frame")


def validate_meta(meta: dict) -> None:
    missing = [k for k in META_REQUIRED if k not in meta]
    if missing:
        raise TracksError(f"tracks meta is missing {missing}")
    if meta["schema"] != META_SCHEMA:
        raise TracksError(f"unknown tracks meta schema {meta['schema']!r} (expected {META_SCHEMA})")
    for key in ("frame_width", "frame_height", "n_frames"):
        if not isinstance(meta[key], int) or meta[key] <= 0:
            raise TracksError(f"tracks meta {key} must be a positive integer, got {meta[key]!r}")
    fps = meta["fps"]
    if fps is not None and (isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0):
        raise TracksError(f"tracks meta fps must be null or a positive number, got {fps!r}")


def sort_rows(rows: list) -> list:
    return sorted(rows, key=lambda r: (r["frame_idx"], r["track_id"]))


def write_tracks(path: Path, rows: list) -> None:
    rows = sort_rows(rows)
    validate_rows(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r["video_id"], r["frame_idx"], r["track_id"],
                        *(f"{r[k]:.2f}" for k in ("x1", "y1", "x2", "y2")), f"{r['conf']:.4f}", r["class_id"]])


def read_tracks(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise TracksError(f"{path}: header {reader.fieldnames} != {list(COLUMNS)}")
        rows = [parse_row(row, f"{path.name} line {i + 2}") for i, row in enumerate(reader)]
    validate_rows(rows)
    return rows


def write_meta(path: Path, meta: dict) -> None:
    validate_meta(meta)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_meta(path: Path) -> dict:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise TracksError(f"{path}: not valid JSON ({e})")
    validate_meta(meta)
    return meta


def write_tracks_dir(out_dir: Path, meta: dict, rows: list) -> None:
    validate_meta(meta)
    validate_rows(rows, meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tracks(out_dir / TRACKS_FILE, rows)
    write_meta(out_dir / META_FILE, meta)


def load_tracks_dir(tracks_dir: Path) -> tuple:
    """Return (meta, rows) from a folder holding tracks.csv and tracks_meta.json."""
    tracks_dir = Path(tracks_dir)
    for name in (TRACKS_FILE, META_FILE):
        if not (tracks_dir / name).is_file():
            raise TracksError(f"{tracks_dir / name} not found")
    meta = read_meta(tracks_dir / META_FILE)
    rows = read_tracks(tracks_dir / TRACKS_FILE)
    validate_rows(rows, meta)
    return meta, rows
