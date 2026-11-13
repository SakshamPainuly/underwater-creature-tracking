"""Read a Tritonia sequence's ground truth and per-frame detections; run and score tracking per camera (D-043).

Evaluation frames of a day = frames with at least one MTrackJ click in any camera (each day's own cadence, D-043).
A ground-truth point belongs to its track's camera (gt_points.csv); a detection to the camera its centre lies in.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import metrics as Mx
from .detections import DetectionError, sha256_file
from .sequences import camera_of
from .tracker import Detection, TrackerParams, no_link, track_camera

CAMERAS = range(6)


def load_meta(seq_dir: Path) -> dict:
    seq_dir = Path(seq_dir)
    meta = json.loads((seq_dir / "sequence.json").read_text(encoding="utf-8"))
    for name, key in (("manifest.csv", "manifest_sha256"), ("gt_points.csv", "gt_points_sha256")):
        if sha256_file(seq_dir / name) != meta[key]:
            raise DetectionError(f"{seq_dir / name}: sha256 differs from sequence.json")
    return meta


def read_gt(seq_dir: Path) -> dict:
    """{day: {camera: {t: [(track_id, x, y)]}}}"""
    out: dict = {}
    with open(Path(seq_dir) / "gt_points.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["day"], {}).setdefault(int(r["camera"]), {}).setdefault(int(r["t"]), []).append(
                (int(r["track_id"]), float(r["x"]), float(r["y"])))
    return out


def eval_frames(gt_day: dict) -> list[int]:
    return sorted({t for cam in gt_day.values() for t in cam})


def read_detections(seq_dir: Path, pred_dir: Path, meta: dict, min_conf: float = 0.0) -> dict:
    """{day: {camera: [Detection]}} from YOLO txt files (`cls x y w h conf`, normalised); centre = point.

    Lines below min_conf are skipped while reading (the tracker would drop them anyway).
    """
    out: dict = {d: {c: [] for c in CAMERAS} for d in meta["days"]}
    with open(Path(seq_dir) / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        day, t = r["day"], int(r["t"])
        w, h = meta["days"][day]["width"], meta["days"][day]["height"]
        p = Path(pred_dir) / f"{Path(r['image']).stem}.txt"
        if not p.is_file():
            raise DetectionError(f"missing prediction file {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 6:
                raise DetectionError(f"{p.name}: expected 'cls x y w h conf', got {line!r}")
            conf = float(parts[5])
            if conf < min_conf:
                continue
            x, y = float(parts[1]) * w, float(parts[2]) * h
            out[day][camera_of(x, y)].append(Detection(t, x, y, conf))
    return out


def gt_as_detections(gt: dict) -> dict:
    """Tracker-ceiling reference: the clicks themselves as detections (conf 1)."""
    return {day: {c: [Detection(t, x, y, 1.0) for t, pts in cams.get(c, {}).items() for _, x, y in pts]
                  for c in CAMERAS} for day, cams in gt.items()}


def run(dets: dict, params: TrackerParams, link: bool = True) -> dict:
    """{day: {camera: [track]}} with tracks as lists of Detection."""
    return {day: {c: (track_camera(ds, params) if link else no_link(ds, params.conf)) for c, ds in cams.items()}
            for day, cams in dets.items()}


def as_pred(tracks_day: dict) -> tuple[dict, list[dict]]:
    """Number tracks per day (1..n, camera by camera) -> ({camera: {t: [(id, x, y)]}}, csv rows)."""
    pred, rows, k = {}, [], 0
    for c in sorted(tracks_day):
        cam = pred.setdefault(c, {})
        for tr in tracks_day[c]:
            k += 1
            for d in tr:
                cam.setdefault(d.t, []).append((k, d.x, d.y))
                rows.append({"camera": c, "track_id": k, "t": d.t, "x": round(d.x, 2), "y": round(d.y, 2),
                             "conf": round(d.conf, 5)})
    return pred, rows


def score_all(gt: dict, tracks: dict, match_px: float = 10.0) -> tuple[dict, list[dict]]:
    """Scores per camera, per day (cameras combined), mean over days and all days combined; plus track rows."""
    per_day, all_counts, rows = {}, [], []
    for day in sorted(gt):
        frames = eval_frames(gt[day])
        pred, day_rows = as_pred(tracks.get(day, {}))
        rows += [{"day": day, **r} for r in day_rows]
        cams, counts = {}, []
        for c in CAMERAS:
            cnt = Mx.evaluate(frames, gt[day].get(c, {}), pred.get(c, {}), match_px)
            counts.append(cnt)
            cams[str(c)] = Mx.scores(cnt)
        day_counts = Mx.combine(counts)
        all_counts.append(day_counts)
        per_day[day] = {"eval_frames": len(frames), "cameras": cams, "all_cameras": Mx.scores(day_counts)}
    keys = ("HOTA", "DetA", "AssA", "LocA", "IDF1", "MOTA")
    mean = {k: sum(v["all_cameras"][k] for v in per_day.values()) / len(per_day) for k in keys}
    return {"per_day": per_day, "mean_over_days": mean, "all_days": Mx.scores(Mx.combine(all_counts))}, rows
