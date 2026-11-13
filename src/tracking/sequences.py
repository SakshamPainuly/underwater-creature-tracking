"""Full-day Tritonia frame sequences and their MTrackJ ground-truth points (D-043 phase 1), standard library only.

A sequence is every page of a day's TIFF stack written as a PNG, plus the hand-clicked MTrackJ points of that
day as tracking ground truth. Frames are 15 s apart: sparse-frame/timelapse analysis, not video tracking.

Camera layout (OBSERVED 2026-11-13 from mean pixel intensity of 20130524 and 20130529): each frame is a 3 x 2
montage with dark seams at columns 351 and 703 and at row 240, so a camera is about 352 x 240 px. Camera ids:
0 1 2 on the top row, 3 4 5 on the bottom row. Slugs are tracked per camera (Tritonia Info.txt); a track's
camera is the camera holding most of its points (7 tracks have 1-12 points on a seam, at most 2.5 px from it).
Nothing here writes next to its inputs; build_sequence refuses an output inside the source folder.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

from dataset_audit.mtrackj import MdfFile, MtrackjError, Track
from dataset_audit.tritonia_frames import page_strips, read_page, write_png_gray

SEAM_X = (351.5, 703.5)
SEAM_Y = 240.0
GT_FIELDS = ["day", "camera", "track_id", "cluster_id", "t", "x", "y", "track_flag"]
MANIFEST_FIELDS = ["image", "day", "source_tif", "t", "page_index", "n_gt_points", "png_sha256"]


class SequenceError(ValueError):
    pass


def camera_of(x: float, y: float) -> int:
    col = 0 if x < SEAM_X[0] else 1 if x < SEAM_X[1] else 2
    return col + (0 if y < SEAM_Y else 3)


def track_camera(track: Track) -> int:
    """Camera holding most of the track's points; a tie goes to the lower camera id (deterministic)."""
    if not track.points:
        raise SequenceError(f"track {track.track_id} has no points")
    c = Counter(camera_of(p.x, p.y) for p in track.points)
    return min(c, key=lambda k: (-c[k], k))


def click_cadence(mdf: MdfFile) -> int:
    """Most common frame step between consecutive points of one track (1 = every frame, 2 = every 2nd)."""
    steps = Counter()
    for tr in mdf.tracks:
        ts = sorted(p.t for p in tr.points)
        steps.update(b - a for a, b in zip(ts, ts[1:]))
    if not steps:
        raise SequenceError("no track has two points")
    return min(steps, key=lambda k: (-steps[k], k))


def gt_rows(day: str, mdf: MdfFile) -> list[dict]:
    """One row per clicked point, sorted by frame, camera, track. Fails on two points of one track in one frame."""
    rows = []
    for tr in mdf.tracks:
        cam = track_camera(tr)
        seen = set()
        for p in tr.points:
            if p.t in seen:
                raise SequenceError(f"{day} track {tr.track_id}: two points in frame {p.t}")
            seen.add(p.t)
            rows.append({"day": day, "camera": cam, "track_id": tr.track_id, "cluster_id": tr.cluster_id, "t": p.t,
                         "x": p.x, "y": p.y, "track_flag": tr.flag})
    rows.sort(key=lambda r: (r["t"], r["camera"], r["track_id"]))
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_sequence(root: Path, days: dict[str, str], mdfs: dict[str, MdfFile], out: Path, progress=None) -> dict:
    """Write every page of each day's stack to out/images/<stack>_t####.png, plus manifest.csv and gt_points.csv.

    days maps day -> stack name (e.g. '20130524' -> '20130524all_15spf'). Returns a summary for sequence.json.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or root in out.parents:
        raise SequenceError(f"REFUSED: {out} is inside the source {root}")
    if out.exists():
        raise SequenceError(f"REFUSED: {out} already exists; choose a new folder")
    plans = {}
    for day, stack in sorted(days.items()):
        tif = root / f"{stack}.tif"
        if day not in mdfs:
            raise SequenceError(f"{day}: no MTrackJ file")
        try:
            w, h, strips = page_strips(tif)
        except (MtrackjError, OSError) as e:
            raise SequenceError(str(e)) from e
        gt = gt_rows(day, mdfs[day])
        if gt and gt[-1]["t"] > len(strips):
            raise SequenceError(f"{day}: a point at t={gt[-1]['t']} lies beyond the {len(strips)} pages")
        plans[day] = (stack, tif, w, h, strips, gt)

    (out / "images").mkdir(parents=True)
    manifest, all_gt, per_day = [], [], {}
    for day, (stack, tif, w, h, strips, gt) in plans.items():
        n_at = Counter(r["t"] for r in gt)
        for i, strip in enumerate(strips):
            t = i + 1
            name = f"{stack}_t{t:04d}.png"
            png = out / "images" / name
            write_png_gray(png, w, h, read_page(tif, strip))
            manifest.append({"image": f"images/{name}", "day": day, "source_tif": tif.name, "t": t, "page_index": i,
                             "n_gt_points": n_at.get(t, 0), "png_sha256": _sha(png)})
            if progress and (t % 500 == 0 or t == len(strips)):
                progress(f"{day}: {t}/{len(strips)} frames")
        all_gt += gt
        per_day[day] = {"source_tif": tif.name, "tif_bytes": tif.stat().st_size, "width": w, "height": h,
                        "frames": len(strips), "click_cadence": click_cadence(mdfs[day]),
                        "tracks": len(mdfs[day].tracks), "gt_points": len(gt), "frames_with_points": len(n_at),
                        "tracks_per_camera": dict(sorted(Counter(track_camera(tr) for tr in mdfs[day].tracks).items())),
                        "track_flags": dict(sorted(Counter(tr.flag for tr in mdfs[day].tracks).items()))}
    for name, fields, rows in (("manifest.csv", MANIFEST_FIELDS, manifest), ("gt_points.csv", GT_FIELDS, all_gt)):
        with open(out / name, "x", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            wr.writeheader()
            wr.writerows(rows)
    return {"days": per_day, "camera_layout": {"seam_x": list(SEAM_X), "seam_y": SEAM_Y, "ids": "0 1 2 top, 3 4 5 bottom"},
            "totals": {"frames": len(manifest), "gt_points": len(all_gt),
                       "tracks": sum(d["tracks"] for d in per_day.values())},
            "manifest_sha256": _sha(out / "manifest.csv"), "gt_points_sha256": _sha(out / "gt_points.csv")}
