#!/usr/bin/env python
r"""Trajectory statistics and figures for Tritonia tracks vs the clicked tracks (D-043 phase 5). Local CPU.

    python scripts/tritonia_trajectories.py --sequence data/processed/tritonia_tracking/val \
        --frames D:\uwct_tritonia\sequences\val --tracks D:\uwct_tritonia\runs\T001\selected\tracks.csv \
        --tracks-sha256 <sha> --tracker-id T002 --out D:\uwct_tritonia\runs\T002_val_trajectories

Writes out/trajectories_clicks.csv and out/trajectories_tracker.csv (one row per track: points, duration_s,
path_px, net_px, median step speed px/s), out/summary.json (per day, matched-track comparison, inputs, git) and
out/figures/<day>.svg (+ <day>_preview.png, a raster check image without text). Statistics use each day's clicked
frames only. Px and px/s only (rule 9). Refuses a test sequence and an existing --out.
"""

import argparse
import csv
import json
import statistics as st
import subprocess
import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tracking import data as T
from tracking import trajectories as J
from tracking.detections import DetectionError, sha256_file

FIELDS = ["day", "camera", "track_id", "points", "first_t", "last_t", "duration_s", "path_px", "net_px",
          "linked_steps", "gaps", "median_step_speed_px_s"]


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def read_png_gray(path: Path) -> tuple[int, int, bytes]:
    """Decode the sequence PNGs (8-bit gray, filter 0 on every row, as write_png_gray writes them)."""
    b = path.read_bytes()
    pos, idat, w, h = 8, b"", None, None
    while pos < len(b):
        n = int.from_bytes(b[pos:pos + 4], "big")
        kind, body = b[pos + 4:pos + 8], b[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h = int.from_bytes(body[:4], "big"), int.from_bytes(body[4:8], "big")
            if body[8:10] != b"\x08\x00":
                raise DetectionError(f"{path.name}: not 8-bit grayscale")
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    if any(raw[y * (w + 1)] != 0 for y in range(h)):
        raise DetectionError(f"{path.name}: unexpected PNG row filter")
    return w, h, b"".join(raw[y * (w + 1) + 1:(y + 1) * (w + 1)] for y in range(h))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True, help="sequence folder holding images/ (for the figures)")
    ap.add_argument("--tracks", type=Path, required=True, help="tracks.csv of the tracker run")
    ap.add_argument("--tracks-sha256", required=True)
    ap.add_argument("--tracker-id", required=True)
    ap.add_argument("--match-px", type=float, default=10.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"REFUSED: {args.out} already exists; choose a new folder")
    try:
        meta = T.load_meta(args.sequence)
        if meta["split"] != "val":
            sys.exit("REFUSED: phase 5 runs on val only")
        if sha256_file(args.tracks) != args.tracks_sha256:
            sys.exit(f"FAILED: {args.tracks} is not the tracks file named by --tracks-sha256")
        with open(args.sequence / "gt_points.csv", newline="", encoding="utf-8") as f:
            gt_rows = list(csv.DictReader(f))
        with open(args.tracks, newline="", encoding="utf-8") as f:
            tr_rows = list(csv.DictReader(f))
        with open(args.sequence / "manifest.csv", newline="", encoding="utf-8") as f:
            manifest = {(r["day"], int(r["t"])): r for r in csv.DictReader(f)}
    except (DetectionError, OSError, KeyError, ValueError) as e:
        sys.exit(f"FAILED: {e}")

    (args.out / "figures").mkdir(parents=True)
    rows_out = {"clicks": [], "tracker": []}
    summary = {"tracker_id": args.tracker_id, "per_day": {}}
    for day in sorted(meta["days"]):
        info = meta["days"][day]
        cad = info["click_cadence"]
        pick = lambda rows, idk: [(int(r["camera"]), int(r[idk]), int(r["t"]), float(r["x"]), float(r["y"]))
                                  for r in rows if r["day"] == day]
        gt_all = J.group_tracks(pick(gt_rows, "track_id"))
        tr_all = J.group_tracks(pick(tr_rows, "track_id"))
        frames = {t for pts in gt_all.values() for t, _, _ in pts}
        restrict = lambda tracks: {k: [p for p in v if p[0] in frames] for k, v in tracks.items()}
        gt_e, tr_e = restrict(gt_all), restrict(tr_all)
        day_stats = {}
        for name, tracks in (("clicks", gt_e), ("tracker", tr_e)):
            stats = {}
            for (cam, tid), pts in sorted(tracks.items()):
                s = J.track_stats(pts, cad)
                if s:
                    stats[(cam, tid)] = s
                    rows_out[name].append({"day": day, "camera": cam, "track_id": tid, **s})
            day_stats[name] = stats
        matched = J.match_tracks(gt_e, tr_e, frames, args.match_px)
        good = [(g, p, hits, n) for g, (p, hits, n) in matched.items() if hits >= 0.5 * n
                and g in day_stats["clicks"] and p in day_stats["tracker"]
                and day_stats["clicks"][g]["median_step_speed_px_s"] and day_stats["tracker"][p]["median_step_speed_px_s"]]
        ratio = [day_stats["tracker"][p]["median_step_speed_px_s"] / day_stats["clicks"][g]["median_step_speed_px_s"]
                 for g, p, _, _ in good]
        summary["per_day"][day] = {
            "click_cadence": cad, "eval_frames": len(frames),
            "clicks": J.summarize(list(day_stats["clicks"].values())),
            "tracker": J.summarize(list(day_stats["tracker"].values())),
            "matched": {"clicked_tracks": len(gt_e), "with_a_tracker_track_covering_half": len(good),
                        "median_speed_ratio_tracker_over_clicks": round(st.median(ratio), 4) if ratio else None}}
        mid = sorted(frames)[len(frames) // 2]
        m = manifest[(day, mid)]
        png = args.frames / m["image"]
        if sha256_file(png) != m["png_sha256"]:
            sys.exit(f"FAILED: {png} differs from the manifest")
        w, h, px = read_png_gray(png)
        sub = (f"Tracker {args.tracker_id}: conf 0.16, gate 20 px, max gap 60, min length 10 (val-selected, T001).")
        (args.out / "figures" / f"{day}.svg").write_text(
            J.day_svg(day, w, h, px, mid, gt_all, tr_all, cad, sub), encoding="utf-8")
        (args.out / "figures" / f"{day}_preview.png").write_bytes(J.preview_png(w, h, px, gt_all, tr_all, cad))
        print(f"{day}: clicks {summary['per_day'][day]['clicks']['tracks']} tracks, tracker "
              f"{summary['per_day'][day]['tracker']['tracks']}, matched {len(good)}", flush=True)
    for name, rows in rows_out.items():
        with open(args.out / f"trajectories_{name}.csv", "x", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            wr.writeheader()
            wr.writerows(rows)
    summary.update({"decision": "D-043", "phase": 5, "split": "val", "git": git_state(),
                    "inputs": {"sequence_manifest_sha256": meta["manifest_sha256"],
                               "gt_points_sha256": meta["gt_points_sha256"], "tracks_sha256": args.tracks_sha256},
                    "units": "px, px/s (15 s per frame); no physical units; speeds not comparable between cameras",
                    "note": "statistics on clicked frames only; steps longer than 2 x cadence are gaps, not path"})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
