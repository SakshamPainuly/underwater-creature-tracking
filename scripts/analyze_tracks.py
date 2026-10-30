#!/usr/bin/env python
"""Trajectory analytics for one tracks folder (D-028). Standard library only; local PC or Colab.

    python scripts/analyze_tracks.py --tracks <folder with tracks.csv + tracks_meta.json>

Writes to <tracks>/analytics/ (or --out): analytics.json (overall + per track + boxes per frame),
track_summary.csv, trajectories.svg, occupancy.svg. Units are px and frames; seconds only when the
meta has a known fps. Descriptive statistics only: no IDF1/HOTA/MOTA (no ground-truth track IDs).
Exit code: 0 = written, 1 = refused, 2 = usage error.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.config import RAW_ROOTS
from tracking import analytics as A
from tracking.schema import TracksError, check_output_dir, load_tracks_dir
from tracking.svg import occupancy_svg, trajectories_svg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="default: <tracks>/analytics")
    parser.add_argument("--min-track-frames", type=int, default=A.DEFAULT_MIN_TRACK_FRAMES)
    parser.add_argument("--stationary-px-per-frame", type=float, default=A.DEFAULT_STATIONARY_PX_PER_FRAME)
    parser.add_argument("--grid", type=int, nargs=2, default=(16, 9), metavar=("NX", "NY"))
    args = parser.parse_args()

    try:
        meta, rows = load_tracks_dir(args.tracks)
        out = check_output_dir(args.out or args.tracks / "analytics", RAW_ROOTS)
    except TracksError as e:
        print(f"REFUSED: {e}")
        sys.exit(1)

    report = A.summarize(meta, rows, args.min_track_frames, args.stationary_px_per_frame)
    grid = A.occupancy_grid(rows, meta["frame_width"], meta["frame_height"], *args.grid)
    report["occupancy_grid"] = {"nx": args.grid[0], "ny": args.grid[1], "counts": grid}

    out.mkdir(parents=True, exist_ok=True)
    (out / "analytics.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    with open(out / "track_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=A.SUMMARY_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(report["per_track"])
    (out / "trajectories.svg").write_text(trajectories_svg(meta, rows), encoding="utf-8")
    (out / "occupancy.svg").write_text(occupancy_svg(meta, grid), encoding="utf-8")

    tag = "SYNTHETIC " if meta["synthetic"] else ""
    print(f"{tag}TRACK ANALYTICS -- {meta['video_id']} ({report['units']})")
    print(json.dumps(report["overall"], indent=1))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
