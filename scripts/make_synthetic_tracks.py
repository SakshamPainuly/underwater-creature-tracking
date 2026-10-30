#!/usr/bin/env python
"""Write a SYNTHETIC tracks folder (no video) for pipeline demos and tests (D-028). Local PC, seconds.

    python scripts/make_synthetic_tracks.py --out runs/tracking_demo/synthetic_s0
    python scripts/analyze_tracks.py --tracks runs/tracking_demo/synthetic_s0

The scenario is scripted (src/tracking/synthetic.py): it is not lobster data and its analytics are
not results. --fps is only for showing the seconds columns; leave it unset otherwise.
Exit code: 0 = written, 1 = refused, 2 = usage error.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.config import RAW_ROOTS
from tracking.schema import TracksError, check_output_dir, write_tracks_dir
from tracking.synthetic import make_scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="output folder (not inside data/raw)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jitter-px", type=float, default=1.0, help="0 = exact, hand-checkable values")
    parser.add_argument("--fps", type=float, default=None, help="optional; synthetic only")
    args = parser.parse_args()

    try:
        out = check_output_dir(args.out, RAW_ROOTS)
        meta, rows = make_scenario(seed=args.seed, jitter_px=args.jitter_px, fps=args.fps)
        write_tracks_dir(out, meta, rows)
    except TracksError as e:
        print(f"REFUSED: {e}")
        sys.exit(1)
    print(f"SYNTHETIC tracks: {len(rows)} boxes, {len(meta['scenario']['tracks'])} tracks, "
          f"{meta['n_frames']} frames -> {out}")


if __name__ == "__main__":
    main()
