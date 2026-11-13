#!/usr/bin/env python
r"""Extract full-day Tritonia frame sequences + MTrackJ ground truth for tracking (D-043 phase 1).

    python scripts/make_tritonia_sequence.py --split val --split-version v8 \
        --out D:\uwct_tritonia\sequences\val [--root D:\uwct_tritonia\raw\tritonia]

--root defaults to UWCT_TRITONIA_ROOT and is only read. Every page of each day that --split-version puts in
--split becomes out/images/<stack>_t####.png; out/ also gets manifest.csv, gt_points.csv and sequence.json.
The test split is refused unless --final names the decision whose final test was approved (CLAUDE.md rule 7).
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.config import RAW_ROOTS
from dataset_audit.mtrackj import MtrackjError, read_mdf
from dataset_audit.versioning import version_dir
from tracking.sequences import SequenceError, build_sequence

DECISION = "D-043"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=os.environ.get("UWCT_TRITONIA_ROOT"))
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--split-version", required=True, help="frozen Tritonia version whose day split is used (v8)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--final", help="decision id of the approved final test (required for --split test)")
    args = ap.parse_args()
    if args.root is None:
        sys.exit("no source: pass --root or set UWCT_TRITONIA_ROOT")
    if args.split == "test" and args.final != DECISION:
        sys.exit(f"REFUSED: the test split needs --final {DECISION}, only after the final test is approved (rule 7)")
    out = args.out.resolve()
    for raw in RAW_ROOTS:
        raw = Path(raw).resolve()
        if out == raw or raw in out.parents:
            sys.exit(f"REFUSED: {out} is inside the raw folder {raw} (rule 1)")

    with open(version_dir(args.split_version) / "split_manifest.csv", newline="", encoding="utf-8") as f:
        days = sorted(r["session_id"] for r in csv.DictReader(f) if r["split"] == args.split)
    try:
        mdfs = {m.name[:8]: read_mdf(m) for m in sorted(args.root.glob("*.tif.mdf")) if m.name[:8] in days}
        stacks = {m.name[:8]: m.name[:-len(".tif.mdf")] for m in sorted(args.root.glob("*.tif.mdf")) if m.name[:8] in days}
        if sorted(stacks) != days:
            sys.exit(f"FAILED: MTrackJ files for {sorted(set(days) - set(stacks))} not found in {args.root}")
        summary = build_sequence(args.root, stacks, mdfs, out, progress=lambda s: print(s, flush=True))
    except (SequenceError, MtrackjError, OSError) as e:
        sys.exit(f"FAILED: {e}")
    summary = {"decision": DECISION, "split_version": args.split_version, "split": args.split, "root": str(args.root),
               "note": "sparse-frame/timelapse frames, 15 s apart; ground truth = MTrackJ clicks", **summary}
    (out / "sequence.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["totals"]), "\nwrote", out)


if __name__ == "__main__":
    main()
