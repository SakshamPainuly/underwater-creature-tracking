#!/usr/bin/env python
"""Create the Tritonia sea slug dataset version as a draft (D-030); local PC, standard library.

    python scripts/make_tritonia_version.py --version v3 --decision D-030 \\
        --boxes D:\\uwct_raw\\tritonia_boxes_batch1 --batch D:\\uwct_work\\tritonia_label_batch1 \\
        --seed 0 --val-days 3 --test-days 3

Writes data/processed/<version>/ (image_manifest.csv, split_manifest.csv, VERSION.json with
dataset_kind 'tritonia' and no parent). The split unit is the recording day: every frame of a day
is in one split (CLAUDE.md rule 5). Freezing is a separate step that needs explicit user approval:
    python scripts/freeze_dataset_version.py v3
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.tritonia_dataset import TritoniaError, make_version
from dataset_audit.versioning import VersionError


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--boxes", type=Path, required=True, help="label source from ingest_tritonia_boxes.py")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--val-days", type=int, required=True)
    parser.add_argument("--test-days", type=int, required=True)
    args = parser.parse_args()
    try:
        d = make_version(args.version, args.decision, args.boxes, args.batch, args.seed, args.val_days, args.test_days)
    except (TritoniaError, VersionError, OSError, KeyError) as e:
        sys.exit(f"REFUSED: {e}")
    with open(d / "split_manifest.csv", newline="", encoding="utf-8") as f:
        sessions = list(csv.DictReader(f))
    for split in ("train", "val", "test"):
        mine = [s for s in sessions if s["split"] == split]
        print(f"{split:5}: {len(mine)} days {[s['session_id'] for s in mine]}, "
              f"{sum(int(s['n_included']) for s in mine)} frames, {sum(int(s['n_instances']) for s in mine)} boxes")
    with open(d / "image_manifest.csv", newline="", encoding="utf-8") as f:
        print("label status:", dict(Counter(r["label_status"] for r in csv.DictReader(f))))
    print(f"draft written: {d}  (freeze only with explicit user approval)")


if __name__ == "__main__":
    main()
