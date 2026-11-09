#!/usr/bin/env python
"""Ingest the hand-drawn Tritonia boxes (makesense.ai YOLO export) as a new read-only label source (D-029).

    python scripts/ingest_tritonia_boxes.py --zip D:\\uwct_tritonia\\work\\tritonia_batch1_yolo.zip \\
        --notes D:\\uwct_tritonia\\work\\tritonia_label_batch1\\notes.txt \\
        --batch D:\\uwct_tritonia\\work\\tritonia_label_batch1 --out D:\\uwct_tritonia\\raw\\tritonia_boxes_batch1

Checks every label file against the batch manifest before writing anything: one file per frame
(or the frame marked '| exclude |' in notes.txt), class ids 0/1 only, boxes inside the frame and
not across a camera border. --out must not exist; afterwards it is read-only (CLAUDE.md rule 1).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.tritonia_dataset import TritoniaError, ingest_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True, help="notes.txt (may hold only comments)")
    parser.add_argument("--batch", type=Path, required=True, help="the labelling batch folder (manifest.csv)")
    parser.add_argument("--out", type=Path, required=True, help="new label source folder (must not exist)")
    parser.add_argument("--extra", type=Path, nargs="*", default=[],
                        help="files copied in unchanged as extra/<name> (e.g. suggestions.csv, auto_label_report.json)")
    args = parser.parse_args()
    try:
        report = ingest_boxes(args.zip, args.notes, args.batch, args.out, tuple(args.extra))
    except (TritoniaError, OSError, ValueError, KeyError) as e:
        sys.exit(f"INGEST REFUSED: {e}")
    print(json.dumps(report["counts"], indent=2))
    print(f"excluded frames: {len(report['excluded'])}; notes on kept frames: {len(report['notes'])}")
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
