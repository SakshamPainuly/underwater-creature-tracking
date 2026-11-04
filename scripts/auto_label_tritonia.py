#!/usr/bin/env python
r"""Automatic sea slug labels for the Tritonia batch from the image next to each cross (D-031).

    python scripts/auto_label_tritonia.py --batch D:\uwct_work\tritonia_label_batch1 --out D:\uwct_work\tritonia_auto_labels

Writes <out>/labels/, labels.zip, notes.txt (frames with no box excluded), suggestions.csv and
auto_label_report.json. Hand it to scripts/ingest_tritonia_boxes.py as --zip/--notes.
Accuracy is then measured against the crosses (scripts/tritonia_point_eval.py), never against
these boxes.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.config import RAW_ROOTS
from dataset_audit.tritonia_prelabel import auto_label_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=Path, required=True, help="labelling batch folder (manifest.csv, images/, points/)")
    parser.add_argument("--out", type=Path, required=True, help="new folder for the automatic labels")
    args = parser.parse_args()
    batch, out = args.batch.resolve(), args.out.resolve()
    for forbidden in (*RAW_ROOTS, batch):
        f = Path(forbidden).resolve()
        if out == f or f in out.parents:
            sys.exit(f"REFUSED: output {out} is inside {f}")
    counts = auto_label_batch(batch, out)
    print(json.dumps({k: v for k, v in counts.items() if k != "params"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
