#!/usr/bin/env python
r"""Fixed-size sea slug boxes on every cross of a Tritonia batch (D-036); local PC, standard library.

    python scripts/fixed_label_tritonia.py --batch D:\uwct_tritonia\work\tritonia_label_batch1 --out D:\uwct_tritonia\work\tritonia_fixed_labels

One 20 x 17 px box (the median size of the v3 automatic boxes on the train days) centred on each
MTrackJ cross, clipped to the camera view that holds the cross. No frame is excluded. Writes
<out>/labels/, labels.zip, notes.txt and fixed_label_report.json; hand them to
scripts/ingest_tritonia_boxes.py as --zip/--notes. Accuracy is still measured against the crosses
(scripts/tritonia_point_eval.py), never against these boxes.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.config import RAW_ROOTS
from dataset_audit.tritonia_prelabel import FIXED_BOX_PX, fixed_label_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=Path, required=True, help="labelling batch folder (manifest.csv, points/)")
    parser.add_argument("--out", type=Path, required=True, help="new folder for the labels")
    args = parser.parse_args()
    batch, out = args.batch.resolve(), args.out.resolve()
    for forbidden in (*RAW_ROOTS, batch):
        f = Path(forbidden).resolve()
        if out == f or f in out.parents:
            sys.exit(f"REFUSED: output {out} is inside {f}")
    try:
        counts = fixed_label_batch(batch, out, FIXED_BOX_PX)
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"FAILED: {e}")
    print(json.dumps(counts, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
