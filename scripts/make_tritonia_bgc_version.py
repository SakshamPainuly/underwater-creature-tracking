#!/usr/bin/env python
r"""Draft Tritonia version from a preprocessed batch (D-045): the parent's frames, splits and boxes, new images.

    python scripts/make_tritonia_bgc_version.py --version v9 --decision D-045 --parent v7 \
        --processed /kaggle/working/tritonia_bgc1_v7 \
        --boxes /tmp/raw/tritonia_boxes_fixed2 --fingerprint-scratch /tmp/v9_fingerprint

Writes data/processed/<version>/ (image_manifest.csv, split_manifest.csv copied from the parent,
preprocess_manifest.csv, VERSION.json). Standard library. With --boxes (the parent's label source) it also
prints the dataset fingerprint the version's YOLO view will have once frozen (labels + build manifest only;
nothing is frozen here). Freezing is a separate step that needs explicit user approval:
    python scripts/freeze_dataset_version.py v9
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit import tritonia_bgc as B
from dataset_audit.tritonia_dataset import TritoniaError
from dataset_audit.versioning import VersionError


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--decision", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--boxes", type=Path)
    ap.add_argument("--fingerprint-scratch", type=Path)
    args = ap.parse_args()
    if (args.boxes is None) != (args.fingerprint_scratch is None):
        sys.exit("pass --boxes and --fingerprint-scratch together")
    try:
        d = B.make_bgc_version(args.version, args.decision, args.parent, args.processed)
        print(f"draft {args.version} -> {d}")
        if args.boxes:
            fp = B.draft_fingerprint(args.version, args.boxes, args.fingerprint_scratch)
            print(f"dataset_fingerprint (after freezing): {fp}")
    except (TritoniaError, VersionError, OSError, KeyError) as e:
        sys.exit(f"FAILED: {e}")


if __name__ == "__main__":
    main()
