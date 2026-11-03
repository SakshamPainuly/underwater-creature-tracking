#!/usr/bin/env python
"""Build the YOLO view of a frozen Tritonia dataset version (D-030); per machine, like prepare_yolo_dataset.py.

    python scripts/prepare_tritonia_yolo.py --version v3 \\
        --frames /content/tritonia/frames --boxes /content/tritonia/boxes

--frames holds the batch's clean PNGs (images/ of the labelling batch), --boxes the label source
from ingest_tritonia_boxes.py. Every image and label is checked against the version's sha256
before it is copied. Output: data/yolo/tritonia_<version>/ (default; never committed). Prints the
dataset fingerprint that the experiment config must name.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_audit.config import RAW_ROOTS
from dataset_audit.tritonia_dataset import TritoniaError, build_view, view_dir_for
from dataset_audit.versioning import VersionError
from yolo_prep.build import TARGET_ULTRALYTICS_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="default: data/yolo/tritonia_<version>")
    args = parser.parse_args()
    out = (args.out or view_dir_for(args.version, PROJECT_ROOT)).resolve()
    for raw in RAW_ROOTS:
        raw = Path(raw).resolve()
        if out == raw or raw in out.parents:
            sys.exit(f"REFUSED: {out} is inside the read-only raw root {raw}")
    try:
        summary = build_view(args.version, args.frames, args.boxes, out, TARGET_ULTRALYTICS_VERSION)
    except (TritoniaError, VersionError, OSError, KeyError) as e:
        sys.exit(f"BUILD FAILED: {e}")
    print(json.dumps(summary["splits"], indent=2))
    print(f"built {out}")
    print(f"dataset_fingerprint: {summary['dataset_fingerprint']}")


if __name__ == "__main__":
    main()
