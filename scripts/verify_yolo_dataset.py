#!/usr/bin/env python
"""Confirm Ultralytics itself reads the prepared dataset as intended. No training.

Run on Colab / the lab GPU after `pip install -r requirements-gpu.txt` and
`python scripts/prepare_yolo_dataset.py`. Uses Ultralytics' own label scanner
(the same code path training uses) and checks, per split, that:
    images seen      == build_summary.json images
    backgrounds seen == build_summary.json confirmed negatives (empty label files)
    boxes seen       == build_summary.json instances
Exit code 0 = match, 1 = mismatch, 2 = ultralytics not installed.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo_prep.verify import COUNT_KEYS, counts_match, ultralytics_split_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "yolo" / "lobster_v1")
    args = parser.parse_args()

    try:
        import ultralytics
    except ImportError:
        print("ultralytics is not installed here; run this on Colab / the lab GPU.")
        sys.exit(2)

    expected = json.loads((args.dataset / "build_summary.json").read_text(encoding="utf-8"))
    if ultralytics.__version__ != expected["target_ultralytics_version"]:
        print(f"WARNING: ultralytics {ultralytics.__version__} != pinned {expected['target_ultralytics_version']}")

    data, seen = ultralytics_split_counts(args.dataset / "data.yaml", list(expected["splits"]))
    print(f"names: {data['names']}")
    ok = data["names"] == {0: "lobster"}

    for split, exp in expected["splits"].items():
        match = counts_match(exp, seen[split])
        ok &= match
        print(f"{split:5} seen={seen[split]} expected={ {k: exp[k] for k in COUNT_KEYS} } "
              f"{'OK' if match else 'MISMATCH'}")

    print("VERIFIED" if ok else "VERIFICATION FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
