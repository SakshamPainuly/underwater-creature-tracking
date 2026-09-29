#!/usr/bin/env python
"""Unpack a delivered image ZIP (e.g. Harsh's enhanced images) into a new image source root.

    python scripts/unpack_image_source.py --zip /content/enhanced_2026-09.zip \
        --out /content/raw/enhanced_2026-09

Keeps only images/<folder>/<name>.png (macOS metadata skipped) and refuses a non-empty --out.
Then check the result against the recorded content fingerprint:

    python scripts/check_raw_unchanged.py --root /content/raw/enhanced_2026-09 --full \
        --fingerprint data/processed/raw_fingerprint_enhanced_2026-09.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.config import BIOLOGY_ROOT
from dataset_audit.delivery import DeliveryError, unpack_delivery


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new image source root (must be empty)")
    args = parser.parse_args()
    out, raw = args.out.resolve(), BIOLOGY_ROOT.resolve()
    if out == raw or raw in out.parents:
        sys.exit(f"REFUSED: {out} is inside the raw Biology data ({raw})")
    try:
        n = unpack_delivery(args.zip, out)
    except (DeliveryError, OSError) as e:
        sys.exit(f"UNPACK FAILED: {e}")
    print(f"unpacked {n} images -> {out}")


if __name__ == "__main__":
    main()
