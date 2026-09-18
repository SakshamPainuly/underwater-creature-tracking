#!/usr/bin/env python
"""Freeze a draft dataset version (irreversible). Requires explicit user approval.

    python scripts/freeze_dataset_version.py v1.1

Refuses unless: the version is a draft, its decision exists in docs/DECISIONS.md, and it
extends its frozen parent (every parent session keeps its split and every parent image keeps
its label_status, except changes listed in VERSION.json approved_split_changes /
approved_label_changes). Then records the sha256 of every file in data/processed/FROZEN.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.versioning import VersionError, freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version")
    args = parser.parse_args()
    try:
        entry = freeze(args.version)
    except VersionError as e:
        sys.exit(f"REFUSED: {e}")
    print(f"froze {args.version}:")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
