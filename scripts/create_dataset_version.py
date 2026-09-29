#!/usr/bin/env python
"""Create a draft dataset version as an exact copy of its frozen parent (no rebuild of manifests).

Use it when only the image source changes (D-017): the sessions, split and labels stay the
parent's byte for byte, and VERSION.json records where the images come from.

    python scripts/create_dataset_version.py --version v2 --from-version v1 --decision D-017 \
        --image-source enhanced_2026-09

The decision must already exist in docs/DECISIONS.md. Freezing the draft is a separate,
user-approved step (scripts/freeze_dataset_version.py, CLAUDE.md rule 4).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.config import IMAGE_SOURCES
from dataset_audit.versioning import VersionError, check_extends_parent, open_draft, read_version_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="new draft version, e.g. v2")
    parser.add_argument("--from-version", required=True, help="frozen parent, e.g. v1")
    parser.add_argument("--decision", required=True, help="docs/DECISIONS.md id, e.g. D-017")
    parser.add_argument("--image-source", choices=sorted(IMAGE_SOURCES),
                        help="where the images come from (default: the parent's)")
    args = parser.parse_args()
    if read_version_meta(args.version) is not None:
        sys.exit(f"REFUSED: {args.version} already exists; this script only creates new drafts")
    try:
        d = open_draft(args.version, args.from_version, args.decision, image_source=args.image_source)
        problems = check_extends_parent(args.version)
    except VersionError as e:
        sys.exit(f"REFUSED: {e}")
    if problems:
        sys.exit(f"draft {args.version} does not extend {args.from_version}: {problems[:5]}")
    print(f"created draft {args.version} in {d}")
    print(json.dumps(read_version_meta(args.version), indent=2))


if __name__ == "__main__":
    main()
