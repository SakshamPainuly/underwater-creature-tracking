#!/usr/bin/env python
"""Build the Ultralytics YOLO dataset view for a dataset version (no training).

Builds data/yolo/lobster_<version>/ from:
    data/raw/biology/                      (read-only; override with UWCT_RAW_ROOT)
    <version dir>/image_manifest.csv
    <version dir>/split_manifest.csv

A frozen version is hash-verified against data/processed/FROZEN.json before building.
Draft versions build only with --allow-draft and are labeled "draft" in build_summary.json
(never use draft builds for reported experiments).

Images are linked, not duplicated (symlink where the OS allows, else hardlink).
Run it once per machine (Windows PC, Colab, lab GPU) -- it takes seconds.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.config import BIOLOGY_ROOT, PROJECT_ROOT
from dataset_audit.versioning import VersionError, is_frozen, read_version_meta, verify_frozen, version_dir
from yolo_prep.build import DATASET_BASENAME, BuildError, build_yolo_dataset


def resolve_version(name: str, allow_draft: bool) -> tuple:
    if is_frozen(name):
        verify_frozen(name)
        return version_dir(name), "frozen"
    if read_version_meta(name) is None:
        raise VersionError(f"unknown dataset version {name}")
    if not allow_draft:
        raise VersionError(f"{name} is a draft (not frozen); pass --allow-draft to build it anyway")
    return version_dir(name), "draft"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--out", type=Path, help="default: data/yolo/lobster_<version>")
    parser.add_argument("--link-mode", choices=["auto", "symlink", "hardlink", "copy"], default="auto")
    args = parser.parse_args()

    print(f"raw data root: {BIOLOGY_ROOT}")
    try:
        vdir, status = resolve_version(args.version, args.allow_draft)
        out = args.out or PROJECT_ROOT / "data" / "yolo" / f"{DATASET_BASENAME}_{args.version}"
        summary = build_yolo_dataset(
            vdir / "image_manifest.csv", vdir / "split_manifest.csv", out, args.link_mode,
            version=args.version, version_status=status,
        )
    except (VersionError, BuildError) as e:
        sys.exit(f"BUILD FAILED: {e}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
