#!/usr/bin/env python
"""Build the Ultralytics YOLO dataset view for a dataset version (no training).

Builds data/yolo/lobster_<version>/ from:
    data/raw/biology/                      (read-only; override with UWCT_RAW_ROOT)
    <version dir>/image_manifest.csv
    <version dir>/split_manifest.csv

A frozen version is hash-verified against data/processed/FROZEN.json before building.
Draft versions build only with --allow-draft and are labeled "draft" in build_summary.json
(never use draft builds for reported experiments).

Images come from the version's image source (VERSION.json "image_source", D-017): the raw images
for v1, the enhanced images for v2 (root: UWCT_ENHANCED_ROOT). Labels always come from the raw
data. Images are linked, not duplicated (symlink where the OS allows, else hardlink).
Run it once per machine (Windows PC, Colab, Kaggle) -- it takes seconds.

Mixed versions (VERSION.json dataset_kind "lobster_mixed": v4, Model A, D-031/D-032) are built from
<version dir>/mixed_manifest.csv by yolo_prep/mixed.py instead: images are always copied (JPGs,
rule 2), raw roots come from UWCT_RAW_ROOT and the UWCT_HARSH_*_ROOT variables, and the default
output is data/yolo/lobster_mixed_<version>. --link-mode does not apply to them.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.config import BIOLOGY_ROOT, IMAGE_SOURCES, PROJECT_ROOT
from dataset_audit.versioning import (VersionError, image_source_of, is_frozen, read_version_meta,
                                      verify_frozen, version_dir)
from yolo_prep import mixed
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
        if (read_version_meta(args.version) or {}).get("dataset_kind") == mixed.DATASET_KIND:
            for name, root in mixed.default_raw_roots().items():
                print(f"raw root {name}: {root}")
            out = args.out or PROJECT_ROOT / "data" / "yolo" / f"{mixed.DATASET_KIND}_{args.version}"
            summary = mixed.build_mixed_dataset(vdir / mixed.MANIFEST_FILE, out, args.version, status)
            print(json.dumps(summary, indent=2))
            return
        image_source = image_source_of(args.version)
        print(f"image source: {image_source} ({IMAGE_SOURCES.get(image_source)})")
        out = args.out or PROJECT_ROOT / "data" / "yolo" / f"{DATASET_BASENAME}_{args.version}"
        summary = build_yolo_dataset(
            vdir / "image_manifest.csv", vdir / "split_manifest.csv", out, args.link_mode,
            version=args.version, version_status=status, image_source=image_source,
        )
    except (VersionError, BuildError) as e:
        sys.exit(f"BUILD FAILED: {e}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
