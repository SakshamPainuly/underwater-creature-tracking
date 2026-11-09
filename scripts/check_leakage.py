#!/usr/bin/env python
"""Check a mixed dataset manifest (originals + derived + external images) for split leakage (D-031).

    python scripts/check_leakage.py --manifest data/processed/v4/mixed_manifest.csv \
        --groups data/processed/v4/zenodo_groups.csv [--image-root /path/to/raw/root]

Rules 1-5 (splits, ancestry) need only the CSVs. With --image-root, rule 6 also hashes every
train/val/test image (sha256 and dHash, which needs Pillow): run that on the machine that holds
the data. Prints PASSED / FAILED and exits 1 on failure. Writes nothing. Schema and rules:
src/dataset_audit/leakage.py.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit import leakage as L

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=REPO / "data" / "processed" / "split_manifest.csv",
                        help="session split for origin 'ours' (default: the v1 split)")
    parser.add_argument("--groups", type=Path, action="append", default=[],
                        help="extra group table(s): origin, group_id, split (e.g. zenodo groups)")
    parser.add_argument("--image-root", type=Path, help="root of the manifest's path column; enables rule 6")
    parser.add_argument("--max-distance", type=int, default=4, help="dHash near-duplicate threshold (bits)")
    parser.add_argument("--show", type=int, default=20, help="errors to print (all are counted)")
    args = parser.parse_args()

    rows = L.read_csv(args.manifest)
    groups = L.groups_from_split_manifest(args.split_manifest)
    for g in args.groups:
        extra = L.groups_from_table(g)
        clash = sorted(set(extra) & set(groups))
        if clash:
            raise SystemExit(f"{g}: group(s) already defined elsewhere, e.g. {clash[0]}")
        groups.update(extra)

    errors = L.check_manifest(rows, groups)
    if args.image_root:
        paths = L.image_paths(rows, args.image_root)
        print(f"hashing {len(paths)} images ...", flush=True)
        errors += L.exact_duplicates(rows, {i: L.sha256_of(p) for i, p in paths.items()})
        errors += L.near_duplicates(rows, {i: L.dhash(p) for i, p in paths.items()}, args.max_distance)
    else:
        print("rule 6 (duplicate images) skipped: no --image-root")

    print(json.dumps(L.summary(rows), indent=2))
    for e in errors[: args.show]:
        print("  " + e)
    if errors:
        print(f"LEAKAGE CHECK: FAILED ({len(errors)} error(s))")
        return 1
    print("LEAKAGE CHECK: PASSED" + ("" if args.image_root else " (rules 1-5 only)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
