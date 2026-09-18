#!/usr/bin/env python
"""Phase 1 split generation.

Reads <version dir>/deployment_metadata.csv (produced by run_audit.py) and
writes a session-level, leakage-aware split assignment to:
    <version dir>/split_manifest.csv

Frozen versions (v1) are refused -- the v1 split is never regenerated. Note: this
generator assigns every session from scratch; a new version must keep its parent's
assignments, which scripts/freeze_dataset_version.py enforces before freezing.

No images are copied/moved. data/raw/biology/ is not touched.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.split import build_split
from dataset_audit.versioning import VersionError, add_version_args, assert_writable, output_dir_for


def load_metadata_rows(in_path: Path):
    with open(in_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["n_videos"] = int(r["n_videos"])
        r["n_images"] = int(r["n_images"])
        r["n_labeled"] = int(r["n_labeled"])
        r["n_instances"] = int(r["n_instances"])
        r["lighting_flag"] = r["lighting_flag"] == "True"
    return rows


def run(version_dir: Path) -> None:
    in_path = version_dir / "deployment_metadata.csv"
    out_path = version_dir / "split_manifest.csv"
    if not in_path.is_file():
        raise FileNotFoundError(f"{in_path} not found -- run scripts/run_audit.py first")
    assert_writable(out_path)

    rows = load_metadata_rows(in_path)
    by_session = {r["session_id"]: r for r in rows}
    results = build_split(rows)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "session_id",
                "source_set",
                "split",
                "n_images",
                "n_labeled",
                "n_instances",
                "substrate",
                "visibility",
                "lighting_flag",
                "flags",
            ]
        )
        for res in results:
            meta = by_session[res.session_id]
            writer.writerow(
                [
                    res.session_id,
                    res.source_set,
                    res.split,
                    meta["n_images"],
                    meta["n_labeled"],
                    meta["n_instances"],
                    meta["substrate"],
                    meta["visibility"],
                    meta["lighting_flag"],
                    ";".join(res.flags),
                ]
            )

    print(f"Wrote {out_path}\n")

    # --- summary ---
    totals = defaultdict(lambda: {"sessions": 0, "images": 0, "labeled": 0, "instances": 0})
    for res in results:
        meta = by_session[res.session_id]
        t = totals[res.split]
        t["sessions"] += 1
        t["images"] += meta["n_images"]
        t["labeled"] += meta["n_labeled"]
        t["instances"] += meta["n_instances"]

    print(f"{'split':10} {'sessions':>9} {'images':>8} {'labeled':>8} {'instances':>10}")
    for split_name in ("train", "val", "test", "excluded"):
        t = totals[split_name]
        print(
            f"{split_name:10} {t['sessions']:>9} {t['images']:>8} {t['labeled']:>8} {t['instances']:>10}"
        )

    print("\nFlagged sessions (scientific review recommended):")
    for res in sorted(results, key=lambda r: r.session_id):
        if res.flags:
            print(f"  [{res.split:8}] {res.session_id:28} {res.flags}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_version_args(parser)
    args = parser.parse_args()
    try:
        run(output_dir_for(args))
    except VersionError as e:
        sys.exit(f"REFUSED: {e}")


if __name__ == "__main__":
    main()
