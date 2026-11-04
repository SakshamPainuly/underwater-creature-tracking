#!/usr/bin/env python
"""Write the mixed manifest of the Model A dataset version (D-031) from Harsh's deliveries.

    python scripts/build_model_a_manifest.py --version v4 --from-version v1 --decision D-031

Creates the draft (a copy of its frozen parent, as create_dataset_version.py does) if needed, then
writes into it:
    mixed_manifest.csv   one row per image (schema: src/dataset_audit/model_a.py)
    zenodo_groups.csv    Zenodo group -> split (origin, group_id, split), for check_leakage.py
    model_a_report.json  counts per delivery folder, exclusions, boxes per split

Reads only (raw roots: dataset_audit.config BIOLOGY_ROOT and MODEL_A_ROOTS). Fails, writing
nothing, if a folder and its index disagree, an ancestor is unknown, or a label check fails.
Next: python scripts/check_leakage.py --manifest <dir>/mixed_manifest.csv --groups <dir>/zenodo_groups.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit import model_a as M
from dataset_audit.config import BIOLOGY_ROOT, DATASET_SOURCES, MODEL_A_ROOTS
from dataset_audit.versioning import VersionError, add_version_args, assert_writable, output_dir_for, version_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_version_args(parser)
    args = parser.parse_args()
    source_dirs = {name: (Path(img), Path(lbl) if lbl else None) for name, img, lbl in DATASET_SOURCES}
    try:
        parent = args.from_version or json.loads((version_dir(args.version) / "VERSION.json")
                                                 .read_text(encoding="utf-8"))["parent"]
        v1_rows = M.read_csv(version_dir(parent) / "image_manifest.csv")
        rows, groups, report = M.build(v1_rows, source_dirs, BIOLOGY_ROOT, MODEL_A_ROOTS)
        out = output_dir_for(args)
        targets = [out / "mixed_manifest.csv", out / "zenodo_groups.csv", out / "model_a_report.json"]
        for t in targets:
            assert_writable(t)
    except (M.ModelAError, VersionError, FileNotFoundError) as e:
        sys.exit(f"REFUSED: {e}")
    M.write_csv(targets[0], rows, M.FIELDNAMES)
    M.write_csv(targets[1], groups, ["origin", "group_id", "split"])
    report = {"version": args.version, "parent": parent, **report}
    targets[2].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["by_delivery"], indent=2))
    print(f"wrote {len(rows)} rows -> {targets[0]}")
    print(f"next: python scripts/check_leakage.py --manifest {targets[0]} --groups {targets[1]}")


if __name__ == "__main__":
    main()
