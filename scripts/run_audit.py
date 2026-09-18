#!/usr/bin/env python
"""Phase 1 dataset audit.

Scans data/raw/biology/ (read-only) and 'What's been done.xlsx' (read-only),
and writes a deployment/session-level metadata table into a dataset version directory:
    <version dir>/deployment_metadata.csv
    <version dir>/deployment_metadata.json

Frozen versions (v1) are refused; write a new draft instead:
    python scripts/run_audit.py --version v1.1 --from-version v1 --decision D-XXX

Does not modify anything under data/raw/biology/.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.metadata import load_session_metadata
from dataset_audit.scan import scan_all_sources, sessions_to_rows
from dataset_audit.versioning import VersionError, add_version_args, assert_writable, output_dir_for

FIELDNAMES = [
    "session_id",
    "source_set",
    "n_videos",
    "video_ids",
    "n_images",
    "n_labeled",
    "n_instances",
    "substrate",
    "visibility",
    "lighting_flag",
    "free_notes",
    "done",
    "sent",
    "files_used",
    "spreadsheet_n_images",
    "metadata_matched",
]


def build_rows():
    sessions = scan_all_sources()
    scan_rows = sessions_to_rows(sessions)
    meta = load_session_metadata()

    rows = []
    for row in scan_rows:
        m = meta.get(row["session_id"])
        merged = dict(row)
        if m is None:
            merged.update(
                {
                    "substrate": "",
                    "visibility": "",
                    "lighting_flag": False,
                    "free_notes": "",
                    "done": "",
                    "sent": "",
                    "files_used": "",
                    "spreadsheet_n_images": None,
                    "metadata_matched": False,
                }
            )
        else:
            merged.update(m)
            merged["metadata_matched"] = True
        rows.append(merged)
    return rows, meta, sessions


def run(out_dir: Path) -> None:
    rows, meta, sessions = build_rows()
    csv_path = out_dir / "deployment_metadata.csv"
    json_path = out_dir / "deployment_metadata.json"
    assert_writable(csv_path)
    assert_writable(json_path)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # --- console summary ---
    total_images = sum(r["n_images"] for r in rows)
    total_labeled = sum(r["n_labeled"] for r in rows)
    total_instances = sum(r["n_instances"] for r in rows)
    unmatched = [r["session_id"] for r in rows if not r["metadata_matched"]]
    unlabeled_sessions = [r["session_id"] for r in rows if r["n_labeled"] == 0]
    multi_file = [r["session_id"] for r in rows if r["n_videos"] > 1]

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print()
    print(f"Sessions found:      {len(rows)}")
    print(f"Total images:        {total_images}")
    print(f"Total labeled images:{total_labeled}")
    print(f"Total lobster boxes: {total_instances}")
    print()
    print(f"Sessions with zero labeled images ({len(unlabeled_sessions)}): {unlabeled_sessions}")
    print(f"Multi-file sessions ({len(multi_file)}): {multi_file}")
    if unmatched:
        print(f"WARNING: sessions with no spreadsheet metadata match ({len(unmatched)}): {unmatched}")

    spreadsheet_only = sorted(set(meta.keys()) - {r["session_id"] for r in rows})
    if spreadsheet_only:
        print()
        print(
            f"Sessions documented in the spreadsheet but NOT present in data/raw/biology/ "
            f"({len(spreadsheet_only)}) -- e.g. 'in progress' deployments not yet delivered:"
        )
        for code in spreadsheet_only:
            print(f"  {code}  (done={meta[code]['done']!r}, sent={meta[code]['sent']!r})")


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
