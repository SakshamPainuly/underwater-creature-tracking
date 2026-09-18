#!/usr/bin/env python
"""Pre-YOLO-prep audit: build a per-image manifest (positive/negative/pending)
joined against the frozen session-level split, and run lightweight consistency
checks.

Reads (read-only):
    data/raw/biology/...
    <version dir>/split_manifest.csv

Writes:
    <version dir>/image_manifest.csv
    <version dir>/dataset_consistency_report.json

Frozen versions (v1) are refused; write a new draft instead (see run_audit.py).
Does not touch data/raw/biology/. Does not train anything. Standard library only.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.image_manifest import build_image_manifest, load_split_map
from dataset_audit.versioning import VersionError, add_version_args, assert_writable, output_dir_for

IMAGE_FIELDNAMES = [
    "filename", "source_set", "session_id", "video_id", "split",
    "has_label_file", "n_instances", "label_status", "issues",
]


def check_duplicate_session_assignments(split_manifest_path: Path) -> list:
    with open(split_manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    session_to_splits = defaultdict(set)
    for r in rows:
        session_to_splits[r["session_id"]].add(r["split"])
    return [sid for sid, splits in session_to_splits.items() if len(splits) > 1]


def run(version_dir: Path) -> None:
    split_manifest_path = version_dir / "split_manifest.csv"
    image_manifest_path = version_dir / "image_manifest.csv"
    consistency_report_path = version_dir / "dataset_consistency_report.json"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"{split_manifest_path} not found -- run scripts/run_split.py first")
    assert_writable(image_manifest_path)
    assert_writable(consistency_report_path)

    split_map = load_split_map(split_manifest_path)
    image_rows, anomaly_rows = build_image_manifest(split_map)
    dup_sessions = check_duplicate_session_assignments(split_manifest_path)

    with open(image_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=IMAGE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(image_rows)

    # --- consistency checks ---
    malformed = [r for r in image_rows if r["issues"]]
    unknown_session = [r for r in image_rows if r["split"].startswith("UNKNOWN_SESSION")]
    dup_filenames = [r for r in anomaly_rows if r["type"] == "duplicate_filename"]
    orphan_labels = [r for r in anomaly_rows if r["type"] == "orphan_label_no_matching_image"]

    consistency_report = {
        "total_images_scanned": len(image_rows),
        "malformed_or_invalid_label_images": malformed,
        "images_with_session_not_in_split_manifest": [r["filename"] for r in unknown_session],
        "duplicate_filenames_across_sources": dup_filenames,
        "orphan_label_files_no_matching_image": orphan_labels,
        "sessions_assigned_to_more_than_one_split": dup_sessions,
    }
    with open(consistency_report_path, "w", encoding="utf-8") as f:
        json.dump(consistency_report, f, indent=2)

    print(f"Wrote {image_manifest_path} ({len(image_rows)} images)")
    print(f"Wrote {consistency_report_path}")
    print()

    # --- distribution summary ---
    by_split = defaultdict(lambda: Counter())
    for r in image_rows:
        s = r["split"]
        by_split[s]["total"] += 1
        if r["label_status"] == "positive":
            by_split[s]["positive"] += 1
        elif r["label_status"] in ("negative_confirmed",):
            by_split[s]["negative_confirmed"] += 1
        elif r["label_status"] == "negative_pending":
            by_split[s]["negative_pending"] += 1
        elif r["label_status"] == "invalid":
            by_split[s]["invalid"] += 1
        by_split[s]["instances"] += r["n_instances"]

    print(f"{'split':10} {'total':>6} {'positive':>9} {'neg_confirmed':>14} {'neg_pending':>12} {'invalid':>8} {'instances':>10}")
    for split_name in ("train", "val", "test", "excluded"):
        c = by_split[split_name]
        print(
            f"{split_name:10} {c['total']:>6} {c['positive']:>9} {c['negative_confirmed']:>14} "
            f"{c['negative_pending']:>12} {c['invalid']:>8} {c['instances']:>10}"
        )

    print()
    print("=== Negative (confirmed, Set2000) images by session ===")
    neg_by_session = defaultdict(lambda: Counter())
    for r in image_rows:
        if r["label_status"] == "negative_confirmed":
            neg_by_session[r["session_id"]]["count"] += 1
            neg_by_session[r["session_id"]]["split"] = r["split"]
    for sid in sorted(neg_by_session):
        print(f"  {sid:28} negatives={neg_by_session[sid]['count']:>3}  split={neg_by_session[sid]['split']}")
    total_confirmed_neg = sum(v["count"] for v in neg_by_session.values())
    print(f"  TOTAL confirmed-negative Set2000 images: {total_confirmed_neg}")

    print()
    print("=== Consistency check summary ===")
    print(f"  Malformed/invalid label files: {len(malformed)}")
    print(f"  Images whose session is missing from split_manifest.csv: {len(unknown_session)}")
    print(f"  Duplicate filenames across sources: {len(dup_filenames)}")
    print(f"  Orphan label files (no matching image): {len(orphan_labels)}")
    print(f"  Sessions assigned to more than one split: {len(dup_sessions)}")


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
