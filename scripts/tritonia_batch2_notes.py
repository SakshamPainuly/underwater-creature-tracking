#!/usr/bin/env python
r"""notes.txt for Tritonia batch 2 (D-035): the automatic labels' notes plus one `exclude` line for every
ADDED frame where not every cross got a box. Batch 1 frames keep their v3 treatment.

    python scripts/tritonia_batch2_notes.py --batch D:\uwct_tritonia\work\tritonia_label_batch2 \
        --auto D:\uwct_tritonia\work\tritonia_auto_labels_batch2 --out D:\uwct_tritonia\work\tritonia_batch2_notes

Writes <out>/notes.txt (hand it to scripts/ingest_tritonia_boxes.py as --notes) and
<out>/batch2_filter_report.json (counts). --out must be a new folder.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.config import RAW_ROOTS
from dataset_audit.tritonia_prelabel import incomplete_added_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--auto", type=Path, required=True, help="output of scripts/auto_label_tritonia.py on that batch")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    for f in (*RAW_ROOTS, args.batch, args.auto):
        if Path(f).resolve() in (out, *out.parents):
            sys.exit(f"REFUSED: {out} is inside {f}")
    if out.exists():
        sys.exit(f"REFUSED: {out} already exists")
    meta = json.loads((args.batch / "batch.json").read_text(encoding="utf-8"))
    added = meta["added"]
    with open(args.auto / "suggestions.csv", newline="", encoding="utf-8") as f:
        sug = list(csv.DictReader(f))
    notes = (args.auto / "notes.txt").read_text(encoding="utf-8")
    already = {line.split("|")[0].strip() for line in notes.splitlines()
               if line.strip() and not line.startswith("#") and "| exclude |" in line}
    missing = incomplete_added_frames(sug, added)
    new_lines = [f"{s} | exclude | batch 2 rule (D-035): {n} cross(es) without a box\n"
                 for s, n in missing.items() if s not in already]
    out.mkdir(parents=True)
    (out / "notes.txt").write_text(notes.rstrip("\n") + "\n" + "".join(new_lines), encoding="utf-8", newline="\n")
    kept = [s for s in added if s not in missing]
    crosses = {}
    for r in sug:
        crosses[r["frame"]] = crosses.get(r["frame"], 0) + 1
    report = {"decision": "D-035", "added_frames": len(added), "added_complete_kept": len(kept),
              "added_excluded": len(missing), "added_excluded_already_by_auto_labels": len(missing) - len(new_lines),
              "boxes_in_kept_added_frames": sum(crosses[s] for s in kept),
              "kept_per_day": {d: sum(1 for s in kept if s[:8] == d) for d in sorted({s[:8] for s in added})}}
    (out / "batch2_filter_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
