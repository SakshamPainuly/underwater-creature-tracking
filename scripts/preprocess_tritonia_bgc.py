#!/usr/bin/env python
r"""Background-contrast preprocessing (method bgc1, D-045) of every frame of a frozen Tritonia version.

Kaggle (numpy + OpenCV; notebooks/kaggle_tritonia_preprocess.ipynb):
    python scripts/preprocess_tritonia_bgc.py --version v7 \
        --batch /kaggle/input/.../tritonia_label_batch2 \
        --sequence /kaggle/input/.../val --sequence /kaggle/input/.../test \
        --out /kaggle/working/tritonia_bgc1_v7

--batch is a labelling batch that holds every frame of --version (byte-identical, sha256-checked) and
supplies background frames for its days; each --sequence (every frame of some days: images/, manifest.csv
with image, t, png_sha256) supplies background frames for its days. Inputs are only read. --out (new folder)
gets a batch-shaped folder with the preprocessed frames, their points, manifest.csv, batch.json,
preprocess_manifest.csv and preprocess_report.json. The method and parameters are fixed in
src/dataset_audit/tritonia_bgc.py (PARAMS); there are no tuning options here on purpose.
--plan-only checks that every frame has enough background frames and writes nothing.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit import tritonia_bgc as B
from dataset_audit.config import RAW_ROOTS
from dataset_audit.tritonia_dataset import TritoniaError
from dataset_audit.versioning import VersionError, read_version_meta, verify_frozen, version_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, help="frozen Tritonia version whose frames are processed (v7)")
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--sequence", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    out = args.out.resolve()
    for raw in RAW_ROOTS:
        raw = Path(raw).resolve()
        if out == raw or raw in out.parents:
            sys.exit(f"REFUSED: {out} is inside the read-only raw root {raw}")
    try:
        meta = read_version_meta(args.version)
        if not meta or meta.get("dataset_kind") != "tritonia":
            sys.exit(f"REFUSED: {args.version} is not a Tritonia dataset version")
        verify_frozen(args.version)
        with open(version_dir(args.version) / "image_manifest.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        pools = B.day_pools(args.batch, args.sequence)
        planned = B.plan(rows, pools)
        print(f"plan OK: {len(planned)} frames, neighbours per frame "
              f"{min(len(p['neighbours']) for p in planned)}-{max(len(p['neighbours']) for p in planned)}")
        if args.plan_only:
            return
        report = B.write_processed_batch(args.batch, rows, planned, out, args.version,
                                         inputs=[args.batch, *args.sequence], progress=print)
    except (TritoniaError, VersionError, OSError, KeyError, ValueError) as e:
        sys.exit(f"FAILED: {e}")
    print(json.dumps(report["per_split"], indent=2))
    print(f"wrote {report['frames']} frames -> {out}")


if __name__ == "__main__":
    main()
