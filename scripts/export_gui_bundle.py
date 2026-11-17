#!/usr/bin/env python
"""Export a portable copy of the GUI for another machine (e.g. a MacBook without the dataset).

    python scripts/export_gui_bundle.py --out D:/DeepSight_GUI

Writes the GUI code, the portable species file, the lobster checkpoints and their records, a few
sample images per source and the handoff documents (docs/gui_bundle/) into --out, plus a
MANIFEST.json with the sha256 of every file. --out must be empty or absent and outside data/raw.
CPU, standard library, well under a minute. Exit code: 0 = written, 1 = refused, 2 = usage error.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from detect import DetectError  # noqa: E402
from detect.bundle import SAMPLES_PER_SOURCE, export_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="new or empty folder to write the bundle to")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_SOURCE,
                        help=f"sample images per lobster source (default {SAMPLES_PER_SOURCE})")
    args = parser.parse_args()
    try:
        manifest = export_bundle(args.out, samples=args.samples)
    except DetectError as exc:
        print(f"EXPORT REFUSED: {exc}")
        sys.exit(1)
    total = sum(f["bytes"] for f in manifest["files"])
    print(f"bundle written: {args.out}  ({len(manifest['files'])} files, {total / 1e6:.0f} MB)")
    for line in manifest["weights_not_included"]:
        print(f"  weights not on this machine, slot left empty: {line}")


if __name__ == "__main__":
    main()
