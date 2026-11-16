#!/usr/bin/env python
"""Build the final report data pack (Markdown + CSV + PNG figures) from the committed records.

    .venv/Scripts/python scripts/make_final_report.py      (needs matplotlib: the GUI .venv has it)

Reads experiments/registry.csv, experiments/records/ and data/processed/v4/mixed_manifest.csv;
writes docs/reports/final_report_data.md, docs/reports/final_report_experiments.csv and
docs/reports/figures/final_*.png. CPU, a few seconds. Deterministic: same records -> same files.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reporting.e001_e004 import ReportError  # noqa: E402
from reporting.final_report import build_final_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="repository root (inputs)")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "reports")
    args = parser.parse_args()
    try:
        report = build_final_report(args.root, args.out)
    except ReportError as e:
        print(f"REPORT NOT BUILT: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"REPORT NOT BUILT: {e}. Run it with the GUI environment (.venv), which has matplotlib.")
        sys.exit(1)
    print(f"report: {report}")


if __name__ == "__main__":
    main()
