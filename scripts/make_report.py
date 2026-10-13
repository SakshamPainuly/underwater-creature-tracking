#!/usr/bin/env python
"""Build the E001 vs E004 report (Markdown + PNG figures) from the committed records.

    .venv/Scripts/python scripts/make_report.py            (needs matplotlib: the GUI .venv has it)

Reads experiments/records/E001/ and E004/ only; writes docs/reports/E001_vs_E004_report.md and
docs/reports/figures/. CPU, a few seconds. Deterministic: same records -> same files.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reporting.e001_e004 import ReportError, build_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=PROJECT_ROOT / "experiments" / "records")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "reports")
    args = parser.parse_args()
    try:
        report = build_report(args.records, args.out)
    except ReportError as e:
        print(f"REPORT NOT BUILT: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"REPORT NOT BUILT: {e}. Run it with the GUI environment (.venv), which has matplotlib.")
        sys.exit(1)
    print(f"report: {report}")


if __name__ == "__main__":
    main()
