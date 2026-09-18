#!/usr/bin/env python
"""Add a finished run to experiments/registry.csv (local PC, standard library only).

    python scripts/register_run.py path/to/run_record.json [--dry-run]

Download run_record.json from the run folder on Drive first. Refuses smoke runs, incomplete
records (no commit, unverified dataset, missing val metrics or checkpoint sha256), records with
test metrics, and experiments that are already registered. Update the run log in
docs/EXPERIMENTS.md by hand from the printed row.
Exit code: 0 = registered (or would be, with --dry-run), 1 = refused.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.run_record import RecordError, append_to_registry, registry_row, validate_record

DEFAULT_REGISTRY = PROJECT_ROOT / "experiments" / "registry.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    record = json.loads(args.record.read_text(encoding="utf-8"))
    try:
        if args.dry_run:
            problems = validate_record(record)
            if problems:
                raise RecordError("; ".join(problems))
            row = registry_row(record)
        else:
            row = append_to_registry(args.registry, record)
    except (RecordError, KeyError) as e:
        print(f"REFUSED: {e}")
        sys.exit(1)
    print(json.dumps(row, indent=2))
    print("DRY RUN: registry unchanged" if args.dry_run else f"registered in {args.registry}")


if __name__ == "__main__":
    main()
