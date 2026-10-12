#!/usr/bin/env python
"""Side-by-side markdown comparison of val_eval records (standard library; runs locally).

    python scripts/compare_val_eval.py experiments/records/E001/val_eval_record.json \
        experiments/records/E004/val_eval_record.json [--out comparison.md]

Exit code: 0 = written, 1 = a file is missing or not a val_eval record, 2 = usage error.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.val_eval import EVAL_SCHEMA, compare_markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", type=Path, nargs="+", help="val_eval_record.json files, in column order")
    parser.add_argument("--out", type=Path, help="write the markdown here instead of printing it")
    args = parser.parse_args()

    records = []
    for p in args.records:
        if not p.is_file():
            print(f"{p}: not found")
            sys.exit(1)
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("schema") != EVAL_SCHEMA:
            print(f"{p}: not a val_eval record (schema {r.get('schema')!r})")
            sys.exit(1)
        records.append(r)
    md = compare_markdown(records)
    if args.out:
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
