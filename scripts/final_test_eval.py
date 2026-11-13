#!/usr/bin/env python
"""FINAL TEST EVALUATION of a finished, pre-declared training run -- Colab / lab GPU only.

    python scripts/final_test_eval.py --experiment E001 \
        --runs-root /content/drive/MyDrive/uwct/runs --confirm-final-test [--target test_zenodo_sk]

The only entry point that evaluates on the test split (CLAUDE.md rule 7). Refuses unless the
experiment is declared in experiments/final_candidates.csv at the checked-out (clean) commit, the
run finished with a valid run_record.json and unchanged best.pt, the dataset view verifies, and no
completed final test exists for the run. The operating confidence is chosen on val first; then
best.pt is evaluated with the full data.yaml and split="test". Outputs go to
<runs-root>/<id>/final_test/ on Drive. Log the result in docs/EXPERIMENTS.md afterwards.
Exit code: 0 = evaluated, 1 = refused (test split not evaluated), 2 = usage error.
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent / "src"))

import env_report  # scripts/env_report.py (this script's directory is on sys.path)
from dataset_audit.versioning import VersionError
from training.experiment import ConfigError
from training.final_test import FINAL_RECORD, TARGETS, FinalTestError, run_final_test
from training.train import TrainError


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="experiment id, e.g. E001")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="default: data/yolo/lobster_<dataset_version>")
    parser.add_argument("--target", choices=sorted(TARGETS), default="test",
                        help="test = v1 test sessions (data.yaml, default); test_zenodo_sk = held-out Zenodo sk "
                             "site of Model A views (data_test_zenodo_sk.yaml, D-043). Each target is evaluated once.")
    parser.add_argument("--confirm-final-test", action="store_true",
                        help="required: acknowledges this is the one final evaluation on the test split")
    args = parser.parse_args()
    if not args.confirm_final_test:
        parser.error("refusing to touch the test split without --confirm-final-test")

    try:
        record = run_final_test(args.runs_root, args.experiment, env_collect=env_report.collect,
                                dataset_dir=args.dataset, target=args.target)
    except (FinalTestError, TrainError, ConfigError, VersionError) as e:
        print(f"FINAL TEST REFUSED (test split not evaluated): {e}")
        sys.exit(1)

    m = record["metrics"]
    print("=" * 72 + f"\nFINAL TEST EVALUATION -- {record['experiment_id']} -- {record['target']} -- RESULT\n" + "=" * 72)
    print(json.dumps({"ultralytics_reported": {k: m["ultralytics_reported"][k] for k in
                                               ("precision", "recall", "map50", "map50_95")},
                      "at_val_conf": m["at_val_conf"]}, indent=2))
    print(f"record: {Path(args.runs_root) / args.experiment / TARGETS[args.target]['out_dir'] / FINAL_RECORD}")
    print("NEXT: log this evaluation in docs/EXPERIMENTS.md (Test-evaluation log). Results never "
          "feed back into training, thresholds or model choice.")


if __name__ == "__main__":
    main()
