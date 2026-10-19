#!/usr/bin/env python
"""VAL-ONLY evaluation of a finished training run -- Colab / lab GPU only (D-020).

    python scripts/val_eval.py --experiment E001 --runs-root /content/drive/MyDrive/uwct/runs

Never touches the test split: Ultralytics gets the run's train/val-only data file and predictions
run on images/val/ only. Refuses unless the checkout is clean, the run finished with a valid
run_record.json and an unchanged best.pt, and the dataset view verifies. Outputs go to
<runs-root>/<id>/val_eval/ on Drive; copy val_eval_record.json to experiments/records/<id>/.
Exit code: 0 = evaluated, 1 = refused, 2 = usage error.
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
from training.train import TrainError
from training.val_eval import EVAL_DIR, EVAL_RECORD, ValEvalError, run_val_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="experiment id, e.g. E001")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="default: data/yolo/lobster_<dataset_version>")
    args = parser.parse_args()

    try:
        record = run_val_eval(args.runs_root, args.experiment, env_collect=env_report.collect,
                              dataset_dir=args.dataset)
    except (ValEvalError, TrainError, ConfigError, VersionError) as e:
        print(f"VAL EVAL REFUSED: {e}")
        sys.exit(1)

    o = record["overall"]
    print(f"VAL-ONLY EVALUATION -- {record['experiment_id']}")
    print(json.dumps({"operating_conf": round(record["operating_conf"], 4),
                      **{k: round(o[k], 4) if isinstance(o[k], float) else o[k] for k in
                         ("ap50", "precision", "recall", "f1", "tp", "fp", "fn",
                          "negative_images_with_detection", "detections_on_negative_images")}}, indent=1))
    print(f"record: {Path(args.runs_root) / args.experiment / EVAL_DIR / EVAL_RECORD}")


if __name__ == "__main__":
    main()
