#!/usr/bin/env python
"""Train a YOLO detector from an experiment config -- Colab / lab GPU only.

    python scripts/train_yolo.py --config configs/experiments/E001_yolo_baseline.yaml \
        --runs-root /content/drive/MyDrive/uwct/runs --weights-dir /content/weights --preflight-only
    ... --smoke          1 epoch on 10% of train, validated on val (pipeline check)
    ... (no flag)        the configured training run -> <runs-root>/<id>/
    ... --resume         continue <runs-root>/<id>/ from its last.pt after a disconnect
    ... --finalize-only  recompute val metrics + run_record.json for a finished run

Refuses to run on Windows or without a CUDA GPU. Before training it verifies: pinned ultralytics
version, config (stdlib validator == PyYAML parse), clean git checkout, frozen dataset view
(hashes + fingerprint), Ultralytics' own reading of train/val, and the exact pretrained checkpoint.
Training sees a data file without the test split. See docs/COLAB.md.
Exit code: 0 = ok, 1 = a check failed (nothing trained), 2 = usage error.
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
from training.run_record import RecordError
from training.train import TrainError, run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True,
                        help="persistent run storage outside the repo, e.g. /content/drive/MyDrive/uwct/runs")
    parser.add_argument("--weights-dir", type=Path, required=True,
                        help="where Ultralytics downloads pretrained checkpoints, outside the repo")
    parser.add_argument("--dataset", type=Path, help="default: data/yolo/<lobster|tritonia>_<dataset_version>")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", dest="mode", action="store_const", const="preflight")
    mode.add_argument("--smoke", dest="mode", action="store_const", const="smoke")
    mode.add_argument("--resume", dest="mode", action="store_const", const="resume")
    mode.add_argument("--finalize-only", dest="mode", action="store_const", const="finalize")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="preflight/smoke only: accept uncommitted changes (recorded)")
    args = parser.parse_args()

    try:
        result = run(args.config, args.runs_root, args.weights_dir, args.mode or "full",
                     env_collect=env_report.collect, dataset_dir=args.dataset, allow_dirty=args.allow_dirty)
    except (TrainError, ConfigError, VersionError, RecordError) as e:
        print(f"STOPPED: {e}")
        sys.exit(1)

    print(json.dumps(result, indent=2, default=str))
    if (args.mode or "full") == "preflight":
        print(f"PREFLIGHT OK: {result['model']['name']} sha256={result['model']['checkpoint_sha256']}")
    else:
        print(f"RUN RECORD: {Path(result['run_dir']) / 'run_record.json'}")


if __name__ == "__main__":
    main()
