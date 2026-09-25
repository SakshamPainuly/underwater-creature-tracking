"""FINAL TEST EVALUATION -- the only code in the repository that evaluates a model on the test split.

CLAUDE.md rule 7: the test split is never used for training, tuning, threshold choice, early
stopping or model selection; only pre-declared final candidates are evaluated on it, and every
test evaluation is logged in docs/EXPERIMENTS.md. Training (train.py) never imports this module
and never sees a data file that lists the test split.

Preconditions, all enforced before the test split is touched:
  1. Colab / lab GPU environment (same guard as training), clean git checkout.
  2. The experiment is declared in experiments/final_candidates.csv at the checked-out commit,
     i.e. the declaration was committed before the evaluation ran.
  3. <runs_root>/<id>/run_record.json is a valid record of a finished *full* run, and
     train/weights/best.pt still has the sha256 recorded there.
  4. The dataset view passes the same verification as training, and its fingerprint equals the
     one recorded for the training run; Ultralytics reads the test split with the built counts.
  5. No completed final test exists for this run (one evaluation per run; interrupted attempts are
     counted in attempts.jsonl and reported).

Order of work:
  a. Operating confidence chosen on VAL: best.pt on the run's train/val data file, confidence that
     maximizes F1 on val. Frozen before step b.
  b. FINAL TEST: best.pt on the full data.yaml with split="test" (metrics + Ultralytics plots).
  c. Prediction examples: one test image per test session, drawn at the val-chosen confidence.

Outputs (Drive, outside Git): <run>/final_test/{attempts.jsonl, val_threshold/, test/,
predictions/, final_test_record.json}.
"""

import csv
import json
from pathlib import Path

from dataset_audit.config import BIOLOGY_ROOT, PROJECT_ROOT
from dataset_audit.versioning import sha256_file
from training import experiment as X
from training import run_record as R

TEST_SPLIT = "test"
FINAL_DIR = "final_test"
FINAL_RECORD = "final_test_record.json"
ATTEMPTS_FILE = "attempts.jsonl"
FINAL_SCHEMA = "uwct.final_test_record/1"
FINAL_CANDIDATES_PATH = PROJECT_ROOT / "experiments" / "final_candidates.csv"
FINAL_CANDIDATE_FIELDS = ("experiment_id", "declared_utc", "decision", "reason")
FULL_DATA_KEYS = {"path", "train", "val", "test", "names"}
FINAL_TEST_LIMITATIONS = (
    "v1 test sessions share sites (mostly also dates) with train sessions: this measures "
    "generalization to new sessions, not new sites (D-007).",
    "Test boxes are concentrated: 13_SHRI_12 + 21_SUB6_03 hold 1,135 of 1,512 (D-008).",
    "Substrate/visibility metadata is mostly missing; zExtra is excluded.",
    "ultralytics_reported precision/recall are at the confidence that maximizes F1 on the TEST split "
    "itself (Ultralytics convention); the operating point is at_val_conf, chosen on val.",
)


class FinalTestError(RuntimeError):
    pass


# ---------------------------------------------------------------- stdlib checks (tested locally)

def declared_candidates(path: Path = FINAL_CANDIDATES_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FINAL_CANDIDATE_FIELDS:
            raise FinalTestError(f"{path} header must be {','.join(FINAL_CANDIDATE_FIELDS)}")
        return {r["experiment_id"]: r for r in reader}


def require_declared(exp_id: str, path: Path = FINAL_CANDIDATES_PATH) -> dict:
    row = declared_candidates(path).get(exp_id)
    if row is None:
        raise FinalTestError(
            f"{exp_id} is not declared in {Path(path).name}. Declare it (row + docs/EXPERIMENTS.md), "
            "commit and push, then run the final test from that commit (CLAUDE.md rule 7).")
    if not row.get("declared_utc") or not row.get("decision"):
        raise FinalTestError(f"{exp_id}: declaration needs declared_utc and a decision id")
    return row


def check_run_for_final_test(run_dir: Path) -> dict:
    """The finished full run to evaluate; refuses smoke runs, bad records, changed weights, repeats."""
    run_dir = Path(run_dir)
    record_path = run_dir / "run_record.json"
    if not record_path.is_file():
        raise FinalTestError(f"{record_path} not found; finish (or --finalize-only) the training run first")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    problems = R.validate_record(record)
    if record.get("kind") != "full":
        problems.append("only full training runs are evaluated on test (this is a smoke run)")
    if problems:
        raise FinalTestError("run record not usable: " + "; ".join(problems))
    best = run_dir / "train" / "weights" / "best.pt"
    if not best.is_file() or sha256_file(best) != record["outputs"]["best"]["sha256"]:
        raise FinalTestError(f"{best} is missing or differs from the sha256 in run_record.json")
    if (run_dir / FINAL_DIR / FINAL_RECORD).is_file():
        raise FinalTestError(f"{run_dir.name} already has a completed final test; it is evaluated once")
    return record


def assert_full_data_yaml(data_yaml: Path) -> None:
    keys = X.data_yaml_keys(data_yaml)
    if keys != FULL_DATA_KEYS:
        raise FinalTestError(f"{data_yaml} must define {sorted(FULL_DATA_KEYS)} (found {sorted(keys)})")


def _mean_curve(curve) -> list:
    """Mean over classes of a (classes x points) curve; accepts a flat list for one class."""
    rows = curve if curve and isinstance(curve[0], (list, tuple)) else [curve]
    return [sum(col) / len(rows) for col in zip(*rows)]


def operating_conf(px, f1_curve) -> float:
    """Confidence maximizing the (class-mean) F1 curve -- used on VAL only."""
    f1 = _mean_curve(f1_curve)
    if not f1 or len(f1) != len(px):
        raise FinalTestError("F1 curve unavailable or inconsistent; cannot choose the operating point")
    return float(px[max(range(len(f1)), key=f1.__getitem__)])


def value_at(px, curve, conf: float) -> float:
    values = _mean_curve(curve)
    i = min(range(len(px)), key=lambda k: abs(px[k] - conf))
    return float(values[i])


def select_prediction_examples(build_rows: list) -> list:
    """One test image per test session (the middle one by filename); deterministic."""
    by_session = {}
    for r in build_rows:
        if r["split"] == TEST_SPLIT:
            by_session.setdefault(r["session_id"], []).append(r["filename"])
    return [sorted(files)[len(files) // 2] for _, files in sorted(by_session.items())]


def build_final_record(*, run_record: dict, declaration: dict, git: dict, dataset: dict,
                       test_counts: dict, conf: dict, metrics: dict, artifacts: list,
                       attempts: int, environment: dict) -> dict:
    record = {
        "schema": FINAL_SCHEMA,
        "created_utc": R.utc_now(),
        "label": "FINAL TEST EVALUATION",
        "split": TEST_SPLIT,
        "experiment_id": run_record["experiment"]["id"],
        "run_name": run_record["run_name"],
        "declaration": declaration,
        "training_git_commit": run_record["git"]["commit"],
        "evaluation_git": git,
        "run_record_best_sha256": run_record["outputs"]["best"]["sha256"],
        "dataset": {**dataset, "test_counts": test_counts},
        "operating_conf": conf,
        "metrics": metrics,
        "artifacts": artifacts,
        "attempts_including_this_one": attempts,
        "environment": environment,
        "limitations": list(FINAL_TEST_LIMITATIONS),
        "log_to": "docs/EXPERIMENTS.md -> Test-evaluation log",
    }
    missing = [k for k in ("precision", "recall", "map50", "map50_95") if not isinstance(
        metrics.get("ultralytics_reported", {}).get(k), (int, float))]
    if missing:
        raise FinalTestError(f"final test metrics incomplete: {missing}")
    return record


# ---------------------------------------------------------------- GPU side (Colab / lab only)

def _tolist(x):
    return x.tolist() if hasattr(x, "tolist") else x


def _box_metrics(result) -> dict:
    box = result.box
    return {
        "precision": float(box.mp), "recall": float(box.mr),
        "map50": float(box.map50), "map50_95": float(box.map),
        "results_dict": {k: float(v) for k, v in result.results_dict.items()},
    }


def _curves(result) -> dict | None:
    box = result.box
    px, p, r, f1 = (getattr(box, a, None) for a in ("px", "p_curve", "r_curve", "f1_curve"))
    if any(v is None for v in (px, p, r, f1)):
        return None
    return {"px": _tolist(px), "p": _tolist(p), "r": _tolist(r), "f1": _tolist(f1)}


def _append_attempt(out_dir: Path, event: dict) -> int:
    path = out_dir / ATTEMPTS_FILE
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": R.utc_now(), **event}) + "\n")
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if '"event": "start"' in line)


def run_final_test(runs_root: Path, exp_id: str, env_collect, dataset_dir: Path | None = None) -> dict:
    from training.train import model_class, require_gpu_environment  # training never imports this module
    from yolo_prep.verify import counts_match, ultralytics_split_counts

    ultralytics = require_gpu_environment()
    runs_root = X.check_output_location(runs_root, PROJECT_ROOT, BIOLOGY_ROOT)
    run_dir = runs_root / exp_id

    git = R.git_state(PROJECT_ROOT)
    if not git["commit"] or git["dirty"]:
        raise FinalTestError("the final test needs a clean git checkout at a recorded commit")
    declaration = require_declared(exp_id)
    run_record = check_run_for_final_test(run_dir)
    cfg = X.validate_config(run_record["experiment"]["config"])
    if ultralytics.__version__ != run_record["ultralytics_version"]:
        raise FinalTestError(f"ultralytics {ultralytics.__version__} != training {run_record['ultralytics_version']}")

    dataset_dir = Path(dataset_dir or PROJECT_ROOT / "data" / "yolo" / f"lobster_{cfg['dataset_version']}")
    dataset = X.verify_built_dataset(dataset_dir, cfg)
    if dataset["fingerprint"] != run_record["dataset"]["fingerprint"]:
        raise FinalTestError("dataset fingerprint differs from the one the model was trained with")
    full_yaml = dataset_dir / "data.yaml"
    assert_full_data_yaml(full_yaml)
    summary = json.loads((dataset_dir / "build_summary.json").read_text(encoding="utf-8"))
    expected_test = summary["splits"][TEST_SPLIT]
    data, seen = ultralytics_split_counts(full_yaml, [TEST_SPLIT])
    if data["names"] != dict(enumerate(X.CLASS_NAMES)) or not counts_match(expected_test, seen[TEST_SPLIT]):
        raise FinalTestError(f"Ultralytics reads the test split differently: {seen[TEST_SPLIT]} vs {expected_test}")

    Model = model_class(cfg["model"])  # RTDETR or YOLO: same family as training

    out_dir = run_dir / FINAL_DIR
    out_dir.mkdir(exist_ok=True)
    best = str(run_dir / "train" / "weights" / "best.pt")
    imgsz = run_record["train_args"]["imgsz"]
    batch = run_record["train_args"].get("batch", 16)

    # a. operating confidence from VAL, with the run's train/val-only data file
    trainval = run_dir / "data_trainval.yaml"
    X.assert_trainval_only(trainval)
    val_result = Model(best).val(data=str(trainval), split=X.VAL_SPLIT, imgsz=imgsz, batch=batch,
                                project=str(out_dir), name="val_threshold", exist_ok=True, plots=False)
    val_curves = _curves(val_result)
    if val_curves is None:
        raise FinalTestError("Ultralytics did not expose P/R/F1 curves; cannot fix the operating point on val")
    conf = operating_conf(val_curves["px"], val_curves["f1"])

    # b. FINAL TEST EVALUATION
    attempts = _append_attempt(out_dir, {"event": "start", "git_commit": git["commit"], "conf_from_val": conf})
    print("=" * 72 + f"\nFINAL TEST EVALUATION -- {exp_id} -- split={TEST_SPLIT!r} -- attempt {attempts}\n" + "=" * 72)
    test_result = Model(best).val(data=str(full_yaml), split=TEST_SPLIT, imgsz=imgsz, batch=batch,
                                 project=str(out_dir), name="test", exist_ok=True, plots=True)
    test_curves = _curves(test_result)
    cm = getattr(test_result, "confusion_matrix", None)

    # c. prediction examples at the val-chosen confidence
    with open(dataset_dir / "build_manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    examples = select_prediction_examples(rows)
    Model(best).predict(source=[str(dataset_dir / "images" / TEST_SPLIT / f) for f in examples], conf=conf,
                       imgsz=imgsz, save=True, project=str(out_dir), name="predictions", exist_ok=True)
    _append_attempt(out_dir, {"event": "end"})

    metrics = {
        "ultralytics_reported": _box_metrics(test_result),
        "at_val_conf": None if test_curves is None else {
            "conf": conf,
            "precision": value_at(test_curves["px"], test_curves["p"], conf),
            "recall": value_at(test_curves["px"], test_curves["r"], conf),
            "f1": value_at(test_curves["px"], test_curves["f1"], conf),
        },
        "confusion_matrix": None if cm is None else {
            "matrix": _tolist(cm.matrix), "axes": "rows = predicted, cols = true; classes + background (Ultralytics)",
        },
        "speed_ms_per_image": getattr(test_result, "speed", None),
        "val_for_threshold": _box_metrics(val_result),
    }
    artifacts = [R.file_entry(p) for p in sorted(out_dir.rglob("*"))
                 if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".csv", ".json")]
    record = build_final_record(
        run_record=run_record, declaration=declaration, git=git, dataset=dataset, test_counts=expected_test,
        conf={"value": conf, "chosen_on": X.VAL_SPLIT, "rule": "max mean F1 on val"},
        metrics=metrics, artifacts=artifacts, attempts=attempts, environment=env_collect(),
    )
    (out_dir / FINAL_RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
