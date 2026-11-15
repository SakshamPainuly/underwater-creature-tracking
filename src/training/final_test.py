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
     train/weights/best.pt still has the sha256 recorded there -- or, where a decision replaced lost
     weights (experiments/records/<id>/weights_recovery.json, D-027), the sha256 named there; such
     weights must then reproduce the recorded val metrics in step a before the test split is touched.
  4. The dataset view passes the same verification as training, and its fingerprint equals the
     one recorded for the training run; Ultralytics reads the test split with the built counts.
  5. No completed final test exists for this run and target (one evaluation per run and target;
     interrupted attempts are counted in attempts.jsonl and reported).

Targets (D-043): "test" = the v1 test sessions in data.yaml (every dataset); "test_zenodo_sk" = the
held-out Zenodo sk site of Model A mixed views (v4, D-031/D-032), in its own data_test_zenodo_sk.yaml
whose test entry is images/test_zenodo_sk, so the two scores never mix. Ultralytics always reads the
data file's "test" entry; each target has its own output folder and once-only guard.

Order of work:
  a. Operating confidence chosen on VAL: best.pt on the run's train/val data file, confidence that
     maximizes F1 on val. Frozen before step b.
  b. FINAL TEST: best.pt on the full data.yaml with split="test" (metrics + Ultralytics plots).
  c. Prediction examples: one test image per test session, drawn at the val-chosen confidence.

Outputs (Drive / Kaggle, outside Git): <run>/final_test/ (target "test") or
<run>/final_test_zenodo_sk/, each with {attempts.jsonl, val_threshold/, test/, predictions/,
final_test_record.json}.
"""

import csv
import json
from pathlib import Path

from dataset_audit.config import PROJECT_ROOT, RAW_ROOTS
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
RECORDS_DIR = PROJECT_ROOT / "experiments" / "records"
RECOVERY_FILE = "weights_recovery.json"
RECOVERY_FIELDS = ("experiment_id", "decision", "registered_best_sha256", "path", "sha256", "val_tolerance")
VAL_CHECK_METRICS = ("precision", "recall", "map50", "map50_95")
FULL_DATA_KEYS = {"path", "train", "val", "test", "names"}
FINAL_TEST_LIMITATIONS = (
    "v1 test sessions share sites (mostly also dates) with train sessions: this measures "
    "generalization to new sessions, not new sites (D-007).",
    "Test boxes are concentrated: 13_SHRI_12 + 21_SUB6_03 hold 1,135 of 1,512 (D-008).",
    "Substrate/visibility metadata is mostly missing; zExtra is excluded.",
    "ultralytics_reported precision/recall are at the confidence that maximizes F1 on the TEST split "
    "itself (Ultralytics convention); the operating point is at_val_conf, chosen on val.",
)
ZENODO_SK_LIMITATIONS = (
    "Zenodo sk is an external new site (other country, camera setup and image size; D-031): the only "
    "new-site measurement, but one site only.",
    "sk labels come from the Zenodo dataset, not from our annotators; 428 images, of which 143 contain a "
    "lobster (143 boxes, D-032), so false positives on lobster-free images weigh heavily.",
    "Training used Zenodo mpa (the other Zenodo site) but no sk image or sk-derived image (D-031).",
    "ultralytics_reported precision/recall are at the confidence that maximizes F1 on sk itself "
    "(Ultralytics convention); the operating point is at_val_conf, chosen on our val split.",
)
TARGETS = {  # target -> data file in the view, build split (images/<split>), output folder, limitations
    "test": {"data_file": "data.yaml", "build_split": "test", "out_dir": FINAL_DIR,
             "limitations": FINAL_TEST_LIMITATIONS},
    "test_zenodo_sk": {"data_file": "data_test_zenodo_sk.yaml", "build_split": "test_zenodo_sk",
                       "out_dir": "final_test_zenodo_sk", "limitations": ZENODO_SK_LIMITATIONS},
}


class FinalTestError(RuntimeError):
    pass


# ---------------------------------------------------------------- stdlib checks (tested locally)

def target_spec(target: str) -> dict:
    if target not in TARGETS:
        raise FinalTestError(f"unknown final-test target {target!r}; choose one of {sorted(TARGETS)}")
    return TARGETS[target]


def expected_target_counts(summary: dict, target: str) -> dict:
    """The built counts of the target's split from build_summary.json (eval-only splits for sk)."""
    split = target_spec(target)["build_split"]
    recorded = {**summary.get("splits", {}), **summary.get("eval_only_splits", {})}
    if split not in recorded:
        raise FinalTestError(f"this dataset view has no {split!r} split (target {target!r} needs a view "
                             "that builds it, e.g. Model A v4)")
    return {k: v for k, v in recorded[split].items() if k != "data_file"}


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


def load_weights_recovery(exp_id: str, records_dir: Path = RECORDS_DIR) -> dict | None:
    """The committed stand-in for a registered best.pt that was lost (D-027), or None if there is none."""
    path = Path(records_dir) / exp_id / RECOVERY_FILE
    if not path.is_file():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in RECOVERY_FIELDS if rec.get(k) in (None, "")]
    if missing or rec["experiment_id"] != exp_id:
        raise FinalTestError(f"{path}: needs {list(RECOVERY_FIELDS)} for {exp_id} (missing {missing})")
    return rec


def check_run_for_final_test(run_dir: Path, recovery: dict | None = None, out_dirname: str = FINAL_DIR) -> dict:
    """The finished full run to evaluate; refuses smoke runs, bad records, changed weights, repeats
    (a repeat = a completed record in out_dirname, the target's output folder)."""
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
    sha = sha256_file(best) if best.is_file() else None
    if recovery is None:
        if sha != record["outputs"]["best"]["sha256"]:
            raise FinalTestError(f"{best} is missing or differs from the sha256 in run_record.json")
    elif recovery["registered_best_sha256"] != record["outputs"]["best"]["sha256"]:
        raise FinalTestError(f"{RECOVERY_FILE} ({recovery['decision']}) replaces a different best.pt than this record's")
    elif sha != recovery["sha256"]:
        raise FinalTestError(f"{best} is missing or differs from the sha256 in {RECOVERY_FILE} ({recovery['decision']})")
    if (run_dir / out_dirname / FINAL_RECORD).is_file():
        raise FinalTestError(f"{run_dir.name} already has a completed final test in {out_dirname}/; "
                             "each target is evaluated once")
    return record


def check_recovered_val(val_metrics: dict, run_record: dict, tolerance: float) -> dict:
    """Recovered weights must reproduce the registered val metrics (D-027); returns |difference| per metric."""
    diffs = {k: abs(float(val_metrics[k]) - float(run_record["val_metrics"][k])) for k in VAL_CHECK_METRICS}
    off = {k: d for k, d in diffs.items() if d > tolerance}
    if off:
        raise FinalTestError(f"recovered weights do not reproduce the registered val metrics (tolerance "
                             f"{tolerance}): {off} -- they are not the registered model; test split not evaluated")
    return diffs


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


def select_prediction_examples(build_rows: list, split: str = TEST_SPLIT) -> list:
    """One image per session of the split (the middle one by filename); deterministic. Mixed views
    (v4) have no session_id column; their group_id (session or source group) is used instead."""
    by_session = {}
    for r in build_rows:
        if r["split"] == split:
            group = r.get("session_id") or r.get("group_id") or ""
            by_session.setdefault(group, []).append(r["filename"])
    return [sorted(files)[len(files) // 2] for _, files in sorted(by_session.items())]


def build_final_record(*, run_record: dict, declaration: dict, git: dict, dataset: dict,
                       test_counts: dict, conf: dict, metrics: dict, artifacts: list,
                       attempts: int, environment: dict, weights_recovery: dict | None = None,
                       target: str = "test") -> dict:
    spec = target_spec(target)
    record = {
        "schema": FINAL_SCHEMA,
        "created_utc": R.utc_now(),
        "label": "FINAL TEST EVALUATION",
        "split": TEST_SPLIT,
        "target": target,
        "data_file": spec["data_file"],
        "experiment_id": run_record["experiment"]["id"],
        "run_name": run_record["run_name"],
        "declaration": declaration,
        "training_git_commit": run_record["git"]["commit"],
        "evaluation_git": git,
        "run_record_best_sha256": run_record["outputs"]["best"]["sha256"],
        "evaluated_best_sha256": (weights_recovery or {}).get("sha256", run_record["outputs"]["best"]["sha256"]),
        "weights_recovery": weights_recovery,
        "dataset": {**dataset, "test_counts": test_counts},
        "operating_conf": conf,
        "metrics": metrics,
        "artifacts": artifacts,
        "attempts_including_this_one": attempts,
        "environment": environment,
        "limitations": list(spec["limitations"]),
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


def run_final_test(runs_root: Path, exp_id: str, env_collect, dataset_dir: Path | None = None,
                   target: str = "test") -> dict:
    from training.train import model_class, require_gpu_environment  # training never imports this module
    from yolo_prep.verify import counts_match, ultralytics_split_counts

    spec = target_spec(target)
    ultralytics = require_gpu_environment()
    runs_root = X.check_output_location(runs_root, PROJECT_ROOT, RAW_ROOTS)
    run_dir = runs_root / exp_id

    git = R.git_state(PROJECT_ROOT)
    if not git["commit"] or git["dirty"]:
        raise FinalTestError("the final test needs a clean git checkout at a recorded commit")
    declaration = require_declared(exp_id)
    recovery = load_weights_recovery(exp_id)
    run_record = check_run_for_final_test(run_dir, recovery, spec["out_dir"])
    cfg = X.validate_config(run_record["experiment"]["config"])
    if ultralytics.__version__ != run_record["ultralytics_version"]:
        raise FinalTestError(f"ultralytics {ultralytics.__version__} != training {run_record['ultralytics_version']}")

    dataset_dir = Path(dataset_dir or PROJECT_ROOT / "data" / "yolo" / f"lobster_{cfg['dataset_version']}")
    dataset = X.verify_built_dataset(dataset_dir, cfg)
    if dataset["fingerprint"] != run_record["dataset"]["fingerprint"]:
        raise FinalTestError("dataset fingerprint differs from the one the model was trained with")
    full_yaml = dataset_dir / spec["data_file"]
    assert_full_data_yaml(full_yaml)
    summary = json.loads((dataset_dir / "build_summary.json").read_text(encoding="utf-8"))
    expected_test = expected_target_counts(summary, target)
    data, seen = ultralytics_split_counts(full_yaml, [TEST_SPLIT])
    if data["names"] != dict(enumerate(X.CLASS_NAMES)) or not counts_match(expected_test, seen[TEST_SPLIT]):
        raise FinalTestError(f"Ultralytics reads the test split differently: {seen[TEST_SPLIT]} vs {expected_test}")

    Model = model_class(cfg["model"])  # RTDETR or YOLO: same family as training

    out_dir = run_dir / spec["out_dir"]
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
    if recovery is not None:
        diffs = check_recovered_val(_box_metrics(val_result), run_record, float(recovery["val_tolerance"]))
        recovery = {**recovery, "val_check_abs_diff": diffs}
        print(f"recovered weights ({recovery['decision']}) reproduce the registered val metrics: {diffs}")

    # b. FINAL TEST EVALUATION
    attempts = _append_attempt(out_dir, {"event": "start", "git_commit": git["commit"], "conf_from_val": conf,
                                         "target": target})
    print("=" * 72 + f"\nFINAL TEST EVALUATION -- {exp_id} -- target={target!r} ({spec['data_file']}, "
          f"split={TEST_SPLIT!r}) -- attempt {attempts}\n" + "=" * 72)
    test_result = Model(best).val(data=str(full_yaml), split=TEST_SPLIT, imgsz=imgsz, batch=batch,
                                 project=str(out_dir), name="test", exist_ok=True, plots=True)
    test_curves = _curves(test_result)
    cm = getattr(test_result, "confusion_matrix", None)

    # c. prediction examples at the val-chosen confidence
    with open(dataset_dir / "build_manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    examples = select_prediction_examples(rows, spec["build_split"])
    Model(best).predict(source=[str(dataset_dir / "images" / spec["build_split"] / f) for f in examples], conf=conf,
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
        weights_recovery=recovery, target=target,
    )
    (out_dir / FINAL_RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
