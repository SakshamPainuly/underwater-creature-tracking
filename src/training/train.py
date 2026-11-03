"""Ultralytics training runner (YOLO and RT-DETR) -- Colab / lab GPU only (CLAUDE.md rule 3, section 6).

torch, ultralytics and yaml are imported inside functions; importing this module locally is
harmless, but every entry point refuses to run on Windows or without a CUDA GPU.

Modes (scripts/train_yolo.py):
    preflight  all checks, loads (downloads) the pretrained checkpoint, Ultralytics label scan of
               train/val; writes nothing to the runs folder
    smoke      preflight + 1 epoch on 10% of train, validated on val; run folder <id>_smoke_<utc>
    full       preflight + the configured training; run folder <runs_root>/<id> (must not exist)
    resume     continue <runs_root>/<id> from train/weights/last.pt; same commit, config, dataset
    finalize   (re)compute val metrics and run_record.json for a finished run

Run folder layout (<runs_root> is on Google Drive, outside the repository):
    <run>/config.yaml            copy of the experiment config
    <run>/data_trainval.yaml     Ultralytics data file with train + val only (no test entry)
    <run>/preflight.json         everything checked before the first launch
    <run>/launches.jsonl         start/end of each launch (resumes after Colab disconnects)
    <run>/train/                 Ultralytics output: args.yaml, results.csv, weights/{best,last}.pt
    <run>/val_best/              Ultralytics validation of best.pt on the val split
    <run>/run_record.json        CLAUDE.md section 5 run record
"""

import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

from dataset_audit.config import PROJECT_ROOT, RAW_ROOTS
from dataset_audit.versioning import sha256_file
from training import experiment as X
from training import run_record as R
from yolo_prep.verify import counts_match, ultralytics_split_counts

MODES = ("preflight", "smoke", "full", "resume", "finalize")
TRAIN_SUBDIR = "train"
VAL_SUBDIR = "val_best"
PREFLIGHT_FILE = "preflight.json"
LAUNCHES_FILE = "launches.jsonl"
RECORD_FILE = "run_record.json"
TRAINVAL_FILE = "data_trainval.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments" / "registry.csv"


class TrainError(RuntimeError):
    pass


def require_gpu_environment():
    """Refuse local Windows / CPU-only machines; return the ultralytics module."""
    if platform.system() == "Windows":
        raise TrainError("training never runs on the local Windows PC (CLAUDE.md rule 3); use Colab / the lab GPU")
    try:
        import torch
        import ultralytics
        import yaml  # noqa: F401  (installed with ultralytics; used for the config cross-check)
    except ImportError as e:
        raise TrainError(f"{e.name} is not installed: run `pip install -r requirements-gpu.txt` on the GPU machine")
    if not torch.cuda.is_available():
        raise TrainError("no CUDA GPU visible (Colab: Runtime > Change runtime type > GPU)")
    return ultralytics


def load_config_checked(path: Path) -> tuple:
    """Stdlib parse + validation, cross-checked against PyYAML: both must read the same values."""
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    ours = X.parse_config_text(text)
    theirs = yaml.safe_load(text)
    if json.dumps(ours, sort_keys=True) != json.dumps(theirs, sort_keys=True):
        raise TrainError(f"{path}: PyYAML reads this config differently from the local validator; "
                         "rewrite the differing values (see src/training/experiment.py)")
    X.validate_config(ours, config_file=Path(path).name)
    return ours, sha256_file(path)


def model_class(model_name: str):
    """Ultralytics class for the config's checkpoint family (RTDETR or YOLO); every load goes through it."""
    from ultralytics import RTDETR, YOLO

    family = X.model_family(model_name)
    if family is None:
        raise TrainError(f"{model_name}: not a supported model family ({X.MODEL_FAMILY_PREFIXES})")
    return RTDETR if family == "rtdetr" else YOLO


def load_pretrained(name: str, weights_dir: Path, ultralytics_version: str) -> tuple:
    """Load the exact official checkpoint `name`; never substitute another model."""
    try:
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES
        listed = name in GITHUB_ASSETS_NAMES
    except ImportError:
        listed = None
    weights_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(weights_dir)  # Ultralytics downloads checkpoints (incl. its AMP-check model) into the cwd
    try:
        model = model_class(name)(name)
    except Exception as e:  # noqa: BLE001 -- any failure means the checkpoint is unavailable
        raise TrainError(f"could not load {name} with ultralytics {ultralytics_version}: {e}\n"
                         "STOP: report this; no other model is substituted.")
    ckpt = Path(getattr(model, "ckpt_path", None) or weights_dir / name).resolve()
    if ckpt.name != name or not ckpt.is_file():
        raise TrainError(f"loaded checkpoint {ckpt} is not {name}; STOP, no substitution")
    info = {"name": name, "family": X.model_family(name), "checkpoint_path": str(ckpt),
            "checkpoint_sha256": sha256_file(ckpt), "in_ultralytics_asset_list": listed}
    return model, info


def _append_launch(run_dir: Path, event: dict) -> None:
    with open(run_dir / LAUNCHES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": R.utc_now(), **event}) + "\n")


def _read_launches(run_dir: Path) -> list:
    path = run_dir / LAUNCHES_FILE
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _epochs_completed(train_dir: Path) -> int | None:
    results = train_dir / "results.csv"
    if not results.is_file():
        return None
    lines = [l for l in results.read_text(encoding="utf-8").splitlines() if l.strip()]
    return max(len(lines) - 1, 0)


def validate_best(run_dir: Path, trainval_yaml: Path, train_args: dict, model_name: str) -> dict:
    """Validate best.pt on the val split -- the only split used here (CLAUDE.md rule 7)."""
    X.assert_trainval_only(trainval_yaml)
    best = run_dir / TRAIN_SUBDIR / "weights" / "best.pt"
    if not best.is_file():
        raise TrainError(f"{best} not found; training did not finish an epoch")
    metrics = model_class(model_name)(str(best)).val(
        data=str(trainval_yaml), split=X.VAL_SPLIT, imgsz=train_args["imgsz"],
        batch=train_args.get("batch", 16), project=str(run_dir), name=VAL_SUBDIR, exist_ok=True,
    )
    box = metrics.box
    return {
        "split": X.VAL_SPLIT,
        "weights": "best.pt",
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "ultralytics_results_dict": {k: float(v) for k, v in metrics.results_dict.items()},
        "note": "Ultralytics val defaults (conf=0.001, iou=0.7); operating threshold not chosen yet.",
    }


def finalize(run_dir: Path, preflight: dict, env_collect) -> dict:
    import yaml

    trainval = run_dir / TRAINVAL_FILE
    train_dir = run_dir / TRAIN_SUBDIR
    val = validate_best(run_dir, trainval, preflight["train_args"], preflight["model"]["name"])
    args_yaml = train_dir / "args.yaml"
    outputs = {
        "best": R.file_entry(train_dir / "weights" / "best.pt"),
        "last": R.file_entry(train_dir / "weights" / "last.pt"),
        "results_csv": R.file_entry(train_dir / "results.csv"),
        "ultralytics_args_yaml": R.file_entry(args_yaml),
    }
    record = R.build_record(
        kind=preflight["kind"], experiment=preflight["experiment"], run_name=preflight["run_name"],
        run_dir=run_dir, git=preflight["git"], dataset=preflight["dataset"], model=preflight["model"],
        ultralytics_version=preflight["ultralytics_version"], train_args=preflight["train_args"],
        ultralytics_args=yaml.safe_load(args_yaml.read_text(encoding="utf-8")) if args_yaml.is_file() else None,
        environment=preflight["environment"],
        launches=R.launch_summary(_read_launches(run_dir)), epochs_completed=_epochs_completed(train_dir),
        outputs=outputs, val_metrics=val,
    )
    record["environment_at_finalize"] = env_collect()
    R.write_record(record, run_dir / RECORD_FILE)
    return record


def run(config_path: Path, runs_root: Path, weights_dir: Path, mode: str, env_collect,
        dataset_dir: Path | None = None, allow_dirty: bool = False) -> dict:
    if mode not in MODES:
        raise TrainError(f"mode must be one of {MODES}")
    ultralytics = require_gpu_environment()
    config_path = Path(config_path).resolve()
    cfg, cfg_sha = load_config_checked(config_path)
    if ultralytics.__version__ != cfg["ultralytics_version"]:
        raise TrainError(f"ultralytics {ultralytics.__version__} installed, config pins "
                         f"{cfg['ultralytics_version']}: pip install -r requirements-gpu.txt")
    runs_root = X.check_output_location(runs_root, PROJECT_ROOT, RAW_ROOTS)
    weights_dir = X.check_output_location(weights_dir, PROJECT_ROOT, RAW_ROOTS)

    git = R.git_state(PROJECT_ROOT)
    if not git["commit"]:
        raise TrainError("not a git checkout: the exact commit must be recorded")
    if git["dirty"] and (mode in ("full", "resume", "finalize") or not allow_dirty):
        raise TrainError("the git working tree has uncommitted changes; full runs need a clean checkout "
                         "(--allow-dirty is accepted for preflight/smoke only and is recorded)")

    dataset_dir = Path(dataset_dir or X.default_dataset_dir(cfg["dataset_version"]))
    dataset = X.verify_built_dataset(dataset_dir, cfg)
    class_names = dataset.get("class_names", X.CLASS_NAMES)  # lobster views: ["lobster"]

    exp_id = cfg["id"]
    if mode in ("resume", "finalize"):
        run_dir = runs_root / exp_id
        preflight_path = run_dir / PREFLIGHT_FILE
        if not preflight_path.is_file():
            raise TrainError(f"{preflight_path} not found; nothing to {mode}")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        for what, now, then in (("config sha256", cfg_sha, preflight["experiment"]["config_sha256"]),
                                ("git commit", git["commit"], preflight["git"]["commit"]),
                                ("dataset fingerprint", dataset["fingerprint"], preflight["dataset"]["fingerprint"]),
                                ("ultralytics", ultralytics.__version__, preflight["ultralytics_version"])):
            if now != then:
                raise TrainError(f"cannot {mode}: {what} changed ({then} -> {now})")
        if mode == "finalize":
            return finalize(run_dir, preflight, env_collect)
        last = run_dir / TRAIN_SUBDIR / "weights" / "last.pt"
        if not last.is_file():
            raise TrainError(f"{last} not found; nothing to resume")
        os.chdir(weights_dir)
        _append_launch(run_dir, {"event": "start", "mode": "resume", "git_commit": git["commit"]})
        model_class(cfg["model"])(str(last)).train(resume=True)
        _append_launch(run_dir, {"event": "end", "mode": "resume"})
        return finalize(run_dir, preflight, env_collect)

    if mode == "full":
        if cfg["parent"] != "none" and cfg["parent"] not in R.registered_experiments(REGISTRY_PATH):
            raise TrainError(f"parent {cfg['parent']} is not in {REGISTRY_PATH.name} at this commit; "
                             f"register and review the parent run before launching {exp_id}")
        run_name = exp_id
    elif mode == "smoke":
        run_name = f"{exp_id}_smoke_{R.utc_now().replace(':', '').replace('-', '')[:15]}"
    else:
        run_name = f"{exp_id}_preflight"
    run_dir = runs_root / run_name
    if mode != "preflight" and run_dir.exists():
        raise TrainError(f"{run_dir} already exists; use --resume, or a new experiment id")
    # All checks run against a temporary data file; the run folder on Drive is created only after
    # every check passed, so a failed check never leaves a half-made run folder behind.
    tmp = tempfile.TemporaryDirectory()
    try:
        check_yaml = X.write_trainval_yaml(dataset_dir, Path(tmp.name) / TRAINVAL_FILE, class_names)
        data, seen = ultralytics_split_counts(check_yaml, X.TRAIN_SPLITS)
        verification = {s: {"seen": seen[s], "match": counts_match(dataset["counts"][s], seen[s])}
                        for s in X.TRAIN_SPLITS}
        if data["names"] != dict(enumerate(class_names)) or not all(v["match"] for v in verification.values()):
            raise TrainError(f"Ultralytics does not read the dataset as built: {verification}, names={data['names']}")
        dataset = {**dataset, "ultralytics_verification": verification}

        model, model_info = load_pretrained(cfg["model"], weights_dir, ultralytics.__version__)
        if mode != "preflight":
            run_dir.mkdir(parents=True)
            trainval = X.write_trainval_yaml(dataset_dir, run_dir / TRAINVAL_FILE, class_names)
        else:
            trainval = check_yaml
        train_args = X.build_train_args(cfg, trainval, run_dir, TRAIN_SUBDIR, smoke=(mode == "smoke"))
        preflight = {
            "kind": "smoke" if mode == "smoke" else "full",
            "created_utc": R.utc_now(),
            "run_name": run_name,
            "experiment": {"id": exp_id, "slug": cfg["slug"], "parent": cfg["parent"], "changed": cfg["changed"],
                           "config_path": config_path.relative_to(PROJECT_ROOT).as_posix(),
                           "config_sha256": cfg_sha, "config": cfg},
            "git": {**git, "allow_dirty": allow_dirty},
            "dataset": dataset,
            "model": model_info,
            "ultralytics_version": ultralytics.__version__,
            "train_args": train_args,
            "environment": env_collect(),
        }
        if mode == "preflight":
            return preflight

        shutil.copyfile(config_path, run_dir / "config.yaml")
        (run_dir / PREFLIGHT_FILE).write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
        _append_launch(run_dir, {"event": "start", "mode": mode, "git_commit": git["commit"]})
        model.train(**train_args)
        _append_launch(run_dir, {"event": "end", "mode": mode})
        return finalize(run_dir, preflight, env_collect)
    finally:
        tmp.cleanup()
