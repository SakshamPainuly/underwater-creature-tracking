"""run_record.json and experiments/registry.csv (CLAUDE.md section 5). Standard library only.

A run record is written by train.py on the GPU machine into the run folder (off-repo, on Drive).
It is registered on the local PC with scripts/register_run.py, which appends one row to
experiments/registry.csv -- so git work stays local and Colab never commits.
"""

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dataset_audit.versioning import sha256_file

SCHEMA = "uwct.run_record/1"
RUN_KINDS = ("full", "smoke")
REGISTRY_FIELDS = (
    "experiment_id", "run_name", "created_utc", "parent", "changed",
    "dataset_version", "dataset_fingerprint", "model", "model_sha256", "imgsz", "epochs_completed",
    "seed", "git_commit", "git_dirty", "ultralytics", "torch", "cuda", "gpu",
    "val_precision", "val_recall", "val_map50", "val_map50_95",
    "best_weights_sha256", "run_dir",
)
VAL_METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
V1_LIMITATIONS = (
    "v1 val/test sessions share sites (mostly also dates) with train sessions: results measure "
    "generalization to new sessions, not new sites (docs/DATASET.md, D-007).",
    "Substrate/visibility metadata is mostly missing; zExtra is excluded.",
    "Val is concentrated: 21_MACK_10 holds 465 of 1,010 val boxes (D-008).",
    "This record holds overall val metrics only; per-condition results and false positives on "
    "negative images come from the evaluation step.",
)

TRITONIA_LIMITATIONS = (
    "Tritonia v3 (D-030): 150 hand-boxed frames from one setup over 16 days, split by day; val and "
    "test measure new days, not new sites or cameras (docs/DATASET_TRITONIA.md).",
    "Frames are 15 s apart: sparse-frame/timelapse analysis, not video tracking. No calibration: no "
    "physical units.",
    "Boxes were drawn by one labeller (batch 1); MTrackJ points were hints only.",
    "This record holds overall val metrics only; per-class results come from a separate evaluation.",
)


def limitations_for(dataset: dict) -> list:
    return list(TRITONIA_LIMITATIONS if dataset.get("dataset_kind") == "tritonia" else V1_LIMITATIONS)


class RecordError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_state(repo_root: Path) -> dict:
    def git(*args):
        try:
            r = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def file_entry(path: Path) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    return {"path": str(p.resolve()), "size": p.stat().st_size, "sha256": sha256_file(p)}


def launch_summary(events: list) -> dict:
    """Pair launch start/end events (launches.jsonl). Unfinished launches were interrupted."""
    launches, current = [], None
    for e in events:
        if e["event"] == "start":
            if current is not None:
                launches.append({**current, "ended_utc": None, "seconds": None, "outcome": "interrupted"})
            current = {"mode": e["mode"], "started_utc": e["time"], "git_commit": e.get("git_commit")}
        elif e["event"] == "end" and current is not None:
            started = datetime.fromisoformat(current["started_utc"])
            seconds = (datetime.fromisoformat(e["time"]) - started).total_seconds()
            launches.append({**current, "ended_utc": e["time"], "seconds": seconds, "outcome": e.get("outcome", "completed")})
            current = None
    if current is not None:
        launches.append({**current, "ended_utc": None, "seconds": None, "outcome": "running_or_interrupted"})
    completed = [l["seconds"] for l in launches if l["seconds"] is not None]
    return {"launches": launches, "train_seconds_recorded": sum(completed),
            "note": "Interrupted launches have no end time; their duration is not included."}


def build_record(*, kind: str, experiment: dict, run_name: str, run_dir: Path, git: dict, dataset: dict,
                 model: dict, ultralytics_version: str, train_args: dict, ultralytics_args: dict | None,
                 environment: dict, launches: dict, epochs_completed: int | None, outputs: dict,
                 val_metrics: dict) -> dict:
    if kind not in RUN_KINDS:
        raise RecordError(f"kind must be one of {RUN_KINDS}")
    record = {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "kind": kind,
        "experiment": experiment,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "git": git,
        "dataset": dataset,
        "model": model,
        "ultralytics_version": ultralytics_version,
        "seed": experiment["config"]["seed"],
        "train_args": train_args,
        "ultralytics_args": ultralytics_args,
        "environment": environment,
        "duration": launches,
        "epochs_completed": epochs_completed,
        "outputs": outputs,
        "val_metrics": val_metrics,
        "test_metrics": None,  # only declared final evaluations produce test metrics (CLAUDE.md rule 7)
        "limitations": limitations_for(dataset),
    }
    problems = validate_record(record)
    if problems:
        raise RecordError("run record incomplete: " + "; ".join(problems))
    return record


def validate_record(record: dict) -> list:
    problems = []
    if record.get("schema") != SCHEMA:
        problems.append(f"schema is not {SCHEMA}")
    if record.get("kind") not in RUN_KINDS:
        problems.append("unknown kind")
    if record.get("test_metrics") is not None:
        problems.append("training records never contain test metrics")
    git = record.get("git") or {}
    if not git.get("commit"):
        problems.append("git commit missing")
    ds = record.get("dataset") or {}
    if ds.get("status") != "frozen" or not ds.get("fingerprint"):
        problems.append("dataset must be a verified frozen version with a fingerprint")
    if ds.get("splits_used_for_training") != ["train", "val"]:
        problems.append("dataset.splits_used_for_training must be ['train', 'val']")
    if not (record.get("model") or {}).get("checkpoint_sha256"):
        problems.append("pretrained checkpoint sha256 missing")
    val = record.get("val_metrics") or {}
    if val.get("split") != "val" or any(not isinstance(val.get(k), (int, float)) for k in VAL_METRIC_KEYS):
        problems.append("val metrics (split=val, precision/recall/map50/map50_95) missing")
    best = (record.get("outputs") or {}).get("best")
    if not best or not best.get("sha256"):
        problems.append("best checkpoint path/sha256 missing")
    return problems


def registry_row(record: dict) -> dict:
    exp, env, val = record["experiment"], record["environment"] or {}, record["val_metrics"]
    gpus = env.get("gpus") or []
    return {
        "experiment_id": exp["id"],
        "run_name": record["run_name"],
        "created_utc": record["created_utc"],
        "parent": exp["parent"],
        "changed": exp["changed"],
        "dataset_version": record["dataset"]["version"],
        "dataset_fingerprint": record["dataset"]["fingerprint"],
        "model": record["model"]["name"],
        "model_sha256": record["model"]["checkpoint_sha256"],
        "imgsz": record["train_args"].get("imgsz"),
        "epochs_completed": record["epochs_completed"],
        "seed": record["seed"],
        "git_commit": record["git"]["commit"],
        "git_dirty": record["git"]["dirty"],
        "ultralytics": record["ultralytics_version"],
        "torch": env.get("torch"),
        "cuda": env.get("cuda_version"),
        "gpu": gpus[0]["name"] if gpus else None,
        "val_precision": val["precision"],
        "val_recall": val["recall"],
        "val_map50": val["map50"],
        "val_map50_95": val["map50_95"],
        "best_weights_sha256": record["outputs"]["best"]["sha256"],
        "run_dir": record["run_dir"],
    }


def append_to_registry(registry: Path, record: dict) -> dict:
    """Append one row for a full run; refuses smoke runs, incomplete records and duplicates."""
    problems = validate_record(record)
    if record.get("kind") != "full":
        problems.append("only full runs are registered (smoke runs are pipeline checks)")
    if problems:
        raise RecordError("; ".join(problems))
    row = registry_row(record)
    registry = Path(registry)
    exists = registry.is_file() and registry.stat().st_size > 0
    if exists:
        with open(registry, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != REGISTRY_FIELDS:
                raise RecordError(f"{registry} header differs from REGISTRY_FIELDS")
            for r in reader:
                if r["run_name"] == row["run_name"] or r["experiment_id"] == row["experiment_id"]:
                    raise RecordError(f"{row['experiment_id']} / {row['run_name']} is already registered")
    with open(registry, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return row


def registered_experiments(registry: Path) -> set:
    registry = Path(registry)
    if not registry.is_file():
        return set()
    with open(registry, newline="", encoding="utf-8") as f:
        return {r["experiment_id"] for r in csv.DictReader(f)}


def write_record(record: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
