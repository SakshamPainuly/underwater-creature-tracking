"""VAL-ONLY evaluation of a finished training run: secondary metrics and a confusion matrix (D-020).

Never touches the test split (CLAUDE.md rule 7). Images come from the build manifest rows whose
split is "val", Ultralytics only sees the run's train/val data file (no `test` key), and every image
path is checked to lie under images/val/.

Measures, for one run's best.pt on its own val split:
  - operating confidence: the same rule as final_test.py (max mean F1 on Ultralytics' val curves),
    so E001 reproduces the 0.298 in its final_test_record.json;
  - own matcher (stdlib, unit-tested): greedy by confidence, one ground-truth box per prediction,
    IoU >= 0.5 (MATCH_IOU, fixed by D-020 before any result existed). At the operating confidence:
    TP / FP / FN (the single-class confusion matrix), precision, recall, F1;
  - false positives on confirmed-negative val images: images with >= 1 detection, detections;
  - the same counts per session and per lighting condition (lighting_flag in the version's
    split_manifest.csv: red-light/night keywords in the deployment notes, or not flagged), plus
    AP50 per group (101-point interpolation, own matcher);
  - P/R/F1 vs confidence and the precision-recall curve (101 points each) for plots;
  - model complexity (parameters, GFLOPs at the training imgsz) and speed: median ms per image
    over the batch-1 val predictions (first image dropped as warm-up) plus Ultralytics' batched val
    speed, on the GPU named in the record's environment. Speeds compare only on the same GPU type.

Outputs (Drive, outside Git): <run>/val_eval/{ultralytics_val/, predictions_val.jsonl,
per_image_val.csv, val_eval_record.json, confusion_matrix.png, curves.png}. The record (< 50 KB) is archived in
experiments/records/<id>/ (D-013, D-020).
"""

import csv
import json
import statistics
from pathlib import Path

from dataset_audit.config import PROJECT_ROOT, RAW_ROOTS
from dataset_audit.versioning import sha256_file, version_dir
from training import experiment as X
from training import run_record as R

VAL_SPLIT = X.VAL_SPLIT
EVAL_DIR = "val_eval"
EVAL_RECORD = "val_eval_record.json"
PREDICTIONS_FILE = "predictions_val.jsonl"
PER_IMAGE_FILE = "per_image_val.csv"
EVAL_SCHEMA = "uwct.val_eval_record/1"
MATCH_IOU = 0.5          # D-020: fixed before any result
PREDICT_CONF = 0.001     # predictions kept for the sweeps (Ultralytics val default)
PREDICT_NMS_IOU = 0.7    # Ultralytics val default
SPEED_WARMUP = 1          # first prediction(s) excluded from the speed medians (CUDA warm-up)
GRID = [i / 100 for i in range(101)]  # confidence and recall grids: 0.00 .. 1.00
NEGATIVE_STATUS = "negative_confirmed"
VAL_EVAL_LIMITATIONS = (
    "Val only. Val chose each run's checkpoint (best epoch) and every operating confidence, so these "
    "numbers are optimistic for unseen data; they compare runs, they do not estimate deployment quality.",
    "One run per arm (seed 0): small differences may be run-to-run noise.",
    "Val sessions share sites with train sessions: new sessions, not new sites (D-007).",
    "Val boxes are concentrated: 21_MACK_10 holds 465 of 1,010 boxes.",
    "Red-light val is 2 sessions / 31 boxes (19_SUB1_04, 19_SUB2_07): per-lighting results are fragile.",
    "Substrate/visibility metadata is mostly missing; zExtra is excluded.",
    "Own-matcher numbers (IoU >= 0.5, greedy) can differ slightly from Ultralytics' (different matching "
    "and batching); both are recorded.",
)


class ValEvalError(RuntimeError):
    pass


# ---------------------------------------------------------------- stdlib logic (tested locally)

def xywh_to_xyxy(b) -> tuple:
    cx, cy, w, h = b
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b) -> float:
    """IoU of two xyxy boxes. Normalized coordinates are fine: IoU is invariant to axis scaling."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_yolo_labels(path: Path) -> list:
    """Ground-truth xyxy boxes (normalized) from a YOLO label file; a missing/empty file = no boxes."""
    path = Path(path)
    if not path.is_file():
        return []
    boxes = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValEvalError(f"{path}:{n}: expected 'class cx cy w h', got {line!r}")
        boxes.append(xywh_to_xyxy([float(v) for v in parts[1:]]))
    return boxes


def match_image(gts: list, preds: list, iou_thr: float = MATCH_IOU) -> list:
    """Greedy matching for one image. preds: [(conf, xyxy)]. Returns [(conf, is_tp)] by conf desc.

    Each prediction, highest confidence first, takes the unmatched ground-truth box it overlaps most
    (IoU >= iou_thr); ties keep the earlier box. Thresholding the result at any confidence c gives
    the same TP/FP as matching only the predictions >= c, because matching is greedy in that order.
    """
    taken = [False] * len(gts)
    out = []
    for conf, box in sorted(preds, key=lambda p: -p[0]):
        best, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if not taken[j]:
                v = iou(box, g)
                if v >= best_iou and (best < 0 or v > best_iou):
                    best, best_iou = j, v
        if best >= 0:
            taken[best] = True
        out.append((conf, best >= 0))
    return out


def counts_at(images: list, conf: float) -> dict:
    """TP/FP/FN and derived metrics over images ({'n_gt', 'matches'}) at a confidence threshold."""
    tp = fp = n_gt = 0
    neg_images = neg_with_det = neg_dets = 0
    for im in images:
        kept = [is_tp for c, is_tp in im["matches"] if c >= conf]
        t = sum(kept)
        tp += t
        fp += len(kept) - t
        n_gt += im["n_gt"]
        if im.get("negative"):
            neg_images += 1
            neg_dets += len(kept)
            neg_with_det += bool(kept)
    fn = n_gt - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {
        "conf": conf, "images": len(images), "gt_boxes": n_gt, "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "negative_images": neg_images, "negative_images_with_detection": neg_with_det,
        "detections_on_negative_images": neg_dets,
    }


def ap50(images: list) -> float | None:
    """AP at MATCH_IOU, 101-point interpolated (COCO style), own matcher. None if no ground truth."""
    n_gt = sum(im["n_gt"] for im in images)
    if not n_gt:
        return None
    ranked = sorted((m for im in images for m in im["matches"]), key=lambda m: -m[0])
    prec, rec, tp = [], [], 0
    for k, (_, is_tp) in enumerate(ranked, 1):
        tp += is_tp
        prec.append(tp / k)
        rec.append(tp / n_gt)
    for k in range(len(prec) - 2, -1, -1):  # precision envelope
        prec[k] = max(prec[k], prec[k + 1])
    total, k = 0.0, 0
    for r in GRID:
        while k < len(rec) and rec[k] < r:
            k += 1
        total += prec[k] if k < len(rec) else 0.0
    return total / len(GRID)


def pr_curve(images: list) -> dict:
    """Interpolated precision at recall 0.00 .. 1.00 (the curve ap50 integrates)."""
    n_gt = sum(im["n_gt"] for im in images)
    ranked = sorted((m for im in images for m in im["matches"]), key=lambda m: -m[0])
    prec, rec, tp = [], [], 0
    for k, (_, is_tp) in enumerate(ranked, 1):
        tp += is_tp
        prec.append(tp / k)
        rec.append(tp / n_gt if n_gt else 0.0)
    for k in range(len(prec) - 2, -1, -1):
        prec[k] = max(prec[k], prec[k + 1])
    out, k = [], 0
    for r in GRID:
        while k < len(rec) and rec[k] < r:
            k += 1
        out.append(round(prec[k], 4) if k < len(rec) else 0.0)
    return {"recall": GRID, "precision": out}


def conf_curves(images: list) -> dict:
    """Own-matcher precision, recall and F1 at confidence 0.00 .. 1.00."""
    rows = [counts_at(images, c) for c in GRID]
    return {"conf": GRID, **{k: [round(r[k], 4) for r in rows] for k in ("precision", "recall", "f1")}}


def group_metrics(images: list, key: str, conf: float) -> dict:
    groups = {}
    for im in images:
        groups.setdefault(str(im[key]), []).append(im)
    return {g: {**counts_at(ims, conf), "ap50": ap50(ims), "sessions": sorted({i["session_id"] for i in ims})}
            for g, ims in sorted(groups.items())}


def confusion_matrix(c: dict) -> dict:
    """Single-class detection confusion matrix (rows = predicted, cols = true, as Ultralytics)."""
    return {
        "labels": ["lobster", "background"],
        "matrix": [[c["tp"], c["fp"]], [c["fn"], None]],
        "note": "rows = predicted, cols = true. background/background (true negatives) is undefined "
                "for a detector and left empty. FP = detections matching no box at IoU >= 0.5 "
                "(includes duplicates and poorly placed boxes); FN = lobster boxes not found.",
    }


def row_session(r: dict) -> str:
    """Session of a build-manifest row. Mixed views (v4) have no session_id column: for our own
    original images their group_id is the session (dataset_audit.model_a); any other val image
    (Zenodo or derived) has no session in split_manifest.csv and is refused."""
    if "session_id" in r:
        return r["session_id"]
    if (r.get("origin"), r.get("kind")) != ("ours", "original"):
        raise ValEvalError(f"{r['filename']}: val image of origin {r.get('origin')!r} / kind "
                           f"{r.get('kind')!r} has no session; val_eval covers our original images only")
    return r["group_id"]


def val_rows(build_rows: list, split_rows: list) -> list:
    """Val images with session and lighting; refuses anything that is not val."""
    lighting = {r["session_id"]: r["lighting_flag"] for r in split_rows}
    out = []
    for r in build_rows:
        if r["split"] != VAL_SPLIT:
            continue
        session = row_session(r)
        if session not in lighting:
            raise ValEvalError(f"{r['filename']}: session {session} not in split_manifest.csv")
        out.append({"filename": r["filename"], "session_id": session,
                    "lighting": "red_light_or_night" if lighting[session] == "True" else "not_flagged",
                    "negative": r["label_status"] == NEGATIVE_STATUS})
    if not out:
        raise ValEvalError("no val images in build_manifest.csv")
    return out


def assert_val_path(path: Path, dataset_dir: Path) -> None:
    # Resolve the folder, not the file: view images are symlinks/hardlinks to the raw files.
    val_dir = (Path(dataset_dir) / "images" / VAL_SPLIT).resolve()
    if Path(path).parent.resolve() != val_dir:
        raise ValEvalError(f"{path} is not in {val_dir}; val_eval reads val images only")


def evaluate(rows: list, gts: dict, preds: dict) -> list:
    """Per-image matches. gts: filename -> [xyxy]; preds: filename -> [(conf, xyxy)]."""
    unknown = set(preds) - {r["filename"] for r in rows}
    if unknown:
        raise ValEvalError(f"predictions for images outside val: {sorted(unknown)[:3]}")
    images = []
    for r in rows:
        g = gts.get(r["filename"], [])
        if r["negative"] and g:
            raise ValEvalError(f"{r['filename']}: confirmed negative but has label boxes")
        images.append({**r, "n_gt": len(g), "matches": match_image(g, preds.get(r["filename"], []))})
    return images


def speed_summary(speeds: list, warmup: int = SPEED_WARMUP) -> dict:
    """Median ms per image for each Ultralytics speed stage over per-image results, warm-up dropped."""
    kept = speeds[warmup:]
    if not kept:
        raise ValEvalError(f"need more than {warmup} timed predictions for a speed summary")
    stages = sorted(set().union(*kept))
    med = {k: statistics.median(s[k] for s in kept if k in s) for k in stages}
    return {"images_timed": len(kept), "warmup_dropped": warmup, "median_ms": med,
            "median_ms_total": sum(med.values())}


def complexity(info, imgsz: int) -> dict:
    """Parameters and GFLOPs from Ultralytics' model info tuple (layers, params, gradients, GFLOPs)."""
    if not isinstance(info, (tuple, list)) or len(info) < 4:
        raise ValEvalError(f"unexpected Ultralytics model info {info!r}")
    layers, params, _, gflops = info[:4]
    return {"layers": int(layers), "parameters": int(params),
            "gflops": float(gflops) if gflops else None, "gflops_imgsz": imgsz}


def summarize(images: list, conf: float) -> dict:
    overall = counts_at(images, conf)
    return {
        "match_iou": MATCH_IOU,
        "operating_conf": conf,
        "overall": {**overall, "ap50": ap50(images)},
        "confusion_matrix": confusion_matrix(overall),
        "per_session": group_metrics(images, "session_id", conf),
        "per_lighting": group_metrics(images, "lighting", conf),
        "curves": {"vs_conf": conf_curves(images), "pr": pr_curve(images)},
    }


def per_image_rows(images: list, conf: float) -> list:
    """One row per val image at the operating confidence (written to per_image_val.csv, off-repo)."""
    return [{"filename": im["filename"], "session_id": im["session_id"], "lighting": im["lighting"],
             "negative": im["negative"], "n_gt": im["n_gt"],
             **{k: v for k, v in counts_at([im], conf).items() if k in ("tp", "fp", "fn")}}
            for im in images]


def check_run(run_dir: Path) -> dict:
    """The finished full run whose best.pt is evaluated; refuses smoke runs and changed weights."""
    record_path = Path(run_dir) / "run_record.json"
    if not record_path.is_file():
        raise ValEvalError(f"{record_path} not found; finish (or --finalize-only) the training run first")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    problems = R.validate_record(record)
    if record.get("kind") != "full":
        problems.append("only full training runs are evaluated (this is a smoke run)")
    if problems:
        raise ValEvalError("run record not usable: " + "; ".join(problems))
    best = Path(run_dir) / "train" / "weights" / "best.pt"
    if not best.is_file() or sha256_file(best) != record["outputs"]["best"]["sha256"]:
        raise ValEvalError(f"{best} is missing or differs from the sha256 in run_record.json")
    return record


# ---------------------------------------------------------------- GPU side (Colab / lab only)

def _plots(summary: dict, exp_id: str, out_dir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = summary["confusion_matrix"]["matrix"]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow([[cm[0][0], cm[0][1]], [cm[1][0], 0]], cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, "n/a" if cm[i][j] is None else str(cm[i][j]), ha="center", va="center")
    ax.set_xticks([0, 1], ["lobster", "background"])
    ax.set_yticks([0, 1], ["lobster", "background"])
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    ax.set_title(f"{exp_id} val, conf {summary['operating_conf']:.3f}, IoU>={MATCH_IOU}")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    c, pr = summary["curves"]["vs_conf"], summary["curves"]["pr"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for k in ("precision", "recall", "f1"):
        a1.plot(c["conf"], c[k], label=k)
    a1.axvline(summary["operating_conf"], ls="--", c="grey")
    a1.set_xlabel("confidence")
    a1.legend()
    a2.plot(pr["recall"], pr["precision"])
    a2.set_xlabel("recall")
    a2.set_ylabel("precision")
    a2.set_title(f"AP50 {summary['overall']['ap50']:.3f}")
    fig.suptitle(f"{exp_id} val")
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=150)
    plt.close(fig)
    return [out_dir / "confusion_matrix.png", out_dir / "curves.png"]


def run_val_eval(runs_root: Path, exp_id: str, env_collect, dataset_dir: Path | None = None) -> dict:
    from training.final_test import _box_metrics, _curves, operating_conf, value_at
    from training.train import model_class, require_gpu_environment

    ultralytics = require_gpu_environment()
    runs_root = X.check_output_location(runs_root, PROJECT_ROOT, RAW_ROOTS)
    run_dir = runs_root / exp_id
    git = R.git_state(PROJECT_ROOT)
    if not git["commit"] or git["dirty"]:
        raise ValEvalError("val_eval needs a clean git checkout at a recorded commit")
    run_record = check_run(run_dir)
    cfg = X.validate_config(run_record["experiment"]["config"])
    if ultralytics.__version__ != run_record["ultralytics_version"]:
        raise ValEvalError(f"ultralytics {ultralytics.__version__} != training {run_record['ultralytics_version']}")
    out_dir = run_dir / EVAL_DIR
    if (out_dir / EVAL_RECORD).is_file():
        raise ValEvalError(f"{out_dir / EVAL_RECORD} exists; move it away to re-run")

    dataset_dir = Path(dataset_dir or PROJECT_ROOT / "data" / "yolo" / f"lobster_{cfg['dataset_version']}")
    dataset = X.verify_built_dataset(dataset_dir, cfg)
    if dataset["fingerprint"] != run_record["dataset"]["fingerprint"]:
        raise ValEvalError("dataset fingerprint differs from the one the model was trained with")
    with open(dataset_dir / "build_manifest.csv", newline="", encoding="utf-8") as f:
        build_rows = list(csv.DictReader(f))
    with open(version_dir(cfg["dataset_version"]) / "split_manifest.csv", newline="", encoding="utf-8") as f:
        rows = val_rows(build_rows, list(csv.DictReader(f)))

    out_dir.mkdir(exist_ok=True)
    Model = model_class(cfg["model"])
    best = str(run_dir / "train" / "weights" / "best.pt")
    imgsz = run_record["train_args"]["imgsz"]
    batch = run_record["train_args"].get("batch", 16)

    # a. Ultralytics val on the train/val-only data file: operating confidence + its own plots
    trainval = run_dir / "data_trainval.yaml"
    X.assert_trainval_only(trainval)
    val_result = Model(best).val(data=str(trainval), split=X.VAL_SPLIT, imgsz=imgsz, batch=batch,
                                project=str(out_dir), name="ultralytics_val", exist_ok=True, plots=True)
    curves = _curves(val_result)
    if curves is None:
        raise ValEvalError("Ultralytics did not expose P/R/F1 curves; cannot choose the operating point")
    conf = operating_conf(curves["px"], curves["f1"])

    # b. predictions on val images only, kept down to PREDICT_CONF for the sweeps
    sources = [dataset_dir / "images" / VAL_SPLIT / r["filename"] for r in rows]
    for p in sources:
        assert_val_path(p, dataset_dir)
    model, preds, speeds = Model(best), {}, []
    with open(out_dir / PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        for path in sources:
            res = model.predict(source=str(path), conf=PREDICT_CONF, iou=PREDICT_NMS_IOU, imgsz=imgsz,
                                      max_det=300, verbose=False)[0]
            boxes = [(float(c), xywh_to_xyxy([float(v) for v in b]))
                     for c, b in zip(res.boxes.conf.tolist(), res.boxes.xywhn.tolist())]
            preds[path.name] = boxes
            speeds.append(dict(res.speed))
            f.write(json.dumps({"filename": path.name, "boxes": [[round(c, 5), *[round(v, 6) for v in b]]
                                                                 for c, b in boxes]}) + "\n")
    gts = {r["filename"]: read_yolo_labels(dataset_dir / "labels" / VAL_SPLIT / (Path(r["filename"]).stem + ".txt"))
           for r in rows}
    images = evaluate(rows, gts, preds)
    summary = summarize(images, conf)
    per_image = per_image_rows(images, conf)
    with open(out_dir / PER_IMAGE_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    plot_files = _plots(summary, exp_id, out_dir)
    env = env_collect()

    record = {
        "schema": EVAL_SCHEMA,
        "created_utc": R.utc_now(),
        "label": "VAL-ONLY EVALUATION (no test split)",
        "split": VAL_SPLIT,
        "experiment_id": exp_id,
        "run_name": run_record["run_name"],
        "training_git_commit": run_record["git"]["commit"],
        "evaluation_git": git,
        "run_record_best_sha256": run_record["outputs"]["best"]["sha256"],
        "dataset": dataset,
        "settings": {"imgsz": imgsz, "predict_conf": PREDICT_CONF, "predict_nms_iou": PREDICT_NMS_IOU,
                     "match_iou": MATCH_IOU, "operating_conf_rule": "max mean F1 on Ultralytics val curves "
                     "(same as final_test.py)"},
        "ultralytics_val": {**_box_metrics(val_result), "at_operating_conf": {
            k: value_at(curves["px"], curves[k], conf) for k in ("p", "r", "f1")}},
        **summary,
        # unfused; verbose=True because Ultralytics' model_info returns None when verbose is False
        "complexity": complexity(Model(best).model.info(verbose=True, imgsz=imgsz), imgsz),
        "speed": {"batch1_predict": speed_summary(speeds),
                  "ultralytics_val_ms_per_image": {k: float(v) for k, v in val_result.speed.items()},
                  "ultralytics_val_batch": batch,
                  "gpu": [g["name"] for g in env.get("gpus", [])]},
        "artifacts": [R.file_entry(p) for p in [out_dir / PREDICTIONS_FILE, out_dir / PER_IMAGE_FILE, *plot_files]],
        "environment": env,
        "limitations": list(VAL_EVAL_LIMITATIONS),
    }
    (out_dir / EVAL_RECORD).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


# ---------------------------------------------------------------- comparison (stdlib)

def compare_markdown(records: list) -> str:
    """Side-by-side markdown tables for two or more val_eval records."""
    ids = [r["experiment_id"] for r in records]
    head = lambda first: f"| {first} | " + " | ".join(ids) + " |\n|---|" + "---|" * len(ids) + "\n"
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
    lines = ["### Overall (val)\n", head("metric")]
    rows = [
        ("dataset version", lambda r: r["dataset"]["version"]),
        ("Ultralytics P (its own conf)", lambda r: f3(r["ultralytics_val"]["precision"])),
        ("Ultralytics R (its own conf)", lambda r: f3(r["ultralytics_val"]["recall"])),
        ("Ultralytics mAP50", lambda r: f3(r["ultralytics_val"]["map50"])),
        ("Ultralytics mAP50-95", lambda r: f3(r["ultralytics_val"]["map50_95"])),
        ("operating conf (chosen on val)", lambda r: f3(r["operating_conf"])),
        ("AP50, own matcher", lambda r: f3(r["overall"]["ap50"])),
        ("precision @ conf", lambda r: f3(r["overall"]["precision"])),
        ("recall @ conf", lambda r: f3(r["overall"]["recall"])),
        ("F1 @ conf", lambda r: f3(r["overall"]["f1"])),
        ("TP / FP / FN @ conf", lambda r: "{tp} / {fp} / {fn}".format(**r["overall"])),
        ("negative images with a detection", lambda r: "{negative_images_with_detection} of {negative_images}"
         .format(**r["overall"])),
        ("detections on negative images", lambda r: str(r["overall"]["detections_on_negative_images"])),
        ("parameters (M)", lambda r: f"{r['complexity']['parameters'] / 1e6:.2f}"),
        ("GFLOPs (at imgsz)", lambda r: "n/a" if r["complexity"]["gflops"] is None
         else f"{r['complexity']['gflops']:.1f} ({r['complexity']['gflops_imgsz']})"),
        ("inference ms/img, batch 1 (median)",
         lambda r: f"{r['speed']['batch1_predict']['median_ms'].get('inference', float('nan')):.1f}"),
        ("total ms/img, batch 1 (median)", lambda r: f"{r['speed']['batch1_predict']['median_ms_total']:.1f}"),
        ("GPU", lambda r: ", ".join(r["speed"]["gpu"]) or "n/a"),
    ]
    lines += [f"| {name} | " + " | ".join(fn(r) for r in records) + " |\n" for name, fn in rows]
    for title, key in (("Per lighting", "per_lighting"), ("Per session", "per_session")):
        lines += [f"\n### {title} (val, each run at its own operating conf)\n",
                  head(key.split("_")[1] + " (boxes)")]
        for g in sorted(set().union(*(r[key] for r in records))):
            n = next(r[key][g]["gt_boxes"] for r in records if g in r[key])
            cells = []
            for r in records:
                m = r[key].get(g)
                cells.append("n/a" if m is None else
                             f"AP50 {f3(m['ap50'])}, P {f3(m['precision'])}, R {f3(m['recall'])}, FP {m['fp']}")
            lines.append(f"| {g} ({n}) | " + " | ".join(cells) + " |\n")
    lines.append("\nLimitations: " + " ".join(records[0]["limitations"]) + "\n")
    return "".join(lines)
