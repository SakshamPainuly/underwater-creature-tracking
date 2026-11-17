"""Final report data pack: the project's numbers, tables and figures, built from committed files.

Inputs (all committed): experiments/registry.csv, experiments/records/<id>/ (run_record.json,
results.csv, final_test*_record.json, val_eval_record.json), data/processed/v4/mixed_manifest.csv.
Every number in the pack is computed from those files, except the few quoted in QUOTED below,
each of which names its source (and tests check that the source still says it).

Output: docs/reports/final_report_data.md, docs/reports/final_report_experiments.csv and PNG
figures in docs/reports/figures/final_*.png. Pixels, counts, confidences, seconds and hours only;
no physical units (CLAUDE.md rule 9); no tracking metric for lobsters (no ground-truth track IDs).
Test numbers are reported, never used to choose anything (rule 7).

The tables are standard library; the figures import matplotlib lazily (the GUI .venv has it).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .e001_e004 import GRID, INK, INK_2, SURFACE, ReportError, _fmt, _plt, _save, _table

# ---------------------------------------------------------------------------------------------
# What the pack covers (fixed by the accepted decisions, not by results)
# ---------------------------------------------------------------------------------------------

FINAL = "E027"  # D-043
V1_BASELINES = ("E001", "E004", "E005")  # raw YOLO, enhanced YOLO (D-017), RT-DETR (D-015/D-025)
TESTED = ("E001", "E004", "E027")  # lobster runs whose final-test record is in the repository
MODEL_A_FIRST = "E007"  # D-032
ROUND1 = {  # D-038 point 2 / D-039: model size x seed
    "yolo26s": ("E011", "E016", "E019"),
    "yolo26m": ("E012", "E014", "E017"),
    "yolo26l": ("E013", "E015"),  # E018 (seed 2): not run (time)
}
ROUND1_NOT_RUN = ("E018",)
ROUND2 = (  # D-040 point 2: (combination, what it adds to the 30-epoch schedule, yolo26m run, yolo26l run)
    ("C1", "schedule only (control)", "E020", "E021"),
    ("C2", "+ AdamW lr0 0.001", "E022", "E023"),
    ("C3", "+ mixup 0.1", "E024", "E025"),
    ("C4", "+ AdamW lr0 0.001, weight_decay 0.001, mixup 0.1", "E026", "E027"),
    ("C5", "+ freeze 10 (backbone frozen)", "E028", "E029"),
)
TRITONIA = ("E006", "E008", "E010") + tuple(f"E0{n}" for n in range(30, 47))  # sea-slug runs (Anu)

PHASES = (  # compute table: phase label -> experiments, platform
    ("v1 baselines: E001, E004, E005", ("E001", "E004", "E005"), "Colab T4"),
    ("Model A first run: E007", ("E007",), "Kaggle T4"),
    ("Model A tuning round 1: E011–E019", tuple(e for runs in ROUND1.values() for e in runs), "Kaggle 2×T4"),
    ("Model A tuning round 2: E020–E029", tuple(e for row in ROUND2 for e in row[2:]), "Kaggle 2×T4"),
    ("Tritonia sea slugs (Anu): E006–E046", TRITONIA, "Kaggle T4"),
)

# Numbers that are not in a committed machine-readable record. Each names where it is written;
# tests/test_final_report.py checks that the named document still contains the quoted text.
QUOTED = {
    "E005_test": {
        "values": {"mAP50": 0.8788, "mAP50-95": 0.6604, "precision": 0.8411, "recall": 0.7712, "F1": 0.8046,
                   "conf": 0.619},
        "source": "docs/EXPERIMENTS.md, Test-evaluation log (E005 record is on Drive, not in the repository)",
        "check": ("docs/EXPERIMENTS.md", "mAP50 0.8788; mAP50-95 0.6604; precision @ val-chosen conf 0.619 = 0.8411"),
    },
    "round1_session_h": {
        "value": 7.4, "source": "D-040 context 1 (Kaggle round-1 notebook log)",
        "check": ("docs/DECISIONS.md", "queue finished about 7.4 h after Cell 1"),
    },
    "round2_session_h": {
        "value": 9.0, "source": "D-043 context 1 (Kaggle round-2 notebook log)",
        "check": ("docs/DECISIONS.md", "last run finished about 9.0 h later"),
    },
    "quota_after_round1_h": {
        "value": 12.0, "source": "D-040 context 5 (Kaggle quota meter, read by the user)",
        "check": ("docs/DECISIONS.md", "Kaggle quota after round 1: 12:00 of 30 h used"),
    },
    "tritonia_test_point_ap": {
        "values": {"E006": 0.248, "E010": 0.2145},
        "source": "docs/EXPERIMENTS.md, Test-evaluation log (point AP against the MTrackJ crosses, D-033)",
        "check": ("docs/EXPERIMENTS.md", "test point AP 0.2145"),
    },
}

FIGURES = (
    "final_round1.png",
    "final_round2.png",
    "final_training_curves.png",
    "final_test_comparison.png",
    "final_confusion.png",
    "final_gpu_hours.png",
)

# Categorical slots 1-3 of the reference palette, validated all-pairs on the light surface
# (aqua is below 3:1 contrast, so every bar carries a direct value label and every figure a table).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MODEL_COLORS = {"E001": BLUE, "E004": ORANGE, "E027": AQUA}
SIZE_COLORS = {"yolo26s": BLUE, "yolo26m": ORANGE, "yolo26l": AQUA}
NEUTRAL = "#8a8984"


# ---------------------------------------------------------------------------------------------
# Loading (stdlib)
# ---------------------------------------------------------------------------------------------

def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReportError(f"Missing record file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _history(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{k.strip(): float(v) for k, v in row.items() if v not in (None, "")} for row in csv.DictReader(f)]


def load_inputs(root: Path) -> dict[str, Any]:
    """Everything the pack needs, read once. Raises ReportError naming any missing file."""
    records = root / "experiments" / "records"
    registry_path = root / "experiments" / "registry.csv"
    if not registry_path.is_file():
        raise ReportError(f"Missing {registry_path}")
    with open(registry_path, newline="", encoding="utf-8") as f:
        registry = {row["experiment_id"]: row for row in csv.DictReader(f)}

    wanted = set(V1_BASELINES) | {MODEL_A_FIRST} | set(TRITONIA)
    wanted |= {e for runs in ROUND1.values() for e in runs} | {e for row in ROUND2 for e in row[2:]}
    missing = sorted(e for e in wanted if e not in registry)
    if missing:
        raise ReportError("Not registered in experiments/registry.csv: " + ", ".join(missing))

    runs = {}
    for exp in sorted(wanted):
        folder = records / exp  # keyed by folder: E042-E046 records still carry their pre-renumbering IDs
        runs[exp] = {"run": _json(folder / "run_record.json"), "history": _history(folder / "results.csv")}

    tests = {exp: _json(records / exp / "final_test_record.json") for exp in TESTED}
    for exp, rec in tests.items():
        if rec.get("experiment_id") != exp or rec.get("split") != "test":
            raise ReportError(f"{exp}/final_test_record.json is not {exp}'s test record")
    sk = _json(records / FINAL / "final_test_zenodo_sk_record.json")
    if sk.get("target") != "test_zenodo_sk":
        raise ReportError(f"{FINAL}/final_test_zenodo_sk_record.json is not a Zenodo sk record")
    val_eval = {exp: _json(records / exp / "val_eval_record.json") for exp in ("E001", "E004", FINAL)}
    for exp, rec in val_eval.items():
        if rec.get("experiment_id") != exp or rec.get("split") != "val":
            raise ReportError(f"{exp}/val_eval_record.json is not {exp}'s val-only record")

    manifest_path = root / "data" / "processed" / "v4" / "mixed_manifest.csv"
    if not manifest_path.is_file():
        raise ReportError(f"Missing {manifest_path}")
    with open(manifest_path, newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    return {"registry": registry, "runs": runs, "tests": tests, "sk": sk, "val_eval": val_eval,
            "manifest": manifest}


# ---------------------------------------------------------------------------------------------
# Derived numbers (stdlib, pure)
# ---------------------------------------------------------------------------------------------

def val_map(inputs: dict, exp: str, key: str = "map50_95") -> float:
    return float(inputs["runs"][exp]["run"]["val_metrics"][key])


def train_hours(inputs: dict, exp: str) -> float:
    return float(inputs["runs"][exp]["run"]["duration"]["train_seconds_recorded"]) / 3600


def interrupted_launches(inputs: dict, exp: str) -> int:
    launches = inputs["runs"][exp]["run"]["duration"].get("launches", [])
    return sum(1 for launch in launches if launch.get("outcome") != "completed")


def round1_summary(inputs: dict) -> dict[str, dict[str, float]]:
    """Seed mean and range of val mAP50-95 per model size (D-038 point 4)."""
    out = {}
    for size, runs in ROUND1.items():
        vals = [val_map(inputs, e) for e in runs]
        out[size] = {"mean": sum(vals) / len(vals), "range": max(vals) - min(vals), "n": len(vals),
                     "min": min(vals), "max": max(vals)}
    return out


def round2_bar(inputs: dict) -> float:
    """D-040 point 4: yolo26m seed mean + its seed range."""
    m = round1_summary(inputs)["yolo26m"]
    return m["mean"] + m["range"]


def test_headline(rec: dict) -> dict[str, float]:
    m = rec["metrics"]
    rep, op = m["ultralytics_reported"], m["at_val_conf"]
    return {"mAP50": rep["map50"], "mAP50-95": rep["map50_95"], "precision": op["precision"],
            "recall": op["recall"], "F1": op["f1"], "conf": op["conf"]}


def test_confusion(rec: dict) -> dict[str, int]:
    """TP / FP / FN from Ultralytics' matrix (rows = predicted, cols = true)."""
    cm = rec["metrics"]["confusion_matrix"]["matrix"]
    return {"TP": int(cm[0][0]), "FP": int(cm[0][1]), "FN": int(cm[1][0])}


def v4_composition(inputs: dict) -> dict[tuple[str, str, str], int]:
    """(split, origin, kind) -> images, for the images in v4 (excluded rows are counted separately)."""
    out: dict[tuple[str, str, str], int] = {}
    for row in inputs["manifest"]:
        key = (row["split"], row["origin"], row["kind"])
        out[key] = out.get(key, 0) + 1
    return out


def v4_exclusions(inputs: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in inputs["manifest"]:
        if row["exclude_reason"]:
            out[row["exclude_reason"]] = out.get(row["exclude_reason"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def phase_hours(inputs: dict, lobster_only: bool = False) -> list[dict[str, Any]]:
    rows = []
    for label, exps, platform in PHASES:
        if lobster_only and exps == TRITONIA:
            continue
        rows.append({"phase": label, "platform": platform, "runs": len(exps),
                     "gpu_hours": sum(train_hours(inputs, e) for e in exps),
                     "interrupted": sum(interrupted_launches(inputs, e) for e in exps)})
    return rows


def experiments_rows(inputs: dict, lobster_only: bool = False) -> list[dict[str, str]]:
    """One row per registered run covered by the pack, for the CSV export (deterministic order)."""
    order = list(V1_BASELINES) + [MODEL_A_FIRST] + [e for runs in ROUND1.values() for e in runs]
    order += [e for row in ROUND2 for e in row[2:]] + ([] if lobster_only else list(TRITONIA))
    group = {e: "v1 baseline" for e in V1_BASELINES}
    group[MODEL_A_FIRST] = "Model A first run"
    group.update({e: "Model A round 1" for runs in ROUND1.values() for e in runs})
    group.update({e: "Model A round 2" for row in ROUND2 for e in row[2:]})
    group.update({e: "Tritonia" for e in TRITONIA})
    rows = []
    for exp in sorted(order, key=lambda e: (order.index(e),)):
        reg, run = inputs["registry"][exp], inputs["runs"][exp]["run"]
        args = run.get("train_args") or {}
        rows.append({
            "experiment_id": exp, "group": group[exp], "parent": reg["parent"], "changed": reg["changed"],
            "dataset_version": reg["dataset_version"], "model": reg["model"], "seed": reg["seed"],
            "epochs_completed": reg["epochs_completed"], "epochs_max": str(args.get("epochs", "")),
            "imgsz": reg["imgsz"], "optimizer": str(args.get("optimizer", "auto")),
            "lr0": str(args.get("lr0", "")), "weight_decay": str(args.get("weight_decay", "")),
            "mixup": str(args.get("mixup", "")), "freeze": str(args.get("freeze", "")),
            "val_precision": reg["val_precision"], "val_recall": reg["val_recall"],
            "val_map50": reg["val_map50"], "val_map50_95": reg["val_map50_95"],
            "train_hours": f"{train_hours(inputs, exp):.3f}", "gpu": reg["gpu"], "git_commit": reg["git_commit"],
        })
    return rows


# ---------------------------------------------------------------------------------------------
# Figures (matplotlib, imported lazily)
# ---------------------------------------------------------------------------------------------

BOXED = {"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"}


def fig_round1(inputs, path: Path) -> None:
    plt = _plt()
    summ = round1_summary(inputs)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, (size, runs) in enumerate(ROUND1.items()):
        ax.bar(i, summ[size]["mean"] - 0.575, 0.6, bottom=0.575, color=SIZE_COLORS[size], alpha=0.3,
               edgecolor=SURFACE, linewidth=2)
        for j, exp in enumerate(runs):
            v = val_map(inputs, exp)
            x = i - 0.18 + j * 0.18
            ax.scatter([x], [v], s=46, color=SIZE_COLORS[size], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(x + 0.03, v, exp, ha="left", va="center", fontsize=7, color=INK_2, bbox=BOXED, zorder=4)
        ax.text(i, 0.5775, f"mean {summ[size]['mean']:.4f}\nrange {summ[size]['range']:.4f}", ha="center",
                va="bottom", fontsize=8, color=INK)
    for exp, style in (("E001", (0, (4, 3))), ("E007", (0, (1, 2)))):
        v = val_map(inputs, exp)
        ax.axhline(v, color=NEUTRAL, linewidth=1.2, linestyle=style, zorder=1)
        ax.text(2.55, v, f"{exp} {v:.4f}", va="center", ha="left", fontsize=8, color=INK_2, bbox=BOXED)
    ax.set_xticks(range(3), [f"{s}\n({summ[s]['n']} seeds)" for s in ROUND1])
    ax.set_xlim(-0.5, 3.05)
    ax.set_ylim(0.575, 0.622)
    ax.set_ylabel("val mAP50-95 (best.pt)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Round 1: model size × seed (dots = seeds, bar = seed mean)")
    _save(fig, path)


ROUND2_SHORT = {"C1": "control", "C2": "AdamW", "C3": "mixup", "C4": "AdamW + wd\n+ mixup", "C5": "freeze 10"}


def fig_round2(inputs, path: Path) -> None:
    plt = _plt()
    from matplotlib.lines import Line2D

    bar = round2_bar(inputs)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    width = 0.36
    for k, (size, col) in enumerate((("yolo26m", 2), ("yolo26l", 3))):
        xs = [i + (k - 0.5) * (width + 0.03) for i in range(len(ROUND2))]
        vals = [val_map(inputs, row[col]) for row in ROUND2]
        ax.bar(xs, [v - 0.56 for v in vals], width, bottom=0.56, color=SIZE_COLORS[size], label=size,
               edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.0006, f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, color=INK_2, bbox=BOXED,
                    zorder=4)
    ax.axhline(bar, color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK, linewidth=1.2, linestyle=(0, (4, 3))))
    labels.append(f"bar {bar:.5f} (D-040)")
    ax.legend(handles, labels, loc="upper left", ncols=3)
    ax.set_xticks(range(len(ROUND2)), [f"{row[0]} {ROUND2_SHORT[row[0]]}\n{row[2]} | {row[3]}" for row in ROUND2],
                  fontsize=8)
    ax.set_ylim(0.56, 0.645)
    ax.set_ylabel("val mAP50-95 (best.pt)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Round 2: 30-epoch schedule × 5 combinations (one seed each)")
    _save(fig, path)


def fig_training_curves(inputs, path: Path) -> None:
    plt = _plt()
    series = (("E001", "E001 · yolo26s, v1", BLUE, (6, -0.06)),
              ("E013", "E013 · yolo26l, v4, round 1", ORANGE, (14, -0.1)),
              ("E027", "E027 · yolo26l, v4, round 2 C4 (final)", AQUA, (-2, 0.035)))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for exp, label, color, (dx, dy) in series:
        hist = inputs["runs"][exp]["history"]
        xs = [row["epoch"] for row in hist]
        ys = [row["metrics/mAP50-95(B)"] for row in hist]
        ax.plot(xs, ys, color=color, linewidth=2, label=label)
        best = max(range(len(ys)), key=lambda i: ys[i])
        ax.scatter([xs[best]], [ys[best]], s=48, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{exp} best {ys[best]:.4f} @ epoch {int(xs[best])}", (xs[best], ys[best]),
                    (xs[best] + dx, ys[best] + dy), fontsize=8, color=INK_2, bbox=BOXED,
                    arrowprops={"arrowstyle": "-", "color": NEUTRAL, "linewidth": 0.8})
    ax.set_xlabel("epoch")
    ax.set_ylabel("val mAP50-95 (validation after each epoch)")
    ax.set_ylim(0.3, 0.7)
    ax.legend(loc="lower right")
    ax.set_title("Validation mAP50-95 during training")
    _save(fig, path)


def fig_test_comparison(inputs, path: Path) -> None:
    plt = _plt()
    metrics = ["mAP50", "mAP50-95", "precision", "recall", "F1"]
    groups = [(f"{e} v1 test", test_headline(inputs["tests"][e]), MODEL_COLORS[e]) for e in TESTED]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [3, 2], "wspace": 0.12})
    ax = axes[0]
    width = 0.27
    for k, (label, h, color) in enumerate(groups):
        xs = [i + (k - 1) * (width + 0.02) for i in range(len(metrics))]
        vals = [h[m] for m in metrics]
        ax.bar(xs, vals, width, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK_2, rotation=90)
    ax.set_xticks(range(len(metrics)), metrics)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncols=3, fontsize=8)
    ax.set_title("v1 test (348 images, 1,512 boxes)", fontsize=11)

    ax = axes[1]
    pair = ((f"{FINAL} v1 test", test_headline(inputs["tests"][FINAL]), AQUA),
            (f"{FINAL} Zenodo sk", test_headline(inputs["sk"]), NEUTRAL))
    width = 0.38
    for k, (label, h, color) in enumerate(pair):
        xs = [i + (k - 0.5) * (width + 0.03) for i in range(len(metrics))]
        vals = [h[m] for m in metrics]
        ax.bar(xs, vals, width, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK_2, rotation=90)
    ax.set_xticks(range(len(metrics)), metrics, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncols=2, fontsize=8)
    ax.set_title(f"{FINAL}: known sites vs a new site (428 images, 143 boxes)", fontsize=11)
    fig.suptitle("Held-out test results · P / R / F1 at each model's val-chosen confidence", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", y=1.02)
    _save(fig, path)


def fig_confusion(inputs, path: Path) -> None:
    plt = _plt()
    panels = [(f"{e} · v1 test", inputs["tests"][e]) for e in TESTED] + [(f"{FINAL} · Zenodo sk", inputs["sk"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 3.5), gridspec_kw={"wspace": 0.08})
    for n, (ax, (title, rec)) in enumerate(zip(axes, panels)):
        c = test_confusion(rec)
        cells = [[("found (TP)", c["TP"]), ("false box (FP)", c["FP"])], [("missed (FN)", c["FN"]), ("—", None)]]
        top = max(c.values())
        for r in range(2):
            for col in range(2):
                name, value = cells[r][col]
                shade = "#efeeea" if value is None else _mix("#e8f0fb", "#1f5aa6", value / top)
                ax.add_patch(plt.Rectangle((col, 1 - r), 0.97, 0.97, color=shade))
                ink = "#ffffff" if value is not None and value / top > 0.55 else INK
                ax.text(col + 0.485, 1 - r + 0.55, "" if value is None else f"{value:,}", ha="center",
                        va="center", fontsize=13, fontweight="bold", color=ink)
                ax.text(col + 0.485, 1 - r + 0.25, name, ha="center", va="center", fontsize=7.5, color=ink)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.485, 1.485], ["true lobster", "true background"], fontsize=8)
        ax.set_yticks([1.485, 0.485], ["predicted\nlobster", "predicted\nbackground"] if n == 0 else ["", ""],
                      fontsize=8)
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Ultralytics confusion matrices on the test sets (boxes; IoU 0.5; shading per panel)", x=0.01,
                 ha="left", fontsize=11, fontweight="bold", y=1.04)
    _save(fig, path)


def _mix(a: str, b: str, t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def fig_gpu_hours(inputs, path: Path, lobster_only: bool = False) -> None:
    plt = _plt()
    rows = phase_hours(inputs, lobster_only)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows):
        ax.barh(y, row["gpu_hours"], 0.55, color=AQUA if "Model A" in row["phase"] else BLUE,
                edgecolor=SURFACE, linewidth=2)
        runs = f"{row['runs']} run" + ("s" if row["runs"] != 1 else "")
        ax.text(row["gpu_hours"] + 0.25, y, f"{row['gpu_hours']:.1f} GPU-h · {runs} · {row['platform']}",
                va="center", fontsize=8, color=INK_2)
    ax.set_yticks(ys, [row["phase"] for row in rows], fontsize=8)
    ax.set_xlabel("recorded training time (GPU-hours, completed launches only)")
    ax.set_xlim(0, max(r["gpu_hours"] for r in rows) * 1.55)
    ax.grid(axis="y", visible=False)
    ax.set_title("Compute per phase (Model A phases in aqua)")
    _save(fig, path)


def make_figures(inputs, fig_dir: Path, lobster_only: bool = False) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    gpu_hours = lambda i, p: fig_gpu_hours(i, p, lobster_only)  # noqa: E731
    makers = (fig_round1, fig_round2, fig_training_curves, fig_test_comparison, fig_confusion, gpu_hours)
    paths = []
    for name, maker in zip(FIGURES, makers):
        maker(inputs, fig_dir / name)
        paths.append(fig_dir / name)
    return paths


# ---------------------------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------------------------

def _f4(x: float) -> str:
    return f"{x:.4f}"


def render_markdown(inputs: dict, fig_rel: str = "figures", lobster_only: bool = False) -> str:
    """The pack as Markdown. `lobster_only` leaves out the other species' runs (the GUI bundle copy)."""
    reg, tests, sk = inputs["registry"], inputs["tests"], inputs["sk"]
    summ, bar = round1_summary(inputs), round2_bar(inputs)
    fin = tests[FINAL]
    th = {e: test_headline(tests[e]) for e in TESTED}
    skh = test_headline(sk)
    cc = {e: test_confusion(tests[e]) for e in TESTED}
    skc = test_confusion(sk)
    e005 = QUOTED["E005_test"]["values"]
    phases = phase_hours(inputs, lobster_only)
    lobster_h = sum(r["gpu_hours"] for r in phases if "Tritonia" not in r["phase"])
    all_h = sum(r["gpu_hours"] for r in phases)
    r2_vals = {row[c]: val_map(inputs, row[c]) for row in ROUND2 for c in (2, 3)}
    above = [e for e, v in r2_vals.items() if v > bar]
    top_size = max(summ, key=lambda s: summ[s]["mean"])
    counts = fin["dataset"]["counts"]
    tc = fin["dataset"]["test_counts"]
    skcounts = sk["dataset"]["test_counts"]
    best_size_mb = {e: inputs["runs"][e]["run"]["outputs"]["best"]["size"] / 1e6 for e in ("E001", "E027")}
    speed = {e: tests[e]["metrics"]["speed_ms_per_image"]["inference"] for e in ("E001", "E027")}
    ve = inputs["val_eval"]
    cx = {e: ve[e]["complexity"] for e in ("E001", FINAL)}
    b1 = {e: ve[e]["speed"]["batch1_predict"]["median_ms"]["inference"] for e in ("E001", FINAL)}
    fig = lambda name, alt: f"![{alt}]({fig_rel}/{name})"  # noqa: E731

    out: list[str] = []
    w = out.append
    w("# Final report data pack: underwater lobster detection")
    w("")
    w("Underwater Creature Tracking project · class `0: lobster` · built by `scripts/make_final_report.py` from the "
      "committed records (`experiments/registry.csv`, `experiments/records/`, `data/processed/v4/`). Every number is "
      "computed from those files except the few in the *Quoted numbers* table at the end, which name their source. "
      "This is source material for the report and slides, not the report itself. Claims are marked OBSERVED / "
      "INFERRED / LIMITATION (CLAUDE.md §3).")
    w("")
    if lobster_only:
        w("*Lobster-only copy (built with `--lobster-only`): the other species' runs and section 10 are left out; "
          "`docs/EXPERIMENTS.md` and `docs/DECISIONS.md` named below are in the project repository.*")
        w("")
    w("> **Key numbers (OBSERVED)**")
    w(">")
    w(f"> - **Final model {FINAL}** (YOLO26l, dataset v4, round-2 combination C4) on the held-out v1 test: "
      f"mAP50-95 **{th[FINAL]['mAP50-95']:.3f}**, mAP50 {th[FINAL]['mAP50']:.3f}, P / R / F1 "
      f"{th[FINAL]['precision']:.3f} / {th[FINAL]['recall']:.3f} / {th[FINAL]['F1']:.3f} at the val-chosen conf "
      f"{th[FINAL]['conf']:.3f}.")
    w(f"> - **Against the GUI's earlier model E001** (YOLO26s, v1) on the same test: mAP50-95 "
      f"{th['E001']['mAP50-95']:.3f} → {th[FINAL]['mAP50-95']:.3f} "
      f"({th[FINAL]['mAP50-95'] - th['E001']['mAP50-95']:+.4f}), recall {th['E001']['recall']:.3f} → "
      f"{th[FINAL]['recall']:.3f}, missed boxes {cc['E001']['FN']} → {cc[FINAL]['FN']}, false boxes "
      f"{cc['E001']['FP']} → {cc[FINAL]['FP']}.")
    w(f"> - **At a new site** (Zenodo sk, never trained on): mAP50-95 {skh['mAP50-95']:.3f}, precision "
      f"{skh['precision']:.3f} at the same conf: the model does not transfer well to an unfamiliar site.")
    w(f"> - **Tuning:** {sum(len(r) for r in ROUND1.values())} round-1 runs (3 sizes × 3 seeds, 1 skipped for time) "
      f"+ {len(r2_vals)} round-2 runs (5 combinations × 2 sizes); one run cleared the pre-declared bar "
      f"{bar:.5f}: {', '.join(above)}.")
    if lobster_only:
        w(f"> - **Compute:** {lobster_h:.1f} GPU-hours of recorded training for the lobster detector "
          f"(completed launches only).")
    else:
        w(f"> - **Compute:** {lobster_h:.1f} GPU-hours of recorded training for the lobster detector, "
          f"{all_h:.1f} including the Tritonia sea-slug runs (completed launches only).")
    w("")

    # 1 -----------------------------------------------------------------------------------------
    w("## 1. What was built, and the order it happened in")
    w("")
    w("Pipeline: raw frames → audit → session-level split → image manifest → YOLO dataset view → training "
      "(Colab / Kaggle GPU) → val-only selection → one declared test evaluation → local Streamlit GUI for "
      "inspection. Every stage communicates through committed files; every decision is in `docs/DECISIONS.md`.")
    w("")
    w(_table(["phase", "what", "decisions"], [
        ["Data foundation (Sept)", "raw audit, session split, frozen v1, raw fingerprint", "D-001–D-011"],
        ["v1 baselines (Sept–Oct)", "E001 YOLO26s raw; E004 enhanced images (v2); E005 RT-DETR (E003 diverged)",
         "D-012–D-027"],
        ["Val-only evaluation (Oct)", "per-session / per-lighting val metrics, IoU 0.5 matching fixed first", "D-020"],
        ["Model A data (Nov)", "Harsh's mixed data on our split, leakage check, frozen v4", "D-031, D-032"],
        ["Model A tuning (Nov 11–13)", "round 1 size × seed, round 2 combinations, E027 selected", "D-038–D-040, D-043"],
        ["Final test (Nov 13)", "E027 on v1 test and Zenodo sk, once each", "D-043"],
        ["E027 val-only evaluation (Nov 16)", "false positives on empty val images, per session / lighting", "D-020"],
    ] + ([] if lobster_only else [
        ["Tritonia sea slugs (Anu, Nov)", "second species, timelapse, point evaluation", "D-029/D-030, D-033–D-037, D-041/D-042"],
    ]), align="lll"))
    w("")

    # 2 -----------------------------------------------------------------------------------------
    w("## 2. Data")
    w("")
    w("**Split rule (D-001):** the unit is the recording session; all files of a session stay in one split. "
      "The v1 val and test frames are kept unchanged in every later version, so all lobster models are scored "
      "on the same 324 val and 348 test images.")
    w("")
    comp = v4_composition(inputs)
    kinds = ("original", "colour_change", "copy_paste", "diffusion")
    rows = []
    for split in ("train", "val", "test"):
        for origin in ("ours", "zenodo"):
            vals = [comp.get((split, origin, k), 0) for k in kinds]
            if sum(vals):
                label = split if split != "test" else ("test (v1)" if origin == "ours" else "test_zenodo_sk")
                rows.append([label, origin] + [f"{v:,}" for v in vals] + [f"{sum(vals):,}"])
    w("**Dataset v4 (Model A, D-031/D-032), images per split, origin and kind** "
      "(from `data/processed/v4/mixed_manifest.csv`):")
    w("")
    w(_table(["split", "origin", "original", "colour change", "copy-paste", "diffusion", "total"], rows, align="llrrrrr"))
    w("")
    w(f"Train: {counts['train']['images']:,} images ({counts['train']['positive']:,} with lobsters, "
      f"{counts['train']['negative']:,} confirmed empty, {counts['train']['instances']:,} boxes). Val "
      f"{counts['val']['images']} images / {counts['val']['instances']:,} boxes; v1 test {tc['images']} images "
      f"({tc['negative']} empty) / {tc['instances']:,} boxes; Zenodo sk {skcounts['images']} images "
      f"({skcounts['negative']} empty) / {skcounts['instances']} boxes.")
    w("")
    excl = v4_exclusions(inputs)
    w("Not built into v4: " + "; ".join(f"{n:,} {reason.replace('_', ' ')}" for reason, n in excl.items())
      + " (\"ancestor not train\" = a derived image whose source frame or pasted crop comes from a v1 val/test "
      "session, removed by the leakage check).")
    w("")
    w("Zenodo data: dataset 15755673, CC BY 4.0 (credit required). Site `mpa` is used in training; site `sk` "
      "is held out as the only new-site test (D-031).")
    w("")
    w("LIMITATIONS (§3): v1 test sessions share sites (mostly also dates) with train sessions, so the v1 test "
      "measures new sessions, not new sites (D-007); substrate/visibility metadata is mostly missing; zExtra "
      "(436 frames, no labels) is excluded and used only for qualitative GUI inspection.")
    w("")

    # 3 -----------------------------------------------------------------------------------------
    w("## 3. Models and experiments")
    w("")
    w("All lobster runs are YOLO26 (Ultralytics 8.4.155, COCO-pretrained checkpoints) except E005 (RT-DETR-l), "
      "at 640 px. One committed config per run names its parent and what changed; the full list with "
      "hyperparameters is in `final_report_experiments.csv`.")
    w("")
    rows = []
    for exp in list(V1_BASELINES) + [MODEL_A_FIRST]:
        r = reg[exp]
        rows.append([exp, r["model"].replace(".pt", ""), r["dataset_version"], r["epochs_completed"],
                     _f4(float(r["val_precision"])), _f4(float(r["val_recall"])), _f4(float(r["val_map50"])),
                     _f4(float(r["val_map50_95"])), r["changed"][:70] + ("…" if len(r["changed"]) > 70 else "")])
    rows.append([FINAL, "yolo26l", "v4", reg[FINAL]["epochs_completed"], _f4(float(reg[FINAL]["val_precision"])),
                 _f4(float(reg[FINAL]["val_recall"])), _f4(float(reg[FINAL]["val_map50"])),
                 _f4(float(reg[FINAL]["val_map50_95"])), "final model (round 2, C4)"])
    w(_table(["run", "model", "data", "epochs", "val P", "val R", "val mAP50", "val mAP50-95", "change vs parent"],
             rows, align="lllrrrrrl"))
    w("")
    w("Not registered: E002 (imgsz 1280, planned, not run), E003 (RT-DETR with AMP: diverged to NaN, rerun as "
      "E005, D-025), E018 (yolo26l seed 2: not run, round-1 time limit)"
      + ("." if lobster_only else ", E009 (Tritonia, run record lost)."))
    w("")

    # 4 -----------------------------------------------------------------------------------------
    w("## 4. How the model was tuned")
    w("")
    w("**Practices (all fixed before the runs they govern):**")
    w("")
    for item in (
        "One selection metric, declared before any comparison: val mAP50-95 (D-038 point 1). It picks the best "
        "epoch inside a run (Ultralytics fitness) and the best run, so one criterion is used throughout.",
        "Test data never chooses anything: val (324 images) for every choice, the v1 test and Zenodo sk only once, "
        "for the run declared in `experiments/final_candidates.csv` beforehand (CLAUDE.md rule 7).",
        "Seeds measure noise: round 1 ran each model size with 3 seeds, and the seed range became the bar a "
        "change must clear (D-038 point 4, D-040 point 4).",
        "Simplest model wins ties: a bigger model is chosen only if its gain over a smaller one exceeds the seed "
        "range.",
        "Committed config per run (parent + what changed), frozen dataset (v4 fingerprint verified in every run), "
        "run record with git commit, seed, environment and best.pt sha256; every package re-checked by sha256.",
        "Time-boxed queue: two runs in parallel on Kaggle's 2×T4, a run started only if its worst case finished "
        "before the deadline, so no run was ever cut off and skipped runs are recorded as \"not run (time)\".",
        "Ultralytics `model.tune()` (genetic search) was not used: it needs dozens to hundreds of runs (D-038 point 9).",
    ):
        w(f"- {item}")
    w("")
    w("**Round 1 (D-038/D-039): which model size?** yolo26s / m / l × seeds 0 / 1 / 2 on v4, 60-epoch ceiling, "
      "patience 10, otherwise as E007.")
    w("")
    rows = []
    for size, runs in ROUND1.items():
        cells = [f"{e} {_f4(val_map(inputs, e))}" for e in runs] + ["E018 not run (time)"] * (3 - len(runs))
        s = summ[size]
        rows.append([size] + cells + [_f4(s["mean"]), _f4(s["range"])])
    w(_table(["model", "seed 0", "seed 1", "seed 2", "mean", "range"], rows, align="llllrr"))
    w("")
    gaps = {s: summ[top_size]["mean"] - summ[s]["mean"] for s in ROUND1 if s != top_size}
    w(f"OBSERVED: the highest mean is {top_size} ({_f4(summ[top_size]['mean'])}). Gap to yolo26m "
      f"{_f4(gaps.get('yolo26m', 0))} ≤ yolo26m's range {_f4(summ['yolo26m']['range'])}, so no evidence l beats m; "
      f"gap to yolo26s {_f4(gaps.get('yolo26s', 0))} > both ranges. Rule (D-040 point 1) → **yolo26m** selected, "
      f"yolo26l kept for round 2 at the user's request. OBSERVED in `results.csv`: no round-1 run completed its "
      f"schedule (all stopped by patience at epochs 26–41 of 60), so none trained with an annealed learning rate "
      f"or without mosaic (D-040 context 4). That motivated round 2's shorter schedule that always completes.")
    w("")
    w(fig("final_round1.png", "Round 1 val mAP50-95 per model size and seed"))
    w("")
    w(f"**Round 2 (D-040): which settings?** Every run uses a 30-epoch schedule with patience 30 (the learning "
      f"rate decays fully; the last 10 epochs train without mosaic), on yolo26m and yolo26l, seed 0. A run "
      f"replaces the round-1 choice only if its val mAP50-95 is above the bar = yolo26m mean + range = "
      f"**{bar:.5f}**.")
    w("")
    rows = [[row[0], row[1], f"{row[2]} {_f4(r2_vals[row[2]])}", f"{row[3]} {_f4(r2_vals[row[3]])}"] for row in ROUND2]
    w(_table(["", "added to the schedule", "yolo26m", "yolo26l"], rows, align="llll"))
    w("")
    w(f"OBSERVED: only {', '.join(above)} clears the bar (margin {r2_vals[FINAL] - bar:.5f}) → selected and declared "
      f"alone (D-043). C4 is the best combination for both sizes; freezing the backbone (C5) is clearly worse; the "
      f"completed schedule alone (C1) is within round-1 seed noise (INFERRED, single seeds).")
    w("")
    w(fig("final_round2.png", "Round 2 val mAP50-95 per combination"))
    w("")
    hist = inputs["runs"][FINAL]["history"]
    best_i = max(range(len(hist)), key=lambda i: hist[i]["metrics/mAP50-95(B)"])
    w(f"E027 trained {len(hist)} epochs, best epoch {int(hist[best_i]['epoch'])} (val mAP50-95 "
      f"{hist[best_i]['metrics/mAP50-95(B)']:.4f} in `results.csv`; the final validation of best.pt after "
      f"training gives {val_map(inputs, FINAL):.4f}, the value used for selection), "
      f"{train_hours(inputs, FINAL):.2f} h on one T4 (sharing the session with a second run).")
    w("")
    w(fig("final_training_curves.png", "Val mAP50-95 per epoch for E001, E013 and E027"))
    w("")
    w(f"LIMITATION: 19 configurations were compared on the same 324 val images, so E027's val score is optimistic; "
      f"its margin over the bar ({r2_vals[FINAL] - bar:.5f}) is smaller than yolo26m's seed range "
      f"({summ['yolo26m']['range']:.5f}), so another seed could fall below it. Only the test results below are "
      f"unbiased.")
    w("")

    # 5 -----------------------------------------------------------------------------------------
    w("## 5. Final evaluation (held-out, once per model)")
    w("")
    rows = []
    for exp in ("E001", "E004"):
        h = th[exp]
        rows.append([f"{exp} ({reg[exp]['model'].replace('.pt', '')}, {reg[exp]['dataset_version']})",
                     _fmt(h["mAP50"]), _fmt(h["mAP50-95"]), _fmt(h["conf"]), _fmt(h["precision"]), _fmt(h["recall"]),
                     _fmt(h["F1"]), f"{cc[exp]['TP']} / {cc[exp]['FP']} / {cc[exp]['FN']}"])
    rows.append(["E005 (rtdetr-l, v1) ¹", _fmt(e005["mAP50"]), _fmt(e005["mAP50-95"]), _fmt(e005["conf"]),
                 _fmt(e005["precision"]), _fmt(e005["recall"]), _fmt(e005["F1"]), "not comparable ²"])
    h = th[FINAL]
    rows.append([f"**{FINAL}** (yolo26l, v4)", _fmt(h["mAP50"]), f"**{_fmt(h['mAP50-95'])}**", _fmt(h["conf"]),
                 _fmt(h["precision"]), _fmt(h["recall"]), _fmt(h["F1"]),
                 f"{cc[FINAL]['TP']} / {cc[FINAL]['FP']} / {cc[FINAL]['FN']}"])
    rows.append([f"{FINAL} on **Zenodo sk** (new site)", _fmt(skh["mAP50"]), _fmt(skh["mAP50-95"]), _fmt(skh["conf"]),
                 _fmt(skh["precision"]), _fmt(skh["recall"]), _fmt(skh["F1"]),
                 f"{skc['TP']} / {skc['FP']} / {skc['FN']}"])
    w(_table(["model", "mAP50", "mAP50-95", "conf (val)", "P", "R", "F1", "TP / FP / FN"], rows, align="lrrrrrrl"))
    w("")
    w("All rows except the last: the same v1 test (348 images, 27 empty, 1,512 boxes). P / R / F1 at each model's "
      "own val-chosen confidence (max mean F1 on val). TP / FP / FN from Ultralytics' confusion matrix. ¹ quoted "
      "from docs/EXPERIMENTS.md (record on Drive). ² RT-DETR's matrix may use another confidence threshold "
      "(UNKNOWN, EXPERIMENTS.md).")
    w("")
    w(fig("final_test_comparison.png", "Test metrics of E001, E004, E027 and E027 on Zenodo sk"))
    w("")
    w(fig("final_confusion.png", "Confusion matrices on the test sets"))
    w("")
    metric_names = ("mAP50", "mAP50-95", "precision", "recall", "F1")
    v1_rows = {**{e: th[e] for e in TESTED}, "E005": e005}
    best_of = {m: max(v1_rows, key=lambda e: v1_rows[e][m]) for m in metric_names}
    won = [m for m in metric_names if best_of[m] == FINAL]
    lost = [f"{m} ({best_of[m]} {v1_rows[best_of[m]][m]:.3f} vs {v1_rows[FINAL][m]:.3f})"
            for m in metric_names if best_of[m] != FINAL]
    higher = [e for e in TESTED if th[e]["mAP50-95"] > val_map(inputs, e)]
    w(f"- OBSERVED: among the four v1 models, {FINAL} is highest on {', '.join(won)}"
      + (f"; not on {', '.join(lost)}" if lost else "") + f". Against E001 it finds "
      f"{cc[FINAL]['TP'] - cc['E001']['TP']} more lobsters with {cc['E001']['FP'] - cc[FINAL]['FP']} fewer false "
      f"boxes. Test mAP50-95 is higher than val for {', '.join(higher)} (E027 {th[FINAL]['mAP50-95']:.4f} vs "
      f"{val_map(inputs, FINAL):.4f}). INFERRED: the v1 test frames are easier than the val frames for these "
      f"models, so val and test numbers are not interchangeable; compare models within one split.")
    w(f"- OBSERVED: on Zenodo sk, {skc['FP']} false boxes against {skc['TP']} found lobsters, {skc['FN']} of "
      f"{skcounts['instances']} missed. INFERRED: the val-chosen confidence is too low for sk (at sk's own max-F1 "
      f"confidence Ultralytics reports P {sk['metrics']['ultralytics_reported']['precision']:.3f} / R "
      f"{sk['metrics']['ultralytics_reported']['recall']:.3f}, but choosing a threshold on sk is not allowed).")
    w("- INFERRED, not tested: the gain over E001 comes from more data (v4) and the larger model together; there is "
      "no run that separates the two (E007 = v4 with yolo26s: val 0.5993, below E001's 0.6115).")
    w("- UNKNOWN (records keep no per-image predictions): false positives on the empty test images, and per-session "
      "test results for E027.")
    w("")
    w("**Val-only secondary metrics (D-020):** `scripts/val_eval.py` re-runs each model's best.pt on the 324 val "
      "images with its own matcher (IoU ≥ 0.5, greedy) at the val-chosen confidence, and keeps per-image results, "
      "so it can count false positives on the 55 lobster-free val images and split results by session and lighting.")
    w("")
    vrows = []
    lab = {"E001": "E001 (yolo26s, v1)", "E004": "E004 (yolo26s, v2)", FINAL: f"**{FINAL}** (yolo26l, v4)"}
    for e in ("E001", "E004", FINAL):
        o, pl = ve[e]["overall"], ve[e]["per_lighting"]
        red, nf = pl["red_light_or_night"], pl["not_flagged"]
        vrows.append([lab[e], _fmt(ve[e]["operating_conf"]), f"{o['tp']} / {o['fp']} / {o['fn']}",
                      f"{o['precision']:.3f} / {o['recall']:.3f} / {o['f1']:.3f}", f"{o['ap50']:.3f}",
                      f"{o['negative_images_with_detection']} of {o['negative_images']} "
                      f"({o['detections_on_negative_images']})",
                      f"{red['ap50']:.3f} / {red['recall']:.3f}", f"{nf['ap50']:.3f} / {nf['recall']:.3f}"])
    w(_table(["model", "conf (val)", "TP / FP / FN", "P / R / F1", "AP50", "empty images with a detection (dets)",
              "red-light AP50 / R", "not-flagged AP50 / R"], vrows, align="lrlrrlll"))
    w("")
    vo, eo = ve[FINAL]["overall"], ve["E001"]["overall"]
    red_f = ve[FINAL]["per_lighting"]["red_light_or_night"]
    big = {s: m for s, m in ve[FINAL]["per_session"].items() if m["gt_boxes"] >= 50}
    worst_p = min(big.items(), key=lambda kv: kv[1]["precision"])
    w(f"- OBSERVED: at its val-chosen confidence {FINAL} finds {vo['tp'] - eo['tp']} more of the "
      f"{vo['gt_boxes']:,} val boxes than E001 with {vo['fp'] - eo['fp']} more false boxes; own-matcher AP50 "
      f"{vo['ap50']:.3f} vs {eo['ap50']:.3f} (equal at 3 decimals). It puts a detection on {vo['negative_images_with_detection']} of the "
      f"{vo['negative_images']} lobster-free val images (E001 {eo['negative_images_with_detection']}); "
      f"{red_f['negative_images_with_detection']} of those are red-light frames.")
    w(f"- OBSERVED: among the {len(big)} val sessions with at least 50 boxes, the lowest precision is {worst_p[0]} (P {worst_p[1]['precision']:.3f}, "
      f"{worst_p[1]['fp']} false boxes for {worst_p[1]['tp']} found lobsters). Per-session tables: "
      f"`docs/EXPERIMENTS.md`, section \"Val secondary metrics: E027 (D-020)\".")
    w("- LIMITATION: val also chose each checkpoint, each confidence and E027 itself, so these numbers are "
      "optimistic; red-light val is 2 sessions / 31 boxes. They compare models; they do not estimate deployment "
      "quality.")
    w("")

    # 6 -----------------------------------------------------------------------------------------
    w("## 6. The GUI model: before and after")
    w("")
    w(_table(["", "before: E001", "now: E027"], [
        ["model", "YOLO26s", "YOLO26l"],
        ["training data", f"v1: {tests['E001']['dataset']['counts']['train']['images']:,} raw frames",
         f"v4: {counts['train']['images']:,} images (originals + Zenodo mpa + derived copies)"],
        ["val mAP50-95", _f4(val_map(inputs, "E001")), _f4(val_map(inputs, FINAL))],
        ["v1 test mAP50-95 / mAP50", f"{th['E001']['mAP50-95']:.4f} / {th['E001']['mAP50']:.4f}",
         f"{th[FINAL]['mAP50-95']:.4f} / {th[FINAL]['mAP50']:.4f}"],
        ["v1 test P / R / F1 (val conf)", f"{th['E001']['precision']:.3f} / {th['E001']['recall']:.3f} / {th['E001']['F1']:.3f}",
         f"{th[FINAL]['precision']:.3f} / {th[FINAL]['recall']:.3f} / {th[FINAL]['F1']:.3f}"],
        ["v1 test found / false / missed", f"{cc['E001']['TP']} / {cc['E001']['FP']} / {cc['E001']['FN']}",
         f"{cc[FINAL]['TP']} / {cc[FINAL]['FP']} / {cc[FINAL]['FN']}"],
        ["Zenodo sk mAP50-95", "never evaluated", f"{skh['mAP50-95']:.4f}"],
        ["operating confidence (chosen on val)", _fmt(th["E001"]["conf"]), _fmt(th[FINAL]["conf"])],
        ["best.pt size", f"{best_size_mb['E001']:.1f} MB", f"{best_size_mb['E027']:.1f} MB"],
        ["inference on a Tesla T4 (test run) ³", f"{speed['E001']:.1f} ms/image", f"{speed['E027']:.1f} ms/image"],
        ["parameters / GFLOPs (640 px)", f"{cx['E001']['parameters'] / 1e6:.2f} M / {cx['E001']['gflops']:.1f}",
         f"{cx[FINAL]['parameters'] / 1e6:.2f} M / {cx[FINAL]['gflops']:.1f}"],
        ["inference, batch 1, median on a Tesla T4 (val eval) ⁴", f"{b1['E001']:.1f} ms/image",
         f"{b1[FINAL]:.1f} ms/image"],
        ["val: empty images with a detection (of 55)", str(ve["E001"]["overall"]["negative_images_with_detection"]),
         str(ve[FINAL]["overall"]["negative_images_with_detection"])],
    ], align="lll"))
    w("")
    w("³ Ultralytics' own timing during each test run: different sessions (Colab 2026-09-19 vs Kaggle 2026-11-13) and "
      "library builds, so a rough guide to the size difference, not a benchmark.")
    w("⁴ `val_eval.py`, one image at a time, first image dropped as warm-up; same GPU type for both (Colab "
      "2026-10-11 vs Kaggle 2026-11-16), so closer to a like-for-like comparison than ³, still not a benchmark.")
    w("")
    w("The GUI (`scripts/gui_app.py`) now offers E027 (default), E001 and E004, shows each model's recorded scores, "
      "can run E027 and E001 side by side on the same image at their own validated settings, and can browse the "
      "Zenodo sk frames. It computes no metric; per-image counts are observations (rule 7).")
    w("")

    # 7 -----------------------------------------------------------------------------------------
    w("## 7. Compute")
    w("")
    w(_table(["phase", "platform", "runs", "GPU-hours", "interrupted launches not counted"],
             [[r["phase"], r["platform"], str(r["runs"]), f"{r['gpu_hours']:.2f}", str(r["interrupted"])] for r in phases]
             + [["**total**", "", str(sum(r["runs"] for r in phases)), f"**{all_h:.2f}**", ""]], align="llrrr"))
    w("")
    w(fig("final_gpu_hours.png", "GPU-hours per phase"))
    w("")
    w("GPU-hours = recorded training time of completed launches (`run_record.json` → duration), summed per run; two "
      "runs sharing a 2×T4 session count twice. Interrupted launches (E001: 2, E005: 4) have no end time, so the "
      "Colab figure is a lower bound. Excluded: smoke tests, dataset builds, evaluations and the unregistered E003 / "
      f"E009 runs. Kaggle sessions (wall clock, quoted): round 1 about {QUOTED['round1_session_h']['value']} h, "
      f"round 2 about {QUOTED['round2_session_h']['value']} h; Kaggle counts one quota hour per session hour "
      f"whatever the number of GPUs (D-040). Quota used after round 1: {QUOTED['quota_after_round1_h']['value']:.0f} "
      "of 30 h (quoted); after round 2 and the final test: UNKNOWN — read the quota meter.")
    w("")

    # 8 -----------------------------------------------------------------------------------------
    w("## 8. What the selection metric means (D-038 point 8)")
    w("")
    w("mAP50 counts a detection as correct when its box overlaps the annotated box with IoU ≥ 0.5: *was the "
      "lobster found?* mAP50-95 averages the same score over IoU thresholds 0.50, 0.55, …, 0.95, so it also "
      "rewards tight boxes. Trajectories are built from box positions in pixels, so both finding the lobster and "
      "placing the box accurately matter; mAP50-95 measures both, and Ultralytics already uses it to pick the "
      "best epoch, so one criterion runs from epoch to final choice. LIMITATION: neither is a tracking metric "
      "(no ground-truth track IDs for lobsters, CLAUDE.md rule 9); the operating confidence (precision/recall "
      "trade-off) is chosen separately on val.")
    w("")

    # 9 -----------------------------------------------------------------------------------------
    w("## 9. What the final model can and cannot do")
    w("")
    for item in (
        f"OBSERVED: on new recording sessions at known sites it finds about {th[FINAL]['recall'] * 100:.0f}% of "
        f"lobster boxes with about {th[FINAL]['precision'] * 100:.0f}% of its boxes correct (v1 test, val-chosen conf).",
        "OBSERVED: it was trained on a mix of our raw frames, Zenodo mpa frames and three kinds of synthetic or "
        "augmented copies (colour change, copy-paste, diffusion); it is evaluated on raw frames only.",
        f"OBSERVED: at a new site (Zenodo sk) quality drops sharply (mAP50-95 {skh['mAP50-95']:.3f}); expect many "
        "false boxes and missed lobsters on footage from a new location until it is fine-tuned there.",
        "LIMITATION: one class (lobster); pixels only (no calibration, so no sizes in cm, rule 9); boxes, not outlines.",
        "LIMITATION: one seed of the final configuration; leave-site-out cross-validation was deferred (D-038 point 9).",
    ):
        w(f"- {item}")
    w("")

    # 10 ----------------------------------------------------------------------------------------
    if not lobster_only:
        w("## 10. Second species: Tritonia sea slugs (Anu)")
        w("")
        tri_h = next(r["gpu_hours"] for r in phases if "Tritonia" in r["phase"])
        tp = QUOTED["tritonia_test_point_ap"]["values"]
        w(f"{len(TRITONIA)} registered runs (datasets v3–v8, {tri_h:.1f} GPU-hours), scored by point AP against "
          f"the clicked MTrackJ crosses (D-033), not by box mAP. Final tests: E006 point AP {tp['E006']} and E010 "
          f"{tp['E010']} (quoted). The tracking work (T002) was in open PR #27 on 2026-11-13, not yet on the "
          f"integration branch, so it is not covered here.")
        w("")

    # 11 ----------------------------------------------------------------------------------------
    w("## 11. Open gaps (UNKNOWN until measured)")
    w("")
    for item in (
        "False positives on empty test images (27 v1, 285 sk) and per-session test results: the final-test records "
        "keep no per-image predictions; measuring them would need a new decision.",
        "A second seed of E027's configuration, and leave-site-out cross-validation (D-038 point 9).",
        "Kaggle quota left after round 2 and the final test.",
    ):
        w(f"- {item}")
    w("")

    # Sources -------------------------------------------------------------------------------------
    w("## Quoted numbers (not computed from a committed record)")
    w("")
    w(_table(["number", "value", "source"], [
        [key.replace("_", " "), json.dumps(q.get("values", q.get("value"))), q["source"]] for key, q in QUOTED.items()
        if not (lobster_only and key.startswith("tritonia"))
    ], align="lll"))
    w("")
    w("Ultralytics' own plots (PR curves, confusion matrices, sample predictions) are off-repo with the run "
      "packages: E027 val in the round-2 output (`E027_package.zip`, sha256 `55a58487…`), test and sk in the "
      "final-test output (`E027_final_test_package.zip`, sha256 `c72921ca…`).")
    w("")
    return "\n".join(out)


def write_experiments_csv(inputs: dict, path: Path, lobster_only: bool = False) -> None:
    rows = experiments_rows(inputs, lobster_only)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_final_report(root: Path, out_dir: Path, lobster_only: bool = False) -> Path:
    """Write the pack to `out_dir`. `lobster_only` leaves out the other species (copy for the GUI bundle)."""
    inputs = load_inputs(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_figures(inputs, out_dir / "figures", lobster_only)
    write_experiments_csv(inputs, out_dir / "final_report_experiments.csv", lobster_only)
    report = out_dir / "final_report_data.md"
    report.write_text(render_markdown(inputs, lobster_only=lobster_only), encoding="utf-8")
    return report
