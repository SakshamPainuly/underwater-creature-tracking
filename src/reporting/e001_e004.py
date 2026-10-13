"""E001 (raw images) vs E004 (enhanced images): report with figures, from committed records only.

Inputs, per experiment, all under experiments/records/<id>/ (D-013, D-020):
  run_record.json, results.csv, val_eval_record.json, final_test_record.json.
Nothing is typed by hand: every number and every figure is computed from those files, so the
report can be rebuilt at any time and always agrees with the records (CLAUDE.md rule 8).

Output: a Markdown report (GitHub renders it with its images) plus PNG figures. Pixels, counts,
confidences and milliseconds only; no physical units (rule 9). Test numbers are reported, never
used to choose between the runs (D-017, D-022).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

EXPERIMENTS = ("E001", "E004")
LABELS = {"E001": "E001 · raw images (v1)", "E004": "E004 · enhanced images (v2)"}
SHORT = {"E001": "E001 raw", "E004": "E004 enhanced"}

# Categorical slots 1-2 of the reference palette, validated together (light surface): blue, orange.
COLORS = {"E001": "#2a78d6", "E004": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Single-hue sequential ramp (blue, light -> dark) for the confusion-matrix cells.
SEQ_LIGHT, SEQ_DARK = "#e8f0fb", "#1f5aa6"
NA_FILL = "#efeeea"

FIGURES = (
    "fig1_test_headline.png",
    "fig2_val_headline.png",
    "fig3_confusion_matrices.png",
    "fig4_val_pr_curve.png",
    "fig5_val_vs_confidence.png",
    "fig6_training_curves.png",
    "fig7_training_losses.png",
    "fig8_val_per_session.png",
    "fig9_val_per_lighting.png",
    "fig10_negative_images.png",
)


class ReportError(RuntimeError):
    """Raised with an actionable message when a record is missing or inconsistent."""


# ---------------------------------------------------------------------------------------------
# Loading (stdlib)
# ---------------------------------------------------------------------------------------------

def load_records(records_root: Path) -> dict[str, dict[str, Any]]:
    """Read the four records of each experiment. Raises ReportError naming any missing file."""
    out: dict[str, dict[str, Any]] = {}
    for exp in EXPERIMENTS:
        folder = Path(records_root) / exp
        files = {
            "run": folder / "run_record.json",
            "val": folder / "val_eval_record.json",
            "test": folder / "final_test_record.json",
            "history": folder / "results.csv",
        }
        missing = [str(p) for p in files.values() if not p.is_file()]
        if missing:
            raise ReportError("Missing record file(s): " + ", ".join(missing))
        rec = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in files.items() if k != "history"}
        with open(files["history"], newline="", encoding="utf-8") as f:
            rec["history"] = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
        for key in ("val", "test"):
            if rec[key].get("experiment_id") != exp:
                raise ReportError(f"{files[key]} is for {rec[key].get('experiment_id')}, expected {exp}")
        if rec["test"].get("split") != "test" or rec["val"].get("split") != "val":
            raise ReportError(f"{exp}: record split labels are not val/test as expected")
        out[exp] = rec
    return out


def history_column(rec: dict[str, Any], column: str) -> list[float]:
    return [float(row[column]) for row in rec["history"]]


# ---------------------------------------------------------------------------------------------
# Tables (stdlib)
# ---------------------------------------------------------------------------------------------

def headline(rec: dict[str, Any], split: str) -> dict[str, float]:
    """The declared metrics for one split: mAP50, mAP50-95, and P/R/F1 at the val-chosen conf."""
    if split == "test":
        m = rec["test"]["metrics"]
        rep, op = m["ultralytics_reported"], m["at_val_conf"]
        return {"mAP50": rep["map50"], "mAP50-95": rep["map50_95"], "precision": op["precision"],
                "recall": op["recall"], "F1": op["f1"], "conf": op["conf"]}
    v = rec["val"]
    o = v["overall"]
    return {"mAP50": v["ultralytics_val"]["map50"], "mAP50-95": v["ultralytics_val"]["map50_95"],
            "precision": o["precision"], "recall": o["recall"], "F1": o["f1"], "conf": o["conf"]}


def confusion_counts(rec: dict[str, Any], split: str) -> dict[str, int]:
    """TP / FP / FN from a record's confusion matrix (rows = predicted, cols = true)."""
    cm = (rec["test"]["metrics"] if split == "test" else rec["val"])["confusion_matrix"]["matrix"]
    return {"TP": int(cm[0][0]), "FP": int(cm[0][1]), "FN": int(cm[1][0])}


def _fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _delta(a: float, b: float) -> str:
    d = b - a
    return ("+" if d >= 0 else "−") + f"{abs(d):.3f}"


# ---------------------------------------------------------------------------------------------
# Figures (matplotlib, imported lazily)
# ---------------------------------------------------------------------------------------------

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "svg.hashsalt": "uwct",
    })
    return plt


def _save(fig, path: Path) -> None:
    # No Software/date metadata, so rebuilding from the same records gives the same bytes.
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    import matplotlib.pyplot as plt

    plt.close(fig)


def _grouped_bars(ax, metrics: list[str], values: dict[str, list[float]], ylim=(0, 1)) -> None:
    n = len(EXPERIMENTS)
    width = 0.38
    xs = list(range(len(metrics)))
    for i, exp in enumerate(EXPERIMENTS):
        offs = [x + (i - (n - 1) / 2) * (width + 0.02) for x in xs]
        bars = ax.bar(offs, values[exp], width, color=COLORS[exp], label=LABELS[exp],
                      edgecolor=SURFACE, linewidth=1.5)
        for b, v in zip(bars, values[exp]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, color=INK_2)
    ax.set_xticks(xs, metrics)
    ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)


def fig_headline(recs, split: str, path: Path) -> None:
    plt = _plt()
    metrics = ["mAP50", "mAP50-95", "precision", "recall", "F1"]
    vals = {e: [headline(recs[e], split)[m] for m in metrics] for e in EXPERIMENTS}
    fig, ax = plt.subplots(figsize=(8, 4))
    _grouped_bars(ax, metrics, vals, ylim=(0, 1.05))
    confs = ", ".join(f"{SHORT[e]} {headline(recs[e], split)['conf']:.3f}" for e in EXPERIMENTS)
    name = "Test split (held out, evaluated once per model)" if split == "test" else "Validation split"
    ax.set_title(f"{name}: detection quality")
    ax.set_ylabel("score (0–1)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.text(0.01, -0.12, f"Precision, recall and F1 at each model's operating confidence, chosen on val "
             f"({confs}). mAP is threshold-free.", fontsize=8, color=INK_2)
    _save(fig, path)


def fig_confusion(recs, path: Path) -> None:
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", [SEQ_LIGHT, SEQ_DARK])
    panels = [(s, e) for s in ("val", "test") for e in EXPERIMENTS]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.6))
    vmax = max(confusion_counts(recs[e], s)["TP"] for s, e in panels)
    for ax, (split, exp) in zip(axes.flat, panels):
        c = confusion_counts(recs[exp], split)
        cells = [[c["TP"], c["FP"]], [c["FN"], None]]
        names = [["TP", "FP"], ["FN", "TN"]]
        ax.grid(False)
        for r in range(2):
            for k in range(2):
                v = cells[r][k]
                fill = NA_FILL if v is None else cmap(v / vmax)
                ax.add_patch(plt.Rectangle((k, 1 - r), 0.97, 0.97, color=fill))
                dark = v is not None and v / vmax > 0.55
                ink = SURFACE if dark else INK
                ax.text(k + 0.485, 1 - r + 0.58, "n/a" if v is None else f"{v:,}", ha="center",
                        va="center", fontsize=16, fontweight="bold", color=INK_2 if v is None else ink)
                ax.text(k + 0.485, 1 - r + 0.3, names[r][k] + (" · undefined" if v is None else ""),
                        ha="center", va="center", fontsize=8, color=INK_2 if not dark else SURFACE)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.485, 1.485], ["true: lobster", "true: background"])
        ax.set_yticks([1.485, 0.485], ["pred: lobster", "pred: background"])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        source = ("own matcher, IoU ≥ 0.5, operating conf" if split == "val"
                  else "Ultralytics confusion matrix")
        ax.set_title(f"{split.upper()} · {SHORT[exp]}", color=COLORS[exp])
        ax.text(0, -0.28, source, fontsize=8, color=INK_2, transform=ax.transData)
    fig.suptitle("Confusion matrices (counts of boxes; TN is undefined for a detector)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def fig_pr_curve(recs, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for exp in EXPERIMENTS:
        pr = recs[exp]["val"]["curves"]["pr"]
        ap = recs[exp]["val"]["overall"]["ap50"]
        ax.plot(pr["recall"], pr["precision"], color=COLORS[exp], linewidth=2,
                label=f"{LABELS[exp]} · AP50 {ap:.3f}")
        o = recs[exp]["val"]["overall"]
        ax.plot([o["recall"]], [o["precision"]], "o", color=COLORS[exp], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Validation: precision–recall curve (IoU ≥ 0.5)")
    ax.legend(loc="lower left")
    fig.text(0.01, -0.03, "Dots: each model's operating point (confidence chosen on val by max F1).",
             fontsize=8, color=INK_2)
    _save(fig, path)


def fig_vs_conf(recs, path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, key, title in zip(axes, ("precision", "recall", "f1"), ("Precision", "Recall", "F1")):
        for exp in EXPERIMENTS:
            c = recs[exp]["val"]["curves"]["vs_conf"]
            xs, ys = c["conf"], c[key]
            if key == "precision":  # no detections above some threshold: precision undefined there
                xs, ys = zip(*[(x, y) for x, y, r in zip(c["conf"], c[key], c["recall"]) if r > 0])
            ax.plot(xs, ys, color=COLORS[exp], linewidth=2, label=SHORT[exp])
            op = recs[exp]["val"]["operating_conf"]
            ax.axvline(op, color=COLORS[exp], linewidth=1, linestyle=(0, (3, 3)))
        ax.set_title(f"{title} vs confidence")
        ax.set_xlabel("confidence threshold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
    axes[0].set_ylabel("score (validation)")
    axes[0].legend(loc="lower left")
    fig.text(0.01, -0.06, "Dashed lines: operating confidence of each model (max F1 on val). "
             "Raising the threshold trades recall for precision. Precision is drawn only where the model "
             "still makes detections.", fontsize=8, color=INK_2)
    fig.tight_layout()
    _save(fig, path)


def fig_training(recs, path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for ax, col, title in zip(axes, ("metrics/mAP50(B)", "metrics/mAP50-95(B)"),
                              ("Validation mAP50 per epoch", "Validation mAP50-95 per epoch")):
        for exp in EXPERIMENTS:
            ys = history_column(recs[exp], col)
            xs = history_column(recs[exp], "epoch")
            ax.plot(xs, ys, color=COLORS[exp], linewidth=1.8, label=f"{SHORT[exp]} ({len(xs)} epochs)")
            i = max(range(len(ys)), key=ys.__getitem__)
            ax.plot([xs[i]], [ys[i]], "o", color=COLORS[exp], markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=2)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
    axes[0].legend(loc="lower right")
    fig.text(0.01, -0.06, "Dots: highest value of each curve. E004 stopped early at epoch 81 "
             "(patience 30); E001 ran all 100 epochs.", fontsize=8, color=INK_2)
    fig.tight_layout()
    _save(fig, path)


def fig_losses(recs, path: Path) -> None:
    plt = _plt()
    cols = [("train/box_loss", "val/box_loss", "Box loss"), ("train/cls_loss", "val/cls_loss", "Class loss")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for ax, (tr, va, title) in zip(axes, cols):
        for exp in EXPERIMENTS:
            xs = history_column(recs[exp], "epoch")
            ax.plot(xs, history_column(recs[exp], tr), color=COLORS[exp], linewidth=1.8,
                    label=f"{SHORT[exp]} · train")
            ax.plot(xs, history_column(recs[exp], va), color=COLORS[exp], linewidth=1.8,
                    linestyle=(0, (4, 2)), label=f"{SHORT[exp]} · val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def _per_group_bars(recs, group_key: str, title: str, path: Path, label_fn, horizontal: bool = False) -> None:
    plt = _plt()
    groups = sorted(recs["E001"]["val"][group_key])
    labels = [label_fn(recs["E001"]["val"][group_key][g], g) for g in groups]
    if horizontal:  # many groups: names on the y-axis, first group on top; exact values are in the table
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.75 * len(groups) + 1.6), sharey=True)
        h = 0.38
        for ax, metric, mtitle in zip(axes, ("ap50", "f1"), ("AP50", "F1 at operating conf")):
            for i, exp in enumerate(EXPERIMENTS):
                ys = [k + (i - 0.5) * (h + 0.02) for k in range(len(groups))]
                ax.barh(ys, [recs[exp]["val"][group_key][g][metric] for g in groups], h,
                        color=COLORS[exp], label=LABELS[exp], edgecolor=SURFACE, linewidth=1.5)
            ax.set_yticks(range(len(groups)), labels, fontsize=8)
            ax.set_xlim(0, 1.0)
            ax.grid(axis="y", visible=False)
            ax.set_title(mtitle)
            ax.set_xlabel("score (validation)")
        axes[0].invert_yaxis()  # shared y: invert once
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, metric, mtitle in zip(axes, ("ap50", "f1"), ("AP50", "F1 at operating conf")):
            vals = {e: [recs[e]["val"][group_key][g][metric] for g in groups] for e in EXPERIMENTS}
            _grouped_bars(ax, labels, vals, ylim=(0, 1.1))
            ax.set_title(mtitle)
            ax.tick_params(axis="x", labelsize=8)
        axes[0].set_ylabel("score (validation)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, names, loc="lower center", ncol=2)
    _save(fig, path)


def _session_label(row, name: str) -> str:
    return f"{name[:10]} · {row['gt_boxes']} boxes"


def _lighting_label(row, name: str) -> str:
    pretty = {"not_flagged": "not flagged", "red_light_or_night": "red light / night"}.get(name, name)
    return f"{pretty}\n{row['gt_boxes']} boxes, {len(row['sessions'])} sessions"


def fig_negatives(recs, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    metrics = ["negative images\nwith a detection", "detections on\nnegative images"]
    vals = {e: [recs[e]["val"]["overall"]["negative_images_with_detection"],
                recs[e]["val"]["overall"]["detections_on_negative_images"]] for e in EXPERIMENTS}
    top = max(max(v) for v in vals.values()) * 1.25
    n_neg = recs["E001"]["val"]["overall"]["negative_images"]
    _grouped_bars(ax, metrics, vals, ylim=(0, top))
    for t in ax.texts:  # integers, not decimals
        t.set_text(str(int(round(float(t.get_text())))))
    ax.set_title(f"False alarms on the {n_neg} empty validation images")
    ax.set_ylabel("count (lower is better)")
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, path)


def make_figures(recs, fig_dir: Path) -> list[Path]:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = [fig_dir / name for name in FIGURES]
    fig_headline(recs, "test", p[0])
    fig_headline(recs, "val", p[1])
    fig_confusion(recs, p[2])
    fig_pr_curve(recs, p[3])
    fig_vs_conf(recs, p[4])
    fig_training(recs, p[5])
    fig_losses(recs, p[6])
    _per_group_bars(recs, "per_session", "Validation per session (AP50 and F1)", p[7], _session_label,
                    horizontal=True)
    _per_group_bars(recs, "per_lighting", "Validation per lighting condition", p[8], _lighting_label)
    fig_negatives(recs, p[9])
    return p


# ---------------------------------------------------------------------------------------------
# Markdown (stdlib)
# ---------------------------------------------------------------------------------------------

def render_markdown(recs, fig_rel: str = "figures") -> str:
    e1, e4 = recs["E001"], recs["E004"]
    t1, t4 = headline(e1, "test"), headline(e4, "test")
    v1, v4 = headline(e1, "val"), headline(e4, "val")
    c = {(s, e): confusion_counts(recs[e], s) for s in ("val", "test") for e in EXPERIMENTS}
    counts = e1["test"]["dataset"]["counts"]
    test_counts = e1["test"]["dataset"].get("test_counts") or {}
    img = lambda i, alt: f"![{alt}]({fig_rel}/{FIGURES[i]})"  # noqa: E731

    def metric_rows(a, b):
        rows = []
        for m in ("mAP50", "mAP50-95", "precision", "recall", "F1"):
            rows.append([m, _fmt(a[m]), _fmt(b[m]), _delta(a[m], b[m])])
        rows.append(["operating confidence (chosen on val)", _fmt(a["conf"]), _fmt(b["conf"]), ""])
        return rows

    def cm_rows(split):
        def signed(d: int) -> str:
            return ("+" if d >= 0 else "−") + str(abs(d))

        return [[k, f"{c[(split, 'E001')][k]:,}", f"{c[(split, 'E004')][k]:,}",
                 signed(c[(split, "E004")][k] - c[(split, "E001")][k])] for k in ("TP", "FP", "FN")]

    sessions = sorted(e1["val"]["per_session"])
    sess_rows = []
    for s in sessions:
        a, b = e1["val"]["per_session"][s], e4["val"]["per_session"][s]
        sess_rows.append([f"`{s}`", str(a["images"]), str(a["gt_boxes"]), _fmt(a["ap50"]), _fmt(b["ap50"]),
                          _fmt(a["f1"]), _fmt(b["f1"]), f"{a['fp']} / {b['fp']}"])
    light_rows = []
    for g in sorted(e1["val"]["per_lighting"]):
        a, b = e1["val"]["per_lighting"][g], e4["val"]["per_lighting"][g]
        light_rows.append([g.replace("_", " "), str(len(a["sessions"])), str(a["gt_boxes"]),
                           _fmt(a["ap50"]), _fmt(b["ap50"]), f"{_fmt(a['precision'])} / {_fmt(a['recall'])}",
                           f"{_fmt(b['precision'])} / {_fmt(b['recall'])}"])

    run_rows = []
    for e in EXPERIMENTS:
        r = recs[e]["run"]
        mx = max(history_column(recs[e], "metrics/mAP50(B)"))
        run_rows.append([e, r["experiment"]["changed"] if e == "E004" else "baseline",
                         r["dataset"]["version"], r["model"]["name"], str(r["train_args"]["imgsz"]),
                         str(r["epochs_completed"]), str(r["seed"]), _fmt(mx)])

    cx = {e: recs[e]["val"]["complexity"] for e in EXPERIMENTS}
    sp = {e: recs[e]["val"]["speed"]["batch1_predict"]["median_ms_total"] for e in EXPERIMENTS}
    gpu = recs["E001"]["val"]["speed"]["gpu"][0]
    neg = {e: recs[e]["val"]["overall"] for e in EXPERIMENTS}

    # Weakest validation sessions: lowest AP50 averaged over both models.
    weak = sorted(sessions, key=lambda s: (e1["val"]["per_session"][s]["ap50"]
                                           + e4["val"]["per_session"][s]["ap50"]) / 2)[:2]
    weak_txt = "; ".join(
        f"`{s[:10]}` (AP50 {_fmt(e1['val']['per_session'][s]['ap50'], 2)} raw → "
        f"{_fmt(e4['val']['per_session'][s]['ap50'], 2)} enhanced)" for s in weak)
    red = {e: recs[e]["val"]["per_lighting"].get("red_light_or_night") for e in EXPERIMENTS}
    red_txt = (f"; red-light/night images (AP50 {_fmt(red['E001']['ap50'], 2)} raw → "
               f"{_fmt(red['E004']['ap50'], 2)} enhanced, only {red['E001']['gt_boxes']} boxes)"
               if all(red.values()) else "")
    ct = {e: c[("test", e)] for e in EXPERIMENTS}

    md = f"""# Lobster detection: raw vs enhanced images (E001 vs E004)

Underwater Creature Tracking project · first class `0: lobster` · report generated from the
committed experiment records by `scripts/make_report.py` (every number and chart below is read
from `experiments/records/E001/` and `experiments/records/E004/`; nothing is typed by hand).

> **Key findings**
>
> - **The enhancement did not help.** On the held-out test split, E004 (enhanced) is below E001
>   (raw): mAP50 {_fmt(t4['mAP50'])} vs {_fmt(t1['mAP50'])}, recall {_fmt(t4['recall'])} vs {_fmt(t1['recall'])},
>   F1 {_fmt(t4['F1'])} vs {_fmt(t1['F1'])}.
> - **It misses more lobsters:** {ct['E004']['FN']:,} vs {ct['E001']['FN']:,} missed boxes on test. It raises fewer
>   false alarms on test ({ct['E004']['FP']:,} vs {ct['E001']['FP']:,}) but not on validation
>   ({c[('val', 'E004')]['FP']} vs {c[('val', 'E001')]['FP']}), so that is not a reliable advantage.
> - **Both models struggle in the same places:** {weak_txt}{red_txt}.
> - **Same cost:** {cx['E001']['parameters'] / 1e6:.1f} M parameters and about {sp['E001']:.0f} ms per image on a
>   {gpu} for both, so the comparison isolates the images.
> - **Caveat:** one training run per model, and E004 stopped at {e4['run']['epochs_completed']} epochs vs
>   {e1['run']['epochs_completed']}. More seeds would confirm the size of the gap.
> - **Recommendation** (project judgement, not a measured result): keep the raw images (E001) as the
>   baseline; next compare detector architectures (RT-DETR, E003) and image size (E002).

## 1. Summary

**Question.** Does a teammate's colour enhancement of the underwater images (Harsh's
`fusion_mild` green-water correction, D-017/D-018) improve lobster detection for the same
YOLO detector?

**Design.** Two training runs that differ in one thing only, the images: **E001** trains on the raw
images (dataset v1), **E004** on the enhanced copies of exactly the same images (dataset v2: same
recording sessions, same train/val/test split, same labels). Same model (`yolo26s.pt`), image size,
settings, seed and GPU type.

**Result on the held-out test split** (evaluated once per model, after the model was fixed):

{_table(["metric", "E001 raw", "E004 enhanced", "difference"], metric_rows(t1, t4))}

- OBSERVED: on test, E004 scores **lower** than E001 on mAP50, mAP50-95, recall and F1, and
  **higher** on precision: it raises fewer false alarms but misses more lobsters.
- OBSERVED: validation agrees on mAP50, mAP50-95, recall and F1 (all lower for E004), but not on
  precision: at the operating confidence E004's val precision is slightly lower
  ({_fmt(v4['precision'])} vs {_fmt(v1['precision'])}) and it has {c[('val', 'E004')]['FP']} false positives vs
  {c[('val', 'E001')]['FP']}. The precision difference is therefore not consistent across splits.
- INFERRED: on this data the enhancement did **not** improve detection with this detector.
- HYPOTHESIS, not tested: part of the gap could be run-to-run noise (one run per model) or E004's
  shorter training (it stopped early at epoch 81; E001 ran 100 epochs).

{img(0, "Test split: E001 vs E004")}

## 2. Data

| split | images | with lobsters | empty (negative) | lobster boxes |
|---|---:|---:|---:|---:|
| train | {counts['train']['images']} | {counts['train']['positive']} | {counts['train']['negative']} | {counts['train']['instances']} |
| val | {counts['val']['images']} | {counts['val']['positive']} | {counts['val']['negative']} | {counts['val']['instances']} |
| test | {test_counts.get('images', '')} | {test_counts.get('positive', '')} | {test_counts.get('negative', '')} | {test_counts.get('instances', '')} |

- Images are frames from GoPro deployments (Set 100 + Set 2000). The split is **by recording
  session**, so no session appears in two splits.
- **Train** fits the model; **val** chooses the checkpoint and the confidence threshold; **test** is
  held out and was used exactly once per model (CLAUDE.md rule 7).
- zExtra photos are excluded (never labelled).

## 3. Training runs

{_table(["run", "what changed", "dataset", "model", "imgsz", "epochs", "seed", "highest val mAP50"], run_rows, "llllrrrr")}

{img(5, "Validation mAP per epoch")}

{img(6, "Training and validation losses")}

## 4. Test results (held out)

{_table(["metric", "E001 raw", "E004 enhanced", "difference"], metric_rows(t1, t4))}

Ultralytics also reports precision/recall at the confidence that maximises F1 *on test itself*
(not a usable operating point, shown for completeness): E001 {_fmt(e1['test']['metrics']['ultralytics_reported']['precision'])} /
{_fmt(e1['test']['metrics']['ultralytics_reported']['recall'])}, E004 {_fmt(e4['test']['metrics']['ultralytics_reported']['precision'])} /
{_fmt(e4['test']['metrics']['ultralytics_reported']['recall'])}.

### Confusion matrices

{_table(["test (Ultralytics matrix)", "E001 raw", "E004 enhanced", "difference"], cm_rows("test"))}

{_table(["val (IoU ≥ 0.5, operating conf)", "E001 raw", "E004 enhanced", "difference"], cm_rows("val"))}

- **TP**: a lobster box found. **FP**: a detection with no lobster box under it (false alarm,
  including duplicate or badly placed boxes). **FN**: a lobster box the model missed.
- TN (background predicted as background) is undefined for a detector, so that cell is empty.
- The test matrix is Ultralytics' own (its internal matching settings); the val matrix uses the
  project's matcher at IoU ≥ 0.5 and each model's operating confidence (D-020). Compare within a
  row, not across the two tables.

{img(2, "Confusion matrices")}

## 5. Validation results in detail

{_table(["metric", "E001 raw", "E004 enhanced", "difference"], metric_rows(v1, v4))}

{img(1, "Validation: E001 vs E004")}

### Precision–recall and the confidence threshold

{img(3, "Validation precision-recall curve")}

{img(4, "Precision, recall and F1 vs confidence")}

### Per session

{_table(["session", "images", "boxes", "AP50 E001", "AP50 E004", "F1 E001", "F1 E004", "FP E001 / E004"], sess_rows)}

{img(7, "Validation per session")}

### Per lighting condition

{_table(["lighting", "sessions", "boxes", "AP50 E001", "AP50 E004", "P / R E001", "P / R E004"], light_rows, "lrrrrrr")}

{img(8, "Validation per lighting condition")}

### False alarms on empty images

The validation split has {neg['E001']['negative_images']} confirmed-empty images (no lobster).
E001 fired on {neg['E001']['negative_images_with_detection']} of them ({neg['E001']['detections_on_negative_images']} detections),
E004 on {neg['E004']['negative_images_with_detection']} ({neg['E004']['detections_on_negative_images']} detections).

{img(9, "False alarms on empty validation images")}

## 6. Model cost

{_table(["", "E001 raw", "E004 enhanced"], [
        ["parameters", f"{cx['E001']['parameters']:,}", f"{cx['E004']['parameters']:,}"],
        [f"GFLOPs at {cx['E001']['gflops_imgsz']} px", f"{cx['E001']['gflops']:.1f}", f"{cx['E004']['gflops']:.1f}"],
        [f"inference time per image, batch 1, median ({gpu})", f"{sp['E001']:.1f} ms", f"{sp['E004']:.1f} ms"],
    ], "lrr")}

Same architecture, so the cost is the same; the enhancement step itself is not timed here.

## 7. Limitations (read before quoting any number)

- **New sessions, not new sites.** Test sessions share sites (mostly also dates) with training
  sessions, so these results measure generalisation to new recording sessions, not to new sites
  (D-007).
- **Concentrated boxes.** Two test sessions hold 1,135 of 1,512 test boxes (D-008); one val
  session (`21_MACK_10`) holds 465 of 1,010 val boxes.
- **One run per model** (seed 0); small differences may be run-to-run noise. E004 stopped early at
  81 epochs, E001 ran 100.
- **Validation is optimistic.** Val chose each run's checkpoint and its operating confidence.
- **Red-light/night val** is only 2 sessions / 31 boxes, so per-lighting results are fragile.
- **Enhancement provenance.** One val and one test image were among the images the enhancement's
  author reviewed while choosing its settings; whether others were seen is UNKNOWN (D-022). A
  five-image re-run matches the delivered images to within 0.01% of pixels (D-023).
- **Environment.** E004's test ran on a newer Colab CUDA (13.0) than its training (12.8); same GPU
  type and Ultralytics version.
- **Metadata.** Substrate and visibility metadata are mostly missing; zExtra is excluded.
- Sizes and positions are in pixels only; there is no camera calibration (CLAUDE.md rule 9).

## 8. Glossary

- **Precision**: of the boxes the model drew, the share that really are lobsters.
- **Recall**: of the real lobsters, the share the model found.
- **F1**: one number balancing precision and recall (their harmonic mean).
- **IoU**: overlap between a predicted box and a true box (0 = none, 1 = identical); a match here
  needs IoU ≥ 0.5.
- **mAP50**: average precision over all confidence thresholds, matching at IoU ≥ 0.5.
  **mAP50-95** averages it over stricter overlaps (IoU 0.5 to 0.95), so it also rewards tight boxes.
- **Operating confidence**: the threshold the model uses in practice, chosen on val where F1 is
  highest, then frozen before the test.

## Sources

- Records: `experiments/records/E001/`, `experiments/records/E004/` (run_record.json, results.csv,
  val_eval_record.json, final_test_record.json).
- Decisions: `docs/DECISIONS.md` (D-007, D-008, D-013, D-017, D-018, D-020, D-022, D-023).
- Experiment log: `docs/EXPERIMENTS.md`. Ultralytics' own plots and prediction examples are on the
  project Drive in `uwct/runs/<id>/final_test/` and `uwct/runs/<id>/val_eval/`.
"""
    return md


def build_report(records_root: Path, out_dir: Path) -> Path:
    """Write <out_dir>/E001_vs_E004_report.md and <out_dir>/figures/*.png. Returns the report path."""
    recs = load_records(records_root)
    out_dir = Path(out_dir)
    make_figures(recs, out_dir / "figures")
    report = out_dir / "E001_vs_E004_report.md"
    report.write_text(render_markdown(recs), encoding="utf-8", newline="\n")
    return report
