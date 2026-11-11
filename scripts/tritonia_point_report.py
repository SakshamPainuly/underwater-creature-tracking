#!/usr/bin/env python
r"""Graphs and annotated frames for one Tritonia point evaluation (D-033). Scores nothing new.

    python scripts/tritonia_point_report.py --batch <batch> --scores <run>/val_point_scores.json \
        --pred-dir <run>/val_point_predictions --out <run>/val_point_report

Reads the scores JSON and the prediction files that scripts/tritonia_point_eval.py wrote, and the
crosses of the batch. The split and the threshold are taken from the scores JSON (on test: the val
threshold it was run with); this script has no threshold option, so it can never choose one.
Writes into a new --out folder (--replace replaces an earlier report folder made by this script):
  point_pr_curve.png        point precision-recall curve over all confidences (area = point AP)
  point_conf_curves.png     point P, R and F1 against the confidence threshold
  point_per_day.png         point P, R and F1 per recording day at the threshold
  point_confusion_matrix.png  matched / false boxes and missed crosses at the threshold
  frames/<stem>.jpg         every frame of the split: boxes and crosses at the threshold
  report.json               the numbers behind the graphs
These are point scores against the MTrackJ crosses: not mAP (D-033).
Needs matplotlib and Pillow (GPU environment; both are in Colab and Kaggle).
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.config import RAW_ROOTS
from dataset_audit import tritonia_point_eval as E
from dataset_audit.tritonia_prelabel import read_points

SCHEMA = "uwct.tritonia_point_report/1"
GREEN, RED, CYAN, YELLOW = (0, 220, 0), (255, 40, 40), (0, 200, 255), (255, 230, 0)


def _plots(out: Path, report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split, conf, at = report["split"], report["conf"], report["at_conf"]
    tag = f"E006 point eval vs crosses -- {split}"

    pr = report["pr_curve"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([p["recall"] for p in pr], [p["precision"] for p in pr], color="#1f77b4", lw=2,
            label=f"sea_slug  point AP {report['point_ap']:.3f}")
    ax.scatter([at["recall"]], [at["precision"]], color="#d62728", zorder=3, label=f"conf {conf:.2f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="point recall", ylabel="point precision",
           title=f"Point precision-recall ({tag})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "point_pr_curve.png", dpi=150)
    plt.close(fig)

    cc = report["conf_curves"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for key, color in (("precision", "#1f77b4"), ("recall", "#ff7f0e"), ("f1", "#2ca02c")):
        ax.plot([c["conf"] for c in cc], [c[key] for c in cc], color=color, lw=2, label=f"point {key}")
    ax.axvline(conf, color="#7f7f7f", ls="--", label=f"threshold {conf:.2f} (chosen on val)")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="confidence threshold", ylabel="score",
           title=f"Point P / R / F1 vs confidence ({tag})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "point_conf_curves.png", dpi=150)
    plt.close(fig)

    days = sorted(report["per_day"])
    fig, ax = plt.subplots(figsize=(1.8 * len(days) + 3, 4.5))
    width = 0.27
    for k, (key, color) in enumerate((("precision", "#1f77b4"), ("recall", "#ff7f0e"), ("f1", "#2ca02c"))):
        xs = [i + (k - 1) * width for i in range(len(days))]
        vals = [report["per_day"][d][key] for d in days]
        ax.bar(xs, vals, width, color=color, label=f"point {key}")
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([f"{d}\n{report['per_day'][d]['crosses']} crosses" for d in days])
    ax.set(ylim=(0, 1.1), ylabel="score", title=f"Per recording day at conf {conf:.2f} ({tag})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "point_per_day.png", dpi=150)
    plt.close(fig)

    tp, fp, fn = at["matched"], at["predictions"] - at["matched"], at["crosses"] - at["matched"]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow([[tp, fp], [fn, 0]], cmap="Blues")
    for (r, c), text in {(0, 0): f"{tp}\nmatched", (0, 1): f"{fp}\nfalse box", (1, 0): f"{fn}\nmissed cross",
                         (1, 1): "--\n(not defined)"}.items():
        ax.text(c, r, text, ha="center", va="center", fontsize=12,
                color="white" if r == 0 and c == 0 and tp > max(fp, fn) / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["sea_slug (cross)", "background"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["sea_slug (box)", "background"])
    ax.set(xlabel="True (MTrackJ crosses)", ylabel="Predicted",
           title=f"Point confusion matrix, conf {conf:.2f} ({split})")
    fig.tight_layout()
    fig.savefig(out / "point_confusion_matrix.png", dpi=150)
    plt.close(fig)


def _draw_frame(src: Path, dst: Path, outcome: dict, match_px: float, title: str) -> None:
    from PIL import Image, ImageDraw

    im = Image.open(src).convert("RGB")
    d = ImageDraw.Draw(im)
    for color, key in ((GREEN, "matched_boxes"), (RED, "false_boxes")):
        for c, (xa, ya, xb, yb) in outcome[key]:
            d.rectangle((xa, ya, xb, yb), outline=color, width=2)
            d.text((xa, max(ya - 11, 0)), f"{c:.2f}", fill=color)
    r = match_px
    for x, y in outcome["found_crosses"]:
        d.ellipse((x - r, y - r, x + r, y + r), outline=CYAN, width=1)
    for x, y in outcome["missed_crosses"]:
        d.line((x - 6, y - 6, x + 6, y + 6), fill=YELLOW, width=2)
        d.line((x - 6, y + 6, x + 6, y - 6), fill=YELLOW, width=2)
    d.rectangle((0, 0, im.width, 14), fill=(0, 0, 0))
    d.text((3, 1), title, fill=(255, 255, 255))
    im.save(dst, quality=92)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=Path, required=True, help="labelling batch (manifest.csv, images/, points/)")
    ap.add_argument("--scores", type=Path, required=True, help="JSON written by scripts/tritonia_point_eval.py")
    ap.add_argument("--pred-dir", type=Path, required=True, help="the prediction files those scores used")
    ap.add_argument("--out", type=Path, required=True, help="new folder for the report")
    ap.add_argument("--replace", action="store_true", help="replace an earlier report folder made by this script")
    args = ap.parse_args()

    out = args.out.resolve()
    for f in RAW_ROOTS:
        if Path(f).resolve() in (out, *out.parents):
            sys.exit(f"REFUSED: {out} is inside the raw root {f}")
    if out.exists():
        if not (args.replace and (out / "report.json").is_file()
                and json.loads((out / "report.json").read_text(encoding="utf-8")).get("schema") == SCHEMA):
            sys.exit(f"REFUSED: {out} already exists (--replace only replaces a report folder made by this script)")
        shutil.rmtree(out)

    scored = json.loads(args.scores.read_text(encoding="utf-8"))
    split, match_px, conf = scored["split"], scored["match_px"], scored["at_conf"]["conf"]
    batch = args.batch.resolve()
    with open(batch / "manifest.csv", newline="", encoding="utf-8") as f:
        stems = [Path(r["image"]).stem for r in csv.DictReader(f)]
    meta = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    (w, h), = {(s["width"], s["height"]) for s in meta["source_stacks"].values()}
    frames = {s: (E.read_predictions(args.pred_dir / f"{s}.txt", w, h), read_points(batch / "points" / f"{s}.csv"))
              for s in E.frames_in_splits(stems, {split})}
    if len(frames) != scored["frames"]:
        sys.exit(f"REFUSED: {len(frames)} {split} frames in the batch, the scores JSON has {scored['frames']}")

    report = {"schema": SCHEMA, "split": split, "final": scored.get("final"), "scores_file": str(args.scores),
              "conf": conf, "conf_source": "best_f1_on_val" if split == "val" else "val threshold (--conf)",
              "match_px": match_px, "frames": len(frames), "point_ap": E.point_ap(frames, match_px),
              "at_conf": E.scores(frames, conf, match_px), "pr_curve": E.pr_curve(frames, match_px),
              "conf_curves": E.conf_curves(frames, match_px),
              "per_day": {d: E.scores({s: v for s, v in frames.items() if s[:8] == d}, conf, match_px)
                          for d in sorted({s[:8] for s in frames})},
              "per_frame": {}, "limitation": scored.get("limitation")}
    if report["point_ap"] != scored["point_ap"] or report["at_conf"] != scored["at_conf"]:
        sys.exit("REFUSED: these predictions do not reproduce the scores JSON (wrong --pred-dir or batch?)")

    out.mkdir(parents=True)
    (out / "frames").mkdir()
    for s, (preds, crosses) in frames.items():
        o = E.frame_outcome(preds, crosses, conf, match_px)
        report["per_frame"][s] = {k: len(v) for k, v in o.items()}
        title = (f"{s} | {split} | conf {conf:.2f} | green = matched box, red = false box, "
                 f"cyan = found cross, yellow X = missed cross")
        _draw_frame(batch / "images" / f"{s}.png", out / "frames" / f"{s}.jpg", o, match_px, title)
    _plots(out, report)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("split", "frames", "conf", "point_ap", "at_conf")}, indent=2))
    print("report:", out)


if __name__ == "__main__":
    main()
