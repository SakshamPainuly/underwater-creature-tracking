#!/usr/bin/env python
r"""Point-based scores of a Tritonia detector against the MTrackJ crosses (D-031).

Validation (choose the confidence threshold here, never on test):
    python scripts/tritonia_point_eval.py --batch <batch> --split val --weights best.pt --pred-dir <new dir> --out val_points.json

Declared final test only (rule 7; log it in docs/EXPERIMENTS.md), with the threshold chosen on val:
    python scripts/tritonia_point_eval.py --batch <batch> --split test --final E006 --conf 0.25 --weights best.pt --pred-dir <new dir> --out test_points.json

--weights runs the model (GPU environment: Ultralytics) on every batch frame of the split and writes
YOLO txt files with confidences into --pred-dir; without --weights, --pred-dir must already hold them.
Frames are those of the D-030 day split; every frame of the split is scored, including frames the
automatic labels had to exclude, because their crosses are still ground truth.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.config import RAW_ROOTS
from dataset_audit import tritonia_point_eval as E
from dataset_audit.tritonia_prelabel import read_points


def predict(weights: Path, images: list[Path], out_dir: Path, imgsz: int) -> dict:
    """Run Ultralytics once per frame (conf 0.001, so the val sweep sees low scores too)."""
    from ultralytics import YOLO  # GPU-side only (CLAUDE.md §7)
    model = YOLO(str(weights))
    out_dir.mkdir(parents=True, exist_ok=False)
    for img in images:
        r = model.predict(str(img), conf=0.001, imgsz=imgsz, verbose=False)[0]
        lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {s:.5f}"
                 for (x, y, w, h), c, s in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        (out_dir / f"{img.stem}.txt").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    return {"weights": str(weights), "conf": 0.001, "imgsz": imgsz}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=Path, required=True, help="labelling batch (manifest.csv, images/, points/)")
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--pred-dir", type=Path, required=True, help="YOLO txt predictions with confidence")
    ap.add_argument("--weights", type=Path, help="run this model first, writing into a new --pred-dir")
    ap.add_argument("--conf", type=float, help="threshold for P/R (required on test: the one chosen on val)")
    ap.add_argument("--final", help="experiment id of the declared final evaluation (required on test)")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size; use the experiment's train imgsz")
    ap.add_argument("--match-px", type=float, default=E.MATCH_PX)
    ap.add_argument("--out", type=Path, required=True, help="new JSON file for the scores")
    args = ap.parse_args()

    if args.split == "test" and (not args.final or args.conf is None):
        sys.exit("REFUSED: test is scored only as a declared final evaluation (--final E###) with the "
                 "threshold chosen on val (--conf); CLAUDE.md rule 7")
    out = args.out.resolve()
    for p in (out, args.pred_dir.resolve()):
        for f in RAW_ROOTS:
            if Path(f).resolve() in (p, *p.parents):
                sys.exit(f"REFUSED: {p} is inside the raw root {f}")
    if out.exists():
        sys.exit(f"REFUSED: {out} already exists")

    batch = args.batch.resolve()
    with open(batch / "manifest.csv", newline="", encoding="utf-8") as f:
        stems = [Path(r["image"]).stem for r in csv.DictReader(f)]
    meta = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    (w, h), = {(s["width"], s["height"]) for s in meta["source_stacks"].values()}
    frames_list = E.frames_in_splits(stems, {args.split})

    run = None
    if args.weights:
        run = predict(args.weights, [batch / "images" / f"{s}.png" for s in frames_list], args.pred_dir.resolve(),
                      args.imgsz)
    elif not args.pred_dir.is_dir():
        sys.exit(f"{args.pred_dir} does not exist; pass --weights to create it")

    frames = {s: (E.read_predictions(args.pred_dir / f"{s}.txt", w, h), read_points(batch / "points" / f"{s}.csv"))
              for s in frames_list}
    result = {"split": args.split, "final": args.final, "frames": len(frames), "match_px": args.match_px,
              "prediction_run": run, "point_ap": E.point_ap(frames, args.match_px)}
    if args.conf is None:
        result["best_f1_on_val"] = E.best_f1_conf(frames, args.match_px)
        conf = result["best_f1_on_val"]["conf"]
    else:
        conf = args.conf
    result["at_conf"] = E.scores(frames, conf, args.match_px)
    days = sorted({s[:8] for s in frames})
    result["per_day"] = {d: E.scores({s: v for s, v in frames.items() if s[:8] == d}, conf, args.match_px)
                         for d in days}
    result["limitation"] = ("crosses are the ground truth; precision is a lower bound because it is UNKNOWN "
                            "whether every slug was clicked (D-029). Not mAP.")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("split", "frames", "point_ap", "at_conf")}, indent=2))


if __name__ == "__main__":
    main()
