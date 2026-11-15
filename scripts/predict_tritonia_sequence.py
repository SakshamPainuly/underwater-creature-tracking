#!/usr/bin/env python
r"""Run a registered Tritonia detector on every frame of a sequence (D-044 phase 2). GPU side (Ultralytics).

    python scripts/predict_tritonia_sequence.py --sequence <dir with sequence.json, manifest.csv, images/> \
        --weights E010/train/weights/best.pt --weights-sha256 <registered sha> --out <new dir> [--limit 50]

Checks first: the sequence against its manifest (every frame's sha256), the weights against the registered
sha256, and refuses a test sequence without --final D-044 (rule 7). Then writes out/predictions/<frame>.txt
(`cls x y w h conf`, normalised, conf >= 0.001, as the D-033 point eval) and out/predictions.json.
--limit N predicts only the first N frames (smoke test, CLAUDE.md §6); its output is not a result.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracking.detections import DetectionError, check_weights, load_sequence, sha256_file, summarize_predictions

CONF = 0.001


def git_state() -> dict:
    repo = Path(__file__).resolve().parents[1]
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--weights-sha256", required=True, help="the sha256 registered in experiments/registry.csv")
    ap.add_argument("--experiment", default="E010", help="experiment id of the weights (recorded only)")
    ap.add_argument("--imgsz", type=int, default=640, help="the experiment's train imgsz")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="smoke test: only the first N frames")
    ap.add_argument("--final", help="D-044, only for the approved final test")
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists():
        sys.exit(f"REFUSED: {out} already exists; choose a new folder")
    if args.sequence.resolve() in out.parents or out == args.sequence.resolve():
        sys.exit(f"REFUSED: {out} is inside the sequence {args.sequence}")
    t0 = time.time()
    try:
        meta, rows = load_sequence(args.sequence, final=args.final)
        wsha = check_weights(args.weights, args.weights_sha256)
    except DetectionError as e:
        sys.exit(f"FAILED: {e}")
    if args.limit:
        rows = rows[:args.limit]
    t_check = time.time() - t0

    from ultralytics import YOLO  # GPU side only (CLAUDE.md §7)
    import ultralytics
    import torch
    model = YOLO(str(args.weights))
    pdir = out / "predictions"
    pdir.mkdir(parents=True)
    images = [str(args.sequence / r["image"]) for r in rows]
    t1 = time.time()
    for k in range(0, len(images), args.batch):
        chunk = images[k:k + args.batch]
        for img, r in zip(chunk, model.predict(chunk, conf=CONF, imgsz=args.imgsz, batch=args.batch, verbose=False)):
            if Path(r.path).name != Path(img).name:
                sys.exit(f"FAILED: result order mismatch ({r.path} vs {img})")
            lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {s:.5f}" for (x, y, w, h), c, s in
                     zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            (pdir / f"{Path(img).stem}.txt").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
        if (k // args.batch) % 50 == 0 or k + args.batch >= len(images):
            print(f"[{min(k + args.batch, len(images))}/{len(images)}] frames", flush=True)
    t_pred = time.time() - t1
    stems = [Path(r["image"]).stem for r in rows]
    record = {
        "decision": "D-044", "phase": 2, "smoke": bool(args.limit), "split": meta["split"],
        "final": args.final, "sequence": {"split_version": meta["split_version"], "days": sorted(meta["days"]),
                                          "manifest_sha256": meta["manifest_sha256"],
                                          "gt_points_sha256": meta["gt_points_sha256"]},
        "experiment": args.experiment, "weights": {"path": str(args.weights), "sha256": wsha},
        "predict_args": {"conf": CONF, "imgsz": args.imgsz, "batch": args.batch},
        "env": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "git": git_state(), "seconds": {"checks": round(t_check, 1), "predict": round(t_pred, 1)},
        "summary_conf_0.001": summarize_predictions(pdir, stems, CONF),
        "summary_conf_0.16": summarize_predictions(pdir, stems, 0.16),
        "note": "sparse-frame/timelapse frames 15 s apart; detections only, not yet tracks; not accuracy",
    }
    (out / "predictions.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("smoke", "seconds", "summary_conf_0.16")}), "\nwrote", out)


if __name__ == "__main__":
    main()
