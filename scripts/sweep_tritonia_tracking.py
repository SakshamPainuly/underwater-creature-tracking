#!/usr/bin/env python
r"""Run a declared tracker-setting grid on the val detections and apply the declared selection (D-043 phase 4).

    python scripts/sweep_tritonia_tracking.py --config configs/tracking/T001_tritonia_val_grid.json \
        --detections D:\uwct_tritonia\runs\d043_val_detections\d043_val_detections --out D:\uwct_tritonia\runs\T001

Checks the sequence and the detections against the hashes in the config, runs every combination (local CPU),
writes out/grid_results.csv (one row per setting), out/selected/tracks.csv + scores.json (the selected setting)
and out/run_record.json (inputs, git state, selection, close-pair scores, references). Val only; refuses an
existing --out. Sparse-frame/timelapse analysis (15 s per frame), not video tracking.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tracking import data as T
from tracking import metrics as Mx
from tracking.detections import DetectionError, sha256_file
from tracking.sweep import KEYS, SweepError, close_pair_frames, expand_grid, load_grid_config, select
from tracking.tracker import TrackerParams

SUMMARY = ("HOTA", "DetA", "AssA", "LocA", "IDF1", "MOTA")


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def close_pair_scores(gt: dict, tracks: dict, match_px: float) -> dict:
    counts, n = [], 0
    for day in sorted(gt):
        pred, _ = T.as_pred(tracks.get(day, {}))
        for c in T.CAMERAS:
            frames = close_pair_frames(gt[day].get(c, {}), match_px)
            n += len(frames)
            counts.append(Mx.evaluate(frames, gt[day].get(c, {}), pred.get(c, {}), match_px))
    s = Mx.scores(Mx.combine(counts))
    return {"camera_frames": n, **{k: s[k] for k in ("IDF1", "MOTA", "IDSW", "TP", "FN", "FP")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--detections", type=Path, required=True, help="folder with predictions/ and predictions.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"REFUSED: {args.out} already exists; choose a new folder")
    t0 = time.time()
    try:
        cfg = load_grid_config(args.config)
        seq = REPO / cfg["sequence"]
        meta = T.load_meta(seq)
        if (meta["split"], meta["manifest_sha256"], meta["gt_points_sha256"]) != \
                ("val", cfg["sequence_manifest_sha256"], cfg["gt_points_sha256"]):
            sys.exit("FAILED: the sequence is not the val sequence named in the config")
        pj = args.detections / "predictions.json"
        rec = json.loads(pj.read_text(encoding="utf-8"))
        want = cfg["detections"]
        if (sha256_file(pj), rec["weights"]["sha256"], rec["experiment"], rec.get("smoke")) != \
                (want["predictions_json_sha256"], want["weights_sha256"], want["experiment"], False):
            sys.exit("FAILED: the detections are not the ones named in the config")
        gt = T.read_gt(seq)
        dets = T.read_detections(seq, args.detections / "predictions", meta, min_conf=min(cfg["grid"]["conf"]))
    except (DetectionError, SweepError, OSError, KeyError, ValueError) as e:
        sys.exit(f"FAILED: {e}")

    settings = expand_grid(cfg["grid"])
    rows, results = [], []
    for k, p in enumerate(settings, 1):
        sc, _ = T.score_all(gt, T.run(dets, TrackerParams(**p)), cfg["match_px"])
        m = sc["mean_over_days"]
        r = {"params": p, **{x: m[x] for x in SUMMARY}, "IDSW": sc["all_days"]["IDSW"],
             "per_day_HOTA": {d: v["all_cameras"]["HOTA"] for d, v in sc["per_day"].items()}}
        results.append(r)
        rows.append({**p, **{x: round(m[x], 6) for x in SUMMARY}, "IDSW": r["IDSW"],
                     **{f"HOTA_{d}": round(h, 6) for d, h in r["per_day_HOTA"].items()}})
        if k % 9 == 0 or k == len(settings):
            print(f"[{k}/{len(settings)}] best mean HOTA so far {max(x['HOTA'] for x in results):.4f}", flush=True)
    chosen = select(results)
    sel_params = TrackerParams(**chosen["params"])
    sel_tracks = T.run(dets, sel_params)
    sel_scores, track_rows = T.score_all(gt, sel_tracks, cfg["match_px"])
    refs = {}
    for name, tr in (("no_link", T.run(dets, sel_params, link=False)),
                     ("clicks_as_detections", T.run(T.gt_as_detections(gt), TrackerParams(0.0, sel_params.gate_px,
                                                                                         sel_params.max_gap,
                                                                                         sel_params.min_len)))):
        s, _ = T.score_all(gt, tr, cfg["match_px"])
        refs[name] = {x: s["mean_over_days"][x] for x in SUMMARY}

    args.out.mkdir(parents=True)
    with open(args.out / "grid_results.csv", "x", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    (args.out / "selected").mkdir()
    with open(args.out / "selected" / "tracks.csv", "x", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["day", "camera", "track_id", "t", "x", "y", "conf"], lineterminator="\n")
        wr.writeheader()
        wr.writerows(track_rows)
    (args.out / "selected" / "scores.json").write_text(json.dumps(
        {"id": cfg["id"], "params": chosen["params"], "match_px": cfg["match_px"], **sel_scores}, indent=1) + "\n",
        encoding="utf-8")
    record = {
        "id": cfg["id"], "decision": cfg["decision"], "phase": 4, "split": "val",
        "config": {"path": args.config.as_posix(), "sha256": sha256_file(args.config)}, "git": git_state(),
        "inputs": {"sequence_manifest_sha256": meta["manifest_sha256"], "gt_points_sha256": meta["gt_points_sha256"],
                   "predictions_json_sha256": sha256_file(pj), "experiment": rec["experiment"],
                   "weights_sha256": rec["weights"]["sha256"], "predict_commit": rec["git"]["commit"]},
        "settings_run": len(settings), "selection_rule": cfg["selection"],
        "selected": {"params": chosen["params"], "mean_over_days": {x: chosen[x] for x in SUMMARY},
                     "per_day_HOTA": chosen["per_day_HOTA"], "IDSW": chosen["IDSW"],
                     "tied_within": sum(r["HOTA"] >= max(x["HOTA"] for x in results) - 0.001 for r in results),
                     "close_pair_frames": close_pair_scores(gt, sel_tracks, cfg["match_px"])},
        "references_at_selected": refs, "grid_results_sha256": sha256_file(args.out / "grid_results.csv"),
        "seconds": round(time.time() - t0, 1),
        "note": "val only; sparse-frame/timelapse analysis; scored on clicked frames; px, no physical units",
    }
    (args.out / "run_record.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    s = record["selected"]
    print(f"selected {s['params']} | mean HOTA {s['mean_over_days']['HOTA']:.4f} IDF1 {s['mean_over_days']['IDF1']:.4f} "
          f"| {len(settings)} settings in {record['seconds']} s | wrote {args.out}")


if __name__ == "__main__":
    main()
