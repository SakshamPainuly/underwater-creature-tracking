#!/usr/bin/env python
r"""Track Tritonia detections per camera and score them against the MTrackJ tracks (D-044 phase 3). Local CPU.

    python scripts/track_tritonia.py --sequence data/processed/tritonia_tracking/val \
        --detections <unzipped d043_val_detections> --conf 0.16 --gate-px 12 --max-gap 8 --min-len 3 --out <new dir>

--mode tracker (default) links E010's detections; --mode no-link makes every detection its own track (floor);
--mode gt feeds the clicks themselves through the tracker (ceiling: tracker alone; --detections not used, --conf
ignored). Scores (HOTA, IDF1, MOTA, ...) only on clicked frames, per camera, per day, mean over days. Writes
out/tracks.csv and out/scores.json. Refuses a test sequence without --final D-044 (rule 7). Val tuning only
through the grid declared in docs/EXPERIMENTS.md (phase 4). Sparse-frame/timelapse analysis, not video tracking.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracking import data as T
from tracking.detections import FINAL_DECISION, DetectionError, sha256_file
from tracking.tracker import TrackerParams


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", type=Path, required=True, help="folder with sequence.json, manifest.csv, gt_points.csv")
    ap.add_argument("--detections", type=Path, help="folder with predictions/ and predictions.json (phase 2)")
    ap.add_argument("--mode", choices=("tracker", "no-link", "gt"), default="tracker")
    ap.add_argument("--conf", type=float, default=0.16)
    ap.add_argument("--gate-px", type=float, default=12.0)
    ap.add_argument("--max-gap", type=int, default=8)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--match-px", type=float, default=10.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--final", help=f"{FINAL_DECISION}, only for the approved final test")
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"REFUSED: {args.out} already exists; choose a new folder")
    try:
        meta = T.load_meta(args.sequence)
        if meta["split"] == "test" and args.final != FINAL_DECISION:
            sys.exit(f"REFUSED: test sequence needs --final {FINAL_DECISION} (rule 7)")
        params = TrackerParams(conf=args.conf, gate_px=args.gate_px, max_gap=args.max_gap,
                               min_len=1 if args.mode == "no-link" else args.min_len)
        gt = T.read_gt(args.sequence)
        source = {"mode": args.mode}
        if args.mode == "gt":
            dets = T.gt_as_detections(gt)
        else:
            if args.detections is None:
                sys.exit("--detections is required unless --mode gt")
            rec = json.loads((args.detections / "predictions.json").read_text(encoding="utf-8"))
            if rec["sequence"]["manifest_sha256"] != meta["manifest_sha256"] or rec.get("smoke"):
                sys.exit("FAILED: predictions are for another sequence or are a smoke run")
            dets = T.read_detections(args.sequence, args.detections / "predictions", meta, min_conf=params.conf)
            source.update(experiment=rec["experiment"], weights_sha256=rec["weights"]["sha256"],
                          predictions_json_sha256=sha256_file(args.detections / "predictions.json"),
                          predict_commit=rec["git"]["commit"])
        tracks = T.run(dets, params, link=args.mode != "no-link")
        result, rows = T.score_all(gt, tracks, args.match_px)
    except (DetectionError, OSError, KeyError, ValueError) as e:
        sys.exit(f"FAILED: {e}")
    args.out.mkdir(parents=True)
    with open(args.out / "tracks.csv", "x", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["day", "camera", "track_id", "t", "x", "y", "conf"], lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    out = {"decision": FINAL_DECISION, "phase": 3, "split": meta["split"], "final": args.final,
           "sequence": {"manifest_sha256": meta["manifest_sha256"], "gt_points_sha256": meta["gt_points_sha256"]},
           "source": source, "params": {"conf": params.conf, "gate_px": params.gate_px, "max_gap": params.max_gap,
                                        "min_len": params.min_len, "match_px": args.match_px},
           "tracks": len({(r["day"], r["track_id"]) for r in rows}), **result,
           "note": "sparse-frame/timelapse analysis (15 s per frame); scored on clicked frames only; px, no units"}
    (args.out / "scores.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    m = result["mean_over_days"]
    print(f"{args.mode}: mean over days HOTA {m['HOTA']:.3f} DetA {m['DetA']:.3f} AssA {m['AssA']:.3f} "
          f"IDF1 {m['IDF1']:.3f} MOTA {m['MOTA']:.3f} | tracks {out['tracks']} | wrote {args.out}")


if __name__ == "__main__":
    main()
