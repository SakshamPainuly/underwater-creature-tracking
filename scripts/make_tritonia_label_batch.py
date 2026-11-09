#!/usr/bin/env python
"""Build a Tritonia box-labelling batch (D-029): frames to label, point hints, manifest. Read-only on the source.

    python scripts/make_tritonia_label_batch.py --root D:\\uwct_tritonia\\raw\\tritonia \\
        --out D:\\uwct_tritonia\\work\\tritonia_label_batch1 [--n 150 --seed 0 --min-gap 40]

--root defaults to UWCT_TRITONIA_ROOT. --out must not exist and must be outside --root. Writes:
    images/<stack>_t<NNNN>.png   clean frame: the image that gets boxed
    hints/<stack>_t<NNNN>.png    same frame with a cross at each MTrackJ point (never labelled)
    points/<stack>_t<NNNN>.csv   that frame's points (track, point, x, y, t)
    labels.txt                   class list for the labelling tool: lobster, sea_slug (ids 0, 1)
    manifest.csv                 one row per frame: source stack, t, page index, points, PNG sha256
    batch.json                   arguments, totals and the source stacks' sizes
Frames are spread evenly over the labelled stacks, only frames with >= 1 point, at least --min-gap
frames apart within a stack (15 s per frame). Labelling steps: docs/TRITONIA_LABELLING.md.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.mtrackj import MtrackjError, read_mdf
from dataset_audit.tritonia_frames import draw_hints, page_strips, pick_frames, read_page, write_png_gray

CLASS_LIST = ["lobster", "sea_slug"]  # same ids as CLASS_NAMES once class 1 is added (D-029)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=os.environ.get("UWCT_TRITONIA_ROOT"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-gap", type=int, default=40, help="frames between picks in one stack (40 = 10 min)")
    args = parser.parse_args()
    if args.root is None:
        sys.exit("no source: pass --root or set UWCT_TRITONIA_ROOT")
    root, out = Path(args.root).resolve(), args.out.resolve()
    if out == root or root in out.parents:
        sys.exit(f"REFUSED: {out} is inside the read-only source {root}")
    if out.exists():
        sys.exit(f"REFUSED: {out} already exists; choose a new batch folder")

    try:
        mdfs = {m.name[:-len(".tif.mdf")]: read_mdf(m) for m in sorted(root.glob("*.tif.mdf"))}
        picks = pick_frames(mdfs, args.n, args.seed, args.min_gap)
    except (MtrackjError, ValueError, OSError) as e:
        sys.exit(f"FAILED: {e}")

    for sub in ("images", "hints", "points"):
        (out / sub).mkdir(parents=True)
    (out / "labels.txt").write_text("\n".join(CLASS_LIST) + "\n", encoding="utf-8")
    rows, stacks, cache = [], {}, {}
    for stack, t in picks:
        tif = root / f"{stack}.tif"
        try:
            if stack not in cache:
                cache.clear()  # one stack's page table at a time
                cache[stack] = page_strips(tif)
                stacks[stack] = {"n_frames": len(cache[stack][2]), "width": cache[stack][0],
                                 "height": cache[stack][1], "tif_bytes": tif.stat().st_size}
            w, h, strips = cache[stack]
            if t > len(strips):
                raise MtrackjError(f"{tif}: t={t} beyond its {len(strips)} pages")
            pixels = read_page(tif, strips[t - 1])
        except (MtrackjError, OSError) as e:
            sys.exit(f"FAILED: {e}")
        stem = f"{stack}_t{t:04d}"
        pts = [(tr.track_id, p) for tr in mdfs[stack].tracks for p in tr.points if p.t == t]
        write_png_gray(out / "images" / f"{stem}.png", w, h, pixels)
        write_png_gray(out / "hints" / f"{stem}.png", w, h, draw_hints(pixels, w, h, [(p.x, p.y) for _, p in pts]))
        with open(out / "points" / f"{stem}.csv", "x", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["track", "point", "x", "y", "t"])
            wr.writerows([tid, p.point_id, p.x, p.y, p.t] for tid, p in pts)
        rows.append({"image": f"images/{stem}.png", "source_tif": tif.name, "t": t, "page_index": t - 1,
                     "n_points": len(pts), "tracks": " ".join(str(tid) for tid, _ in pts),
                     "png_sha256": hashlib.sha256((out / "images" / f"{stem}.png").read_bytes()).hexdigest()})
        print(f"[{len(rows)}/{len(picks)}] {stem}: {len(pts)} points", flush=True)

    with open(out / "manifest.csv", "x", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    totals = {"frames": len(rows), "points": sum(r["n_points"] for r in rows),
              "stacks": len({r["source_tif"] for r in rows})}
    (out / "batch.json").write_text(json.dumps(
        {"root": str(root), "n": args.n, "seed": args.seed, "min_gap": args.min_gap, "classes": CLASS_LIST,
         "totals": totals, "source_stacks": stacks}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {totals['frames']} frames ({totals['points']} points, {totals['stacks']} stacks) -> {out}")


if __name__ == "__main__":
    main()
