#!/usr/bin/env python
r"""Build Tritonia batch 2 (D-035): batch 1 unchanged plus more frames from the TRAINING days only.

    python scripts/make_tritonia_batch2.py --root D:\uwct_tritonia\raw\tritonia \
        --base-batch D:\uwct_tritonia\work\tritonia_label_batch1 --split-version v3 \
        --out D:\uwct_tritonia\work\tritonia_label_batch2 [--per-day 300 --min-gap 5 --seed 0]

--root defaults to UWCT_TRITONIA_ROOT and is only read. The 150 batch 1 frames are copied (sha256
checked against the batch 1 manifest) with their points, so v3's val and test frames come through
byte for byte. New frames come only from the days that --split-version puts in train, at least
--min-gap frames from every batch 1 frame and from each other. Same layout as batch 1 (images/,
points/, labels.txt, manifest.csv, batch.json) without hints/. batch.json lists the added frames.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.mtrackj import MtrackjError, read_mdf
from dataset_audit.tritonia_dataset import session_of, stem_of
from dataset_audit.tritonia_frames import page_strips, pick_extra_frames, read_page, write_png_gray
from dataset_audit.versioning import version_dir

CLASS_LIST = ["lobster", "sea_slug"]
DECISION = "D-035"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=os.environ.get("UWCT_TRITONIA_ROOT"))
    ap.add_argument("--base-batch", type=Path, required=True)
    ap.add_argument("--split-version", required=True, help="frozen Tritonia version whose day split is kept (v3)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-day", type=int, default=300)
    ap.add_argument("--min-gap", type=int, default=5, help="frames between picks in one stack (5 = 75 s)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.root is None:
        sys.exit("no source: pass --root or set UWCT_TRITONIA_ROOT")
    root, base, out = (Path(p).resolve() for p in (args.root, args.base_batch, args.out))
    for src in (root, base):
        if out == src or src in out.parents:
            sys.exit(f"REFUSED: {out} is inside the input {src}")
    if out.exists():
        sys.exit(f"REFUSED: {out} already exists; choose a new batch folder")

    with open(base / "manifest.csv", newline="", encoding="utf-8") as f:
        base_rows = list(csv.DictReader(f))
    base_meta = json.loads((base / "batch.json").read_text(encoding="utf-8"))
    with open(version_dir(args.split_version) / "split_manifest.csv", newline="", encoding="utf-8") as f:
        split = {r["session_id"]: r["split"] for r in csv.DictReader(f)}
    train_days = sorted(d for d, s in split.items() if s == "train")

    try:
        mdfs = {m.name[:-len(".tif.mdf")]: read_mdf(m) for m in sorted(root.glob("*.tif.mdf"))}
        stack_of = {session_of(name + "_t0000"): name for name in mdfs}
        train_stacks = [stack_of[d] for d in train_days]
        taken = {}
        for r in base_rows:
            taken.setdefault(r["source_tif"][:-len(".tif")], []).append(int(r["t"]))
        picks = pick_extra_frames(mdfs, train_stacks, args.per_day, args.seed, args.min_gap, taken)
    except (MtrackjError, ValueError, OSError, KeyError) as e:
        sys.exit(f"FAILED: {e}")

    (out / "images").mkdir(parents=True)
    (out / "points").mkdir()
    (out / "labels.txt").write_text("\n".join(CLASS_LIST) + "\n", encoding="utf-8")
    for r in base_rows:  # batch 1, unchanged
        stem = stem_of(r["image"])
        src = base / "images" / f"{stem}.png"
        if _sha(src) != r["png_sha256"]:
            sys.exit(f"FAILED: {src} differs from the batch 1 manifest (sha256)")
        shutil.copyfile(src, out / "images" / f"{stem}.png")
        shutil.copyfile(base / "points" / f"{stem}.csv", out / "points" / f"{stem}.csv")
        if _sha(out / "images" / f"{stem}.png") != r["png_sha256"]:
            sys.exit(f"FAILED: copy of {src} does not verify")

    rows, stacks, cache, added = list(base_rows), dict(base_meta["source_stacks"]), {}, []
    for k, (stack, t) in enumerate(picks, 1):
        tif = root / f"{stack}.tif"
        try:
            if stack not in cache:
                cache.clear()
                cache[stack] = page_strips(tif)
                info = {"n_frames": len(cache[stack][2]), "width": cache[stack][0], "height": cache[stack][1],
                        "tif_bytes": tif.stat().st_size}
                if tif.name in stacks and stacks[tif.name] != info:
                    sys.exit(f"FAILED: {tif.name} differs from what batch 1 recorded: {stacks[tif.name]} vs {info}")
                stacks[tif.name] = info
            w, h, strips = cache[stack]
            if t > len(strips):
                raise MtrackjError(f"{tif}: t={t} beyond its {len(strips)} pages")
            pixels = read_page(tif, strips[t - 1])
        except (MtrackjError, OSError) as e:
            sys.exit(f"FAILED: {e}")
        stem = f"{stack}_t{t:04d}"
        pts = [(tr.track_id, p) for tr in mdfs[stack].tracks for p in tr.points if p.t == t]
        write_png_gray(out / "images" / f"{stem}.png", w, h, pixels)
        with open(out / "points" / f"{stem}.csv", "x", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["track", "point", "x", "y", "t"])
            wr.writerows([tid, p.point_id, p.x, p.y, p.t] for tid, p in pts)
        rows.append({"image": f"images/{stem}.png", "source_tif": tif.name, "t": t, "page_index": t - 1,
                     "n_points": len(pts), "tracks": " ".join(str(tid) for tid, _ in pts),
                     "png_sha256": _sha(out / "images" / f"{stem}.png")})
        added.append(stem)
        if k % 100 == 0 or k == len(picks):
            print(f"[{k}/{len(picks)}] {stem}", flush=True)

    rows.sort(key=lambda r: r["image"])
    with open(out / "manifest.csv", "x", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(base_rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    per_day = {d: sum(1 for s in added if session_of(s) == d) for d in train_days}
    totals = {"frames": len(rows), "points": sum(int(r["n_points"]) for r in rows),
              "stacks": len({r["source_tif"] for r in rows}), "base_frames": len(base_rows), "added_frames": len(added)}
    (out / "batch.json").write_text(json.dumps(
        {"root": str(root), "decision": DECISION, "n": len(rows), "seed": args.seed, "min_gap": args.min_gap,
         "per_day": args.per_day, "classes": CLASS_LIST, "totals": totals,
         "base_batch": {"path": str(base), "manifest_sha256": _sha(base / "manifest.csv"), "frames": len(base_rows)},
         "split_version": args.split_version, "train_days": train_days, "added_per_day": per_day,
         "added": added, "source_stacks": dict(sorted(stacks.items()))}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"totals": totals, "added_per_day": per_day}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
