#!/usr/bin/env python
"""Inspect the Tritonia source (D-029): TIFF stacks + MTrackJ tracks, read-only, standard library only.

    python scripts/inspect_tritonia.py --root E:\\Tritonia [--json out/tritonia_inspection.json]

--root defaults to the UWCT_TRITONIA_ROOT environment variable. For every `<name>.tif` it counts
pages and reads the first page's size from the TIFF headers (no pixel data), parses
`<name>.tif.mdf` when present, and checks every point's frame number and position against the
stack. Prints one line per stack and the totals; --json also writes them (never inside --root).
Exit code 1 if any MDF fails to parse or has a frame number beyond its stack.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.mtrackj import MtrackjError, read_mdf, summarize, tiff_pages


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=os.environ.get("UWCT_TRITONIA_ROOT"))
    parser.add_argument("--json", type=Path, help="also write the inspection as JSON (outside --root)")
    args = parser.parse_args()
    if args.root is None:
        sys.exit("no source: pass --root or set UWCT_TRITONIA_ROOT")
    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"{root} is not a folder")
    if args.json is not None:
        out = args.json.resolve()
        if out == root or root in out.parents:
            sys.exit(f"REFUSED: {out} is inside the read-only source {root}")

    stacks, failed = [], False
    tifs = sorted(root.glob("*.tif"))
    if not tifs:
        sys.exit(f"no .tif files in {root}")
    for tif in tifs:
        row = {"tif": tif.name, "tif_bytes": tif.stat().st_size}
        try:
            row["n_frames"], row["width"], row["height"] = tiff_pages(tif)
        except (MtrackjError, OSError, ValueError) as e:  # struct.error is a ValueError
            row["error"] = str(e)
            failed = True
            stacks.append(row)
            print(f"{tif.name}: TIFF ERROR {e}")
            continue
        mdf = tif.with_name(tif.name + ".mdf")
        if mdf.exists():
            try:
                row["mdf"] = mdf.name
                row.update(summarize(read_mdf(mdf), row["n_frames"], row["width"], row["height"]))
            except (MtrackjError, OSError, UnicodeDecodeError) as e:
                row["error"] = str(e)
                failed = True
            failed = failed or bool(row.get("t_beyond_stack"))
        else:
            row["mdf"] = None
        stacks.append(row)
        if "error" in row:
            desc = f"MDF ERROR {row['error']}"
        elif row["mdf"] is None:
            desc = "no mdf (unlabelled)"
        else:
            desc = (f"{row['n_tracks']} tracks, {row['n_points']} points, t {row['t_min']}-{row['t_max']}, "
                    f"{len(row['points_outside_frame'])} outside frame, {row['t_beyond_stack']} beyond stack")
        print(f"{tif.name}: {row['n_frames']} frames {row['width']}x{row['height']}, {desc}")

    ok = [s for s in stacks if "error" not in s]
    labelled = [s for s in ok if s.get("mdf")]
    totals = {
        "stacks": len(stacks),
        "stacks_with_mdf": len(labelled),
        "frames": sum(s["n_frames"] for s in ok),
        "frames_in_labelled_stacks": sum(s["n_frames"] for s in labelled),
        "tracks": sum(s["n_tracks"] for s in labelled),
        "points": sum(s["n_points"] for s in labelled),
        "points_outside_frame": sum(len(s["points_outside_frame"]) for s in labelled),
        "frame_sizes": sorted({f"{s['width']}x{s['height']}" for s in ok}),
        "errors": len(stacks) - len(ok),
    }
    print("TOTAL " + ", ".join(f"{k} {v}" for k, v in totals.items()))
    if args.json is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"root": str(root), "stacks": stacks, "totals": totals}, indent=2) + "\n",
                       encoding="utf-8")
        print(f"wrote {out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
