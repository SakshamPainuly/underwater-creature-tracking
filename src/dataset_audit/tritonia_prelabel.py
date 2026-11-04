"""Suggested sea slug boxes from the image near each MTrackJ cross (proposal, D-031 draft).

A suggestion is a pre-label for the labeller, never a label by itself: the user reviews every frame.
The box size comes from the pixels (the bright blob next to the cross), never from an assumed slug
size; a cross without a clear blob gets no box and a flag, so it is decided by hand.

Method, per cross (all in image px, inside the camera view holding the cross):
1. 3x3 mean smoothing; the brightest smoothed pixel within SEARCH_PX of the cross is the peak.
2. Local background = median, noise = 1.4826 * MAD of a ring around the peak (the blob's
   immediate surroundings, not distant shadows); each ring in RINGS_PX (inner, outer radius) is
   tried and the one where the peak stands out most is used (a large blob fills the small ring).
3. Contrast = peak - background; below max(MIN_CONTRAST, NOISE_K * noise) -> no box, flag "no clear blob".
4. The blob = pixels connected to the peak (4-neighbours, smoothed) at or above a share of the
   contrast (BLOB_LEVELS, tried in order: a blob merged with a neighbouring bright patch is retried
   at a stricter level), grown at most MAX_HALF_PX from the peak; still too large at the strictest
   level -> no box, flag "blob not compact".
5. Box = blob bounding box + PAD_PX, clipped to the camera view (so it never crosses a border).
6. The on-screen timestamp is white text with a black outline: a box whose surroundings (+2 px) hold
   both a pixel >= TEXT_BRIGHT and one <= TEXT_DARK is dropped, flag "looks like timestamp text".
   (OBSERVED on the 34 hand-labelled frames: only the timestamp box had both, 239 and 8; slug boxes
   had a darkest pixel of 36 or more.)
Standard library only; deterministic.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .tritonia_dataset import TILE_COLS, TILE_ROWS

SEARCH_PX = 15
RINGS_PX = ((10, 16), (16, 24))
MIN_CONTRAST = 12
NOISE_K = 2.5
BLOB_LEVELS = (0.5, 0.65, 0.8)
MAX_HALF_PX = 20
PAD_PX = 1
MIN_SIDE_PX = 4
DUPLICATE_IOU = 0.5
TEXT_BRIGHT, TEXT_DARK = 230, 20

PARAMS = {k: globals()[k] for k in ("SEARCH_PX", "RINGS_PX", "MIN_CONTRAST", "NOISE_K", "BLOB_LEVELS", "MAX_HALF_PX",
                                    "PAD_PX", "MIN_SIDE_PX", "DUPLICATE_IOU", "TEXT_BRIGHT", "TEXT_DARK")}


def tile_bounds(x: float, y: float, width: int, height: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the camera view holding (x, y); x1/y1 exclusive."""
    xs = [round(width * i / TILE_COLS) for i in range(TILE_COLS + 1)]
    ys = [round(height * j / TILE_ROWS) for j in range(TILE_ROWS + 1)]
    i = min(TILE_COLS - 1, max(0, sum(1 for b in xs[1:-1] if x >= b)))
    j = min(TILE_ROWS - 1, max(0, sum(1 for b in ys[1:-1] if y >= b)))
    return xs[i], ys[j], xs[i + 1], ys[j + 1]


def _median(values: list[int]) -> float:
    v = sorted(values)
    n = len(v)
    return float(v[n // 2]) if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _grow(smooth, pi: int, pj: int, thr: float, tile: tuple[int, int, int, int]) -> list | None:
    """Pixels 4-connected to the peak with smooth >= thr; None if the blob reaches MAX_HALF_PX."""
    tx0, ty0, tx1, ty1 = tile
    seen, stack, region = {(pi, pj)}, [(pi, pj)], []
    while stack:
        i, j = stack.pop()
        region.append((i, j))
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if (a, b) in seen or not (tx0 <= a < tx1 and ty0 <= b < ty1):
                continue
            seen.add((a, b))
            if smooth(a, b) >= thr:
                if abs(a - pi) > MAX_HALF_PX or abs(b - pj) > MAX_HALF_PX:
                    return None
                stack.append((a, b))
    return region


def suggest_box(pixels: bytes, width: int, height: int, x: float, y: float) -> dict:
    """Suggestion for one cross: {"box": (xa, ya, xb, yb) or None, "flag": str, "peak", "contrast", "noise"}."""
    tx0, ty0, tx1, ty1 = tile_bounds(x, y, width, height)

    def raw(i: int, j: int) -> int:
        return pixels[j * width + i]

    def smooth(i: int, j: int) -> float:
        vals = [raw(a, b) for b in range(max(ty0, j - 1), min(ty1, j + 2))
                for a in range(max(tx0, i - 1), min(tx1, i + 2))]
        return sum(vals) / len(vals)

    cx, cy = int(round(x)), int(round(y))
    best = None
    for j in range(max(ty0, cy - SEARCH_PX), min(ty1, cy + SEARCH_PX + 1)):
        for i in range(max(tx0, cx - SEARCH_PX), min(tx1, cx + SEARCH_PX + 1)):
            d2 = (i - x) ** 2 + (j - y) ** 2
            if d2 > SEARCH_PX ** 2:
                continue
            key = (smooth(i, j), -d2)
            if best is None or key > best[0]:
                best = (key, i, j)
    out = {"box": None, "flag": "", "peak": None, "contrast": 0.0, "noise": 0.0}
    if best is None:
        out["flag"] = "cross outside the frame"
        return out
    (peak_val, _), pi, pj = best
    out["peak"] = (pi, pj)
    options = []
    for r_in, r_out in RINGS_PX:
        window = [raw(i, j) for j in range(max(ty0, pj - r_out), min(ty1, pj + r_out + 1))
                  for i in range(max(tx0, pi - r_out), min(tx1, pi + r_out + 1))
                  if r_in ** 2 <= (i - pi) ** 2 + (j - pj) ** 2 <= r_out ** 2]
        if window:
            bg_r = _median(window)
            noise_r = 1.4826 * _median([abs(v - bg_r) for v in window])
            margin = peak_val - bg_r - max(MIN_CONTRAST, NOISE_K * noise_r)
            options.append((margin, bg_r, noise_r))
    _, bg, noise = max(options)
    contrast = peak_val - bg
    out["contrast"], out["noise"] = round(contrast, 2), round(noise, 2)
    if contrast < max(MIN_CONTRAST, NOISE_K * noise):
        out["flag"] = "no clear blob"
        return out

    region = None
    for level in BLOB_LEVELS:
        region = _grow(smooth, pi, pj, bg + contrast * level, (tx0, ty0, tx1, ty1))
        if region is not None:
            break
    if region is None:
        out["flag"] = "blob not compact"
        return out
    xa = min(i for i, _ in region) - PAD_PX
    xb = max(i for i, _ in region) + 1 + PAD_PX
    ya = min(j for _, j in region) - PAD_PX
    yb = max(j for _, j in region) + 1 + PAD_PX
    # grow to the minimum side around the centre, then clip to the camera view
    if xb - xa < MIN_SIDE_PX:
        c = (xa + xb) / 2
        xa, xb = c - MIN_SIDE_PX / 2, c + MIN_SIDE_PX / 2
    if yb - ya < MIN_SIDE_PX:
        c = (ya + yb) / 2
        ya, yb = c - MIN_SIDE_PX / 2, c + MIN_SIDE_PX / 2
    box = (float(max(tx0, xa)), float(max(ty0, ya)), float(min(tx1, xb)), float(min(ty1, yb)))
    around = [raw(i, j) for j in range(max(0, int(box[1]) - 2), min(height, int(box[3]) + 2))
              for i in range(max(0, int(box[0]) - 2), min(width, int(box[2]) + 2))]
    if max(around) >= TEXT_BRIGHT and min(around) <= TEXT_DARK:
        out["flag"] = "looks like timestamp text"
        return out
    if box != (float(xa), float(ya), float(xb), float(yb)):
        out["flag"] = "touches camera border"
    out["box"] = box
    return out


def suggest_frame(pixels: bytes, width: int, height: int, points: list[tuple[float, float]]) -> list[dict]:
    """One suggestion per cross; a box duplicating an earlier one (IoU > DUPLICATE_IOU) is dropped and flagged."""
    results = []
    for (x, y) in points:
        s = suggest_box(pixels, width, height, x, y)
        s["cross"] = (x, y)
        if s["box"] is not None:
            for r in results:
                if r["box"] is not None and iou(r["box"], s["box"]) > DUPLICATE_IOU:
                    s["box"], s["flag"] = None, "same blob as another cross"
                    break
        results.append(s)
    return results


def read_points(path: Path) -> list[tuple[float, float]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [(float(r["x"]), float(r["y"])) for r in csv.DictReader(f)]


def compare(suggested: list[dict], hand_boxes: list[tuple[float, ...]]) -> list[dict]:
    """Per cross: the hand box nearest the cross (centre distance) and how the suggestion relates to it.

    "inside" = share of the suggested box covered by that hand box (1.0 = fully inside);
    hand boxes in batch 1 are often looser than the slug, so IoU alone understates agreement.
    """
    rows = []
    for s in suggested:
        x, y = s["cross"]
        hand = min(hand_boxes, key=lambda b: ((b[0] + b[2]) / 2 - x) ** 2 + ((b[1] + b[3]) / 2 - y) ** 2,
                   default=None)
        row = {"cross_x": x, "cross_y": y, "flag": s["flag"], "suggested": s["box"], "hand": hand,
               "iou": None, "inside": None}
        if hand is not None and s["box"] is not None:
            b = s["box"]
            ix = max(0.0, min(b[2], hand[2]) - max(b[0], hand[0]))
            iy = max(0.0, min(b[3], hand[3]) - max(b[1], hand[1]))
            row["iou"] = round(iou(b, hand), 3)
            row["inside"] = round(ix * iy / ((b[2] - b[0]) * (b[3] - b[1])), 3)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- automatic labels (D-031, crosses plan)

def auto_label_batch(batch_dir: Path, out_dir: Path) -> dict:
    """Suggested boxes as the labels of every batch frame: <out>/labels/<stem>.txt, notes.txt,
    suggestions.csv (one row per cross) and labels.zip (the layout the ingest reads).

    A frame where no cross got a box is marked `exclude` in notes.txt: an empty label file would
    say "no slug", which the crosses contradict. Nothing is reviewed by a person (D-031).
    """
    import zipfile

    from .tritonia_labeller import format_label, read_gray_png
    batch_dir, out_dir = Path(batch_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; automatic labels are written once")
    with open(batch_dir / "manifest.csv", newline="", encoding="utf-8") as f:
        stems = [Path(r["image"]).stem for r in csv.DictReader(f)]
    (out_dir / "labels").mkdir(parents=True)
    rows, notes, counts = [], [], {"frames": len(stems), "crosses": 0, "boxes": 0, "excluded": 0}
    for stem in stems:
        w, h, px = read_gray_png(batch_dir / "images" / f"{stem}.png")
        found = suggest_frame(px, w, h, read_points(batch_dir / "points" / f"{stem}.csv"))
        boxes = [r["box"] for r in found if r["box"] is not None]
        counts["crosses"] += len(found)
        counts["boxes"] += len(boxes)
        for r in found:
            rows.append({"frame": stem, "cross_x": r["cross"][0], "cross_y": r["cross"][1],
                         "box": "" if r["box"] is None else " ".join(f"{v:.1f}" for v in r["box"]),
                         "flag": r["flag"], "contrast": r["contrast"], "noise": r["noise"]})
        if not boxes:
            notes.append(f"{stem} | exclude | automatic labels: no cross got a box\n")
            counts["excluded"] += 1
            continue
        (out_dir / "labels" / f"{stem}.txt").write_text(format_label(boxes, w, h), encoding="utf-8", newline="\n")
    (out_dir / "notes.txt").write_text(
        "# frame | note or exclude | reason   (automatic labels, D-031; see suggestions.csv)\n" + "".join(notes),
        encoding="utf-8", newline="\n")
    with open(out_dir / "suggestions.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    with zipfile.ZipFile(out_dir / "labels.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((out_dir / "labels").glob("*.txt")):
            z.write(p, p.name)
    counts["flags"] = {}
    for r in rows:
        if r["flag"]:
            counts["flags"][r["flag"]] = counts["flags"].get(r["flag"], 0) + 1
    counts["params"] = PARAMS
    import json
    (out_dir / "auto_label_report.json").write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n",
                                                    encoding="utf-8")
    return counts
