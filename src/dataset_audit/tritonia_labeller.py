"""Logic behind scripts/label_tritonia.py (D-029 batch 1): boxes <-> YOLO lines, saving, export ZIP.

No GUI code here, so it is tested without a window. Every box is class 1 (sea_slug); a box is
checked with the same rule as the ingest (tritonia_dataset.box_problems), so the labeller can never
produce a file that the ingest would refuse for a box. The export ZIP has one <stem>.txt per
labelled frame at its top level, the layout scripts/ingest_tritonia_boxes.py reads.
"""

from __future__ import annotations

import os
import struct
import zipfile
import zlib
from collections import Counter
from pathlib import Path

from .tritonia_dataset import CLASS_NAMES, box_problems, parse_notes, parse_yolo_lines

SEA_SLUG = CLASS_NAMES.index("sea_slug")
MIN_BOX_PX = 3


def normalise_px(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, ...]:
    """Corner order fixed and clipped to the frame (a drag may start or end outside it)."""
    xa, xb = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    ya, yb = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    return xa, ya, xb, yb


def px_to_yolo(box: tuple[float, ...], width: int, height: int, cls: int = SEA_SLUG) -> tuple:
    xa, ya, xb, yb = box
    return (cls, (xa + xb) / 2 / width, (ya + yb) / 2 / height, (xb - xa) / width, (yb - ya) / height)


def yolo_to_px(box: tuple, width: int, height: int) -> tuple[float, ...]:
    _, x, y, w, h = box
    return ((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)


def check_new_box(box: tuple[float, ...], width: int, height: int) -> list[str]:
    """Why a drawn box (pixel corners) cannot be kept; empty list = keep it."""
    xa, ya, xb, yb = box
    if xb - xa < MIN_BOX_PX or yb - ya < MIN_BOX_PX:
        return [f"box smaller than {MIN_BOX_PX} px"]
    return [p.split(": ", 1)[-1] for p in box_problems([px_to_yolo(box, width, height)], width, height)]


def find_handle(boxes_px: list[tuple[float, ...]], x: float, y: float, tol: float) -> tuple[int, str] | None:
    """Box edge or corner under (x, y): (box index, edges) with edges a subset of "lrtb", or None.

    tol is the grab distance in image px; it shrinks for a small box so its inside stays free for
    drawing. The smallest box wins where boxes overlap. Inside a box (away from its edges) is None,
    so a new box can still be drawn inside an old one.
    """
    best = None
    for i, (xa, ya, xb, yb) in enumerate(boxes_px):
        t = min(tol, (xb - xa) / 3, (yb - ya) / 3)
        if not (xa - t <= x <= xb + t and ya - t <= y <= yb + t):
            continue
        edges = ("l" if abs(x - xa) <= t else "r" if abs(x - xb) <= t else "") + \
                ("t" if abs(y - ya) <= t else "b" if abs(y - yb) <= t else "")
        area = (xb - xa) * (yb - ya)
        if edges and (best is None or area < best[0]):
            best = (area, i, edges)
    return None if best is None else (best[1], best[2])


def resize_box(box: tuple[float, ...], edges: str, x: float, y: float, width: int, height: int) -> tuple[float, ...]:
    """Move the grabbed edges of box to (x, y); corners re-sorted and clipped like a drawn box."""
    xa, ya, xb, yb = box
    if "l" in edges:
        xa = x
    if "r" in edges:
        xb = x
    if "t" in edges:
        ya = y
    if "b" in edges:
        yb = y
    return normalise_px(xa, ya, xb, yb, width, height)


def format_label(boxes_px: list[tuple[float, ...]], width: int, height: int) -> str:
    lines = []
    for b in boxes_px:
        cls, x, y, w, h = px_to_yolo(b, width, height)
        lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return "".join(line + "\n" for line in lines)


def save_label(path: Path, boxes_px: list[tuple[float, ...]], width: int, height: int) -> None:
    """Write atomically, so a crash never leaves half a label file."""
    path = Path(path)
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text(format_label(boxes_px, width, height), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def load_label(path: Path, width: int, height: int) -> list[tuple[float, ...]]:
    path = Path(path)
    if not path.is_file():
        return []
    return [yolo_to_px(b, width, height) for b in parse_yolo_lines(path.read_text(encoding="utf-8"), path.name)]


def excluded_stems(notes_path: Path, stems: set[str]) -> set[str]:
    if not Path(notes_path).is_file():
        return set()
    notes = parse_notes(Path(notes_path).read_text(encoding="utf-8-sig"), stems)
    return {s for s, ns in notes.items() if any(n["kind"] == "exclude" for n in ns)}


def append_note(notes_path: Path, stem: str, kind: str, reason: str) -> None:
    reason = " ".join(reason.replace("|", "/").split())
    if kind not in ("note", "exclude") or not reason:
        raise ValueError("a note needs kind 'note' or 'exclude' and a reason")
    notes_path = Path(notes_path)
    text = notes_path.read_text(encoding="utf-8-sig") if notes_path.is_file() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    notes_path.write_text(text + f"{stem} | {kind} | {reason}\n", encoding="utf-8", newline="\n")


def progress(labels_dir: Path, stems: list[str], excluded: set[str]) -> dict:
    done = {s for s in stems if (Path(labels_dir) / f"{s}.txt").is_file()}
    return {"frames": len(stems), "labelled": len(done - excluded), "excluded": len(excluded),
            "left": len([s for s in stems if s not in done and s not in excluded])}


def read_gray_png(path: Path) -> tuple[int, int, bytes]:
    """Pixels of an 8-bit grayscale PNG as written by tritonia_frames.write_png_gray (filter 0 rows)."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    pos, idat, w = 8, b"", None
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, colour = struct.unpack(">IIBB", body[:10])
            if (depth, colour) != (8, 0):
                raise ValueError(f"{path}: not 8-bit grayscale")
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    rows = [raw[y * (w + 1):(y + 1) * (w + 1)] for y in range(h)]
    if any(r[0] != 0 for r in rows):
        raise ValueError(f"{path}: filtered rows; only unfiltered PNGs from the batch builder are supported")
    return w, h, b"".join(r[1:] for r in rows)


def stretch(pixels: bytes, low_pct: float = 1.0, high_pct: float = 99.5) -> bytes:
    """Display-only contrast stretch: the low..high percentile range is spread over 0..255."""
    counts = Counter(pixels)
    total, seen, lo, hi = len(pixels), 0, None, 255
    for v in range(256):
        seen += counts.get(v, 0)
        if lo is None and seen >= total * low_pct / 100:
            lo = v
        if seen >= total * high_pct / 100:
            hi = v
            break
    lo = lo or 0
    if hi <= lo:
        return pixels
    table = bytes(min(255, max(0, round((v - lo) * 255 / (hi - lo)))) for v in range(256))
    return pixels.translate(table)


def stretch_tiles(pixels: bytes, width: int, height: int, cols: int, rows: int) -> bytes:
    """stretch() applied to each camera tile on its own (the 6 cameras differ in brightness)."""
    out = bytearray(pixels)
    xs = [round(width * i / cols) for i in range(cols + 1)]
    ys = [round(height * j / rows) for j in range(rows + 1)]
    for j in range(rows):
        for i in range(cols):
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            tile = b"".join(pixels[y * width + x0:y * width + x1] for y in range(y0, y1))
            done = stretch(tile)
            tw = x1 - x0
            for k, y in enumerate(range(y0, y1)):
                out[y * width + x0:y * width + x1] = done[k * tw:(k + 1) * tw]
    return bytes(out)


def pgm(width: int, height: int, pixels: bytes) -> bytes:
    """Binary PGM (Tk reads it natively: tk.PhotoImage(data=..., format='PPM'))."""
    return f"P5 {width} {height} 255\n".encode() + pixels


def export_zip(labels_dir: Path, stems: list[str], zip_path: Path) -> int:
    """Pack every <stem>.txt that exists into a new ZIP (never overwrites); return how many."""
    zip_path = Path(zip_path)
    if zip_path.exists():
        raise FileExistsError(f"{zip_path} already exists")
    files = [Path(labels_dir) / f"{s}.txt" for s in stems if (Path(labels_dir) / f"{s}.txt").is_file()]
    with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)
    return len(files)


# ---------------------------------------------------------------- suggested boxes (D-033 proposal)
# Each box has an origin: "hand" (drawn by the user), "suggested" (from tritonia_prelabel, kept as
# it was) or "suggested_resized" (from tritonia_prelabel, resized by the user). A frame that opened
# with suggestions has <labels>/_provenance/<stem>.json; a frame without one was labelled by hand.
ORIGINS = ("hand", "suggested", "suggested_resized")
PROVENANCE_DIR = "_provenance"
PROVENANCE_FIELDS = ["frame", "source", "boxes", "hand", "suggested_kept", "suggested_resized",
                     "suggested_deleted", "suggested_total", "crosses_flagged"]


def provenance_path(labels_dir: Path, stem: str) -> Path:
    return Path(labels_dir) / PROVENANCE_DIR / f"{stem}.json"


def save_provenance(labels_dir: Path, stem: str, origins: list[str], suggested_total: int,
                    suggested_deleted: int, flags: list[dict], params: dict) -> None:
    """Per-frame record of where each saved box came from (same order as the label file)."""
    import json
    if any(o not in ORIGINS for o in origins):
        raise ValueError(f"unknown box origin in {origins}")
    path = provenance_path(labels_dir, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"frame": stem, "origins": origins, "suggested_total": suggested_total,
                               "suggested_deleted": suggested_deleted, "flags": flags,
                               "suggester": params}, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_provenance(labels_dir: Path, stem: str, n_boxes: int) -> dict | None:
    """The frame's record, or None if it has none or it no longer matches the label file."""
    import json
    path = provenance_path(labels_dir, stem)
    if not path.is_file():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    return rec if len(rec.get("origins", [])) == n_boxes else None


def provenance_rows(labels_dir: Path, stems: list[str], width: int, height: int) -> list[dict]:
    """One row per labelled frame: how many boxes came from the user and from the suggester."""
    rows = []
    for s in stems:
        label = Path(labels_dir) / f"{s}.txt"
        if not label.is_file():
            continue
        n = len(load_label(label, width, height))
        rec = load_provenance(labels_dir, s, n)
        origins = rec["origins"] if rec else ["hand"] * n
        rows.append({"frame": s, "source": "suggested+reviewed" if rec else "hand",
                     "boxes": n, "hand": origins.count("hand"),
                     "suggested_kept": origins.count("suggested"),
                     "suggested_resized": origins.count("suggested_resized"),
                     "suggested_deleted": rec["suggested_deleted"] if rec else 0,
                     "suggested_total": rec["suggested_total"] if rec else 0,
                     "crosses_flagged": len(rec["flags"]) if rec else 0})
    return rows


def export_provenance_csv(labels_dir: Path, stems: list[str], width: int, height: int, csv_path: Path) -> int:
    """Write provenance_rows to a new CSV (never overwrites); return the row count."""
    import csv
    csv_path = Path(csv_path)
    if csv_path.exists():
        raise FileExistsError(f"{csv_path} already exists")
    rows = provenance_rows(labels_dir, stems, width, height)
    with open(csv_path, "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PROVENANCE_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return len(rows)
