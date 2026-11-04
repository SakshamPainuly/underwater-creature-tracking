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
