"""Tritonia labelling batch (D-029): pick frames, extract them from TIFF stacks to PNG, standard library only.

The stacks are uncompressed 8-bit grayscale TIFFs (OBSERVED: compression none, one strip per page).
`read_page` reads one page's bytes; `write_png_gray` writes them as a PNG with zlib. Hint images
copy the frame and draw a cross at each MTrackJ point so the labeller can find the slugs; the clean
frame is what gets boxed. Nothing here writes next to its inputs; callers pass an output folder.
"""

from __future__ import annotations

import random
import struct
import zlib
from pathlib import Path

from .mtrackj import MdfFile, MtrackjError

_TAG_WIDTH, _TAG_HEIGHT, _TAG_BITS, _TAG_COMPRESSION = 256, 257, 258, 259
_TAG_PHOTOMETRIC, _TAG_STRIP_OFFSETS, _TAG_SPP, _TAG_STRIP_BYTES = 262, 273, 277, 279


def _ifd_tags(f, e: str, offset: int, path: Path) -> tuple[dict, int]:
    f.seek(offset)
    (n,) = struct.unpack(e + "H", f.read(2))
    entries = f.read(12 * n)
    tags = {}
    for i in range(n):
        tag, typ, count = struct.unpack(e + "HHI", entries[12 * i:12 * i + 8])
        raw = entries[12 * i + 8:12 * i + 12]
        if typ == 3 and count == 1:
            tags[tag] = struct.unpack(e + "H", raw[:2])[0]
        elif typ == 4 and count == 1:
            tags[tag] = struct.unpack(e + "I", raw)[0]
        elif tag in (_TAG_STRIP_OFFSETS, _TAG_STRIP_BYTES):
            raise MtrackjError(f"{path}: page at {offset} has {count} strips; only single-strip pages are supported")
    (nxt,) = struct.unpack(e + "I", f.read(4))
    return tags, nxt


def page_strips(path: Path) -> tuple[int, int, list[tuple[int, int]]]:
    """(width, height, [(strip offset, byte count) per page]) for an uncompressed 8-bit gray TIFF."""
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8 or head[:2] not in (b"II", b"MM"):
            raise MtrackjError(f"{path}: not a TIFF file")
        e = "<" if head[:2] == b"II" else ">"
        magic, offset = struct.unpack(e + "HI", head[2:8])
        if magic != 42:
            raise MtrackjError(f"{path}: TIFF magic {magic} is not 42 (BigTIFF is not supported)")
        pages, seen, size = [], set(), None
        while offset:
            if offset in seen:
                raise MtrackjError(f"{path}: IFD chain loops at offset {offset}")
            seen.add(offset)
            tags, nxt = _ifd_tags(f, e, offset, path)
            w, h = tags.get(_TAG_WIDTH), tags.get(_TAG_HEIGHT)
            if size is None:
                size = (w, h)
            if (w, h) != size:
                raise MtrackjError(f"{path}: page {len(pages)} is {w}x{h}, page 0 is {size[0]}x{size[1]}")
            if (tags.get(_TAG_COMPRESSION, 1), tags.get(_TAG_BITS, 1), tags.get(_TAG_SPP, 1),
                    tags.get(_TAG_PHOTOMETRIC)) != (1, 8, 1, 1):
                raise MtrackjError(f"{path}: page {len(pages)} is not uncompressed 8-bit grayscale "
                                   "(BlackIsZero); this reader does not decode it")
            off, count = tags.get(_TAG_STRIP_OFFSETS), tags.get(_TAG_STRIP_BYTES)
            if off is None or count != w * h:
                raise MtrackjError(f"{path}: page {len(pages)} strip is {count} bytes, expected {w * h}")
            pages.append((off, count))
            offset = nxt
    if not pages:
        raise MtrackjError(f"{path}: TIFF has no pages")
    return size[0], size[1], pages


def read_page(path: Path, strip: tuple[int, int]) -> bytes:
    off, count = strip
    with open(path, "rb") as f:
        f.seek(off)
        data = f.read(count)
    if len(data) != count:
        raise MtrackjError(f"{path}: strip at {off} is truncated ({len(data)} of {count} bytes)")
    return data


def write_png_gray(path: Path, width: int, height: int, pixels: bytes) -> None:
    """Write 8-bit grayscale pixels (row-major, width * height bytes) as a PNG. Refuses to overwrite."""
    if len(pixels) != width * height:
        raise ValueError(f"expected {width * height} pixel bytes, got {len(pixels)}")
    raw = b"".join(b"\x00" + pixels[y * width:(y + 1) * width] for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    with open(path, "xb") as f:
        f.write(png)


def draw_hints(pixels: bytes, width: int, height: int, points: list[tuple[float, float]],
               arm: int = 6) -> bytes:
    """Copy of the frame with a black-outlined white cross at each point (for finding slugs only)."""
    img = bytearray(pixels)

    def put(x: int, y: int, v: int) -> None:
        if 0 <= x < width and 0 <= y < height:
            img[y * width + x] = v

    for px, py in points:
        cx, cy = round(px), round(py)
        for d in range(-arm - 1, arm + 2):
            for o in (-1, 1):
                put(cx + d, cy + o, 0)
                put(cx + o, cy + d, 0)
        for d in range(-arm, arm + 1):
            put(cx + d, cy, 255)
            put(cx, cy + d, 255)
    return bytes(img)


def pick_frames(mdfs: dict[str, MdfFile], total: int, seed: int, min_gap: int) -> list[tuple[str, int]]:
    """Spread `total` frames over the stacks as evenly as possible (stacks sorted by name; which ones get
    one extra is drawn with `seed`). Within a stack, candidates are frames with at least one point, drawn
    at random and kept only if at least `min_gap` frames from every frame already kept."""
    names = sorted(mdfs)
    if not names:
        raise ValueError("no labelled stacks")
    rng = random.Random(seed)
    base, extra = divmod(total, len(names))
    plus = set(rng.sample(names, extra))
    picks = []
    for name in names:
        want = base + (name in plus)
        cands = sorted({p.t for p in mdfs[name].points})
        rng.shuffle(cands)
        kept = []
        for t in cands:
            if all(abs(t - k) >= min_gap for k in kept):
                kept.append(t)
                if len(kept) == want:
                    break
        if len(kept) < want:
            raise ValueError(f"{name}: only {len(kept)} frames with points are {min_gap}+ frames apart; "
                             f"wanted {want}. Lower min_gap or the total.")
        picks += [(name, t) for t in sorted(kept)]
    return picks


def pick_extra_frames(mdfs: dict[str, MdfFile], days: list[str], per_day: int, seed: int, min_gap: int,
                      taken: dict[str, list[int]]) -> list[tuple[str, int]]:
    """Batch 2 (D-035): up to `per_day` more frames per stack in `days` (the training days only).

    Candidates are frames with at least one point that are at least `min_gap` frames from every frame
    already in `taken` (batch 1) and from every frame kept here. Drawn at random with one generator seeded
    with `seed`, stacks in sorted order. A stack with fewer candidates gives what it has (reported by the
    caller); val and test days are never touched because only `days` is read.
    """
    rng = random.Random(seed)
    picks = []
    for name in sorted(days):
        if name not in mdfs:
            raise ValueError(f"{name}: no MTrackJ file for this training day")
        cands = sorted({p.t for p in mdfs[name].points})
        rng.shuffle(cands)
        kept, base = [], list(taken.get(name, []))
        for t in cands:
            if all(abs(t - k) >= min_gap for k in kept + base):
                kept.append(t)
                if len(kept) == per_day:
                    break
        picks += [(name, t) for t in sorted(kept)]
    return picks
