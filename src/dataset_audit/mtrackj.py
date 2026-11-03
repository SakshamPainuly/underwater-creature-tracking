"""Read MTrackJ data files (`.mdf`) and count TIFF stack pages, standard library only (D-029).

MTrackJ (FIJI plugin) stores manual tracks as text. The lines this reader accepts:

    MTrackJ 1.5.0 Data File
    Displaying ...                      (display settings, ignored)
    Reference <x> <y> <z>               (optional, kept as read; meaning not yet confirmed)
    Assembly <id> <colour>
    Cluster <id> <colour>
    Track <id> <colour> <flag>          (flag true/false, kept as read; meaning not yet confirmed)
    Point <id> <x> <y> <z> <t> <c>      (x, y in image pixels; t = 1-based frame number)
    End of MTrackJ Data File

Anything else is an error, so a file in an unexpected format is reported, not half-read.
Nothing here writes files; the readers only open their inputs for reading.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path


class MtrackjError(ValueError):
    pass


@dataclass(frozen=True)
class Point:
    point_id: int
    x: float
    y: float
    z: float
    t: int
    c: float


@dataclass
class Track:
    cluster_id: int
    track_id: int
    colour: str
    flag: str
    points: list[Point] = field(default_factory=list)


@dataclass
class MdfFile:
    version: str
    reference: tuple[float, ...] | None
    tracks: list[Track]

    @property
    def points(self) -> list[Point]:
        return [p for tr in self.tracks for p in tr.points]


def _num(token: str, what: str, lineno: int) -> float:
    try:
        return float(token)
    except ValueError:
        raise MtrackjError(f"line {lineno}: {what} {token!r} is not a number") from None


def parse_mdf(text: str) -> MdfFile:
    """Parse the text of one `.mdf` file. Raises MtrackjError with the line number on bad input."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("MTrackJ ") or not lines[0].endswith(" Data File"):
        raise MtrackjError("line 1: not an MTrackJ data file (expected 'MTrackJ <version> Data File')")
    version = lines[0][len("MTrackJ "):-len(" Data File")]
    reference, tracks, cluster, track, ended = None, [], None, None, False
    for lineno, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if not parts:
            continue
        if ended:
            raise MtrackjError(f"line {lineno}: content after 'End of MTrackJ Data File'")
        key = parts[0]
        if key == "Displaying":
            continue
        if line.strip() == "End of MTrackJ Data File":
            ended = True
        elif key == "Reference":
            reference = tuple(_num(v, "reference value", lineno) for v in parts[1:])
        elif key == "Assembly":
            cluster, track = None, None
        elif key == "Cluster":
            if len(parts) < 2:
                raise MtrackjError(f"line {lineno}: Cluster line without an id")
            cluster, track = int(_num(parts[1], "cluster id", lineno)), None
        elif key == "Track":
            if cluster is None:
                raise MtrackjError(f"line {lineno}: Track outside a Cluster")
            if len(parts) != 4:
                raise MtrackjError(f"line {lineno}: expected 'Track <id> <colour> <flag>', got {line!r}")
            track = Track(cluster, int(_num(parts[1], "track id", lineno)), parts[2], parts[3])
            tracks.append(track)
        elif key == "Point":
            if track is None:
                raise MtrackjError(f"line {lineno}: Point outside a Track")
            if len(parts) != 7:
                raise MtrackjError(f"line {lineno}: expected 'Point <id> <x> <y> <z> <t> <c>', got {line!r}")
            pid, x, y, z, t, c = (_num(v, n, lineno) for v, n in
                                  zip(parts[1:], ("point id", "x", "y", "z", "t", "c")))
            if t != int(t) or t < 1:
                raise MtrackjError(f"line {lineno}: frame t={parts[5]} is not a whole number >= 1")
            track.points.append(Point(int(pid), x, y, z, int(t), c))
        else:
            raise MtrackjError(f"line {lineno}: unknown line type {key!r}")
    if not ended:
        raise MtrackjError("file ends without 'End of MTrackJ Data File' (truncated?)")
    keys = [(tr.cluster_id, tr.track_id) for tr in tracks]
    if len(set(keys)) != len(keys):
        raise MtrackjError("the same (cluster, track) id occurs more than once")
    return MdfFile(version, reference, tracks)


def read_mdf(path: Path) -> MdfFile:
    try:
        return parse_mdf(Path(path).read_text(encoding="utf-8"))
    except MtrackjError as e:
        raise MtrackjError(f"{path}: {e}") from None


_TIFF_WIDTH, _TIFF_HEIGHT = 256, 257


def tiff_pages(path: Path) -> tuple[int, int, int]:
    """(page count, width, height of the first page) of a classic TIFF, by walking its IFD chain.

    Reads headers only, never pixel data. BigTIFF and a looping IFD chain are errors.
    """
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8 or head[:2] not in (b"II", b"MM"):
            raise MtrackjError(f"{path}: not a TIFF file")
        e = "<" if head[:2] == b"II" else ">"
        magic, offset = struct.unpack(e + "HI", head[2:8])
        if magic != 42:
            raise MtrackjError(f"{path}: TIFF magic {magic} is not 42 (BigTIFF is not supported)")
        pages, seen, size = 0, set(), None
        while offset:
            if offset in seen:
                raise MtrackjError(f"{path}: IFD chain loops at offset {offset}")
            seen.add(offset)
            f.seek(offset)
            (n,) = struct.unpack(e + "H", f.read(2))
            entries = f.read(12 * n)
            if size is None:
                tags = {}
                for i in range(n):
                    tag, typ, _count = struct.unpack(e + "HHI", entries[12 * i:12 * i + 8])
                    raw = entries[12 * i + 8:12 * i + 12]
                    tags[tag] = struct.unpack(e + "H", raw[:2])[0] if typ == 3 else struct.unpack(e + "I", raw)[0]
                if _TIFF_WIDTH not in tags or _TIFF_HEIGHT not in tags:
                    raise MtrackjError(f"{path}: first page has no width/height tag")
                size = (tags[_TIFF_WIDTH], tags[_TIFF_HEIGHT])
            (offset,) = struct.unpack(e + "I", f.read(4))
            pages += 1
    if size is None:
        raise MtrackjError(f"{path}: TIFF has no pages")
    return pages, size[0], size[1]


def summarize(mdf: MdfFile, n_frames: int | None = None, width: int | None = None,
              height: int | None = None) -> dict:
    """Counts plus the checks against the image stack when its frame count and size are given."""
    pts = mdf.points
    per_track = sorted(len(tr.points) for tr in mdf.tracks)
    out = {
        "mtrackj_version": mdf.version,
        "reference": list(mdf.reference) if mdf.reference else None,
        "n_tracks": len(mdf.tracks),
        "n_points": len(pts),
        "points_per_track_min": per_track[0] if per_track else 0,
        "points_per_track_max": per_track[-1] if per_track else 0,
        "track_flags": {f: sum(tr.flag == f for tr in mdf.tracks) for f in sorted({tr.flag for tr in mdf.tracks})},
        "t_min": min((p.t for p in pts), default=None),
        "t_max": max((p.t for p in pts), default=None),
    }
    if n_frames is not None:
        out["t_beyond_stack"] = sum(p.t > n_frames for p in pts)
    if width is not None and height is not None:
        out["points_outside_frame"] = [
            {"track": tr.track_id, "point": p.point_id, "x": p.x, "y": p.y, "t": p.t}
            for tr in mdf.tracks for p in tr.points
            if not (0 <= p.x < width and 0 <= p.y < height)
        ]
    return out
