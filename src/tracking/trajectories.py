"""Trajectory statistics and overlay figures for Tritonia tracks (D-044 phase 5), standard library only.

Statistics are computed on each day's evaluation frames only (the clicked frames), so clicked and tracked paths are
sampled alike. A step is two consecutive points of one track at most 2 x the day's click cadence apart; longer
gaps are not part of the path. Units: px and px/s (15 s per frame); no physical units (rule 9). Speeds are not
comparable between cameras (angled cameras, parallax; Tritonia Info.txt).

Figures: one SVG per day, two panels (clicks above, tracker below) on a lightened frame of that day. Colour by the
dataviz method: per camera the 3 longest tracks take categorical slots 1-3 (validated all-pairs on the light
surface; aqua is below 3:1, so coloured tracks carry a direct label), every other track is neutral gray ("other").
Tracks are ranked by spatial extent (bounding-box diagonal), so the coloured ones are those that range furthest;
ranking by points or by path length coloured stationary, jittering detections (checked on the 20130529 preview).
"""

from __future__ import annotations

import base64
import math
import statistics as st
import struct
import zlib
from collections import defaultdict

SECONDS_PER_FRAME = 15.0
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a")  # dataviz reference palette, light, slots 1-3
OTHER = "#52514e"
SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"


def group_tracks(points: list[tuple]) -> dict:
    """[(camera, track_id, t, x, y)] -> {(camera, track_id): [(t, x, y)] sorted by t}."""
    out = defaultdict(list)
    for cam, tid, t, x, y in points:
        out[(cam, tid)].append((t, x, y))
    return {k: sorted(v) for k, v in out.items()}


def track_stats(pts: list[tuple], cadence: int) -> dict | None:
    """Stats of one track's points [(t, x, y)] (already restricted to evaluation frames); None if < 2 points."""
    if len(pts) < 2:
        return None
    steps = [(b[0] - a[0], math.hypot(b[1] - a[1], b[2] - a[2])) for a, b in zip(pts, pts[1:])]
    linked = [(dt, d) for dt, d in steps if dt <= 2 * cadence]
    speeds = [d / (dt * SECONDS_PER_FRAME) for dt, d in linked]
    span = pts[-1][0] - pts[0][0]
    return {"points": len(pts), "first_t": pts[0][0], "last_t": pts[-1][0],
            "duration_s": span * SECONDS_PER_FRAME, "path_px": round(sum(d for _, d in linked), 2),
            "net_px": round(math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]), 2),
            "linked_steps": len(linked), "gaps": len(steps) - len(linked),
            "median_step_speed_px_s": round(st.median(speeds), 4) if speeds else None}


def summarize(stats: list[dict]) -> dict:
    """Medians over tracks with at least one linked step."""
    s = [x for x in stats if x and x["median_step_speed_px_s"] is not None]
    med = lambda k: round(st.median(x[k] for x in s), 4) if s else None
    return {"tracks": len(s), "median_points": med("points"), "median_duration_s": med("duration_s"),
            "median_path_px": med("path_px"), "median_net_px": med("net_px"),
            "median_step_speed_px_s": med("median_step_speed_px_s")}


def match_tracks(gt: dict, pred: dict, frames: set, match_px: float = 10.0) -> dict:
    """For each clicked track, the tracker track (same camera) with most points within match_px on shared frames."""
    by_cam_t = defaultdict(list)
    for (cam, pid), pts in pred.items():
        for t, x, y in pts:
            if t in frames:
                by_cam_t[(cam, t)].append((pid, x, y))
    out = {}
    for (cam, gid), pts in gt.items():
        hits = defaultdict(int)
        for t, x, y in pts:
            for pid, px, py in by_cam_t.get((cam, t), []):
                if math.hypot(px - x, py - y) <= match_px:
                    hits[pid] += 1
        if hits:
            pid = min(hits, key=lambda p: (-hits[p], p))
            out[(cam, gid)] = ((cam, pid), hits[pid], len(pts))
    return out


# ------------------------------------------------------------------ figures

def png_gray_bytes(width: int, height: int, pixels: bytes) -> bytes:
    raw = b"".join(b"\x00" + pixels[y * width:(y + 1) * width] for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def lighten(pixels: bytes, keep: float = 0.45, to: int = 252) -> bytes:
    """Blend toward the light surface so marks keep their contrast on the photo."""
    return bytes(int(to - (to - p) * keep) for p in pixels)


def extent(pts: list[tuple]) -> float:
    """Bounding-box diagonal in px: near 0 for a detection that jitters in place, however long it lasts."""
    xs, ys = [p[1] for p in pts], [p[2] for p in pts]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def colour_tracks(tracks: dict) -> dict:
    """{(camera, id): colour}: per camera the 3 tracks with the largest extent take slots 1-3, others gray."""
    by_cam = defaultdict(list)
    for (cam, tid), pts in tracks.items():
        by_cam[cam].append((-round(extent(pts), 6), tid))
    out = {}
    for cam, items in by_cam.items():
        for rank, (_, tid) in enumerate(sorted(items)):
            out[(cam, tid)] = SLOTS[rank] if rank < len(SLOTS) else OTHER
    return out


def _segments(pts: list[tuple], max_dt: int) -> list[list[tuple]]:
    segs, cur = [], [pts[0]]
    for a, b in zip(pts, pts[1:]):
        if b[0] - a[0] > max_dt:
            segs.append(cur)
            cur = []
        cur.append(b)
    segs.append(cur)
    return [s for s in segs if len(s) >= 2]


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _panel(y0: float, title: str, bg_b64: str, w: int, h: int, tracks: dict, max_dt: int) -> list[str]:
    col = colour_tracks(tracks)
    out = [f'<text x="0" y="{y0 + 18}" class="t1">{_esc(title)}</text>',
           f'<image x="0" y="{y0 + 28}" width="{w}" height="{h}" href="data:image/png;base64,{bg_b64}"/>']
    g = y0 + 28
    order = sorted(tracks, key=lambda k: (col[k] != OTHER, k))  # gray first, coloured on top
    for k in order:
        c = col[k]
        for seg in _segments(tracks[k], max_dt):
            pts = " ".join(f"{x:.1f},{y + g:.1f}" for _, x, y in seg)
            if c != OTHER:
                out.append(f'<polyline points="{pts}" class="halo"/>')
            out.append(f'<polyline points="{pts}" stroke="{c}" class="{"tr" if c != OTHER else "ot"}"/>')
        if c != OTHER:
            t, x, y = tracks[k][-1]
            right = x > w - 30  # keep the label inside the panel
            lx, anchor = (x - 4, "end") if right else (x + 4, "start")
            ly = y + g - 4 if y > 14 else y + g + 14
            out.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" class="lab">{k[1]}</text>')
    for xs in (351.5, 703.5):
        out.append(f'<line x1="{xs}" y1="{g}" x2="{xs}" y2="{g + h}" class="seam"/>')
    out.append(f'<line x1="0" y1="{g + 240}" x2="{w}" y2="{g + 240}" class="seam"/>')
    return out


def day_svg(day: str, w: int, h: int, bg_pixels: bytes, bg_t: int, gt: dict, pred: dict, cadence: int,
            subtitle: str) -> str:
    """Two panels: clicks (MTrackJ) above, tracker below, on the lightened frame bg_t of the day."""
    b64 = base64.b64encode(png_gray_bytes(w, h, lighten(bg_pixels))).decode("ascii")
    max_dt = 2 * cadence
    ph = h + 40
    legend_y = 2 * ph + 24
    total_h = legend_y + 76
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -4 {w} {total_h + 4}" width="{w}" height="{total_h + 4}" '
             f'font-family="Segoe UI, Arial, sans-serif">',
             f'<style>.t1{{font-size:15px;font-weight:600;fill:{INK}}} .t2{{font-size:12px;fill:{INK2}}} '
             f'.halo{{fill:none;stroke:{SURFACE};stroke-width:4;stroke-linejoin:round;stroke-linecap:round}} '
             f'.tr{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}} '
             f'.ot{{fill:none;stroke-width:1.25;stroke-opacity:0.7;stroke-linejoin:round}} '
             f'.lab{{font-size:11px;fill:{INK};paint-order:stroke;stroke:{SURFACE};stroke-width:3}} '
             f'.seam{{stroke:{INK2};stroke-width:1;stroke-dasharray:3 3}}</style>',
             f'<rect x="0" y="-4" width="{w}" height="{total_h + 4}" fill="{SURFACE}"/>']
    parts += _panel(0, f"{day} · clicked tracks (MTrackJ, ground truth) · {len(gt)} tracks", b64, w, h, gt, max_dt)
    parts += _panel(ph, f"{day} · tracker T002 on E010 detections · {len(pred)} tracks", b64, w, h, pred, max_dt)
    lx = 0
    parts.append(f'<text x="0" y="{legend_y}" class="t2">Per camera (dashed seams), the 3 tracks with the largest extent '
                 f'are coloured and labelled with their id; all other tracks are gray. Gaps longer than '
                 f'{max_dt} frames are not drawn.</text>')
    for c, name in ((SLOTS[0], "largest extent"), (SLOTS[1], "2nd"), (SLOTS[2], "3rd"), (OTHER, "other tracks")):
        parts.append(f'<line x1="{lx}" y1="{legend_y + 20}" x2="{lx + 22}" y2="{legend_y + 20}" stroke="{c}" '
                     f'stroke-width="{2 if c != OTHER else 1.25}"/>')
        parts.append(f'<text x="{lx + 28}" y="{legend_y + 24}" class="t2">{name}</text>')
        lx += 28 + 8 * len(name) + 24
    parts.append(f'<text x="0" y="{legend_y + 46}" class="t2">{_esc(subtitle)}</text>')
    parts.append(f'<text x="0" y="{legend_y + 64}" class="t2">Background: frame t={bg_t}, lightened. '
                 f'Sparse-frame/timelapse, 15 s per frame; positions in px.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def preview_png(w: int, h: int, bg_pixels: bytes, gt: dict, pred: dict, cadence: int) -> bytes:
    """Raster check image (no text): clicks above, tracker below, same colours as the SVG."""
    H = 2 * h + 8
    img = bytearray(w * H * 3)
    bg = lighten(bg_pixels)
    for panel in (0, 1):
        for y in range(h):
            for x in range(w):
                v = bg[y * w + x]
                o = ((panel * (h + 8) + y) * w + x) * 3
                img[o:o + 3] = bytes((v, v, v))

    def rgb(hexs):
        return tuple(int(hexs[i:i + 2], 16) for i in (1, 3, 5))

    def put(x, y, c):
        if 0 <= x < w and 0 <= y < H:
            o = (y * w + x) * 3
            img[o:o + 3] = bytes(c)

    def line(x0, y0, x1, y1, c, thick):
        n = max(1, int(max(abs(x1 - x0), abs(y1 - y0))))
        for i in range(n + 1):
            x, y = x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n
            for dx in range(-thick, thick + 1):
                for dy in range(-thick, thick + 1):
                    put(int(round(x)) + dx, int(round(y)) + dy, c)

    for panel, tracks in ((0, gt), (1, pred)):
        col = colour_tracks(tracks)
        off = panel * (h + 8)
        for k in sorted(tracks, key=lambda k: (col[k] != OTHER, k)):
            c = rgb(col[k])
            for seg in _segments(tracks[k], 2 * cadence):
                for a, b in zip(seg, seg[1:]):
                    line(a[1], a[2] + off, b[1], b[2] + off, c, 1 if col[k] != OTHER else 0)
    raw = b"".join(b"\x00" + bytes(img[y * w * 3:(y + 1) * w * 3]) for y in range(H))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, H, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
