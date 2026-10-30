"""Trajectory and occupancy plots as standalone SVG (D-028). Standard library only, deterministic.

Open the .svg files in any browser. Gaps in a track are drawn dashed (the position there is
interpolated, not observed).
"""

from xml.sax.saxutils import escape

from tracking.analytics import centroid, group_by_track

PLOT_WIDTH = 960
COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
          "#7f7f7f", "#bcbd22")


def _header(w: float, h: float, title: str) -> list:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
            f'viewBox="0 0 {w:.0f} {h:.0f}" font-family="sans-serif" font-size="12">',
            f"<title>{escape(title)}</title>",
            f'<rect x="0" y="0" width="{w:.0f}" height="{h:.0f}" fill="#ffffff"/>']


def _caption(meta: dict, plot_h: float, text: str) -> str:
    tag = "SYNTHETIC — " if meta["synthetic"] else ""
    return (f'<text x="8" y="{plot_h + 20:.0f}" fill="#333333">{escape(tag + meta["video_id"] + " — " + text)}'
            f"</text>")


def trajectories_svg(meta: dict, rows: list) -> str:
    s = PLOT_WIDTH / meta["frame_width"]
    w, h = PLOT_WIDTH, meta["frame_height"] * s
    out = _header(w, h + 30, f"Trajectories {meta['video_id']}")
    out.append(f'<rect x="0" y="0" width="{w:.0f}" height="{h:.1f}" fill="#f4f6f8" stroke="#999999"/>')
    for n, (tid, obs) in enumerate(group_by_track(rows).items()):
        color = COLORS[n % len(COLORS)]
        pts = [(r["frame_idx"], *centroid(r)) for r in obs]
        segment = [pts[0]]
        for prev, cur in zip(pts, pts[1:]):
            if cur[0] - prev[0] > 1:  # gap: close the solid run, draw the jump dashed
                out.append(_polyline(segment, s, color))
                out.append(_polyline([prev, cur], s, color, dashed=True))
                segment = [cur]
            else:
                segment.append(cur)
        out.append(_polyline(segment, s, color))
        _, x0, y0 = pts[0]
        _, x1, y1 = pts[-1]
        out.append(f'<circle cx="{x0 * s:.1f}" cy="{y0 * s:.1f}" r="3" fill="{color}"/>')
        out.append(f'<text x="{x1 * s + 4:.1f}" y="{y1 * s - 4:.1f}" fill="{color}">ID {tid}</text>')
    out.append(_caption(meta, h, "track centroids in px (dot = first frame, dashed = gap)"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _polyline(pts: list, s: float, color: str, dashed: bool = False) -> str:
    coords = " ".join(f"{x * s:.1f},{y * s:.1f}" for _, x, y in pts)
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.5"{dash}/>'


def occupancy_svg(meta: dict, grid: list) -> str:
    ny, nx = len(grid), len(grid[0])
    s = PLOT_WIDTH / meta["frame_width"]
    w, h = PLOT_WIDTH, meta["frame_height"] * s
    cw, ch = w / nx, h / ny
    peak = max((c for row in grid for c in row), default=0)
    out = _header(w, h + 30, f"Occupancy {meta['video_id']}")
    for j, row in enumerate(grid):
        for i, c in enumerate(row):
            opacity = c / peak if peak else 0
            out.append(f'<rect x="{i * cw:.1f}" y="{j * ch:.1f}" width="{cw:.1f}" height="{ch:.1f}" '
                       f'fill="#08519c" fill-opacity="{opacity:.3f}" stroke="#dddddd"><title>{c}</title></rect>')
    out.append(_caption(meta, h, f"box centroids per cell ({nx}x{ny} grid, darkest = {peak})"))
    out.append("</svg>")
    return "\n".join(out) + "\n"
