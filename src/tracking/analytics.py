"""Per-track and per-video trajectory analytics from a tracks file (D-028). Standard library only.

Units: pixels and frames. Seconds appear only when the tracks meta has a known fps; there is no
calibration, so no physical units (CLAUDE.md rule 9). These are descriptive statistics of the
tracker's output, not tracking accuracy: IDF1/HOTA/MOTA need ground-truth track IDs.

Definitions (box centroid = ((x1 + x2) / 2, (y1 + y2) / 2)):
  - span_frames = last_frame - first_frame + 1; missing_frames = span_frames - observed_frames;
    a gap is a run of missing frames between two observations of the same track.
  - path_px = sum of centroid distances between consecutive observations. Across a gap the
    straight line is used, so path_px is a lower bound on the true path there.
  - mean_speed_px_per_frame = path_px / (last_frame - first_frame); None for a one-frame track.
  - max_step_px_per_frame = max over consecutive observations of distance / frames elapsed.
  - straightness = net_displacement_px / path_px (1 = straight line); None when path_px is 0.
  - stationary_frames = frames elapsed in steps slower than `stationary_px_per_frame`
    (a dwell proxy; its threshold is a parameter, not a biological claim).
  - short tracks (< `min_track_frames` observations) are a fragmentation proxy: an ID switch or a
    brief false positive both produce them.
"""

import math
import statistics

DEFAULT_MIN_TRACK_FRAMES = 10
DEFAULT_STATIONARY_PX_PER_FRAME = 1.0


def centroid(row: dict) -> tuple:
    return ((row["x1"] + row["x2"]) / 2, (row["y1"] + row["y2"]) / 2)


def group_by_track(rows: list) -> dict:
    """track_id -> its rows sorted by frame (track ids in ascending order)."""
    tracks = {}
    for r in rows:
        tracks.setdefault(r["track_id"], []).append(r)
    return {tid: sorted(tracks[tid], key=lambda r: r["frame_idx"]) for tid in sorted(tracks)}


def _round(v, digits=3):
    return None if v is None else round(v, digits)


def track_summary(obs: list, fps=None, stationary_px_per_frame: float = DEFAULT_STATIONARY_PX_PER_FRAME) -> dict:
    """Statistics of one track's observations (sorted by frame)."""
    frames = [r["frame_idx"] for r in obs]
    points = [centroid(r) for r in obs]
    first, last = frames[0], frames[-1]
    elapsed = last - first
    path = 0.0
    max_step = None
    gaps = []
    stationary = 0
    for (f0, p0), (f1, p1) in zip(zip(frames, points), zip(frames[1:], points[1:])):
        d = math.dist(p0, p1)
        df = f1 - f0
        path += d
        step = d / df
        max_step = step if max_step is None else max(max_step, step)
        if df > 1:
            gaps.append(df - 1)
        if step < stationary_px_per_frame:
            stationary += df
    net = math.dist(points[0], points[-1])
    speed = path / elapsed if elapsed else None
    out = {
        "track_id": obs[0]["track_id"],
        "class_id": statistics.mode(r["class_id"] for r in obs),
        "first_frame": first,
        "last_frame": last,
        "observed_frames": len(obs),
        "span_frames": elapsed + 1,
        "missing_frames": elapsed + 1 - len(obs),
        "n_gaps": len(gaps),
        "max_gap_frames": max(gaps) if gaps else 0,
        "path_px": _round(path),
        "net_displacement_px": _round(net),
        "mean_speed_px_per_frame": _round(speed),
        "max_step_px_per_frame": _round(max_step),
        "straightness": _round(net / path if path > 0 else None),
        "stationary_frames": stationary,
        "mean_box_w_px": _round(statistics.fmean(r["x2"] - r["x1"] for r in obs)),
        "mean_box_h_px": _round(statistics.fmean(r["y2"] - r["y1"] for r in obs)),
        "mean_conf": _round(statistics.fmean(r["conf"] for r in obs), 4),
        "duration_s": None,
        "mean_speed_px_per_s": None,
    }
    if fps:
        out["duration_s"] = _round(elapsed / fps)
        out["mean_speed_px_per_s"] = _round(speed * fps if speed is not None else None)
    return out


def boxes_per_frame(rows: list, n_frames: int) -> list:
    counts = [0] * n_frames
    for r in rows:
        counts[r["frame_idx"]] += 1
    return counts


def occupancy_grid(rows: list, width: int, height: int, nx: int = 16, ny: int = 9) -> list:
    """ny x nx counts of box centroids per cell (one count per box per frame)."""
    grid = [[0] * nx for _ in range(ny)]
    for r in rows:
        cx, cy = centroid(r)
        i = min(max(int(cx / width * nx), 0), nx - 1)
        j = min(max(int(cy / height * ny), 0), ny - 1)
        grid[j][i] += 1
    return grid


def summarize(meta: dict, rows: list, min_track_frames: int = DEFAULT_MIN_TRACK_FRAMES,
              stationary_px_per_frame: float = DEFAULT_STATIONARY_PX_PER_FRAME) -> dict:
    """Video-level analytics: overall counts plus one summary per track."""
    fps = meta.get("fps")
    per_track = [track_summary(obs, fps, stationary_px_per_frame) for obs in group_by_track(rows).values()]
    counts = boxes_per_frame(rows, meta["n_frames"])
    lengths = [t["observed_frames"] for t in per_track]
    overall = {
        "n_frames": meta["n_frames"],
        "n_boxes": len(rows),
        "n_tracks": len(per_track),
        "frames_with_boxes": sum(1 for c in counts if c),
        "max_boxes_in_a_frame": max(counts) if counts else 0,
        "median_track_observed_frames": statistics.median(lengths) if lengths else None,
        "short_tracks": sum(1 for n in lengths if n < min_track_frames),
        "tracks_with_gaps": sum(1 for t in per_track if t["n_gaps"]),
    }
    return {
        "video_id": meta["video_id"],
        "synthetic": meta["synthetic"],
        "fps": fps,
        "units": "px and frames" + ("; seconds from fps" if fps else "; fps unknown, no seconds"),
        "parameters": {"min_track_frames": min_track_frames, "stationary_px_per_frame": stationary_px_per_frame},
        "overall": overall,
        "per_track": per_track,
        "boxes_per_frame": counts,
        "limitations": [
            "Descriptive statistics of the tracker output, not tracking accuracy: no ground-truth track IDs, "
            "so no IDF1/HOTA/MOTA.",
            "No calibration: distances and speeds are in pixels, not physical units.",
            "Paths across gaps use straight lines (lower bound).",
            *(["SYNTHETIC data: these numbers describe a generated scenario, not lobster behaviour."]
              if meta["synthetic"] else []),
        ],
    }


SUMMARY_COLUMNS = ("track_id", "class_id", "first_frame", "last_frame", "observed_frames", "span_frames",
                   "missing_frames", "n_gaps", "max_gap_frames", "path_px", "net_displacement_px",
                   "mean_speed_px_per_frame", "max_step_px_per_frame", "straightness", "stationary_frames",
                   "mean_box_w_px", "mean_box_h_px", "mean_conf", "duration_s", "mean_speed_px_per_s")
