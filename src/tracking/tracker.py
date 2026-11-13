"""Point tracker for Tritonia sparse-frame/timelapse sequences (D-043 phase 3), standard library only.

Per camera, frame by frame: detections at or above `conf` are assigned to open tracks by minimum total distance
(optimal assignment) between each track's last position and the detection, only within `gate_px`. Unassigned
detections open new tracks; a track not extended for more than `max_gap` frames is closed. Tracks with fewer than
`min_len` detections are dropped. Constant-position model: Tritonia moves a median 2-4 px per click (D-043).
`no_link` is the floor reference: every detection is its own track. Deterministic for the same input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .assign import linear_sum_assignment

BIG = 1e9


@dataclass(frozen=True)
class Detection:
    t: int
    x: float
    y: float
    conf: float


@dataclass(frozen=True)
class TrackerParams:
    conf: float
    gate_px: float
    max_gap: int
    min_len: int

    def __post_init__(self):
        if not (0 <= self.conf <= 1 and self.gate_px > 0 and self.max_gap >= 1 and self.min_len >= 1):
            raise ValueError(f"invalid tracker settings {self}")


def track_camera(dets: list[Detection], params: TrackerParams) -> list[list[Detection]]:
    """Tracks of one camera of one day, each a time-ordered list of detections, ordered by (start t, x, y)."""
    by_t: dict[int, list[Detection]] = {}
    for d in dets:
        if d.conf >= params.conf:
            by_t.setdefault(d.t, []).append(d)
    open_tracks: list[list[Detection]] = []
    closed: list[list[Detection]] = []
    for t in sorted(by_t):
        still = []
        for tr in open_tracks:
            (still if t - tr[-1].t <= params.max_gap else closed).append(tr)
        open_tracks = still
        frame = sorted(by_t[t], key=lambda d: (d.x, d.y, -d.conf))
        taken = set()
        if open_tracks and frame:
            cost = [[math.hypot(tr[-1].x - d.x, tr[-1].y - d.y) for d in frame] for tr in open_tracks]
            gated = [[c if c <= params.gate_px else BIG for c in row] for row in cost]
            for i, j in linear_sum_assignment(gated):
                if cost[i][j] <= params.gate_px:
                    open_tracks[i].append(frame[j])
                    taken.add(j)
        open_tracks += [[d] for j, d in enumerate(frame) if j not in taken]
    tracks = [tr for tr in closed + open_tracks if len(tr) >= params.min_len]
    return sorted(tracks, key=lambda tr: (tr[0].t, tr[0].x, tr[0].y))


def no_link(dets: list[Detection], conf: float) -> list[list[Detection]]:
    """Floor reference: every detection at or above conf is a one-point track."""
    return [[d] for d in sorted((d for d in dets if d.conf >= conf), key=lambda d: (d.t, d.x, d.y))]
