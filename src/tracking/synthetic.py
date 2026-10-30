"""Synthetic tracks with known answers, for tests and pipeline demos (D-028). Standard library only.

No image or video is involved and nothing here describes lobster behaviour: the scenario exercises
each analytics case (constant motion, a pause, an occlusion gap, an ID switch, a brief false
positive) so the outputs can be checked against hand-computed values. Every output made from it is
marked synthetic in its meta and analytics.
"""

import random

from tracking.schema import META_SCHEMA

BOX_W, BOX_H = 160, 90


def _x_pause(f: int) -> float:
    # moves 3 px/frame, holds still for frames 100..200, then moves again
    return 300 + 3 * (min(f, 100) + max(0, f - 200))


# track_id -> (frames, centroid(f), object label, conf range)
def _scenario(n_frames: int) -> dict:
    last = n_frames - 1
    return {
        1: (range(0, n_frames), lambda f: (100 + 4 * f, 200), "A: constant 4 px/frame, straight", (0.6, 0.95)),
        2: (range(0, n_frames), lambda f: (_x_pause(f), 600), "B: 3 px/frame with a 100-frame pause",
            (0.6, 0.95)),
        3: ([f for f in range(0, n_frames) if not 120 <= f < 140], lambda f: (200 + 4 * f, 950 - 3 * f),
            "C: 5 px/frame diagonal, occluded frames 120-139", (0.6, 0.95)),
        4: (range(30, 38), lambda f: (1700 - 2 * f, 420 + f), "D: first 8 frames of object D", (0.6, 0.95)),
        5: (range(38, last + 1), lambda f: (1700 - 2 * f, 420 + f), "D again: ID switch from track 4 at frame 38",
            (0.6, 0.95)),
        6: (range(250, 253), lambda f: (960, 300), "E: 3-frame low-confidence blip (false-positive-like)",
            (0.3, 0.4)),
    }


def make_scenario(seed: int = 0, n_frames: int = 300, width: int = 1920, height: int = 1080,
                  jitter_px: float = 1.0, fps=None) -> tuple:
    """Return (meta, rows) for the fixed scenario. jitter_px = 0 gives exact, hand-checkable values."""
    if n_frames < 260:
        raise ValueError("the scenario needs n_frames >= 260")
    rng = random.Random(seed)
    video_id = f"synthetic_s{seed}"
    rows = []
    objects = {}
    for tid, (frames, path, label, (c_lo, c_hi)) in _scenario(n_frames).items():
        objects[str(tid)] = label
        for f in frames:
            cx, cy = path(f)
            if jitter_px:
                cx += rng.gauss(0, jitter_px)
                cy += rng.gauss(0, jitter_px)
            rows.append({
                "video_id": video_id, "frame_idx": f, "track_id": tid, "class_id": 0,
                "x1": round(max(cx - BOX_W / 2, 0), 2), "y1": round(max(cy - BOX_H / 2, 0), 2),
                "x2": round(min(cx + BOX_W / 2, width), 2), "y2": round(min(cy + BOX_H / 2, height), 2),
                "conf": round(rng.uniform(c_lo, c_hi), 4),
            })
    meta = {
        "schema": META_SCHEMA,
        "video_id": video_id,
        "synthetic": True,
        "frame_width": width,
        "frame_height": height,
        "n_frames": n_frames,
        "fps": fps,
        "source": "tracking.synthetic.make_scenario (no video)",
        "detector": {"name": "none: synthetic boxes"},
        "tracker": {"name": "none: scripted track IDs"},
        "scenario": {"seed": seed, "jitter_px": jitter_px, "tracks": objects},
    }
    return meta, rows
