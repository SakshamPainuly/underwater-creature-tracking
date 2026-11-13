"""Declared tracker-setting grid on val and its selection rule (D-043 phase 4), standard library only.

A grid config (configs/tracking/T###_*.json) lists the values of the 4 tracker settings; every combination is run
on the same val detections. Selection (declared before the run, docs/EXPERIMENTS.md): highest mean-over-days HOTA;
results within TIE of the best are tied and broken by higher mean IDF1, then lower conf, smaller gate, smaller
maximum gap, smaller minimum length (the simpler / less filtered setting).
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

from .tracker import TrackerParams

KEYS = ("conf", "gate_px", "max_gap", "min_len")
TIE = 0.001
REQUIRED = ("id", "decision", "split", "sequence", "sequence_manifest_sha256", "gt_points_sha256", "detections",
            "grid", "match_px", "selection")


class SweepError(ValueError):
    pass


def load_grid_config(path: Path) -> dict:
    path = Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise SweepError(f"{path.name}: missing {missing}")
    if not path.name.startswith(cfg["id"] + "_"):
        raise SweepError(f"{path.name}: file name must start with its id {cfg['id']}_")
    if cfg["split"] != "val":
        raise SweepError(f"{path.name}: a grid runs on val only (rule 7)")
    if sorted(cfg["grid"]) != sorted(KEYS):
        raise SweepError(f"{path.name}: grid keys must be {KEYS}")
    for k in KEYS:
        v = cfg["grid"][k]
        if not v or v != sorted(set(v)):
            raise SweepError(f"{path.name}: grid {k} must be a non-empty, sorted list without repeats")
    for p in expand_grid(cfg["grid"]):
        TrackerParams(**p)  # raises on an invalid value
    return cfg


def expand_grid(grid: dict) -> list[dict]:
    """All combinations, in a fixed order (conf, gate, gap, min_len ascending)."""
    return [dict(zip(KEYS, combo)) for combo in itertools.product(*(grid[k] for k in KEYS))]


def select(results: list[dict]) -> dict:
    """results: [{"params": {...}, "HOTA": mean over days, "IDF1": mean over days, ...}] -> the selected one."""
    if not results:
        raise SweepError("no results")
    best = max(r["HOTA"] for r in results)
    tied = [r for r in results if r["HOTA"] >= best - TIE]
    return min(tied, key=lambda r: (-r["IDF1"], *(r["params"][k] for k in KEYS)))


def close_pair_frames(gt_cam: dict, match_px: float) -> list[int]:
    """Frames of one camera where two clicked slugs are within match_px of each other (the ambiguous case)."""
    out = []
    for t, pts in gt_cam.items():
        if any(math.hypot(a[1] - b[1], a[2] - b[2]) <= match_px
               for i, a in enumerate(pts) for b in pts[i + 1:]):
            out.append(t)
    return sorted(out)
