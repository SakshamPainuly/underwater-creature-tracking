"""Point-based detection scores for Tritonia: predicted boxes against the MTrackJ crosses (D-033).

The crosses were clicked by Dr. Wyeth's group; they are the ground truth. The training boxes come
from code (tritonia_prelabel) and are never used for scoring. Box-based metrics (mAP50, mAP50-95)
need hand-drawn boxes and are not computed here.

Matching (declared before any run, D-033): predictions sorted by confidence, highest first; each
takes the nearest still-unmatched cross whose distance to the box (0 if the cross is inside it) is
at most MATCH_PX. One prediction matches at most one cross and one cross at most one prediction.
- point precision = matched predictions / predictions
- point recall    = matched crosses / crosses
- point AP        = area under the precision-recall curve over all confidences (all-point
                    interpolation, as in VOC); a ranking summary, not mAP.
LIMITATION: whether every slug was clicked is UNKNOWN (D-029), so a prediction on an unclicked slug
counts as false: point precision is a lower bound.
"""

from __future__ import annotations

from pathlib import Path

MATCH_PX = 10.0
CONF_GRID = tuple(round(0.01 * i, 2) for i in range(1, 100))  # thresholds tried for the val F1 choice
SPLIT_SEED, N_VAL_DAYS, N_TEST_DAYS = 0, 3, 3  # D-030


def frames_in_splits(stems: list[str], splits: set[str]) -> list[str]:
    """Frames whose recording day is in one of `splits` under the D-030 day split of the batch."""
    from .tritonia_dataset import session_of, split_days
    split_of = split_days(sorted({session_of(s) for s in stems}), SPLIT_SEED, N_VAL_DAYS, N_TEST_DAYS)
    return [s for s in stems if split_of[session_of(s)] in splits]


def read_predictions(path: Path, width: int, height: int) -> list[tuple[float, tuple[float, ...]]]:
    """Ultralytics save_txt + save_conf lines `cls x y w h conf` (normalised) -> [(conf, px box)].

    A missing file means the model predicted nothing on that frame.
    """
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 6:
            raise ValueError(f"{path.name} line {n}: expected 'cls x y w h conf', got {line!r}")
        _, x, y, w, h, conf = (float(p) for p in parts)
        out.append((conf, ((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)))
    return out


def point_to_box(px: float, py: float, box: tuple[float, ...]) -> float:
    xa, ya, xb, yb = box
    dx = max(xa - px, 0.0, px - xb)
    dy = max(ya - py, 0.0, py - yb)
    return (dx * dx + dy * dy) ** 0.5


def match_details(preds: list[tuple[float, tuple[float, ...]]], crosses: list[tuple[float, float]],
                  match_px: float = MATCH_PX) -> list[tuple[float, tuple[float, ...], int | None]]:
    """[(conf, box, index of the matched cross or None)] per prediction, greedy by confidence
    (ties: input order). The matching rule of D-033; match_frame is this without the indices."""
    free = set(range(len(crosses)))
    result = []
    for conf, box in sorted(preds, key=lambda p: -p[0]):
        near = [(point_to_box(*crosses[i], box), i) for i in free]
        near = [d for d in near if d[0] <= match_px]
        hit = min(near)[1] if near else None
        if hit is not None:
            free.discard(hit)
        result.append((conf, box, hit))
    return result


def match_frame(preds: list[tuple[float, tuple[float, ...]]], crosses: list[tuple[float, float]],
                match_px: float = MATCH_PX) -> list[tuple[float, bool]]:
    """[(conf, matched)] per prediction, greedy by confidence (ties: input order)."""
    return [(conf, hit is not None) for conf, _, hit in match_details(preds, crosses, match_px)]


def scores(frames: dict[str, tuple[list, list]], conf: float, match_px: float = MATCH_PX) -> dict:
    """P, R, F1 at one confidence threshold over frames {stem: (preds, crosses)}."""
    tp = n_pred = n_cross = 0
    for preds, crosses in frames.values():
        kept = [p for p in preds if p[0] >= conf]
        m = match_frame(kept, crosses, match_px)
        tp += sum(ok for _, ok in m)
        n_pred += len(kept)
        n_cross += len(crosses)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_cross if n_cross else 0.0
    return {"conf": conf, "predictions": n_pred, "crosses": n_cross, "matched": tp,
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def pr_curve(frames: dict[str, tuple[list, list]], match_px: float = MATCH_PX) -> list[dict]:
    """Raw point precision-recall curve: one point per confidence-ranked prediction (matching done
    per frame, as for point_ap). Starts at recall 0, precision 1. Empty if there are no crosses."""
    ranked, n_cross = [], 0
    for preds, crosses in frames.values():
        ranked += match_frame(preds, crosses, match_px)
        n_cross += len(crosses)
    if not n_cross:
        return []
    ranked.sort(key=lambda m: -m[0])
    tp = fp = 0
    curve = [{"conf": None, "precision": 1.0, "recall": 0.0}]
    for conf, ok in ranked:
        tp += ok
        fp += not ok
        curve.append({"conf": conf, "precision": tp / (tp + fp), "recall": tp / n_cross})
    return curve


def point_ap(frames: dict[str, tuple[list, list]], match_px: float = MATCH_PX) -> float:
    """All-point interpolated AP of the confidence-ranked predictions (matching done per frame)."""
    curve = pr_curve(frames, match_px)
    if len(curve) < 2:
        return 0.0
    recall = [c["recall"] for c in curve]
    precision = [c["precision"] for c in curve]
    for i in range(len(precision) - 2, -1, -1):  # precision envelope
        precision[i] = max(precision[i], precision[i + 1])
    return round(sum((recall[i] - recall[i - 1]) * precision[i] for i in range(1, len(recall))), 4)


def conf_curves(frames: dict[str, tuple[list, list]], match_px: float = MATCH_PX) -> list[dict]:
    """scores() at every CONF_GRID threshold, lowest first: the P, R and F1 curves over confidence."""
    return [scores(frames, c, match_px) for c in CONF_GRID]


def frame_outcome(preds: list, crosses: list, conf: float, match_px: float = MATCH_PX) -> dict:
    """One frame at one threshold, for drawing: matched / false boxes and found / missed crosses."""
    kept = [p for p in preds if p[0] >= conf]
    details = match_details(kept, crosses, match_px)
    hit = {i for _, _, i in details if i is not None}
    return {"matched_boxes": [(c, b) for c, b, i in details if i is not None],
            "false_boxes": [(c, b) for c, b, i in details if i is None],
            "found_crosses": [crosses[i] for i in sorted(hit)],
            "missed_crosses": [p for i, p in enumerate(crosses) if i not in hit]}


def best_f1_conf(frames: dict[str, tuple[list, list]], match_px: float = MATCH_PX) -> dict:
    """The CONF_GRID threshold with the highest F1 (choose on val only: rule 7). Ties -> higher threshold."""
    best = None
    for c in sorted(CONF_GRID, reverse=True):
        s = scores(frames, c, match_px)
        if best is None or s["f1"] > best["f1"]:
            best = s
    return best or scores(frames, 1.0, match_px)
