"""Tracking metrics for points (D-044): HOTA, CLEAR (MOTA, ID switches, MT/PT/ML) and Identity (IDF1).

Definitions follow TrackEval (Luiten et al., HOTA, IJCV 2021; Ristani et al., IDF1, 2016; Bernardin & Stiefelhagen,
CLEAR MOT, 2008) with boxes replaced by points:
- similarity of a prediction and a click at distance d: max(0, 1 - d / match_px) (match_px = 10, as D-033);
- HOTA: global alignment score, optimal per-frame matching on alignment x similarity, 19 alphas 0.05..0.95;
- CLEAR and Identity: a pair counts when d <= match_px.
Only the given evaluation frames are scored (clicked frames, D-044). A sequence is one camera of one day.
Inputs: frames (list of t), gt and pred as {t: [(id, x, y), ...]}. Results carry raw counts so sequences combine
exactly (sums; AssA and LocA weighted by TP), as TrackEval combines sequences.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .assign import linear_sum_assignment

ALPHAS = [round(0.05 * k, 2) for k in range(1, 20)]
EPS = 1e-10
MT_FRAC, ML_FRAC = 0.8, 0.2


def _sim(d: float, match_px: float) -> float:
    return max(0.0, 1.0 - d / match_px)


def _dist(a, b) -> float:
    return math.hypot(a[1] - b[1], a[2] - b[2])


def _max_assign(score: list[list[float]]) -> list[tuple[int, int]]:
    return linear_sum_assignment([[-s for s in row] for row in score])


def hota_counts(frames, gt, pred, match_px: float = 10.0) -> dict:
    gt_cnt, pr_cnt, pot = Counter(), Counter(), defaultdict(float)
    sims = {}
    for t in frames:
        g, p = gt.get(t, []), pred.get(t, [])
        gt_cnt.update(x[0] for x in g)
        pr_cnt.update(x[0] for x in p)
        if not g or not p:
            continue
        s = [[_sim(_dist(a, b), match_px) for b in p] for a in g]
        rows, cols = [sum(r) for r in s], [sum(c) for c in zip(*s)]
        for i, a in enumerate(g):
            for j, b in enumerate(p):
                if s[i][j] > 0:
                    pot[(a[0], b[0])] += s[i][j] / (rows[i] + cols[j] - s[i][j])
        sims[t] = s
    gas = {k: v / (gt_cnt[k[0]] + pr_cnt[k[1]] - v) for k, v in pot.items()}
    n = len(ALPHAS)
    tp, fn, fp, loc = [0] * n, [0] * n, [0] * n, [0.0] * n
    mc = [Counter() for _ in range(n)]
    for t in frames:
        g, p = gt.get(t, []), pred.get(t, [])
        if t not in sims:
            for a in range(n):
                fn[a] += len(g)
                fp[a] += len(p)
            continue
        s = sims[t]
        score = [[gas.get((g[i][0], p[j][0]), 0.0) * s[i][j] for j in range(len(p))] for i in range(len(g))]
        pairs = [(i, j) for i, j in _max_assign(score) if score[i][j] > 0]
        for a, alpha in enumerate(ALPHAS):
            ok = [(i, j) for i, j in pairs if s[i][j] >= alpha - EPS]
            tp[a] += len(ok)
            fn[a] += len(g) - len(ok)
            fp[a] += len(p) - len(ok)
            for i, j in ok:
                mc[a][(g[i][0], p[j][0])] += 1
                loc[a] += s[i][j]
    ass = [sum(c * c / (gt_cnt[k[0]] + pr_cnt[k[1]] - c) for k, c in mc[a].items()) for a in range(n)]
    return {"tp": tp, "fn": fn, "fp": fp, "ass_sum": ass, "loc_sum": loc}


def hota_from_counts(c: dict) -> dict:
    det, ass, loc, h = [], [], [], []
    for a in range(len(ALPHAS)):
        tp, fn, fp = c["tp"][a], c["fn"][a], c["fp"][a]
        d = tp / max(1, tp + fn + fp)
        s = c["ass_sum"][a] / max(1, tp)
        det.append(d)
        ass.append(s)
        loc.append(c["loc_sum"][a] / tp if tp else 0.0)
        h.append(math.sqrt(d * s))
    mean = lambda xs: sum(xs) / len(xs)
    return {"HOTA": mean(h), "DetA": mean(det), "AssA": mean(ass), "LocA": mean(loc),
            "HOTA_alpha": dict(zip((f"{a:.2f}" for a in ALPHAS), (round(x, 6) for x in h)))}


def clear_counts(frames, gt, pred, match_px: float = 10.0) -> dict:
    tp = fn = fp = idsw = 0
    loc = 0.0
    last_any, last_step = {}, {}  # gt id -> pred id matched last ever / at the previous evaluation frame
    gt_cnt, gt_hit = Counter(), Counter()
    for t in frames:
        g, p = gt.get(t, []), pred.get(t, [])
        gt_cnt.update(x[0] for x in g)
        step = {}
        if g and p:
            d = [[_dist(a, b) for b in p] for a in g]
            score = [[(1000.0 if last_step.get(g[i][0]) == p[j][0] else 0.0) + _sim(d[i][j], match_px) + EPS
                      if d[i][j] <= match_px else 0.0 for j in range(len(p))] for i in range(len(g))]
            for i, j in _max_assign(score):
                if score[i][j] <= 0:
                    continue
                gid, pid = g[i][0], p[j][0]
                if gid in last_any and last_any[gid] != pid:
                    idsw += 1
                last_any[gid] = step[gid] = pid
                gt_hit[gid] += 1
                tp += 1
                loc += _sim(d[i][j], match_px)
        m = len(step)
        fn += len(g) - m
        fp += len(p) - m
        last_step = step
    fracs = [gt_hit[k] / gt_cnt[k] for k in gt_cnt]
    return {"tp": tp, "fn": fn, "fp": fp, "idsw": idsw, "loc_sum": loc, "gt_tracks": len(gt_cnt),
            "mt": sum(f >= MT_FRAC for f in fracs), "ml": sum(f < ML_FRAC for f in fracs)}


def clear_from_counts(c: dict) -> dict:
    n_gt = c["tp"] + c["fn"]
    return {"MOTA": (c["tp"] - c["fp"] - c["idsw"]) / max(1, n_gt), "MOTP": c["loc_sum"] / c["tp"] if c["tp"] else 0.0,
            "IDSW": c["idsw"], "TP": c["tp"], "FN": c["fn"], "FP": c["fp"], "GT_tracks": c["gt_tracks"],
            "MT": c["mt"], "PT": c["gt_tracks"] - c["mt"] - c["ml"], "ML": c["ml"]}


def identity_counts(frames, gt, pred, match_px: float = 10.0) -> dict:
    pot = Counter()
    n_gt = n_pr = 0
    for t in frames:
        g, p = gt.get(t, []), pred.get(t, [])
        n_gt += len(g)
        n_pr += len(p)
        for a in g:
            for b in p:
                if _dist(a, b) <= match_px:
                    pot[(a[0], b[0])] += 1
    gids = sorted({k[0] for k in pot})
    pids = sorted({k[1] for k in pot})
    idtp = 0
    if gids:
        pairs = _max_assign([[pot.get((gi, pi), 0) for pi in pids] for gi in gids])
        idtp = sum(pot.get((gids[i], pids[j]), 0) for i, j in pairs)
    return {"idtp": idtp, "idfn": n_gt - idtp, "idfp": n_pr - idtp}


def identity_from_counts(c: dict) -> dict:
    tp, fn, fp = c["idtp"], c["idfn"], c["idfp"]
    return {"IDF1": 2 * tp / max(1, 2 * tp + fn + fp), "IDP": tp / max(1, tp + fp), "IDR": tp / max(1, tp + fn),
            "IDTP": tp, "IDFN": fn, "IDFP": fp}


def evaluate(frames, gt, pred, match_px: float = 10.0) -> dict:
    """Counts of one sequence (one camera of one day)."""
    return {"hota": hota_counts(frames, gt, pred, match_px), "clear": clear_counts(frames, gt, pred, match_px),
            "identity": identity_counts(frames, gt, pred, match_px)}


def combine(counts: list[dict]) -> dict:
    """Sum the raw counts of several sequences (lists elementwise)."""
    out = {}
    for block in ("hota", "clear", "identity"):
        acc = {}
        for c in counts:
            for k, v in c[block].items():
                if isinstance(v, list):
                    acc[k] = [x + y for x, y in zip(acc.get(k, [0] * len(v)), v)]
                else:
                    acc[k] = acc.get(k, 0) + v
        out[block] = acc
    return out


def scores(counts: dict) -> dict:
    return {**hota_from_counts(counts["hota"]), **clear_from_counts(counts["clear"]),
            **identity_from_counts(counts["identity"])}
