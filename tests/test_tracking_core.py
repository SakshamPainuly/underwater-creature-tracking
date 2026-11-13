import itertools
import random
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tracking import metrics as Mx
from tracking.assign import linear_sum_assignment
from tracking.tracker import Detection, TrackerParams, no_link, track_camera


def brute(cost):
    n, m = len(cost), len(cost[0])
    best = None
    if n <= m:
        for cols in itertools.permutations(range(m), n):
            s = sum(cost[i][c] for i, c in enumerate(cols))
            best = s if best is None or s < best else best
    else:
        for rows in itertools.permutations(range(n), m):
            s = sum(cost[r][j] for j, r in enumerate(rows))
            best = s if best is None or s < best else best
    return best


class AssignTest(unittest.TestCase):
    def test_matches_brute_force_on_random_rectangular_matrices(self):
        rng = random.Random(0)
        for _ in range(300):
            n, m = rng.randint(1, 5), rng.randint(1, 5)
            cost = [[rng.choice([rng.random() * 10, rng.randint(0, 3)]) for _ in range(m)] for _ in range(n)]
            pairs = linear_sum_assignment(cost)
            self.assertEqual(len(pairs), min(n, m))
            self.assertEqual(len({i for i, _ in pairs}), len(pairs))
            self.assertEqual(len({j for _, j in pairs}), len(pairs))
            self.assertAlmostEqual(sum(cost[i][j] for i, j in pairs), brute(cost), places=9)

    def test_empty(self):
        self.assertEqual(linear_sum_assignment([]), [])
        self.assertEqual(linear_sum_assignment([[]]), [])


def dets(points):
    return [Detection(t, x, y, c) for t, x, y, c in points]


class TrackerTest(unittest.TestCase):
    P = TrackerParams(conf=0.2, gate_px=10, max_gap=2, min_len=1)

    def test_two_slugs_kept_apart_and_low_conf_ignored(self):
        d = dets([(1, 10, 10, .9), (1, 50, 10, .9), (2, 12, 10, .9), (2, 48, 11, .9), (2, 30, 30, .1),
                  (3, 14, 10, .9), (3, 47, 12, .9)])
        tracks = track_camera(d, self.P)
        self.assertEqual([[(p.t, p.x) for p in tr] for tr in tracks],
                         [[(1, 10), (2, 12), (3, 14)], [(1, 50), (2, 48), (3, 47)]])

    def test_optimal_assignment_beats_greedy(self):
        # tracks at x=0 and x=4; detections at 3 and 7.5, gate 4. Greedy (nearest pair first) gives B->3 (1 px)
        # and leaves A unmatched (7.5 px > gate); the optimal assignment links both: A->3, B->7.5
        d = dets([(1, 0, 0, .9), (1, 4, 0, .9), (2, 3, 0, .9), (2, 7.5, 0, .9)])
        tracks = track_camera(d, TrackerParams(conf=0.2, gate_px=4, max_gap=2, min_len=1))
        self.assertEqual([[p.x for p in tr] for tr in tracks], [[0, 3], [4, 7.5]])

    def test_gate_gap_and_min_len(self):
        d = dets([(1, 10, 10, .9), (2, 30, 10, .9), (5, 10, 11, .9), (6, 10, 12, .9)])
        tracks = track_camera(d, self.P)  # 30 px jump -> new track; gap 1->5 = 4 > 2 -> new track
        self.assertEqual([[p.t for p in tr] for tr in tracks], [[1], [2], [5, 6]])
        self.assertEqual(len(track_camera(d, TrackerParams(0.2, 10, 2, 2))), 1)
        self.assertEqual([[p.t for p in tr] for tr in track_camera(d, TrackerParams(0.2, 10, 4, 1))],
                         [[1, 5, 6], [2]])

    def test_no_link_and_invalid_settings(self):
        self.assertEqual(len(no_link(dets([(1, 1, 1, .5), (2, 1, 1, .5), (2, 9, 9, .1)]), 0.2)), 2)
        with self.assertRaises(ValueError):
            TrackerParams(conf=0.2, gate_px=0, max_gap=2, min_len=1)


def seq(points):
    out = {}
    for t, i, x, y in points:
        out.setdefault(t, []).append((i, x, y))
    return out


class MetricsTest(unittest.TestCase):
    def test_perfect_tracking_scores_one(self):
        g = seq([(t, 1, 10 + t, 10) for t in range(1, 6)] + [(t, 2, 100, 50) for t in range(1, 6)])
        p = seq([(t, 7, 10 + t, 10) for t in range(1, 6)] + [(t, 8, 100, 50) for t in range(1, 6)])
        s = Mx.scores(Mx.evaluate(range(1, 6), g, p))
        for k in ("HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1"):
            self.assertAlmostEqual(s[k], 1.0, places=9, msg=k)
        self.assertEqual((s["IDSW"], s["MT"], s["ML"]), (0, 2, 0))

    def test_one_track_split_in_two(self):
        # hand-computed: DetA 1; AssA = (2*0.5 + 2*0.5)/4 = 0.5 -> HOTA sqrt(0.5); IDF1 = 4/8; MOTA = (4-1)/4
        g = seq([(t, 1, 10, 10) for t in range(1, 5)])
        p = seq([(1, 7, 10, 10), (2, 7, 10, 10), (3, 8, 10, 10), (4, 8, 10, 10)])
        s = Mx.scores(Mx.evaluate(range(1, 5), g, p))
        self.assertAlmostEqual(s["HOTA"], 0.5 ** 0.5, places=9)
        self.assertAlmostEqual(s["DetA"], 1.0)
        self.assertAlmostEqual(s["AssA"], 0.5)
        self.assertAlmostEqual(s["IDF1"], 0.5)
        self.assertAlmostEqual(s["MOTA"], 0.75)
        self.assertEqual(s["IDSW"], 1)

    def test_localisation_offset_drops_high_alphas(self):
        # 2 px off -> similarity 0.8 -> matched for alphas 0.05..0.80 (16 of 19)
        g = seq([(t, 1, 10, 10) for t in range(1, 4)])
        p = seq([(t, 5, 12, 10) for t in range(1, 4)])
        s = Mx.scores(Mx.evaluate(range(1, 4), g, p))
        self.assertAlmostEqual(s["HOTA"], 16 / 19, places=9)
        self.assertAlmostEqual(s["LocA"], 0.8 * 16 / 19, places=9)
        self.assertAlmostEqual(s["MOTA"], 1.0)
        self.assertAlmostEqual(s["IDF1"], 1.0)

    def test_misses_false_positives_and_unscored_frames(self):
        g = seq([(1, 1, 10, 10), (2, 1, 10, 10), (3, 1, 10, 10)])
        p = seq([(1, 5, 10, 10), (2, 6, 200, 200), (3, 5, 10, 10), (4, 9, 50, 50)])  # t=4 is not scored
        s = Mx.scores(Mx.evaluate([1, 2, 3], g, p))
        self.assertEqual((s["TP"], s["FN"], s["FP"], s["IDSW"]), (2, 1, 1, 0))
        self.assertAlmostEqual(s["MOTA"], (2 - 1) / 3)
        self.assertAlmostEqual(s["IDF1"], 4 / 6)
        self.assertAlmostEqual(s["DetA"], 2 / 4)

    def test_combine_equals_evaluating_the_union(self):
        g1, p1 = seq([(t, 1, 10, 10) for t in (1, 2, 3)]), seq([(1, 5, 10, 10), (2, 5, 10, 10), (3, 6, 10, 10)])
        g2, p2 = seq([(t, 2, 90, 90) for t in (1, 2)]), seq([(1, 7, 93, 90), (2, 7, 300, 300)])
        both = Mx.combine([Mx.evaluate([1, 2, 3], g1, p1), Mx.evaluate([1, 2, 3], g2, p2)])
        union_g = {t: g1.get(t, []) + g2.get(t, []) for t in (1, 2, 3)}
        union_p = {t: p1.get(t, []) + p2.get(t, []) for t in (1, 2, 3)}
        ref = Mx.evaluate([1, 2, 3], union_g, union_p)  # far apart: no cross matches, so identical counts
        self.assertEqual(Mx.scores(both), Mx.scores(ref))

    def test_empty_prediction(self):
        s = Mx.scores(Mx.evaluate([1, 2], seq([(1, 1, 5, 5), (2, 1, 5, 5)]), {}))
        self.assertEqual((s["HOTA"], s["MOTA"], s["IDF1"], s["FN"]), (0.0, 0.0, 0.0, 2))


if __name__ == "__main__":
    unittest.main()
