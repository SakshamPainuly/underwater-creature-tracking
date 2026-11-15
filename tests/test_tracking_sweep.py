import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tracking import sweep as W

GRID = {"conf": [0.1, 0.16], "gate_px": [8, 12], "max_gap": [8], "min_len": [1, 3]}


def cfg(**over):
    c = {"id": "T009", "decision": "D-044", "split": "val", "sequence": "x", "sequence_manifest_sha256": "a",
         "gt_points_sha256": "b", "detections": {}, "grid": GRID, "match_px": 10, "selection": "declared"}
    c.update(over)
    return c


class SweepTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, c):
        p = self.tmp / name
        p.write_text(json.dumps(c), encoding="utf-8")
        return p

    def test_expand_grid_order_and_size(self):
        g = W.expand_grid(GRID)
        self.assertEqual(len(g), 8)
        self.assertEqual(g[0], {"conf": 0.1, "gate_px": 8, "max_gap": 8, "min_len": 1})
        self.assertEqual(g[-1], {"conf": 0.16, "gate_px": 12, "max_gap": 8, "min_len": 3})

    def test_config_checks(self):
        self.assertEqual(W.load_grid_config(self.write("T009_x.json", cfg()))["id"], "T009")
        bad = [("T008_x.json", cfg()),                                     # file name != id
               ("T009_x.json", cfg(split="test")),                         # val only
               ("T009_x.json", cfg(grid={**GRID, "conf": [0.16, 0.1]})),   # unsorted
               ("T009_x.json", cfg(grid={**GRID, "gate_px": [0, 8]})),     # invalid value
               ("T009_x.json", {k: v for k, v in cfg().items() if k != "selection"})]
        for name, c in bad:
            with self.assertRaises((W.SweepError, ValueError), msg=str(c)[:60]):
                W.load_grid_config(self.write(name, c))

    def test_selection_rule(self):
        p = lambda conf, gate, gap=8, ml=1: {"conf": conf, "gate_px": gate, "max_gap": gap, "min_len": ml}
        rs = [{"params": p(0.1, 8), "HOTA": 0.3000, "IDF1": 0.40},
              {"params": p(0.16, 8), "HOTA": 0.3005, "IDF1": 0.40},   # tied (within 0.001), same IDF1, higher conf
              {"params": p(0.1, 12), "HOTA": 0.2995, "IDF1": 0.45},   # tied, best IDF1 -> selected
              {"params": p(0.25, 20), "HOTA": 0.2980, "IDF1": 0.90}]  # not tied (0.0025 below)
        self.assertEqual(W.select(rs)["params"], p(0.1, 12))
        rs[2]["IDF1"] = 0.40
        self.assertEqual(W.select(rs)["params"], p(0.1, 8))           # then lower conf, smaller gate
        with self.assertRaises(W.SweepError):
            W.select([])

    def test_close_pair_frames(self):
        cam = {1: [(1, 10, 10), (2, 15, 10)], 2: [(1, 10, 10), (2, 40, 10)], 3: [(1, 10, 10)]}
        self.assertEqual(W.close_pair_frames(cam, 10), [1])

    def test_committed_t001_config_is_valid(self):
        c = W.load_grid_config(REPO / "configs" / "tracking" / "T001_tritonia_val_grid.json")
        self.assertEqual(len(W.expand_grid(c["grid"])), 81)
        self.assertEqual(c["detections"]["experiment"], "E010")


if __name__ == "__main__":
    unittest.main()
