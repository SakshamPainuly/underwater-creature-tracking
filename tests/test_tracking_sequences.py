import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from dataset_audit import mtrackj as M
from test_tritonia_frames import gray_tiff, mdf, png_pixels
from tracking import sequences as S


class CameraTest(unittest.TestCase):
    def test_camera_ids_follow_the_observed_seams(self):
        self.assertEqual(S.camera_of(0, 0), 0)
        self.assertEqual(S.camera_of(351.4, 239.9), 0)
        self.assertEqual(S.camera_of(351.6, 10), 1)
        self.assertEqual(S.camera_of(703.6, 10), 2)
        self.assertEqual(S.camera_of(10, 240.0), 3)
        self.assertEqual(S.camera_of(1055, 479), 5)

    def test_track_camera_is_the_majority_with_a_deterministic_tie(self):
        m = M.parse_mdf(mdf({1: [(10, 10, 1), (10, 10, 2), (360, 10, 3)], 2: [(10, 10, 1), (360, 10, 2)]}))
        self.assertEqual(S.track_camera(m.tracks[0]), 0)
        self.assertEqual(S.track_camera(m.tracks[1]), 0)  # tie 1:1 -> lower id

    def test_click_cadence_is_the_most_common_step(self):
        self.assertEqual(S.click_cadence(M.parse_mdf(mdf({1: [(1, 1, 1), (1, 1, 3), (1, 1, 5), (1, 1, 6)]}))), 2)
        self.assertEqual(S.click_cadence(M.parse_mdf(mdf({1: [(1, 1, 1), (1, 1, 2)], 2: [(1, 1, 4), (1, 1, 6)]}))), 1)

    def test_gt_rows_sorted_and_reject_two_points_in_one_frame(self):
        rows = S.gt_rows("20130524", M.parse_mdf(mdf({2: [(500, 300, 2)], 1: [(10, 10, 2), (11, 10, 1)]})))
        self.assertEqual([(r["t"], r["camera"], r["track_id"]) for r in rows], [(1, 0, 1), (2, 0, 1), (2, 4, 2)])
        with self.assertRaises(S.SequenceError):
            S.gt_rows("d", M.parse_mdf(mdf({1: [(1, 1, 1), (2, 2, 1)]})))


class BuildTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "raw"
        self.root.mkdir()
        self.pages = [bytes([i * 40 + k % 7 for k in range(8 * 4)]) for i in range(3)]
        gray_tiff(self.root / "20130524all_15spf.tif", self.pages, 8, 4)
        text = mdf({1: [(1.0, 1.0, 1), (2.0, 1.0, 3)], 2: [(6.0, 3.0, 2)]})
        (self.root / "20130524all_15spf.tif.mdf").write_text(text, encoding="utf-8")
        self.mdfs = {"20130524": M.parse_mdf(text)}

    def tearDown(self):
        self._tmp.cleanup()

    def test_writes_every_page_with_manifest_and_ground_truth(self):
        out = self.tmp / "seq"
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        s = S.build_sequence(self.root, {"20130524": "20130524all_15spf"}, self.mdfs, out)
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir()}, before)  # inputs untouched
        for i, px in enumerate(self.pages, 1):
            self.assertEqual(png_pixels(out / "images" / f"20130524all_15spf_t{i:04d}.png"), (8, 4, px))
        man = list(csv.DictReader((out / "manifest.csv").read_text(encoding="utf-8").splitlines()))
        self.assertEqual([(r["t"], r["n_gt_points"]) for r in man], [("1", "1"), ("2", "1"), ("3", "1")])
        gt = list(csv.DictReader((out / "gt_points.csv").read_text(encoding="utf-8").splitlines()))
        self.assertEqual([(r["t"], r["track_id"]) for r in gt], [("1", "1"), ("2", "2"), ("3", "1")])
        self.assertEqual(s["totals"], {"frames": 3, "gt_points": 3, "tracks": 2})
        self.assertEqual(s["days"]["20130524"]["click_cadence"], 2)

    def test_refuses_output_inside_source_or_existing(self):
        with self.assertRaises(S.SequenceError):
            S.build_sequence(self.root, {"20130524": "20130524all_15spf"}, self.mdfs, self.root / "seq")
        (self.tmp / "exists").mkdir()
        with self.assertRaises(S.SequenceError):
            S.build_sequence(self.root, {"20130524": "20130524all_15spf"}, self.mdfs, self.tmp / "exists")

    def test_refuses_points_beyond_the_stack(self):
        m = {"20130524": M.parse_mdf(mdf({1: [(1, 1, 1), (1, 1, 9)]}))}
        with self.assertRaises(S.SequenceError):
            S.build_sequence(self.root, {"20130524": "20130524all_15spf"}, m, self.tmp / "seq")

    def test_cli_refuses_test_split_without_final(self):
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "make_tritonia_sequence.py"), "--root",
                            str(self.root), "--split", "test", "--split-version", "v8", "--out", str(self.tmp / "t")],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("REFUSED", r.stderr)
        self.assertFalse((self.tmp / "t").exists())


if __name__ == "__main__":
    unittest.main()
