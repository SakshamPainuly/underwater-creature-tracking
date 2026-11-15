import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from dataset_audit import mtrackj as M
from test_tritonia_frames import gray_tiff, mdf
from tracking import detections as D
from tracking import sequences as S


class DetectionChecksTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        root = self.tmp / "raw"
        root.mkdir()
        gray_tiff(root / "20130524all_15spf.tif", [bytes(32), bytes(range(32))], 8, 4)
        text = mdf({1: [(1.0, 1.0, 1), (2.0, 1.0, 2)]})
        self.seq = self.tmp / "seq"
        summary = S.build_sequence(root, {"20130524": "20130524all_15spf"}, {"20130524": M.parse_mdf(text)}, self.seq)
        self.meta = {"split": "val", "split_version": "v8", **summary}
        (self.seq / "sequence.json").write_text(json.dumps(self.meta), encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_load_sequence_verifies_frames(self):
        meta, rows = D.load_sequence(self.seq)
        self.assertEqual(len(rows), 2)
        (self.seq / rows[1]["image"]).write_bytes(b"changed")
        with self.assertRaises(D.DetectionError):
            D.load_sequence(self.seq)
        D.load_sequence(self.seq, verify=False)  # manifest still consistent

    def test_test_sequence_needs_the_final_decision(self):
        (self.seq / "sequence.json").write_text(json.dumps({**self.meta, "split": "test"}), encoding="utf-8")
        with self.assertRaises(D.DetectionError):
            D.load_sequence(self.seq)
        D.load_sequence(self.seq, final="D-044")

    def test_weights_sha_and_prediction_summary(self):
        w = self.tmp / "best.pt"
        w.write_bytes(b"weights")
        good = D.sha256_file(w)
        self.assertEqual(D.check_weights(w, good), good)
        with self.assertRaises(D.DetectionError):
            D.check_weights(w, "0" * 64)
        p = self.tmp / "pred"
        p.mkdir()
        (p / "a.txt").write_text("1 0.5 0.5 0.02 0.03 0.40000\n1 0.1 0.1 0.02 0.03 0.01000\n", encoding="utf-8")
        (p / "b.txt").write_text("", encoding="utf-8")
        self.assertEqual(D.summarize_predictions(p, ["a", "b"], 0.16),
                         {"frames": 2, "frames_with_detections": 1, "detections": 1, "at_conf": 0.16})
        with self.assertRaises(D.DetectionError):
            D.summarize_predictions(p, ["a", "missing"], 0.16)

    def test_cli_checks_fail_before_any_gpu_import(self):
        w = self.tmp / "best.pt"
        w.write_bytes(b"weights")
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "predict_tritonia_sequence.py"), "--sequence",
                            str(self.seq), "--weights", str(w), "--weights-sha256", "0" * 64, "--out",
                            str(self.tmp / "out")], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("is not the registered", r.stderr)
        self.assertFalse((self.tmp / "out").exists())


if __name__ == "__main__":
    unittest.main()
