import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_prep import build as B


def ultralytics_img2label_paths(img_paths, label_dir="labels", suffix=".txt"):
    """Verbatim copy of ultralytics 8.4.155 ultralytics/data/utils.py::img2label_paths."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}{label_dir}{os.sep}"
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + f"{suffix}" for x in img_paths]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


class YoloBuildTest(unittest.TestCase):
    """Builds a tiny fake dataset (fake PNG bytes; the builder never decodes images)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        t = Path(self._tmp.name)
        self.raw = t / "raw"
        imgs, lbls, extra = self.raw / "imgs", self.raw / "lbls", self.raw / "extra"
        for d in (imgs, lbls, extra):
            d.mkdir(parents=True)
        (imgs / "pos.png").write_bytes(b"png-pos")
        (imgs / "neg.png").write_bytes(b"png-neg")
        (lbls / "pos.txt").write_text("0 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n")
        (extra / "pending.png").write_bytes(b"png-pending")

        self._saved = (B.DATASET_SOURCES, B.BIOLOGY_ROOT)
        B.DATASET_SOURCES = [("Set2000", imgs, lbls), ("zExtra", extra, None)]
        B.BIOLOGY_ROOT = self.raw

        self.split = t / "split_manifest.csv"
        _write_csv(self.split, [{"session_id": "S1", "split": "train"}, {"session_id": "S9", "split": "excluded"}])
        self.manifest = t / "image_manifest.csv"
        self._rows = [
            {"filename": "pos.png", "source_set": "Set2000", "session_id": "S1", "split": "train",
             "label_status": "positive", "n_instances": 2, "issues": ""},
            {"filename": "neg.png", "source_set": "Set2000", "session_id": "S1", "split": "train",
             "label_status": "negative_confirmed", "n_instances": 0, "issues": ""},
            {"filename": "pending.png", "source_set": "zExtra", "session_id": "S9", "split": "excluded",
             "label_status": "negative_pending", "n_instances": 0, "issues": ""},
        ]
        _write_csv(self.manifest, self._rows)
        self.out = t / "yolo" / "lobster_test"

    def tearDown(self):
        B.DATASET_SOURCES, B.BIOLOGY_ROOT = self._saved
        self._tmp.cleanup()

    def build(self, link_mode="copy", out=None):
        return B.build_yolo_dataset(self.manifest, self.split, out or self.out, link_mode,
                                    version="vtest", version_status="frozen")

    def test_positive_negative_and_excluded_handling(self):
        summary = self.build()
        lbl = self.out / "labels" / "train"
        self.assertEqual((lbl / "pos.txt").read_text(), "0 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n")
        self.assertEqual((lbl / "neg.txt").read_text(), "")  # confirmed negative = empty label file
        self.assertFalse(any("pending" in p.name for p in self.out.rglob("*")))  # zExtra never included
        self.assertEqual(summary["splits"]["train"], {"images": 2, "positive": 1, "negative": 1, "instances": 2})
        self.assertIn("names:\n  0: lobster", (self.out / "data.yaml").read_text())

    def test_ultralytics_label_rule_resolves_into_built_labels(self):
        self.build()
        imgs = sorted(str(p) for p in (self.out / "images" / "train").glob("*.png"))
        for label in ultralytics_img2label_paths(imgs):
            self.assertTrue(Path(label).is_file(), label)

    def test_raw_layout_would_not_resolve_labels(self):
        # Why the derived tree exists: raw Set 2000 labels are not where Ultralytics looks.
        label = ultralytics_img2label_paths([str(self.raw / "imgs" / "pos.png")])[0]
        self.assertFalse(Path(label).is_file())

    def test_fingerprint_is_deterministic_and_recorded(self):
        first = self.build()["dataset_fingerprint"]
        second = self.build()["dataset_fingerprint"]
        self.assertEqual(first, second)

    def test_non_png_requires_copy_mode(self):
        (self.raw / "imgs" / "pos.png").rename(self.raw / "imgs" / "pos.jpg")
        self._rows[0]["filename"] = "pos.jpg"
        _write_csv(self.manifest, self._rows)
        with self.assertRaises(B.BuildError):
            self.build(link_mode="hardlink")
        with self.assertRaises(B.BuildError):
            self.build(link_mode="auto")

    def test_refuses_output_inside_raw_root(self):
        with self.assertRaises(B.BuildError):
            self.build(out=self.raw / "yolo")

    def test_refuses_to_delete_foreign_directory(self):
        self.out.mkdir(parents=True)
        (self.out / "someone_elses_file.txt").write_text("keep me")
        with self.assertRaises(B.BuildError):
            self.build()
        self.assertTrue((self.out / "someone_elses_file.txt").is_file())

    def test_split_mismatch_is_rejected(self):
        _write_csv(self.split, [{"session_id": "S1", "split": "val"}, {"session_id": "S9", "split": "excluded"}])
        with self.assertRaises(B.BuildError):
            self.build()


if __name__ == "__main__":
    unittest.main()
