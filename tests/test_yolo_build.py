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


def fake_png(width, height, payload=b""):
    """A PNG signature + IHDR header (all the builder reads) followed by arbitrary bytes."""
    ihdr = (13).to_bytes(4, "big") + b"IHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big")
    return B.PNG_SIGNATURE + ihdr + b"\x08\x02\x00\x00\x00" + payload


class ImageSourceBuildTest(unittest.TestCase):
    """D-017: images from a delivered image source, labels and split from the raw data + manifests."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        t = Path(self._tmp.name)
        self.raw = t / "raw"
        imgs, lbls = self.raw / "imgs", self.raw / "lbls"
        for d in (imgs, lbls):
            d.mkdir(parents=True)
        (imgs / "pos.png").write_bytes(fake_png(1920, 1080, b"raw-pos"))
        (imgs / "neg.png").write_bytes(fake_png(1920, 1088, b"raw-neg"))
        (lbls / "pos.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        # delivered in the sender's own folders, which disagree with our split (both are train here)
        self.enh = t / "enhanced"
        (self.enh / "images" / "test").mkdir(parents=True)
        (self.enh / "images" / "val").mkdir(parents=True)
        (self.enh / "images" / "test" / "pos.png").write_bytes(fake_png(1920, 1080, b"enh-pos"))
        (self.enh / "images" / "val" / "neg.png").write_bytes(fake_png(1920, 1088, b"enh-neg"))

        self._saved = (B.DATASET_SOURCES, B.BIOLOGY_ROOT, B.IMAGE_SOURCES)
        B.DATASET_SOURCES = [("Set2000", imgs, lbls)]
        B.BIOLOGY_ROOT = self.raw
        B.IMAGE_SOURCES = {"biology": self.raw, "enh": self.enh}

        self.split = t / "split_manifest.csv"
        _write_csv(self.split, [{"session_id": "S1", "split": "train"}])
        self.manifest = t / "image_manifest.csv"
        _write_csv(self.manifest, [
            {"filename": "pos.png", "source_set": "Set2000", "session_id": "S1", "split": "train",
             "label_status": "positive", "n_instances": 1, "issues": ""},
            {"filename": "neg.png", "source_set": "Set2000", "session_id": "S1", "split": "train",
             "label_status": "negative_confirmed", "n_instances": 0, "issues": ""},
        ])
        self.out = t / "yolo" / "lobster_test"

    def tearDown(self):
        B.DATASET_SOURCES, B.BIOLOGY_ROOT, B.IMAGE_SOURCES = self._saved
        self._tmp.cleanup()

    def build(self, image_source="enh", out=None):
        return B.build_yolo_dataset(self.manifest, self.split, out or self.out, "copy",
                                    version="vtest", version_status="frozen", image_source=image_source)

    def manifest_rows(self):
        with open(self.out / "build_manifest.csv", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_images_from_source_split_and_labels_from_manifests(self):
        summary = self.build()
        train = self.out / "images" / "train"
        self.assertTrue((train / "pos.png").read_bytes().endswith(b"enh-pos"))  # the sender's test/ folder is ignored
        self.assertTrue((train / "neg.png").read_bytes().endswith(b"enh-neg"))
        self.assertEqual((self.out / "labels" / "train" / "pos.txt").read_text(), "0 0.5 0.5 0.1 0.1\n")
        self.assertEqual(summary["image_source"], "enh")
        rows = {r["filename"]: r for r in self.manifest_rows()}
        self.assertEqual(rows["pos.png"]["image_source"], "enh")
        self.assertEqual(rows["pos.png"]["raw_image"], "images/test/pos.png")
        self.assertEqual(rows["pos.png"]["image_sha256"], B.sha256_file(self.enh / "images" / "test" / "pos.png"))

    def test_fingerprint_identifies_the_images(self):
        enhanced = self.build()["dataset_fingerprint"]
        raw = self.build(image_source="biology")["dataset_fingerprint"]
        self.assertNotEqual(enhanced, raw)
        (self.enh / "images" / "test" / "pos.png").write_bytes(fake_png(1920, 1080, b"enh-pos-v2"))
        self.assertNotEqual(self.build()["dataset_fingerprint"], enhanced)

    def test_raw_source_build_manifest_is_unchanged(self):
        summary = self.build(image_source="biology")
        self.assertEqual(list(self.manifest_rows()[0]),
                         ["split", "filename", "session_id", "label_status", "n_instances", "raw_image"])
        self.assertEqual(summary["image_source"], "biology")

    def test_size_change_is_refused(self):
        (self.enh / "images" / "test" / "pos.png").write_bytes(fake_png(1280, 720))
        with self.assertRaisesRegex(B.BuildError, "differs from the raw image"):
            self.build()

    def test_missing_duplicate_and_unknown_sources_are_refused(self):
        with self.assertRaisesRegex(B.BuildError, "unknown image source"):
            self.build(image_source="nope")
        (self.enh / "images" / "val" / "pos.png").write_bytes(fake_png(1920, 1080))
        with self.assertRaisesRegex(B.BuildError, "more than once"):
            self.build()
        (self.enh / "images" / "val" / "pos.png").unlink()
        (self.enh / "images" / "val" / "neg.png").unlink()
        with self.assertRaisesRegex(B.BuildError, "missing from image source"):
            self.build()
        self.assertFalse(self.out.exists())  # refused before anything was written

    def test_refuses_output_inside_image_source(self):
        with self.assertRaisesRegex(B.BuildError, "inside the raw data folder"):
            self.build(out=self.enh / "yolo")


if __name__ == "__main__":
    unittest.main()
