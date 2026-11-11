"""Tritonia dataset chain on a made-up batch in a temporary folder (no project data is read or written)."""

import csv
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dataset_audit import tritonia_dataset as T
from dataset_audit import versioning as V
from training import experiment as X
from yolo_prep.build import dataset_fingerprint

DAYS = ["20130519", "20130520", "20130521", "20130522", "20130523"]
W, H = 1056, 480
SLUG = "1 0.100000 0.200000 0.050000 0.100000\n"  # inside camera 1


def make_batch(d: Path) -> list[str]:
    """10 frames, 2 per day: images/<stem>.png (arbitrary bytes), manifest.csv, batch.json."""
    (d / "images").mkdir(parents=True)
    rows, stems = [], []
    for day in DAYS:
        for t in (100, 300):
            stem = f"{day}all_15spf_t{t:04d}"
            data = f"png {stem}".encode()
            (d / "images" / f"{stem}.png").write_bytes(data)
            rows.append({"image": f"images/{stem}.png", "source_tif": f"{day}all_15spf.tif", "t": t,
                         "page_index": t - 1, "n_points": 1, "tracks": "1", "png_sha256": T._sha(data)})
            stems.append(stem)
    with open(d / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    stacks = {f"{day}all_15spf": {"n_frames": 500, "width": W, "height": H, "tif_bytes": 1} for day in DAYS}
    (d / "batch.json").write_text(json.dumps({"seed": 0, "n": 10, "min_gap": 40, "classes": T.CLASS_NAMES,
                                              "totals": {}, "source_stacks": stacks}), encoding="utf-8")
    return stems


def make_zip(path: Path, labels: dict) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, text in labels.items():
            z.writestr(name, text)
    return path


class TritoniaChainTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.t = Path(self._tmp.name)
        self.batch = self.t / "batch"
        self.stems = make_batch(self.batch)
        self.notes = self.t / "notes.txt"
        self.notes.write_text(f"# comments are fine\n{self.stems[1]} | exclude | too dark to label\n"
                              f"{self.stems[2]}.png | note | cross without a visible slug\n",
                              encoding="utf-8-sig")  # with a BOM, as Windows Notepad may save it
        labels = {f"{s}.txt": SLUG for s in self.stems if s != self.stems[1]}
        labels[f"{self.stems[3]}.txt"] = ""  # labelled, no slug: confirmed negative
        labels[f"{self.stems[4]}.txt"] = SLUG + "1 0.600000 0.700000 0.040000 0.080000\n"
        self.zip = make_zip(self.t / "export.zip", labels)
        self.processed = self.t / "processed"
        self.processed.mkdir()
        self.decisions = self.t / "DECISIONS.md"
        self.decisions.write_text("## D-030 — test\n", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def ingest(self, zip_path=None, out="boxes"):
        return T.ingest_boxes(zip_path or self.zip, self.notes, self.batch, self.t / out)

    def test_full_chain_ingest_version_freeze_build_verify(self):
        report = self.ingest()
        self.assertEqual(report["counts"]["boxes"], 9)  # 7 single-box frames + 2 boxes, minus the negative
        self.assertEqual(report["counts"]["empty_label_files"], 1)
        self.assertEqual(list(report["excluded"]), [self.stems[1]])
        self.assertEqual(list(report["notes"]), [self.stems[2]])

        vdir = T.make_version("v3", "D-030", self.t / "boxes", self.batch, seed=0, n_val=1, n_test=1,
                              root=self.processed, decisions_path=self.decisions)
        with open(vdir / "image_manifest.csv", newline="", encoding="utf-8") as f:
            images = list(csv.DictReader(f))
        status = {r["filename"][:-4]: r["label_status"] for r in images}
        self.assertEqual(status[self.stems[1]], "excluded")
        self.assertEqual(status[self.stems[3]], "negative_confirmed")
        self.assertEqual(status[self.stems[0]], "positive")
        by_day = {}
        for r in images:
            by_day.setdefault(r["session_id"], set()).add(r["split"])
        self.assertTrue(all(len(s) == 1 for s in by_day.values()))  # a day never spans two splits
        self.assertEqual(sorted(s.pop() for s in by_day.values()), ["test", "train", "train", "train", "val"])

        with self.assertRaisesRegex(T.TritoniaError, "not frozen"):
            T.build_view("v3", self.batch / "images", self.t / "boxes", self.t / "view", "8.4.155", root=self.processed)
        V.freeze("v3", root=self.processed, decisions_path=self.decisions)

        summary = T.build_view("v3", self.batch / "images", self.t / "boxes", self.t / "view", "8.4.155",
                               root=self.processed)
        self.assertEqual(sum(c["images"] for c in summary["splits"].values()), 9)
        self.assertEqual(summary["dataset_fingerprint"], dataset_fingerprint(self.t / "view"))
        self.assertIn("  1: sea_slug", (self.t / "view" / "data.yaml").read_text(encoding="utf-8"))

        cfg = {"dataset_version": "v3", "dataset_fingerprint": summary["dataset_fingerprint"],
               "ultralytics_version": "8.4.155"}
        facts = X.verify_built_dataset(self.t / "view", cfg, processed_root=self.processed)
        self.assertEqual(facts["class_names"], ["lobster", "sea_slug"])
        self.assertEqual(set(facts["counts"]), {"train", "val"})

        (self.t / "view" / "labels" / "train" / next((self.t / "view" / "labels" / "train").iterdir()).name
         ).write_text(SLUG + SLUG, encoding="utf-8")
        with self.assertRaisesRegex(X.ConfigError, "fingerprint"):
            X.verify_built_dataset(self.t / "view", cfg, processed_root=self.processed)

    def test_refuses_bad_exports_and_writes_nothing(self):
        cases = {
            "crosses the camera border at x=352": {f"{self.stems[0]}.txt": "1 0.333333 0.2 0.05 0.1\n"},
            "crosses the camera border at y=240": {f"{self.stems[0]}.txt": "1 0.1 0.5 0.05 0.1\n"},
            "class 2": {f"{self.stems[0]}.txt": "2 0.1 0.2 0.05 0.1\n"},
            "outside the": {f"{self.stems[0]}.txt": "1 0.99 0.2 0.05 0.1\n"},
            "no label file": {},
            "no such frame": {"other_t0001.txt": SLUG},
            "expected 'class x y w h'": {f"{self.stems[0]}.txt": "1 0.1 0.2\n"},
        }
        for expected, override in cases.items():
            labels = {f"{s}.txt": SLUG for s in self.stems if s != self.stems[1]}
            if expected == "no label file":
                del labels[f"{self.stems[0]}.txt"]
            labels.update(override)
            z = make_zip(self.t / f"bad_{len(override)}_{abs(hash(expected))}.zip", labels)
            with self.subTest(expected), self.assertRaisesRegex(T.TritoniaError, expected):
                self.ingest(z, out="refused")
            self.assertFalse((self.t / "refused").exists())

    def test_notes_format_and_split_rule(self):
        with self.assertRaisesRegex(T.TritoniaError, "expected '<frame> \\| exclude\\|note"):
            T.parse_notes(f"{self.stems[0]} - too dark\n", set(self.stems))
        with self.assertRaisesRegex(T.TritoniaError, "not in the batch manifest"):
            T.parse_notes("nope | exclude | x\n", set(self.stems))
        a = T.split_days(DAYS, 0, 1, 1)
        self.assertEqual(a, T.split_days(list(reversed(DAYS)), 0, 1, 1))
        self.assertEqual(sorted(a.values()).count("train"), 3)
        with self.assertRaisesRegex(T.TritoniaError, "leave train"):
            T.split_days(DAYS, 0, 3, 2)

    def test_lobster_paths_unchanged(self):
        self.assertEqual(X.dataset_kind("v1"), "lobster")
        self.assertEqual(X.default_dataset_dir("v1"), REPO / "data" / "yolo" / "lobster_v1")
        out = X.write_trainval_yaml(self.t, self.t / "tv.yaml")
        self.assertTrue(out.read_text(encoding="utf-8").endswith("names:\n  0: lobster\n"))
        self.assertEqual(V.check_extends_parent("v2"), [])
        from training import run_record as R
        self.assertEqual(R.limitations_for({"version": "v1"}), list(R.V1_LIMITATIONS))
        self.assertEqual(R.limitations_for({"dataset_kind": "tritonia"}), list(R.TRITONIA_LIMITATIONS))


class ProjectV3SplitTest(unittest.TestCase):
    def test_d030_day_assignment(self):
        """The days D-030 lists are what the split rule gives for the 16 labelled days, seed 0."""
        days = ["20130519", "20130520", "20130521", "20130522", "20130523", "20130524", "20130525", "20130526",
                "20130527", "20130528", "20130529", "20130530", "20130531", "20130601", "20130602", "20130603"]
        split = T.split_days(days, 0, 3, 3)
        self.assertEqual(sorted(d for d, s in split.items() if s == "val"), D030_VAL)
        self.assertEqual(sorted(d for d, s in split.items() if s == "test"), D030_TEST)


D030_VAL = ["20130524", "20130529", "20130602"]
D030_TEST = ["20130520", "20130521", "20130528"]

if __name__ == "__main__":
    unittest.main()


class ExtensionTest(unittest.TestCase):
    def test_extension_problems(self):
        from dataset_audit.tritonia_dataset import extension_problems
        parent = [{"filename": "a.png", "split": "val", "n_instances": "2"}]
        self.assertEqual(extension_problems(parent, [{"filename": "a.png", "split": "val", "n_instances": 2},
                                                     {"filename": "b.png", "split": "train", "n_instances": 1}]), [])
        self.assertIn("missing", extension_problems(parent, [])[0])
        self.assertIn("['split']", extension_problems(parent, [{"filename": "a.png", "split": "train",
                                                                "n_instances": 2}])[0])
