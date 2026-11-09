"""v4 mixed YOLO view (yolo_prep/mixed.py, D-031/D-032): build, split rules and training-side verification."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.model_a import FIELDNAMES
from dataset_audit.versioning import sha256_file
from training import experiment as X
from yolo_prep import mixed as M
from yolo_prep.build import BuildError, dataset_fingerprint

BOX = "0 0.5 0.5 0.1 0.1\n"


def row(image_id, origin, kind, split, path, label_path, n_boxes, status, exclude_reason=""):
    r = dict.fromkeys(FIELDNAMES, "")
    r.update(image_id=image_id, origin=origin, kind=kind, group_id=f"g_{split}", split=split, path=path,
             label_path=label_path, n_boxes=str(n_boxes), label_status=status, exclude_reason=exclude_reason)
    return r


class MixedViewTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        t = self.t = Path(self._tmp.name)
        self.roots = {"biology": t / "raw" / "biology", "harsh_model_a_2026-11": t / "raw" / "harsh"}
        files = {  # raw file -> bytes / text
            "biology/S/a.png": b"png-a", "biology/S/a.txt": BOX + BOX,
            "biology/S/b.png": b"png-b",                                   # Set 2000 negative: no label file
            "biology/S/v.png": b"png-v", "biology/S/v.txt": BOX,
            "biology/S/t.png": b"png-t", "biology/S/t.txt": BOX,
            "biology/S/x.png": b"png-x",                                   # excluded original
            "harsh_model_a_2026-11/D/a.png": b"diffusion-a", "harsh_model_a_2026-11/D/a.txt": BOX + BOX,
            "harsh_model_a_2026-11/Z/sk.jpg": b"jpg-sk", "harsh_model_a_2026-11/Z/sk.txt": BOX,
            "harsh_model_a_2026-11/Z/mpa.jpg": b"jpg-mpa", "harsh_model_a_2026-11/Z/mpa.txt": "",
        }
        for rel, content in files.items():
            head, _, rest = rel.partition("/")
            p = self.roots[head] / rest
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content) if isinstance(content, bytes) else p.write_text(content, encoding="utf-8")
        self.rows = [
            row("ours/original/a.png", "ours", "original", "train", "biology/S/a.png", "biology/S/a.txt", 2, "positive"),
            row("ours/original/b.png", "ours", "original", "train", "biology/S/b.png", "", 0, "negative_confirmed"),
            row("ours/diffusion/a.png", "ours", "diffusion", "train", "harsh_model_a_2026-11/D/a.png",
                "harsh_model_a_2026-11/D/a.txt", 2, "positive"),
            row("zenodo/original/mpa.jpg", "zenodo", "original", "train", "harsh_model_a_2026-11/Z/mpa.jpg",
                "harsh_model_a_2026-11/Z/mpa.txt", 0, "negative"),
            row("ours/original/v.png", "ours", "original", "val", "biology/S/v.png", "biology/S/v.txt", 1, "positive"),
            row("ours/original/t.png", "ours", "original", "test", "biology/S/t.png", "biology/S/t.txt", 1, "positive"),
            row("zenodo/original/sk.jpg", "zenodo", "original", "test", "harsh_model_a_2026-11/Z/sk.jpg",
                "harsh_model_a_2026-11/Z/sk.txt", 1, "positive"),
            row("ours/original/x.png", "ours", "original", "excluded", "biology/S/x.png", "", 0,
                "negative_confirmed", "label_review_excluded"),
        ]
        self.processed = t / "processed"
        self.vdir = self.processed / "v4"
        self.vdir.mkdir(parents=True)
        self.manifest = self.write_manifest(self.rows)
        (self.vdir / "VERSION.json").write_text(json.dumps({"name": "v4", "parent": "v1", "decision": "D-031",
                                                            "status": "frozen", "dataset_kind": "lobster_mixed"}),
                                                encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def write_manifest(self, rows):
        path = self.vdir / M.MANIFEST_FILE
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerows(rows)
        return path

    def freeze(self):  # register the fixture's files as a frozen v4 (what freeze() records)
        files = {p.name: sha256_file(p) for p in sorted(self.vdir.iterdir()) if p.is_file()}
        (self.processed / "FROZEN.json").write_text(json.dumps({"versions": {"v4": {
            "dir": "v4", "parent": "v1", "decision": "D-031", "frozen_on": "2026-11-08", "files": files}}}),
            encoding="utf-8")

    def build(self, out="view", status="frozen"):
        return M.build_mixed_dataset(self.manifest, self.t / out, "v4", status, roots=self.roots)

    def test_splits_names_labels_and_yaml(self):
        s = self.build()
        view = self.t / "view"
        self.assertEqual(s["splits"]["train"], {"images": 4, "positive": 2, "negative": 2, "instances": 4})
        self.assertEqual(s["splits"]["val"]["images"], 1)
        self.assertEqual(s["splits"]["test"], {"images": 1, "positive": 1, "negative": 0, "instances": 1})
        self.assertEqual(s["eval_only_splits"]["test_zenodo_sk"]["images"], 1)  # never in the v1 test
        self.assertEqual(s["not_included"], {"label_review_excluded": 1})
        # the diffusion copy keeps its source's file name: the prefix keeps both
        self.assertEqual(sorted(p.name for p in (view / "images" / "train").iterdir()),
                         ["ours__diffusion__a.png", "ours__original__a.png", "ours__original__b.png",
                          "zenodo__original__mpa.jpg"])
        self.assertEqual((view / "labels" / "train" / "ours__original__b.txt").read_text(), "")
        self.assertEqual((view / "labels" / "train" / "ours__diffusion__a.txt").read_text(), BOX + BOX)
        self.assertEqual((view / "images" / "test_zenodo_sk" / "zenodo__original__sk.jpg").read_bytes(), b"jpg-sk")
        self.assertFalse(list(view.rglob("*__x.*")))
        self.assertIn("test: images/test\n", (view / "data.yaml").read_text(encoding="utf-8"))
        self.assertIn("test: images/test_zenodo_sk\n",
                      (view / "data_test_zenodo_sk.yaml").read_text(encoding="utf-8"))
        # copies, never links (rule 2), and raw files unchanged
        self.assertFalse((view / "images" / "train" / "zenodo__original__mpa.jpg").is_symlink())
        self.assertEqual((self.roots["biology"] / "S" / "a.png").read_bytes(), b"png-a")
        # deterministic: a second build gives the same fingerprint
        self.assertEqual(self.build("view2")["dataset_fingerprint"], s["dataset_fingerprint"])
        self.assertEqual(s["dataset_fingerprint"], dataset_fingerprint(view))

    def test_training_verification(self):
        self.freeze()
        s = self.build()
        cfg = {"dataset_version": "v4", "dataset_fingerprint": s["dataset_fingerprint"],
               "ultralytics_version": "8.4.155"}
        facts = X.verify_built_dataset(self.t / "view", cfg, processed_root=self.processed)
        self.assertEqual(facts["dataset_kind"], "lobster_mixed")
        self.assertEqual(facts["splits_used_for_training"], ["train", "val"])
        self.assertEqual(facts["counts"]["train"]["images"], 4)

        img = self.t / "view" / "images" / "train" / "ours__original__a.png"
        img.write_bytes(b"changed")
        with self.assertRaisesRegex(X.ConfigError, "sha256"):
            X.verify_built_dataset(self.t / "view", cfg, processed_root=self.processed)
        img.write_bytes(b"png-a")
        (self.t / "view" / "labels" / "val" / "ours__original__v.txt").write_text(BOX + BOX)
        with self.assertRaisesRegex(X.ConfigError, "fingerprint"):
            X.verify_built_dataset(self.t / "view", cfg, processed_root=self.processed)

        self.build("draft_view", status="draft")
        cfg["dataset_fingerprint"] = dataset_fingerprint(self.t / "draft_view")
        with self.assertRaisesRegex(X.ConfigError, "frozen versions only"):
            X.verify_built_dataset(self.t / "draft_view", cfg, processed_root=self.processed)

    def test_refusals(self):
        bad = [dict(r) for r in self.rows]
        bad[4].update(kind="colour_change")  # a derived image in val
        self.write_manifest(bad)
        with self.assertRaisesRegex(BuildError, "unmodified originals only"):
            self.build()

        bad = [dict(r) for r in self.rows]
        bad[0].update(n_boxes="3")  # the label file holds 2 boxes
        self.write_manifest(bad)
        with self.assertRaisesRegex(BuildError, "2 boxes on disk, manifest says 3"):
            self.build()
        self.assertFalse((self.t / "view").exists())  # nothing written before the checks pass

        self.write_manifest(self.rows)
        with self.assertRaisesRegex(BuildError, "read-only raw folder"):
            M.build_mixed_dataset(self.manifest, self.roots["biology"] / "view", "v4", "frozen", roots=self.roots)

        (self.roots["biology"] / "S" / "v.txt").write_text("1 0.5 0.5 0.1 0.1\n")  # class 1 in a lobster label
        with self.assertRaisesRegex(BuildError, "class-0"):
            self.build()


if __name__ == "__main__":
    unittest.main()
