import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit import versioning as V


def _write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


class VersioningTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "processed"
        self.root.mkdir()
        self.decisions = Path(self._tmp.name) / "DECISIONS.md"
        self.decisions.write_text("## D-005 Freeze v1\n## D-010 Correction\n", encoding="utf-8")
        _write_csv(self.root / "split_manifest.csv", ["session_id", "split"],
                   [{"session_id": "S1", "split": "train"}, {"session_id": "S2", "split": "test"}])
        _write_csv(self.root / "image_manifest.csv", ["filename", "label_status"],
                   [{"filename": "a.png", "label_status": "positive"},
                    {"filename": "b.png", "label_status": "negative_confirmed"}])
        files = {f: V.sha256_file(self.root / f) for f in ("split_manifest.csv", "image_manifest.csv")}
        registry = {"versions": {"v1": {"dir": ".", "parent": None, "decision": "D-005",
                                        "frozen_on": "2026-09-17", "files": files}}}
        (self.root / V.REGISTRY_NAME).write_text(json.dumps(registry), encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def draft(self, name="v1.1", decision="D-010"):
        return V.open_draft(name, "v1", decision, root=self.root, decisions_path=self.decisions)

    def test_frozen_files_and_registry_are_not_writable(self):
        for f in ("split_manifest.csv", "image_manifest.csv", V.REGISTRY_NAME):
            with self.assertRaises(V.VersionError):
                V.assert_writable(self.root / f, root=self.root)
        V.assert_writable(self.root / "unrelated.csv", root=self.root)  # not frozen -> allowed

    def test_verify_frozen_detects_tampering(self):
        V.verify_frozen("v1", root=self.root)
        with open(self.root / "split_manifest.csv", "a", encoding="utf-8") as f:
            f.write("S3,val\n")
        with self.assertRaises(V.VersionError):
            V.verify_frozen("v1", root=self.root)

    def test_frozen_version_cannot_be_opened_for_writing(self):
        with self.assertRaises(V.VersionError):
            V.open_draft("v1", None, None, root=self.root, decisions_path=self.decisions)

    def test_draft_requires_recorded_decision(self):
        with self.assertRaises(V.VersionError):
            self.draft(decision="D-999")
        self.assertFalse((self.root / "v1.1").exists())

    def test_draft_records_its_image_source(self):
        self.assertEqual(V.image_source_of("v1", root=self.root), "biology")
        self.draft()  # same images as the parent: VERSION.json keeps its v1-era shape
        self.assertNotIn("image_source", json.loads((self.root / "v1.1" / V.VERSION_FILE).read_text()))
        d = V.open_draft("v2", "v1", "D-010", root=self.root, decisions_path=self.decisions,
                         image_source="enhanced_2026-09")
        self.assertEqual(V.image_source_of("v2", root=self.root), "enhanced_2026-09")
        self.assertEqual((d / "image_manifest.csv").read_bytes(), (self.root / "image_manifest.csv").read_bytes())
        self.assertEqual(V.check_extends_parent("v2", root=self.root), [])
        with self.assertRaises(V.VersionError):  # an existing draft cannot switch image source
            V.open_draft("v2", "v1", "D-010", root=self.root, decisions_path=self.decisions, image_source="biology")
        with self.assertRaises(V.VersionError):
            V.open_draft("v3", "v1", "D-010", root=self.root, decisions_path=self.decisions, image_source="nope")

    def test_draft_starts_as_copy_of_parent_and_extends_it(self):
        d = self.draft()
        self.assertEqual((d / "split_manifest.csv").read_bytes(), (self.root / "split_manifest.csv").read_bytes())
        self.assertEqual(json.loads((d / V.VERSION_FILE).read_text())["status"], "draft")
        self.assertEqual(V.check_extends_parent("v1.1", root=self.root), [])

    def test_moving_a_parent_session_is_rejected_unless_approved(self):
        d = self.draft()
        _write_csv(d / "split_manifest.csv", ["session_id", "split"],
                   [{"session_id": "S1", "split": "val"}, {"session_id": "S2", "split": "test"},
                    {"session_id": "S3", "split": "train"}])
        self.assertTrue(any("S1" in p for p in V.check_extends_parent("v1.1", root=self.root)))
        meta = json.loads((d / V.VERSION_FILE).read_text())
        meta["approved_split_changes"] = ["S1"]
        (d / V.VERSION_FILE).write_text(json.dumps(meta))
        self.assertEqual(V.check_extends_parent("v1.1", root=self.root), [])

    def test_label_status_change_is_rejected(self):
        d = self.draft()
        _write_csv(d / "image_manifest.csv", ["filename", "label_status"],
                   [{"filename": "a.png", "label_status": "positive"},
                    {"filename": "b.png", "label_status": "positive"}])
        self.assertTrue(V.check_extends_parent("v1.1", root=self.root))
        with self.assertRaises(V.VersionError):
            V.freeze("v1.1", root=self.root, decisions_path=self.decisions)

    def test_freeze_then_immutable(self):
        d = self.draft()
        V.freeze("v1.1", root=self.root, decisions_path=self.decisions)
        V.verify_frozen("v1.1", root=self.root)
        with self.assertRaises(V.VersionError):
            V.assert_writable(d / "anything.csv", root=self.root)
        with self.assertRaises(V.VersionError):
            V.freeze("v1.1", root=self.root, decisions_path=self.decisions)

    def test_invalid_version_name(self):
        with self.assertRaises(V.VersionError):
            V.version_dir("latest", root=self.root)


if __name__ == "__main__":
    unittest.main()
