import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
V1_PATH = PROCESSED / "raw_fingerprint.json"
POST_PATH = PROCESSED / "raw_fingerprint_d011_post_cleanup.json"
V1_SHA256 = "7c8711e293eae5cd878b3d593f56ebbf7af27fa1fa17c32108d1a7fdf208f0fe"  # as committed in a737b53
D011_REMOVED = {"Set 2000/batch_2 (2k - imgOnly).zip", "Set 2000/batch_2_2k_labels.zip"}

_spec = importlib.util.spec_from_file_location("check_raw_unchanged", ROOT / "scripts" / "check_raw_unchanged.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class RawFingerprintTest(unittest.TestCase):
    """Reads only the committed fingerprint JSONs, never the raw data (D-009, D-011)."""

    def test_v1_fingerprint_is_unchanged(self):
        self.assertEqual(hashlib.sha256(V1_PATH.read_bytes()).hexdigest(), V1_SHA256)

    def test_post_cleanup_is_v1_minus_exactly_the_two_zips(self):
        v1, post = _load(V1_PATH), _load(POST_PATH)
        self.assertEqual(set(v1["files"]) - set(post["files"]), D011_REMOVED)
        self.assertEqual(set(post["files"]) - set(v1["files"]), set())
        self.assertEqual(post["files"], {k: v for k, v in v1["files"].items() if k not in D011_REMOVED})
        self.assertEqual(post["kind"], v1["kind"])
        self.assertEqual(post["n_files"], len(post["files"]))
        self.assertEqual(post["n_files"], 4407)

    def test_default_fingerprint_is_post_cleanup(self):
        self.assertEqual(C.DEFAULT_FINGERPRINT_PATH.resolve(), POST_PATH.resolve())
        self.assertEqual(C.V1_FINGERPRINT_PATH.resolve(), V1_PATH.resolve())

    def test_write_refuses_v1_fingerprint_and_raw_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            self.assertIsNotNone(C.write_refusal(V1_PATH, [raw]))
            self.assertIsNotNone(C.write_refusal(raw, [raw]))
            self.assertIsNotNone(C.write_refusal(raw / "Set 100" / "fp.json", [raw]))
            self.assertIsNotNone(C.write_refusal(Path(tmp) / "other" / ".." / "raw" / "fp.json", [raw]))
            self.assertIsNone(C.write_refusal(Path(tmp) / "fp.json", [raw]))


if __name__ == "__main__":
    unittest.main()
