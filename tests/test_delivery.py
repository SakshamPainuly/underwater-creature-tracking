import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit import delivery as D


class UnpackDeliveryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.t = Path(self._tmp.name)
        self.zip = self.t / "delivery.zip"
        with zipfile.ZipFile(self.zip, "w") as z:
            z.writestr("Enhanced images/images/train/a.png", b"a")
            z.writestr("Enhanced images/images/test/b.png", b"b")
            z.writestr("Enhanced images/images/.DS_Store", b"x")
            z.writestr("__MACOSX/Enhanced images/images/train/._a.png", b"x")
            z.writestr("Enhanced images/notes.txt", b"x")

    def tearDown(self):
        self._tmp.cleanup()

    def test_keeps_only_delivered_pngs_rerooted_at_images(self):
        out = self.t / "enhanced"
        self.assertEqual(D.unpack_delivery(self.zip, out), 2)
        files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        self.assertEqual(files, ["images/test/b.png", "images/train/a.png"])
        self.assertEqual((out / "images" / "train" / "a.png").read_bytes(), b"a")

    def test_refuses_non_empty_output(self):
        out = self.t / "enhanced"
        out.mkdir()
        (out / "old.png").write_bytes(b"x")
        with self.assertRaises(D.DeliveryError):
            D.unpack_delivery(self.zip, out)

    def test_duplicate_targets_are_refused(self):
        with self.assertRaises(D.DeliveryError):
            D.image_members(["x/images/train/a.png", "y/images/train/a.png"])


if __name__ == "__main__":
    unittest.main()
