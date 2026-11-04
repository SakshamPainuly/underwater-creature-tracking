"""Logic of scripts/label_tritonia.py (no window): boxes, saving, notes, and an export the ingest accepts."""

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_audit import tritonia_dataset as T
from dataset_audit import tritonia_labeller as L
from test_tritonia_dataset import H, W, make_batch


class LabellerLogicTest(unittest.TestCase):
    def test_pixel_box_round_trip_and_class(self):
        box = L.normalise_px(150, 90, 100, 40, W, H)  # dragged up-left
        self.assertEqual(box, (100, 40, 150, 90))
        cls, x, y, w, h = L.px_to_yolo(box, W, H)
        self.assertEqual(cls, 1)
        back = L.yolo_to_px((cls, x, y, w, h), W, H)
        self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(back, box)))
        self.assertEqual(L.normalise_px(-20, 470, 30, 900, W, H), (0, 470, 30, H))  # clipped to the frame

    def test_new_box_checks_match_the_ingest(self):
        self.assertEqual(L.check_new_box((100, 40, 150, 90), W, H), [])
        self.assertIn("crosses the camera border at x=352", L.check_new_box((330, 40, 380, 90), W, H)[0])
        self.assertIn("crosses the camera border at y=240", L.check_new_box((100, 220, 150, 260), W, H)[0])
        self.assertEqual(L.check_new_box((340, 40, 354, 90), W, H), [])  # within the 3 px tolerance
        self.assertIn("smaller", L.check_new_box((100, 40, 101, 90), W, H)[0])

    def test_save_load_notes_progress(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            L.save_label(d / "a.txt", [(100, 40, 150, 90)], W, H)
            self.assertEqual(len(L.load_label(d / "a.txt", W, H)), 1)
            self.assertEqual(L.load_label(d / "missing.txt", W, H), [])
            L.save_label(d / "b.txt", [], W, H)
            self.assertEqual((d / "b.txt").read_text(encoding="utf-8"), "")
            notes = d / "notes.txt"
            notes.write_text("# header", encoding="utf-8-sig")
            L.append_note(notes, "c", "exclude", "too | dark")
            L.append_note(notes, "a", "note", "cross without slug")
            self.assertEqual(L.excluded_stems(notes, {"a", "b", "c"}), {"c"})
            self.assertEqual(L.progress(d, ["a", "b", "c", "e"], {"c"}),
                             {"frames": 4, "labelled": 2, "excluded": 1, "left": 1})
            with self.assertRaises(ValueError):
                L.append_note(notes, "a", "note", "   ")

    def test_contrast_is_display_only_and_per_tile(self):
        from dataset_audit.tritonia_frames import write_png_gray
        w, h = 6, 4
        px = bytes([100, 101, 102] * 2 + [100, 101, 102] * 2 + [10, 20, 30] * 2 + [10, 20, 30] * 2)
        with tempfile.TemporaryDirectory() as d:
            png = Path(d) / "f.png"
            write_png_gray(png, w, h, px)
            before = png.read_bytes()
            self.assertEqual(L.read_gray_png(png), (w, h, px))
            out = L.stretch_tiles(px, w, h, 3, 2)
            self.assertEqual(png.read_bytes(), before)  # the file on disk never changes
        self.assertEqual(len(out), len(px))
        self.assertEqual(max(out), 255)
        self.assertEqual(min(out), 0)
        self.assertEqual(L.stretch(bytes([7] * 50)), bytes([7] * 50))  # flat image: unchanged
        self.assertTrue(L.pgm(w, h, out).startswith(b"P5 6 4 255\n"))

    def test_export_passes_the_real_ingest(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            stems = make_batch(d / "batch")
            labels = d / "labels"
            labels.mkdir()
            notes = d / "batch" / "notes.txt"
            L.append_note(notes, stems[0], "exclude", "too dark")
            for s in stems[1:]:
                L.save_label(labels / f"{s}.txt", [(100, 40, 150, 90), (720, 300, 760, 350)], W, H)
            self.assertEqual(L.export_zip(labels, stems, d / "export.zip"), len(stems) - 1)
            with self.assertRaises(FileExistsError):
                L.export_zip(labels, stems, d / "export.zip")
            report = T.ingest_boxes(d / "export.zip", notes, d / "batch", d / "boxes")
            self.assertEqual(report["counts"]["boxes_per_class"], {"lobster": 0, "sea_slug": 2 * (len(stems) - 1)})
            self.assertEqual(list(report["excluded"]), [stems[0]])


if __name__ == "__main__":
    unittest.main()
