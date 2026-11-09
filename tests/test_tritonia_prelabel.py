"""Suggested boxes (tritonia_prelabel) and box provenance (tritonia_labeller), on synthetic frames."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dataset_audit import tritonia_labeller as L
from dataset_audit import tritonia_prelabel as P

W, H = 1056, 480  # 3 x 2 camera views of 352 x 240


def frame(bg: int = 120, blobs=(), text=()) -> bytes:
    """Flat frame with bright rectangles (x0, y0, x1, y1, value) and outlined 'text' rectangles."""
    px = bytearray([bg]) * (W * H)
    for x0, y0, x1, y1, v in blobs:
        for y in range(y0, y1):
            px[y * W + x0:y * W + x1] = bytes([v]) * (x1 - x0)
    for x0, y0, x1, y1 in text:  # black outline, white inside
        for y in range(y0 - 2, y1 + 2):
            px[y * W + x0 - 2:y * W + x1 + 2] = bytes([5]) * (x1 - x0 + 4)
        for y in range(y0, y1):
            px[y * W + x0:y * W + x1] = bytes([250]) * (x1 - x0)
    return bytes(px)


class SuggestTest(unittest.TestCase):
    def test_box_fits_the_blob_beside_the_cross(self):
        px = frame(blobs=[(100, 50, 116, 60, 200)])  # 16 x 10 px blob
        r = P.suggest_box(px, W, H, 98, 47)  # cross a few px off, as hand clicks are
        self.assertEqual(r["flag"], "")
        self.assertEqual(r["box"], (99.0, 49.0, 117.0, 61.0))  # blob + 1 px pad

    def test_flat_view_gives_no_box(self):
        r = P.suggest_box(frame(), W, H, 100, 50)
        self.assertIsNone(r["box"])
        self.assertEqual(r["flag"], "no clear blob")

    def test_faint_blob_below_min_contrast_gives_no_box(self):
        r = P.suggest_box(frame(blobs=[(100, 50, 110, 60, 120 + P.MIN_CONTRAST - 2)]), W, H, 104, 54)
        self.assertIsNone(r["box"])

    def test_large_bright_area_is_not_a_slug(self):
        r = P.suggest_box(frame(blobs=[(60, 20, 200, 150, 220)]), W, H, 120, 80)
        self.assertIsNone(r["box"])
        self.assertIn(r["flag"], ("blob not compact", "no clear blob"))

    def test_timestamp_text_is_dropped(self):
        r = P.suggest_box(frame(text=[(100, 50, 112, 58)]), W, H, 105, 53)
        self.assertIsNone(r["box"])
        self.assertEqual(r["flag"], "looks like timestamp text")

    def test_box_never_crosses_a_camera_border(self):
        r = P.suggest_box(frame(blobs=[(345, 60, 352, 70, 200), (352, 60, 360, 70, 200)]), W, H, 348, 65)
        self.assertIsNotNone(r["box"])
        self.assertLessEqual(r["box"][2], 352)
        self.assertEqual(L.check_new_box(r["box"], W, H), [])

    def test_two_crosses_on_one_blob_give_one_box(self):
        px = frame(blobs=[(100, 50, 116, 60, 200)])
        res = P.suggest_frame(px, W, H, [(106, 55), (109, 56)])
        self.assertEqual(sum(r["box"] is not None for r in res), 1)
        self.assertEqual(res[1]["flag"], "same blob as another cross")

    def test_tile_bounds(self):
        self.assertEqual(P.tile_bounds(10, 10, W, H), (0, 0, 352, 240))
        self.assertEqual(P.tile_bounds(352, 240, W, H), (352, 240, 704, 480))
        self.assertEqual(P.tile_bounds(1055, 479, W, H), (704, 240, 1056, 480))


class SplitFilterTest(unittest.TestCase):
    def test_frames_in_splits_follows_the_d030_day_split(self):
        sys.path.insert(0, str(REPO / "scripts"))
        import label_tritonia as M
        days = ["201305%02d" % d for d in range(19, 32)] + ["2013060%d" % d for d in (1, 2, 3)]
        stems = [f"{d}all_15spf_t{t:04d}" for d in days for t in (1, 50)]
        got = {s[:8] for s in M.frames_in_splits(stems, {"val", "test"})}
        self.assertEqual(got, {"20130524", "20130529", "20130602", "20130520", "20130521", "20130528"})
        self.assertEqual(len(M.frames_in_splits(stems, {"train"})), 20)


class ProvenanceTest(unittest.TestCase):
    def test_round_trip_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            L.save_label(d / "a.txt", [(100, 40, 120, 60), (200, 40, 220, 60), (300, 40, 320, 60)], W, H)
            L.save_provenance(d, "a", ["suggested", "suggested_resized", "hand"], 3, 1,
                              [{"x": 5.0, "y": 6.0, "flag": "no clear blob"}], P.PARAMS)
            L.save_label(d / "b.txt", [(100, 40, 120, 60)], W, H)  # labelled by hand, no record
            self.assertEqual(L.load_provenance(d, "a", 3)["origins"][1], "suggested_resized")
            self.assertIsNone(L.load_provenance(d, "a", 2))  # record no longer matches the labels
            out = d / "prov.csv"
            self.assertEqual(L.export_provenance_csv(d, ["a", "b", "c"], W, H, out), 2)
            with open(out, newline="", encoding="utf-8") as f:
                rows = {r["frame"]: r for r in csv.DictReader(f)}
            self.assertEqual((rows["a"]["source"], rows["a"]["suggested_kept"], rows["a"]["suggested_resized"],
                              rows["a"]["hand"], rows["a"]["suggested_deleted"], rows["a"]["crosses_flagged"]),
                             ("suggested+reviewed", "1", "1", "1", "1", "1"))
            self.assertEqual((rows["b"]["source"], rows["b"]["hand"]), ("hand", "1"))
            with self.assertRaises(FileExistsError):
                L.export_provenance_csv(d, ["a"], W, H, out)
            with self.assertRaises(ValueError):
                L.save_provenance(d, "a", ["guessed"], 1, 0, [], {})

    def test_export_zip_ignores_the_provenance_folder(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            L.save_label(d / "a.txt", [(100, 40, 120, 60)], W, H)
            L.save_provenance(d, "a", ["suggested"], 1, 0, [], P.PARAMS)
            import zipfile
            L.export_zip(d, ["a"], d / "x.zip")
            with zipfile.ZipFile(d / "x.zip") as z:
                self.assertEqual(z.namelist(), ["a.txt"])


if __name__ == "__main__":
    unittest.main()
