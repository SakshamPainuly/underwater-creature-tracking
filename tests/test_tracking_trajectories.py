import sys
import unittest
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tracking import trajectories as J


class StatsTest(unittest.TestCase):
    def test_track_stats_links_steps_within_twice_the_cadence(self):
        pts = [(1, 0, 0), (2, 3, 4), (3, 6, 8), (10, 6, 8)]  # 5 px per frame, then a 7-frame gap
        s = J.track_stats(pts, cadence=1)
        self.assertEqual((s["points"], s["linked_steps"], s["gaps"]), (4, 2, 1))
        self.assertEqual(s["path_px"], 10.0)
        self.assertEqual(s["net_px"], 10.0)
        self.assertEqual(s["duration_s"], 9 * 15.0)
        self.assertAlmostEqual(s["median_step_speed_px_s"], 5 / 15, places=4)
        self.assertIsNone(J.track_stats([(1, 0, 0)], 1))
        self.assertEqual(J.track_stats([(1, 0, 0), (3, 0, 2)], cadence=1)["linked_steps"], 1)  # dt 2 <= 2

    def test_summarize_skips_tracks_without_linked_steps(self):
        a = J.track_stats([(1, 0, 0), (2, 0, 3)], 1)
        b = J.track_stats([(1, 0, 0), (9, 0, 3)], 1)  # only a gap
        self.assertEqual(J.summarize([a, b])["tracks"], 1)

    def test_match_tracks_picks_most_overlapping_track(self):
        gt = {(0, 1): [(1, 10, 10), (2, 10, 10), (3, 10, 10)]}
        pred = {(0, 7): [(1, 11, 10)], (0, 8): [(2, 10, 12), (3, 10, 12)], (1, 9): [(1, 10, 10)]}
        self.assertEqual(J.match_tracks(gt, pred, {1, 2, 3}), {(0, 1): ((0, 8), 2, 3)})


class FigureTest(unittest.TestCase):
    def test_colours_follow_extent_per_camera(self):
        # extent n px (moving 1 px per frame); a long track jittering in place (id 5) stays gray
        tracks = {(0, i): [(t, t, 1) for t in range(n + 1)] for i, n in enumerate([2, 9, 5, 7, 0])}
        tracks[(0, 5)] = [(t, 50 + (t % 2) * 1.5, 50) for t in range(200)]
        tracks[(1, 99)] = [(0, 1, 1)]
        c = J.colour_tracks(tracks)
        self.assertEqual([c[(0, i)] for i in (1, 3, 2)], list(J.SLOTS))
        self.assertEqual({c[(0, 0)], c[(0, 4)], c[(0, 5)]}, {J.OTHER})
        self.assertEqual(c[(1, 99)], J.SLOTS[0])

    def test_svg_is_well_formed_and_png_preview_decodes(self):
        w, h = 20, 10
        px = bytes(range(200))
        gt = {(0, 1): [(1, 2, 2), (2, 5, 5), (9, 8, 8)]}
        pred = {(0, 4): [(1, 2, 3), (2, 6, 5)]}
        svg = J.day_svg("20130529", w, h, px, 2, gt, pred, 1, "test <run> & co.")
        root = ET.fromstring(svg)
        self.assertTrue(root.tag.endswith("svg"))
        self.assertEqual(sum(1 for e in root.iter() if e.tag.endswith("polyline") and "stroke" in e.attrib), 2)
        png = J.preview_png(w, h, px, gt, pred, 1)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        idat = png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8]
        self.assertEqual(len(zlib.decompress(idat)), (2 * h + 8) * (w * 3 + 1))


class ReportChartTest(unittest.TestCase):
    def test_bar_chart_is_well_formed_with_selective_labels(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("report", REPO / "scripts" / "tritonia_tracking_report.py")
        R = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(R)
        svg = R.bar_chart("t", "s", ["a", "b"], [[(0.5, R.BLUE), (0.25, R.ORANGE)], [(0.0, R.BLUE)]],
                          [("val", R.BLUE), ("test", R.ORANGE)], {(0, 1)}, "note", ymax=0.5)
        root = ET.fromstring(svg)
        bars = [e for e in root.iter() if e.tag.endswith("path")]
        self.assertEqual(len(bars), 2)  # a zero bar draws nothing
        labels = [e.text for e in root.iter() if e.tag.endswith("text") and e.text == "0.250"]
        self.assertEqual(labels, ["0.250"])


if __name__ == "__main__":
    unittest.main()
