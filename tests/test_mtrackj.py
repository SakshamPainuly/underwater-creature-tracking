import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dataset_audit import mtrackj as M

MDF = """MTrackJ 1.5.0 Data File
Displaying true true true 1 2 0 0 100 10 0 0 0 2 1 12 0 true false false false
Reference 1055.0 170.0 1.0
Assembly 1 FF0000
Cluster 1 FF0000
Track 1 FF0000 false
Point 1 174.0 90.0 1.0 1.0 1.0
Point 2 171.0 87.0 1.0 2.0 1.0
Point 4 -1.0 86.0 1.0 3.0 1.0
Track 2 FFFF00 true
Point 1 124.0 480.0 1.0 3.0 1.0
End of MTrackJ Data File
"""


def tiny_tiff(path: Path, pages: int, width: int, height: int, order: str = ">") -> None:
    """A header-only TIFF: `pages` IFDs, each with width (SHORT) and height (LONG) tags."""
    bo = b"MM" if order == ">" else b"II"
    data = bytearray(bo + struct.pack(order + "HI", 42, 8))
    for i in range(pages):
        start = len(data)
        nxt = 0 if i == pages - 1 else start + 2 + 2 * 12 + 4
        data += struct.pack(order + "H", 2)
        data += struct.pack(order + "HHI", 256, 3, 1) + struct.pack(order + "HH", width, 0)
        data += struct.pack(order + "HHII", 257, 4, 1, height)
        data += struct.pack(order + "I", nxt)
    path.write_bytes(bytes(data))


class ParseMdfTest(unittest.TestCase):
    def test_reads_tracks_points_and_metadata(self):
        m = M.parse_mdf(MDF)
        self.assertEqual(m.version, "1.5.0")
        self.assertEqual(m.reference, (1055.0, 170.0, 1.0))
        self.assertEqual([(t.track_id, t.flag, len(t.points)) for t in m.tracks], [(1, "false", 3), (2, "true", 1)])
        self.assertEqual(m.tracks[0].points[2], M.Point(4, -1.0, 86.0, 1.0, 3, 1.0))

    def test_summary_flags_edge_points_and_frames_beyond_stack(self):
        s = M.summarize(M.parse_mdf(MDF), n_frames=2, width=1056, height=480)
        self.assertEqual((s["n_tracks"], s["n_points"], s["t_min"], s["t_max"]), (2, 4, 1, 3))
        self.assertEqual(s["track_flags"], {"false": 1, "true": 1})
        self.assertEqual(s["t_beyond_stack"], 2)
        self.assertEqual([(p["track"], p["point"]) for p in s["points_outside_frame"]], [(1, 4), (2, 1)])

    def test_rejects_bad_input_with_line_numbers(self):
        bad = {
            "not an MTrackJ": "hello\n",
            "unknown line type": MDF.replace("Assembly 1 FF0000", "Foo 1"),
            "Point outside a Track": MDF.replace("Track 1 FF0000 false\n", ""),
            "not a whole number": MDF.replace("Point 2 171.0 87.0 1.0 2.0", "Point 2 171.0 87.0 1.0 2.5"),
            "truncated": MDF.replace("End of MTrackJ Data File\n", ""),
            "more than once": MDF.replace("Track 2 FFFF00", "Track 1 FFFF00"),
            "line 7": MDF.replace("Point 1 174.0 90.0 1.0 1.0 1.0", "Point 1 174.0 x 1.0 1.0 1.0"),
        }
        for expected, text in bad.items():
            with self.subTest(expected), self.assertRaisesRegex(M.MtrackjError, expected):
                M.parse_mdf(text)


class TiffPagesTest(unittest.TestCase):
    def test_counts_pages_both_byte_orders(self):
        with tempfile.TemporaryDirectory() as d:
            for order in (">", "<"):
                p = Path(d) / f"s{order == '<'}.tif"
                tiny_tiff(p, 3, 1056, 480, order)
                self.assertEqual(M.tiff_pages(p), (3, 1056, 480))

    def test_rejects_non_tiff(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.tif"
            p.write_bytes(b"not a tiff")
            with self.assertRaisesRegex(M.MtrackjError, "not a TIFF"):
                M.tiff_pages(p)


class InspectCliTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(REPO / "scripts" / "inspect_tritonia.py"), *args],
                              capture_output=True, text=True)

    def test_reports_and_refuses_json_inside_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "src"
            root.mkdir()
            tiny_tiff(root / "a.tif", 3, 1056, 480)
            (root / "a.tif.mdf").write_text(MDF, encoding="utf-8")
            tiny_tiff(root / "b.tif", 2, 1056, 480)
            before = sorted((p.name, p.stat().st_size) for p in root.iterdir())

            r = self.run_cli("--root", str(root), "--json", str(Path(d) / "out" / "i.json"))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("a.tif: 3 frames 1056x480, 2 tracks, 4 points", r.stdout)
            self.assertIn("b.tif: 2 frames 1056x480, no mdf", r.stdout)
            self.assertTrue((Path(d) / "out" / "i.json").exists())

            r = self.run_cli("--root", str(root), "--json", str(root / "i.json"))
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("REFUSED", r.stdout + r.stderr)
            self.assertEqual(sorted((p.name, p.stat().st_size) for p in root.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
