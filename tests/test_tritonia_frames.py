import csv
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dataset_audit import mtrackj as M
from dataset_audit import tritonia_frames as F


def gray_tiff(path: Path, pages: list[bytes], width: int, height: int, order: str = ">") -> None:
    """Uncompressed 8-bit grayscale TIFF, one strip per page: IFD then pixels, page after page."""
    data = bytearray((b"MM" if order == ">" else b"II") + struct.pack(order + "HI", 42, 8))
    tags = 7
    for i, px in enumerate(pages):
        start = len(data)
        pix_at = start + 2 + 12 * tags + 4
        nxt = 0 if i == len(pages) - 1 else pix_at + len(px)
        data += struct.pack(order + "H", tags)
        for tag, typ, val in ((256, 3, width), (257, 3, height), (258, 3, 8), (259, 3, 1), (262, 3, 1),
                              (273, 4, pix_at), (279, 4, len(px))):
            data += struct.pack(order + "HHI", tag, typ, 1)
            data += struct.pack(order + "HH", val, 0) if typ == 3 else struct.pack(order + "I", val)
        data += struct.pack(order + "I", nxt) + px
    path.write_bytes(bytes(data))


def png_pixels(path: Path) -> tuple[int, int, bytes]:
    b = path.read_bytes()
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    w = h = None
    while pos < len(b):
        (n,) = struct.unpack(">I", b[pos:pos + 4])
        kind, body = b[pos + 4:pos + 8], b[pos + 8:pos + 8 + n]
        assert struct.unpack(">I", b[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(kind + body)
        if kind == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    return w, h, b"".join(raw[y * (w + 1) + 1:(y + 1) * (w + 1)] for y in range(h))


def mdf(points_by_track: dict[int, list[tuple[float, float, int]]]) -> str:
    lines = ["MTrackJ 1.5.0 Data File", "Assembly 1 FF0000", "Cluster 1 FF0000"]
    for tid, pts in points_by_track.items():
        lines.append(f"Track {tid} FF0000 true")
        lines += [f"Point {i} {x} {y} 1.0 {t}.0 1.0" for i, (x, y, t) in enumerate(pts, 1)]
    return "\n".join(lines + ["End of MTrackJ Data File", ""])


class FramesTest(unittest.TestCase):
    def test_page_extraction_round_trips_through_png(self):
        w, h = 7, 5
        pages = [bytes((i * 31 + k) % 256 for k in range(w * h)) for i in range(3)]
        with tempfile.TemporaryDirectory() as d:
            for order in (">", "<"):
                tif = Path(d) / f"s{order == '<'}.tif"
                gray_tiff(tif, pages, w, h, order)
                pw, ph, strips = F.page_strips(tif)
                self.assertEqual((pw, ph, len(strips)), (w, h, 3))
                self.assertEqual(M.tiff_pages(tif), (3, w, h))
                png = Path(d) / f"p{order == '<'}.png"
                F.write_png_gray(png, w, h, F.read_page(tif, strips[2]))
                self.assertEqual(png_pixels(png), (w, h, pages[2]))
                with self.assertRaises(FileExistsError):
                    F.write_png_gray(png, w, h, pages[0])

    def test_rejects_compressed_tiff(self):
        with tempfile.TemporaryDirectory() as d:
            tif = Path(d) / "c.tif"
            gray_tiff(tif, [bytes(6)], 3, 2)
            b = bytearray(tif.read_bytes())
            i = b.index(struct.pack(">HHI", 259, 3, 1))
            b[i + 8:i + 10] = struct.pack(">H", 5)  # LZW
            tif.write_bytes(bytes(b))
            with self.assertRaisesRegex(M.MtrackjError, "not uncompressed 8-bit grayscale"):
                F.page_strips(tif)

    def test_hints_draw_cross_and_leave_input_unchanged(self):
        px = bytes(100)
        out = F.draw_hints(px, 10, 10, [(5, 5)], arm=2)
        self.assertEqual(px, bytes(100))
        self.assertEqual(out[5 * 10 + 5], 255)
        self.assertEqual(out[5 * 10 + 7], 255)
        self.assertEqual(out[0], 0)

    def test_pick_frames_spreads_evenly_with_gap(self):
        a = M.parse_mdf(mdf({1: [(1, 1, t) for t in range(1, 200)]}))
        b = M.parse_mdf(mdf({1: [(1, 1, t) for t in range(1, 200)]}))
        picks = F.pick_frames({"a": a, "b": b}, 5, seed=0, min_gap=40)
        self.assertEqual(picks, F.pick_frames({"a": a, "b": b}, 5, seed=0, min_gap=40))
        per = {s: sorted(t for n, t in picks if n == s) for s in "ab"}
        self.assertEqual(sorted(len(v) for v in per.values()), [2, 3])
        for ts in per.values():
            self.assertTrue(all(y - x >= 40 for x, y in zip(ts, ts[1:])))
        with self.assertRaisesRegex(ValueError, "Lower min_gap"):
            F.pick_frames({"a": a}, 6, seed=0, min_gap=40)


class BatchCliTest(unittest.TestCase):
    def test_builds_batch_and_refuses_output_inside_source(self):
        w, h = 8, 6
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "raw"
            root.mkdir()
            pages = [bytes([i]) * (w * h) for i in range(4)]
            gray_tiff(root / "day1.tif", pages, w, h)
            (root / "day1.tif.mdf").write_text(mdf({1: [(2.0, 3.0, 2)], 2: [(6.0, 1.0, 2), (5.0, 1.0, 4)]}),
                                               encoding="utf-8")
            before = sorted((p.name, p.stat().st_size) for p in root.iterdir())
            cmd = [sys.executable, str(REPO / "scripts" / "make_tritonia_label_batch.py"), "--root", str(root)]

            r = subprocess.run(cmd + ["--out", str(root / "batch")], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("REFUSED", r.stdout + r.stderr)

            out = Path(d) / "batch"
            r = subprocess.run(cmd + ["--out", str(out), "--n", "2", "--min-gap", "1"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out / "manifest.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([(r["t"], r["n_points"]) for r in rows], [("2", "2"), ("4", "1")])
            self.assertEqual(png_pixels(out / "images" / "day1_t0002.png"), (w, h, pages[1]))
            self.assertNotEqual(png_pixels(out / "hints" / "day1_t0002.png")[2], pages[1])
            self.assertEqual((out / "labels.txt").read_text(encoding="utf-8"), "lobster\nsea_slug\n")
            self.assertEqual(sorted((p.name, p.stat().st_size) for p in root.iterdir()), before)

            r = subprocess.run(cmd + ["--out", str(out), "--n", "2", "--min-gap", "1"], capture_output=True, text=True)
            self.assertIn("already exists", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
