import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracking import analytics as A
from tracking import schema as S
from tracking.svg import occupancy_svg, trajectories_svg
from tracking.synthetic import make_scenario


def row(frame, tid, cx, cy, w=10, h=10, conf=0.9, video="v"):
    return {"video_id": video, "frame_idx": frame, "track_id": tid, "class_id": 0, "conf": conf,
            "x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2}


def meta(n_frames=10, fps=None, video="v"):
    return {"schema": S.META_SCHEMA, "video_id": video, "synthetic": True, "frame_width": 100,
            "frame_height": 100, "n_frames": n_frames, "fps": fps, "detector": {}, "tracker": {}}


class SchemaTest(unittest.TestCase):
    def test_round_trip_sorted(self):
        rows = [row(1, 2, 50, 50), row(0, 1, 20, 20), row(1, 1, 23, 24)]
        with tempfile.TemporaryDirectory() as d:
            S.write_tracks_dir(Path(d), meta(), rows)
            m, back = S.load_tracks_dir(Path(d))
        self.assertEqual(m, meta())
        self.assertEqual([(r["frame_idx"], r["track_id"]) for r in back], [(0, 1), (1, 1), (1, 2)])
        self.assertAlmostEqual(back[1]["x1"], 18.0)

    def test_rejects_bad_rows(self):
        bad = {
            "duplicate": [row(0, 1, 20, 20), row(0, 1, 30, 30)],
            "box": [{**row(0, 1, 20, 20), "x2": 10}],
            "conf": [row(0, 1, 20, 20, conf=1.5)],
            "two videos": [row(0, 1, 20, 20), row(1, 1, 20, 20, video="w")],
            "frame range": [row(10, 1, 20, 20)],
            "outside frame": [row(0, 1, 99, 50, w=20)],
        }
        for name, rows in bad.items():
            with self.subTest(name), self.assertRaises(S.TracksError):
                S.validate_rows(rows, meta())

    def test_rejects_bad_meta_and_header(self):
        for change in ({"fps": 0}, {"fps": True}, {"n_frames": 0}, {"schema": "x"}):
            with self.subTest(change), self.assertRaises(S.TracksError):
                S.validate_meta({**meta(), **change})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / S.TRACKS_FILE
            p.write_text("frame,id\n0,1\n", encoding="utf-8")
            with self.assertRaises(S.TracksError):
                S.read_tracks(p)
            p.write_text(",".join(S.COLUMNS) + "\nv,zero,1,0,0,1,1,0.5,0\n", encoding="utf-8")
            with self.assertRaisesRegex(S.TracksError, "line 2"):
                S.read_tracks(p)

    def test_output_guard(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "raw"
            with self.assertRaises(S.TracksError):
                S.check_output_dir(raw / "out", [raw])
            with self.assertRaises(S.TracksError):
                S.check_output_dir(raw, raw)
            self.assertEqual(S.check_output_dir(Path(d) / "out", [raw]), (Path(d) / "out").resolve())


class AnalyticsTest(unittest.TestCase):
    def test_hand_computed_track(self):
        # 3-4-5 steps, a 2-frame gap, then a still frame
        obs = [row(0, 1, 0, 0), row(1, 1, 3, 4), row(4, 1, 12, 16), row(5, 1, 12, 16)]
        t = A.track_summary(obs)
        self.assertEqual((t["observed_frames"], t["span_frames"], t["missing_frames"]), (4, 6, 2))
        self.assertEqual((t["n_gaps"], t["max_gap_frames"]), (1, 2))
        self.assertEqual(t["path_px"], 20.0)
        self.assertEqual(t["net_displacement_px"], 20.0)
        self.assertEqual(t["mean_speed_px_per_frame"], 4.0)
        self.assertEqual(t["max_step_px_per_frame"], 5.0)
        self.assertEqual(t["straightness"], 1.0)
        self.assertEqual(t["stationary_frames"], 1)
        self.assertIsNone(t["duration_s"])

    def test_single_frame_and_fps(self):
        t = A.track_summary([row(3, 7, 5, 5)])
        self.assertIsNone(t["mean_speed_px_per_frame"])
        self.assertIsNone(t["straightness"])
        t = A.track_summary([row(0, 1, 0, 0), row(10, 1, 30, 40)], fps=5)
        self.assertEqual((t["duration_s"], t["mean_speed_px_per_s"]), (2.0, 25.0))

    def test_occupancy_grid_edges(self):
        grid = A.occupancy_grid([row(0, 1, 5, 5), row(0, 2, 99.9, 99.9, w=0.1, h=0.1)], 100, 100, 4, 2)
        self.assertEqual(grid, [[1, 0, 0, 0], [0, 0, 0, 1]])


class SyntheticScenarioTest(unittest.TestCase):
    def setUp(self):
        self.meta, self.rows = make_scenario(jitter_px=0)
        self.report = A.summarize(self.meta, self.rows)
        self.tracks = {t["track_id"]: t for t in self.report["per_track"]}

    def test_valid_and_deterministic(self):
        S.validate_meta(self.meta)
        S.validate_rows(self.rows, self.meta)
        self.assertEqual(make_scenario(seed=3), make_scenario(seed=3))
        self.assertNotEqual(make_scenario(seed=3)[1], make_scenario(seed=4)[1])

    def test_known_answers(self):
        o = self.report["overall"]
        self.assertEqual((o["n_tracks"], o["n_boxes"], o["max_boxes_in_a_frame"]), (6, 1153, 5))
        self.assertEqual((o["short_tracks"], o["tracks_with_gaps"]), (2, 1))
        self.assertEqual((self.tracks[1]["path_px"], self.tracks[1]["mean_speed_px_per_frame"]), (1196.0, 4.0))
        self.assertEqual((self.tracks[2]["path_px"], self.tracks[2]["stationary_frames"]), (597.0, 100))
        t3 = self.tracks[3]
        self.assertEqual((t3["n_gaps"], t3["max_gap_frames"], t3["path_px"]), (1, 20, 1495.0))
        self.assertEqual(self.tracks[6]["straightness"], None)
        self.assertIn("SYNTHETIC", self.report["limitations"][-1])
        self.assertIsNone(self.report["per_track"][0]["duration_s"])

    def test_svgs_are_well_formed(self):
        for svg in (trajectories_svg(self.meta, self.rows),
                    occupancy_svg(self.meta, A.occupancy_grid(self.rows, 1920, 1080))):
            root = ET.fromstring(svg)
            self.assertTrue(root.tag.endswith("svg"))
            self.assertIn("SYNTHETIC", svg)
        self.assertIn("stroke-dasharray", trajectories_svg(self.meta, self.rows))  # track 3's gap


class GpuModuleImportTest(unittest.TestCase):
    def test_run_module_imports_without_ultralytics(self):
        from tracking import run
        self.assertIn("bytetrack.yaml", run.TRACKERS)


if __name__ == "__main__":
    unittest.main()
