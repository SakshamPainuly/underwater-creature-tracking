"""Point-based scoring against the crosses, and automatic labels that the real ingest accepts (D-033)."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_audit import tritonia_dataset as T
from dataset_audit import tritonia_point_eval as E
from dataset_audit import tritonia_prelabel as P
from dataset_audit.tritonia_frames import write_png_gray
from dataset_audit.tritonia_labeller import px_to_yolo
from test_tritonia_prelabel import H, W, frame


class MatchTest(unittest.TestCase):
    def test_point_to_box(self):
        box = (10, 10, 20, 20)
        self.assertEqual(E.point_to_box(15, 15, box), 0.0)
        self.assertEqual(E.point_to_box(25, 15, box), 5.0)
        self.assertEqual(E.point_to_box(23, 24, box), 5.0)

    def test_one_to_one_by_confidence(self):
        crosses = [(15, 15), (50, 50)]
        preds = [(0.4, (12, 12, 18, 18)),  # second on cross 1: duplicate -> false
                 (0.9, (10, 10, 20, 20)),  # best on cross 1
                 (0.8, (45, 45, 55, 55)),  # cross 2
                 (0.7, (200, 200, 210, 210))]  # nothing near
        self.assertEqual(E.match_frame(preds, crosses), [(0.9, True), (0.8, True), (0.7, False), (0.4, False)])
        self.assertEqual(E.match_frame([(0.5, (28, 10, 40, 20))], [(15, 15)], 10), [(0.5, False)])  # 13 px away
        self.assertEqual(E.match_frame([(0.5, (24, 10, 40, 20))], [(15, 15)], 10), [(0.5, True)])  # 9 px away

    def test_scores_ap_and_threshold(self):
        frames = {"a": ([(0.9, (10, 10, 20, 20)), (0.3, (200, 200, 210, 210))], [(15, 15), (50, 50)]),
                  "b": ([], [(100, 100)])}
        s = E.scores(frames, 0.5)
        self.assertEqual((s["predictions"], s["matched"], s["crosses"], s["precision"], s["recall"]),
                         (1, 1, 3, 1.0, 0.3333))
        s = E.scores(frames, 0.1)
        self.assertEqual((s["predictions"], s["precision"]), (2, 0.5))
        self.assertEqual(E.point_ap(frames), 0.3333)  # one hit at precision 1, then only a false one
        best = E.best_f1_conf(frames)
        self.assertEqual(best["predictions"], 1)  # dropping the false box raises F1
        self.assertGreaterEqual(best["conf"], 0.31)
        self.assertEqual(E.point_ap({"x": ([], [(1, 1)])}), 0.0)

    def test_curves_and_frame_outcome(self):
        frames = {"a": ([(0.9, (10, 10, 20, 20)), (0.3, (200, 200, 210, 210))], [(15, 15), (50, 50)]),
                  "b": ([], [(100, 100)])}
        pr = E.pr_curve(frames)
        self.assertEqual([(p["precision"], round(p["recall"], 4)) for p in pr],
                         [(1.0, 0.0), (1.0, 0.3333), (0.5, 0.3333)])
        self.assertEqual(E.pr_curve({"x": ([(0.5, (0, 0, 1, 1))], [])}), [])
        cc = E.conf_curves(frames)
        self.assertEqual([c["conf"] for c in cc], list(E.CONF_GRID))
        self.assertEqual(cc[0], E.scores(frames, 0.01))
        o = E.frame_outcome(*frames["a"], conf=0.2)
        self.assertEqual(o, {"matched_boxes": [(0.9, (10, 10, 20, 20))],
                             "false_boxes": [(0.3, (200, 200, 210, 210))],
                             "found_crosses": [(15, 15)], "missed_crosses": [(50, 50)]})
        self.assertEqual(E.frame_outcome(*frames["a"], conf=0.95)["missed_crosses"], [(15, 15), (50, 50)])
        details = E.match_details(*frames["a"])
        self.assertEqual([i for _, _, i in details], [0, None])

    def test_read_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.txt"
            p.write_text("1 0.5 0.5 0.1 0.2 0.75\n\n", encoding="utf-8")
            (conf, box), = E.read_predictions(p, 1000, 500)
            self.assertEqual(conf, 0.75)
            self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(box, (450, 200, 550, 300))))
            self.assertEqual(E.read_predictions(Path(d) / "none.txt", 1000, 500), [])
            p.write_text("1 0.5 0.5 0.1 0.2\n", encoding="utf-8")  # no confidence
            with self.assertRaises(ValueError):
                E.read_predictions(p, 1000, 500)

    def test_frames_in_splits_follows_d030(self):
        days = ["201305%02d" % d for d in range(19, 32)] + ["2013060%d" % d for d in (1, 2, 3)]
        stems = [f"{d}all_15spf_t0001" for d in days]
        self.assertEqual({s[:8] for s in E.frames_in_splits(stems, {"test"})}, {"20130520", "20130521", "20130528"})
        self.assertEqual({s[:8] for s in E.frames_in_splits(stems, {"val"})}, {"20130524", "20130529", "20130602"})


def make_png_batch(d: Path) -> list[str]:
    """3 frames: one blob beside its cross, one cross on a flat view (no box), one of each."""
    (d / "images").mkdir(parents=True)
    (d / "points").mkdir()
    spec = {"20130519all_15spf_t0100": ([(100, 50, 116, 60, 200)], [(98, 47)]),
            "20130519all_15spf_t0200": ([], [(500, 100)]),
            "20130520all_15spf_t0100": ([(600, 300, 612, 310, 210)], [(605, 304), (900, 100)])}
    rows = []
    for stem, (blobs, crosses) in spec.items():
        write_png_gray(d / "images" / f"{stem}.png", W, H, frame(blobs=blobs))
        with open(d / "points" / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
            f.write("track,point,x,y,t\n" + "".join(f"1,1,{x},{y},1\n" for x, y in crosses))
        rows.append({"image": f"images/{stem}.png", "source_tif": f"{stem[:8]}all_15spf.tif", "t": 1,
                     "page_index": 0, "n_points": len(crosses), "tracks": "1",
                     "png_sha256": T.sha256_file(d / "images" / f"{stem}.png")})
    with open(d / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    stacks = {f"{day}all_15spf": {"n_frames": 500, "width": W, "height": H, "tif_bytes": 1}
              for day in ("20130519", "20130520")}
    (d / "batch.json").write_text(json.dumps({"seed": 0, "n": 3, "min_gap": 40, "classes": T.CLASS_NAMES,
                                              "totals": {}, "source_stacks": stacks}), encoding="utf-8")
    return list(spec)


class AutoLabelTest(unittest.TestCase):
    def test_auto_labels_pass_the_real_ingest(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            stems = make_png_batch(d / "batch")
            counts = P.auto_label_batch(d / "batch", d / "auto")
            self.assertEqual((counts["frames"], counts["crosses"], counts["boxes"], counts["excluded"]), (3, 4, 2, 1))
            notes = (d / "auto" / "notes.txt").read_text(encoding="utf-8")
            self.assertIn(f"{stems[1]} | exclude |", notes)  # its only cross got no box
            extra = (d / "auto" / "suggestions.csv", d / "auto" / "auto_label_report.json")
            report = T.ingest_boxes(d / "auto" / "labels.zip", d / "auto" / "notes.txt", d / "batch", d / "boxes",
                                    extra)
            self.assertEqual(report["counts"]["boxes"], 2)
            self.assertEqual(sorted(report["extra_files"]), ["auto_label_report.json", "suggestions.csv"])
            self.assertEqual((d / "boxes" / "extra" / "suggestions.csv").read_bytes(), extra[0].read_bytes())
            with self.assertRaises(T.TritoniaError):  # duplicate names
                T.ingest_boxes(d / "auto" / "labels.zip", d / "auto" / "notes.txt", d / "batch", d / "boxes2",
                               (extra[0], extra[0]))
            self.assertEqual(report["counts"]["excluded"], 1)
            with open(d / "auto" / "suggestions.csv", newline="", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 4)
            with self.assertRaises(FileExistsError):
                P.auto_label_batch(d / "batch", d / "auto")


class FixedBoxTest(unittest.TestCase):
    def test_fixed_box_centred_and_clipped_to_the_camera_view(self):
        self.assertEqual(P.fixed_box(100, 50, W, H, (20, 17)), (90, 41.5, 110, 58.5))
        tx0, ty0, tx1, ty1 = P.tile_bounds(W / 3 - 2, 5, W, H)  # 2 px left of a camera border, near the top
        b = P.fixed_box(W / 3 - 2, 5, W, H, (20, 17))
        self.assertEqual((b[1], b[2]), (ty0, tx1))
        self.assertEqual(T.box_problems([px_to_yolo(b, W, H)], W, H), [])
        with self.assertRaises(ValueError):
            P.fixed_box(W + 1, 5, W, H)

    def test_fixed_labels_pass_the_real_ingest(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            make_png_batch(d / "batch")
            counts = P.fixed_label_batch(d / "batch", d / "fixed", (20, 17))
            self.assertEqual((counts["frames"], counts["crosses"], counts["boxes"]), (3, 4, 4))
            report = T.ingest_boxes(d / "fixed" / "labels.zip", d / "fixed" / "notes.txt", d / "batch",
                                    d / "boxes", (d / "fixed" / "fixed_label_report.json",))
            self.assertEqual((report["counts"]["boxes"], report["counts"]["excluded"]), (4, 0))
            with self.assertRaises(FileExistsError):
                P.fixed_label_batch(d / "batch", d / "fixed")

    def test_crosses_on_one_spot_get_one_box(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            stems = make_png_batch(d / "batch")
            pts = d / "batch" / "points" / f"{stems[2]}.csv"
            pts.write_text(pts.read_text(encoding="utf-8") + "2,1,605,304,1\n", encoding="utf-8")  # same spot
            counts = P.fixed_label_batch(d / "batch", d / "fixed", (20, 17))
            self.assertEqual((counts["crosses"], counts["boxes"], counts["duplicate_crosses"]), (5, 4, 1))
            lines = (d / "fixed" / "labels" / f"{stems[2]}.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(set(lines)))


class CliTest(unittest.TestCase):
    def test_test_split_needs_a_declared_final_and_val_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, str(REPO / "scripts" / "tritonia_point_eval.py"), "--batch", d,
                                "--split", "test", "--pred-dir", d, "--out", str(Path(d) / "o.json")],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("rule 7", r.stderr)

    def test_val_run_on_saved_predictions(self):
        days = ["201305%02d" % d for d in range(19, 32)] + ["2013060%d" % d for d in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "batch" / "points").mkdir(parents=True)
            (d / "pred").mkdir()
            rows = []
            for day in days:
                stem = f"{day}all_15spf_t0001"
                (d / "batch" / "points" / f"{stem}.csv").write_text("track,point,x,y,t\n1,1,100,50,1\n",
                                                                    encoding="utf-8")
                rows.append({"image": f"images/{stem}.png"})
                # val day 20130524: a hit (0.9) and a false box (0.2); other days: one hit each
                lines = "1 0.094697 0.104167 0.015152 0.031250 0.9\n"
                if day == "20130524":
                    lines += "1 0.900000 0.900000 0.015152 0.031250 0.2\n"
                (d / "pred" / f"{stem}.txt").write_text(lines, encoding="utf-8")
            with open(d / "batch" / "manifest.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["image"])
                w.writeheader()
                w.writerows(rows)
            stacks = {f"{day}all_15spf": {"width": W, "height": H} for day in days}
            (d / "batch" / "batch.json").write_text(json.dumps({"source_stacks": stacks}), encoding="utf-8")
            r = subprocess.run([sys.executable, str(REPO / "scripts" / "tritonia_point_eval.py"), "--batch",
                                str(d / "batch"), "--split", "val", "--pred-dir", str(d / "pred"),
                                "--out", str(d / "o.json")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            res = json.loads((d / "o.json").read_text(encoding="utf-8"))
            self.assertEqual(res["frames"], 3)
            self.assertEqual(sorted(res["per_day"]), ["20130524", "20130529", "20130602"])
            self.assertEqual((res["at_conf"]["matched"], res["at_conf"]["crosses"]), (3, 3))
            self.assertEqual(res["at_conf"]["precision"], 1.0)  # val-chosen threshold drops the 0.2 box
            self.assertEqual(res["point_ap"], 1.0)
            r2 = subprocess.run([sys.executable, str(REPO / "scripts" / "tritonia_point_eval.py"), "--batch",
                                 str(d / "batch"), "--split", "val", "--pred-dir", str(d / "pred"),
                                 "--out", str(d / "o.json")], capture_output=True, text=True)
            self.assertIn("already exists", r2.stderr)  # never overwrites a result

if __name__ == "__main__":
    unittest.main()
