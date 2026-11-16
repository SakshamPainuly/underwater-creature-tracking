import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training import val_eval as V

A = (0.1, 0.1, 0.3, 0.3)
B = (0.6, 0.6, 0.8, 0.8)


def img(name, gts, preds, session="S1", lighting="not_flagged", negative=False):
    return {**V.evaluate([{"filename": name, "session_id": session, "lighting": lighting, "negative": negative}],
                         {name: gts}, {name: preds})[0]}


class GeometryTest(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(V.iou(A, A), 1.0)
        self.assertEqual(V.iou(A, B), 0.0)
        self.assertAlmostEqual(V.iou((0, 0, 2, 1), (1, 0, 3, 1)), 1 / 3)

    def test_xywh_to_xyxy(self):
        self.assertEqual(V.xywh_to_xyxy([0.5, 0.5, 0.2, 0.4]), (0.4, 0.3, 0.6, 0.7))

    def test_read_yolo_labels(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.txt"
            p.write_text("0 0.5 0.5 0.2 0.2\n\n", encoding="utf-8")
            self.assertEqual(len(V.read_yolo_labels(p)), 1)
            self.assertEqual(V.read_yolo_labels(Path(d) / "missing.txt"), [])
            p.write_text("0 0.5 0.5\n", encoding="utf-8")
            with self.assertRaises(V.ValEvalError):
                V.read_yolo_labels(p)


class MatchingTest(unittest.TestCase):
    def test_duplicate_prediction_is_one_tp_one_fp(self):
        self.assertEqual(V.match_image([A], [(0.6, A), (0.9, A)]), [(0.9, True), (0.6, False)])

    def test_low_overlap_is_fp(self):
        shifted = (0.2, 0.2, 0.4, 0.4)  # IoU with A = 1/7
        self.assertEqual(V.match_image([A], [(0.9, shifted)]), [(0.9, False)])

    def test_prediction_takes_best_overlapping_box(self):
        near = (0.1, 0.1, 0.3, 0.32)
        out = V.match_image([near, A], [(0.9, A), (0.8, near)])
        self.assertEqual(out, [(0.9, True), (0.8, True)])

    def test_counts_and_negatives(self):
        images = [img("p.png", [A, B], [(0.9, A), (0.2, B), (0.5, (0, 0, 0.05, 0.05))]),
                  img("n.png", [], [(0.4, B), (0.1, A)], negative=True)]
        c = V.counts_at(images, 0.3)
        self.assertEqual((c["tp"], c["fp"], c["fn"]), (1, 2, 1))
        self.assertEqual((c["negative_images"], c["negative_images_with_detection"],
                          c["detections_on_negative_images"]), (1, 1, 1))
        self.assertAlmostEqual(c["precision"], 1 / 3)
        self.assertAlmostEqual(c["recall"], 1 / 2)
        c = V.counts_at(images, 0.95)
        self.assertEqual((c["tp"], c["fp"], c["fn"], c["precision"], c["f1"]), (0, 0, 2, 0.0, 0.0))

    def test_confusion_matrix(self):
        cm = V.confusion_matrix({"tp": 5, "fp": 2, "fn": 3})
        self.assertEqual(cm["matrix"], [[5, 2], [3, None]])


class APTest(unittest.TestCase):
    def test_perfect_detector(self):
        self.assertAlmostEqual(V.ap50([img("a.png", [A, B], [(0.9, A), (0.8, B)])]), 1.0)

    def test_no_detections_and_no_ground_truth(self):
        self.assertEqual(V.ap50([img("a.png", [A], [])]), 0.0)
        self.assertIsNone(V.ap50([img("n.png", [], [(0.9, A)], negative=True)]))

    def test_fp_ranked_first_halves_precision_at_first_recall(self):
        # ranking: FP (0.9), TP (0.8) -> precision 0.5 for every recall level up to 1
        ap = V.ap50([img("a.png", [A], [(0.9, B), (0.8, A)])])
        self.assertAlmostEqual(ap, 0.5)

    def test_curves_shapes(self):
        images = [img("a.png", [A], [(0.9, A)])]
        c, pr = V.conf_curves(images), V.pr_curve(images)
        self.assertEqual(len(c["f1"]), 101)
        self.assertEqual(c["f1"][90], 1.0)
        self.assertEqual(c["f1"][91], 0.0)
        self.assertEqual(pr["precision"], [1.0] * 101)


class ValOnlyTest(unittest.TestCase):
    build = [
        {"split": "train", "filename": "t.png", "session_id": "S0", "label_status": "positive"},
        {"split": "val", "filename": "v.png", "session_id": "S1", "label_status": "positive"},
        {"split": "val", "filename": "n.png", "session_id": "S2", "label_status": "negative_confirmed"},
        {"split": "test", "filename": "x.png", "session_id": "S3", "label_status": "positive"},
    ]
    split = [{"session_id": s, "lighting_flag": f} for s, f in
             (("S0", "False"), ("S1", "True"), ("S2", "False"), ("S3", "False"))]

    def test_val_rows_keep_val_only(self):
        rows = V.val_rows(self.build, self.split)
        self.assertEqual([r["filename"] for r in rows], ["v.png", "n.png"])
        self.assertEqual([r["lighting"] for r in rows], ["red_light_or_night", "not_flagged"])
        self.assertEqual([r["negative"] for r in rows], [False, True])

    def test_mixed_view_rows_use_group_id_as_session(self):
        mixed = [{"split": "train", "filename": "z.png", "origin": "zenodo", "kind": "original",
                  "group_id": "mpa_cam1_110822", "label_status": "positive"},
                 {"split": "val", "filename": "v.png", "origin": "ours", "kind": "original",
                  "group_id": "S1", "label_status": "positive"},
                 {"split": "val", "filename": "n.png", "origin": "ours", "kind": "original",
                  "group_id": "S2", "label_status": "negative_confirmed"}]
        rows = V.val_rows(mixed, self.split)
        self.assertEqual([r["session_id"] for r in rows], ["S1", "S2"])
        self.assertEqual([r["lighting"] for r in rows], ["red_light_or_night", "not_flagged"])
        self.assertEqual([r["negative"] for r in rows], [False, True])
        for origin, kind in (("zenodo", "original"), ("ours", "colour_change")):
            bad = [{**mixed[1], "origin": origin, "kind": kind}]
            with self.assertRaises(V.ValEvalError):
                V.val_rows(bad, self.split)

    def test_unknown_session_and_empty_val_are_refused(self):
        with self.assertRaises(V.ValEvalError):
            V.val_rows(self.build, self.split[:1])
        with self.assertRaises(V.ValEvalError):
            V.val_rows(self.build[:1], self.split)

    def test_predictions_outside_val_are_refused(self):
        rows = V.val_rows(self.build, self.split)
        with self.assertRaises(V.ValEvalError):
            V.evaluate(rows, {}, {"x.png": [(0.9, A)]})

    def test_negative_with_boxes_is_refused(self):
        rows = V.val_rows(self.build, self.split)
        with self.assertRaises(V.ValEvalError):
            V.evaluate(rows, {"n.png": [A]}, {})

    def test_only_val_image_paths_pass(self):
        with tempfile.TemporaryDirectory() as d:
            V.assert_val_path(Path(d) / "images" / "val" / "v.png", Path(d))
            for bad in (Path(d) / "images" / "test" / "x.png", Path(d) / "images" / "val" / "sub" / "v.png",
                        Path(d) / "images" / "val" / ".." / "test" / "x.png"):
                with self.assertRaises(V.ValEvalError):
                    V.assert_val_path(bad, Path(d))

    def test_symlinked_val_image_passes(self):
        # The YOLO view links images to the raw files (link_mode symlink on Colab).
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "raw" / "r.png"
            raw.parent.mkdir()
            raw.write_bytes(b"")
            link = Path(d) / "images" / "val" / "r.png"
            link.parent.mkdir(parents=True)
            try:
                link.symlink_to(raw)
            except OSError:
                self.skipTest("symlinks not permitted on this machine")
            V.assert_val_path(link, Path(d))

    def test_check_run_refuses_missing_record(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(V.ValEvalError):
                V.check_run(Path(d))

    def test_module_is_stdlib_only_at_import(self):
        self.assertNotIn("ultralytics", sys.modules)
        self.assertNotIn("torch", sys.modules)


class SummaryTest(unittest.TestCase):
    def test_summary_and_comparison(self):
        rows = V.val_rows(ValOnlyTest.build, ValOnlyTest.split)
        images = V.evaluate(rows, {"v.png": [A]}, {"v.png": [(0.9, A)], "n.png": [(0.5, B)]})
        s = V.summarize(images, 0.3)
        self.assertEqual(s["overall"]["tp"], 1)
        self.assertEqual(s["overall"]["negative_images_with_detection"], 1)
        self.assertEqual(sorted(s["per_lighting"]), ["not_flagged", "red_light_or_night"])
        self.assertEqual(s["per_session"]["S1"]["ap50"], 1.0)
        self.assertIsNone(s["per_session"]["S2"]["ap50"])
        per_image = V.per_image_rows(images, 0.3)
        self.assertEqual([(r["tp"], r["fp"], r["fn"]) for r in per_image], [(1, 0, 0), (0, 1, 0)])

        rec = {"experiment_id": "E001", "dataset": {"version": "v1"}, "limitations": ["L."],
               "complexity": V.complexity((225, 3_011_043, 0, 8.2), 640),
               "speed": {"batch1_predict": V.speed_summary([{"inference": 9.0}, {"inference": 2.0}]),
                         "gpu": ["Tesla T4"]},
               "ultralytics_val": {"precision": 0.8, "recall": 0.7, "map50": 0.85, "map50_95": 0.6}, **s}
        md = V.compare_markdown([rec, {**rec, "experiment_id": "E004"}])
        self.assertIn("| metric | E001 | E004 |", md)
        self.assertIn("1 of 1", md)
        self.assertIn("S2 (0)", md)
        self.assertIn("| parameters (M) | 3.01 | 3.01 |", md)
        self.assertIn("| inference ms/img, batch 1 (median) | 2.0 | 2.0 |", md)
        self.assertIn("Tesla T4", md)


class CostTest(unittest.TestCase):
    def test_speed_summary_drops_warmup_and_takes_medians(self):
        s = V.speed_summary([{"preprocess": 50, "inference": 900, "postprocess": 5},
                             {"preprocess": 1, "inference": 10, "postprocess": 1},
                             {"preprocess": 3, "inference": 12, "postprocess": 1},
                             {"preprocess": 2, "inference": 30, "postprocess": 2}])
        self.assertEqual(s["images_timed"], 3)
        self.assertEqual(s["median_ms"], {"inference": 12, "postprocess": 1, "preprocess": 2})
        self.assertEqual(s["median_ms_total"], 15)

    def test_speed_summary_needs_timed_images(self):
        with self.assertRaises(V.ValEvalError):
            V.speed_summary([{"inference": 5.0}])

    def test_complexity(self):
        c = V.complexity((225, 3_011_043, 3_011_027, 8.19), 640)
        self.assertEqual(c, {"layers": 225, "parameters": 3_011_043, "gflops": 8.19, "gflops_imgsz": 640})
        self.assertIsNone(V.complexity((1, 2, 3, 0.0), 640)["gflops"])
        with self.assertRaises(V.ValEvalError):
            V.complexity(None, 640)


if __name__ == "__main__":
    unittest.main()
