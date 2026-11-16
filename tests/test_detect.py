"""CPU tests for src/detect: the pure formatting and guard logic only.

No torch, ultralytics or PIL is imported here. Everything tested is deterministic and runs in
milliseconds, so the suite stays usable on the local PC (CLAUDE.md rule 3, section 7).
"""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detect import (  # noqa: E402
    DEFAULT_CONF,
    ENHANCED_SOURCE,
    ENHANCED_ZEXTRA_SOURCE,
    MODELS,
    RAW_SOURCE,
    V4_YOLO_DATASET,
    DEFAULT_YOLO_DATASET,
    dataset_for_split,
    model_by_id,
    recorded_scores,
    checkpoint_image_source,
    index_all_enhanced,
    index_enhanced,
    LIMITATIONS,
    SPLIT_STATUS,
    Detection,
    DetectError,
    TruthBox,
    detections_csv_rows,
    file_sha256,
    label_path_for,
    list_split_images,
    load_ground_truth,
    match_detections,
    parse_yolo_labels,
    refuse_raw_output,
    resolve_weights,
    sweet_spot_for_sha256,
    to_yolo_lines,
)


def box(x1=100.0, y1=200.0, x2=300.0, y2=400.0, conf=0.9, class_id=0, name="lobster"):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_id=class_id, class_name=name)


class GeometryTest(unittest.TestCase):
    def test_width_height_area(self):
        d = box(x1=10, y1=20, x2=40, y2=70)
        self.assertEqual((d.width, d.height, d.area), (30, 50, 1500))

    def test_detection_is_immutable(self):
        with self.assertRaises(Exception):
            box().confidence = 0.1  # type: ignore[misc]


class YoloLinesTest(unittest.TestCase):
    def test_centre_and_size_are_normalised(self):
        # A 200x200 box at (100,200)-(300,400) in a 1000x800 image:
        # centre (200,300) -> 0.2, 0.375; size 200x200 -> 0.2, 0.25
        self.assertEqual(to_yolo_lines([box()], 1000, 800),
                         ["0 0.200000 0.375000 0.200000 0.250000"])

    def test_class_id_is_preserved_and_never_renumbered(self):
        line = to_yolo_lines([box(class_id=3)], 100, 100)[0]
        self.assertTrue(line.startswith("3 "))

    def test_values_are_clamped_into_range(self):
        # A box running off the right/bottom edge must still produce valid normalised values.
        line = to_yolo_lines([box(x1=900, y1=900, x2=1200, y2=1300)], 1000, 1000)[0]
        values = [float(v) for v in line.split()[1:]]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values), line)

    def test_one_line_per_detection(self):
        self.assertEqual(len(to_yolo_lines([box(), box(x1=0, y1=0, x2=10, y2=10)], 500, 500)), 2)

    def test_empty_input_gives_no_lines(self):
        self.assertEqual(to_yolo_lines([], 500, 500), [])

    def test_non_positive_image_size_is_refused(self):
        for width, height in ((0, 100), (100, 0), (-5, 100)):
            with self.assertRaises(DetectError):
                to_yolo_lines([box()], width, height)


class CsvRowsTest(unittest.TestCase):
    def test_rows_are_numbered_and_carry_the_settings(self):
        rows = detections_csv_rows([box(), box(conf=0.5)], "frame.png", conf=0.3, imgsz=640)
        self.assertEqual([r["detection"] for r in rows], [1, 2])
        self.assertTrue(all(r["image"] == "frame.png" for r in rows))
        self.assertTrue(all(r["conf_threshold"] == 0.3 and r["imgsz"] == 640 for r in rows))

    def test_units_are_pixels_only(self):
        # CLAUDE.md rule 9: no cm/m anywhere in the output schema.
        keys = " ".join(detections_csv_rows([box()], "f.png", 0.3, 640)[0])
        self.assertIn("width_px", keys)
        self.assertIn("area_px2", keys)
        for forbidden in ("_cm", "_m", "metre", "meter"):
            self.assertNotIn(forbidden, keys)

    def test_no_detections_gives_no_rows(self):
        self.assertEqual(detections_csv_rows([], "f.png", 0.3, 640), [])


class WeightsResolutionTest(unittest.TestCase):
    def test_missing_checkpoint_explains_where_to_get_it(self):
        with self.assertRaises(DetectError) as ctx:
            resolve_weights("weights/definitely_not_here.pt")
        message = str(ctx.exception)
        self.assertIn("best.pt", message)
        self.assertIn("COLAB.md", message)

    def test_environment_variable_is_used_when_no_argument_given(self):
        with mock.patch.dict("os.environ", {"UWCT_WEIGHTS": "weights/from_env.pt"}):
            with self.assertRaises(DetectError) as ctx:
                resolve_weights()
        self.assertIn("from_env.pt", str(ctx.exception))

    def test_relative_paths_resolve_against_the_repository_root(self):
        with self.assertRaises(DetectError) as ctx:
            resolve_weights("weights/nope.pt")
        self.assertIn(str(ROOT), str(ctx.exception))


class RawOutputGuardTest(unittest.TestCase):
    """CLAUDE.md rule 1: a script that writes must refuse paths inside the raw root."""

    def test_path_inside_the_raw_root_is_refused(self):
        from dataset_audit.config import BIOLOGY_ROOT

        for candidate in (Path(BIOLOGY_ROOT) / "out.csv",
                          Path(BIOLOGY_ROOT) / "Set 100" / "nested" / "out.txt",
                          Path(BIOLOGY_ROOT)):
            with self.assertRaises(DetectError, msg=str(candidate)):
                refuse_raw_output(candidate)

    def test_path_outside_the_raw_root_is_allowed(self):
        self.assertEqual(refuse_raw_output(ROOT / "runs" / "out.csv"),
                         (ROOT / "runs" / "out.csv").resolve())


class DeclaredDefaultsTest(unittest.TestCase):
    def test_default_confidence_is_e001s_operating_point(self):
        # Chosen on val and frozen before the single test evaluation (docs/EXPERIMENTS.md).
        self.assertAlmostEqual(DEFAULT_CONF, 0.298, places=3)

    def test_limitations_state_the_v1_caveats(self):
        text = " ".join(LIMITATIONS).lower()
        for required in ("not new sites", "pixel", "zextra", "unknown", "zenodo sk"):
            self.assertIn(required, text)

    def test_limitation_numbers_match_the_committed_records(self):
        # The new-site sentence quotes E027's test scores; they must be the recorded ones (rule 8).
        scores = recorded_scores("E027")
        text = " ".join(LIMITATIONS)
        self.assertIn(f"{scores.test:.3f}", text)
        self.assertIn(f"{scores.zenodo_sk:.3f}", text)

class LabelPathTest(unittest.TestCase):
    def test_images_folder_becomes_labels_and_suffix_becomes_txt(self):
        path = label_path_for(Path("data/yolo/lobster_v1/images/test/frame_00001.png"))
        self.assertEqual(path, Path("data/yolo/lobster_v1/labels/test/frame_00001.txt"))

    def test_only_the_last_images_segment_is_rewritten(self):
        path = label_path_for(Path("/srv/images/lobster/images/val/a.png"))
        self.assertEqual(path, Path("/srv/images/lobster/labels/val/a.txt"))

    def test_a_path_outside_a_built_split_is_refused(self):
        with self.assertRaises(DetectError):
            label_path_for(Path("data/raw/biology/zExtra Photos/a.png"))


class ParseYoloLabelsTest(unittest.TestCase):
    def test_round_trips_with_the_exporter(self):
        # to_yolo_lines and parse_yolo_labels must be inverses in pixel space.
        original = box(x1=100.0, y1=200.0, x2=300.0, y2=400.0)
        text = "\n".join(to_yolo_lines([original], 1000, 800))
        parsed = parse_yolo_labels(text, 1000, 800)[0]
        for got, want in ((parsed.x1, 100.0), (parsed.y1, 200.0),
                          (parsed.x2, 300.0), (parsed.y2, 400.0)):
            self.assertAlmostEqual(got, want, places=2)

    def test_empty_file_is_a_confirmed_negative_not_an_error(self):
        # An empty label file means "no lobster here"; a MISSING file means unknown (rule 6).
        self.assertEqual(parse_yolo_labels("", 100, 100), [])
        self.assertEqual(parse_yolo_labels("\n  \n", 100, 100), [])

    def test_multiple_boxes_are_all_parsed(self):
        text = "0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.05 0.05\n"
        self.assertEqual(len(parse_yolo_labels(text, 640, 640)), 2)

    def test_class_id_is_preserved(self):
        self.assertEqual(parse_yolo_labels("3 0.5 0.5 0.2 0.2", 100, 100)[0].class_id, 3)

    def test_trailing_confidence_column_is_ignored(self):
        boxes = parse_yolo_labels("0 0.5 0.5 0.2 0.2 0.97", 100, 100)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].width, 20.0, places=6)

    def test_malformed_lines_are_refused_with_the_line_number(self):
        for bad in ("0 0.5 0.5 0.2", "0 0.5 0.5 0.2 banana"):
            with self.assertRaises(DetectError, msg=bad) as ctx:
                parse_yolo_labels(bad, 100, 100)
            self.assertIn("line 1", str(ctx.exception))

    def test_non_positive_image_size_is_refused(self):
        with self.assertRaises(DetectError):
            parse_yolo_labels("0 0.5 0.5 0.2 0.2", 0, 100)


class LoadGroundTruthTest(unittest.TestCase):
    def test_missing_label_file_is_unknown_not_negative(self):
        # CLAUDE.md rule 6: a missing label is never silently read as "no lobster".
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "images" / "test" / "frame.png"
            image.parent.mkdir(parents=True)
            image.touch()
            self.assertIsNone(load_ground_truth(image, 640, 640))

    def test_empty_label_file_is_an_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "images" / "test" / "frame.png"
            image.parent.mkdir(parents=True)
            image.touch()
            label = Path(tmp) / "labels" / "test" / "frame.txt"
            label.parent.mkdir(parents=True)
            label.write_text("", encoding="utf-8")
            self.assertEqual(load_ground_truth(image, 640, 640), [])


def truth(x1=100.0, y1=200.0, x2=300.0, y2=400.0, class_id=0):
    return TruthBox(x1=x1, y1=y1, x2=x2, y2=y2, class_id=class_id)


class MatchDetectionsTest(unittest.TestCase):
    def test_identical_boxes_agree(self):
        result = match_detections([box()], [truth()])
        self.assertEqual(len(result.matched), 1)
        self.assertEqual(result.unmatched_detections, ())
        self.assertEqual(result.unmatched_truth, ())
        self.assertAlmostEqual(result.matched[0][2], 1.0, places=6)

    def test_a_prediction_with_no_label_is_extra(self):
        result = match_detections([box(x1=0, y1=0, x2=10, y2=10)], [truth()])
        self.assertEqual(result.unmatched_detections, (0,))
        self.assertEqual(result.unmatched_truth, (0,))

    def test_a_label_with_no_prediction_is_missed(self):
        result = match_detections([], [truth()])
        self.assertEqual(result.unmatched_truth, (0,))
        self.assertEqual(result.matched, ())

    def test_detections_on_a_confirmed_negative_are_all_extra(self):
        result = match_detections([box(), box(x1=0, y1=0, x2=5, y2=5)], [])
        self.assertEqual(result.unmatched_detections, (0, 1))
        self.assertEqual(result.matched, ())

    def test_overlap_below_the_threshold_does_not_pair(self):
        # 100x100 prediction overlapping a 100x100 label by 50x100 -> IoU = 1/3.
        prediction = box(x1=0, y1=0, x2=100, y2=100)
        label = truth(x1=50, y1=0, x2=150, y2=100)
        self.assertEqual(len(match_detections([prediction], [label], 0.3).matched), 1)
        self.assertEqual(len(match_detections([prediction], [label], 0.5).matched), 0)

    def test_a_different_class_never_pairs(self):
        result = match_detections([box(class_id=0)], [truth(class_id=1)])
        self.assertEqual(result.matched, ())

    def test_each_label_is_claimed_once_by_the_most_confident_prediction(self):
        strong = box(conf=0.9)
        weak = box(x1=105, y1=205, x2=305, y2=405, conf=0.4)
        result = match_detections([weak, strong], [truth()])
        self.assertEqual(len(result.matched), 1)
        self.assertEqual(result.matched[0][0], 1)  # the strong prediction, index 1
        self.assertEqual(result.unmatched_detections, (0,))

    def test_invalid_threshold_is_refused(self):
        for bad in (0.0, 1.5, -0.2):
            with self.assertRaises(DetectError, msg=str(bad)):
                match_detections([box()], [truth()], bad)


class SplitStatusTest(unittest.TestCase):
    def test_every_split_declares_what_it_can_be_used_for(self):
        self.assertEqual(set(SPLIT_STATUS), {"test", "test_zenodo_sk", "val", "train"})

    def test_the_held_out_splits_warn_against_tuning_on_them(self):
        # CLAUDE.md rule 7 must be visible wherever held-out images are shown.
        for split in ("test", "test_zenodo_sk"):
            text = SPLIT_STATUS[split].lower()
            self.assertIn("rule 7", text)
            self.assertIn("threshold selection on test", text)

    def test_zenodo_sk_is_read_from_the_v4_view_and_the_rest_from_v1(self):
        self.assertEqual(dataset_for_split("test_zenodo_sk"), V4_YOLO_DATASET)
        for split in ("train", "val", "test"):
            self.assertEqual(dataset_for_split(split), DEFAULT_YOLO_DATASET)

    def test_unknown_split_is_refused_by_name(self):
        with self.assertRaises(DetectError) as ctx:
            list_split_images("holdout")
        self.assertIn("test", str(ctx.exception))


class SweetSpotTest(unittest.TestCase):
    """Validated slider values are looked up from records by the checkpoint's sha256."""

    def _records(self, tmp, exp="E009", best_sha="ab" * 32, val_conf=0.2763, test_conf=None, imgsz=960):
        import json

        folder = Path(tmp) / exp
        folder.mkdir(parents=True)
        (folder / "run_record.json").write_text(json.dumps({
            "experiment": {"id": exp}, "train_args": {"imgsz": imgsz},
            "outputs": {"best": {"sha256": best_sha}}}), encoding="utf-8")
        if val_conf is not None:
            (folder / "val_eval_record.json").write_text(json.dumps({"operating_conf": val_conf}), encoding="utf-8")
        if test_conf is not None:
            (folder / "final_test_record.json").write_text(json.dumps({"operating_conf": test_conf}), encoding="utf-8")
        return Path(tmp)

    def test_known_checkpoint_gets_its_own_val_chosen_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            spot = sweet_spot_for_sha256("ab" * 32, self._records(tmp))
        self.assertEqual(spot.experiment_id, "E009")
        self.assertEqual(spot.conf, 0.276)  # rounded to the slider's precision
        self.assertEqual(spot.imgsz, 960)
        self.assertEqual((spot.iou, spot.match_iou), (0.7, 0.5))
        self.assertIn("val_eval_record.json", spot.conf_source)

    def test_final_test_record_is_the_fallback_for_the_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            spot = sweet_spot_for_sha256("ab" * 32, self._records(tmp, val_conf=None, test_conf=0.2983))
        self.assertEqual(spot.conf, 0.298)
        self.assertIn("final_test_record.json", spot.conf_source)

    def test_final_test_record_dict_format_is_read(self):
        # final_test_eval.py writes {"value": ..., "chosen_on": "val", ...}, not a bare number.
        with tempfile.TemporaryDirectory() as tmp:
            spot = sweet_spot_for_sha256("ab" * 32, self._records(
                tmp, val_conf=None, test_conf={"value": 0.26926, "chosen_on": "val", "rule": "max mean F1 on val"}))
        self.assertEqual(spot.conf, 0.269)
        self.assertIn("final_test_record.json", spot.conf_source)

    def test_a_confidence_not_chosen_on_val_is_never_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            spot = sweet_spot_for_sha256("ab" * 32, self._records(
                tmp, val_conf=None, test_conf={"value": 0.5, "chosen_on": "test"}))
        self.assertEqual(spot.conf, DEFAULT_CONF)
        self.assertIn("no val-chosen confidence", spot.conf_source)

    def test_unknown_checkpoint_falls_back_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            spot = sweet_spot_for_sha256("cd" * 32, self._records(tmp))
        self.assertFalse(spot.known)
        self.assertEqual(spot.conf, DEFAULT_CONF)
        self.assertIn("not validated", spot.conf_source)

    def test_missing_records_folder_is_not_an_error(self):
        spot = sweet_spot_for_sha256("ab" * 32, Path(tempfile.gettempdir()) / "no_such_records_dir_uwct")
        self.assertFalse(spot.known)

    def test_sha256_of_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.pt"
            path.write_bytes(b"lobster")
            self.assertEqual(file_sha256(path), hashlib.sha256(b"lobster").hexdigest())

    def test_committed_records_map_e001_to_its_operating_point(self):
        records = ROOT / "experiments" / "records"
        import json

        best = json.loads((records / "E001" / "run_record.json").read_text(encoding="utf-8"))["outputs"]["best"]["sha256"]
        spot = sweet_spot_for_sha256(best, records)
        self.assertEqual((spot.experiment_id, spot.conf, spot.imgsz), ("E001", 0.298, 640))

    def test_committed_records_map_e027_to_its_operating_point(self):
        import json

        records = ROOT / "experiments" / "records"
        best = json.loads((records / "E027" / "run_record.json").read_text(encoding="utf-8"))["outputs"]["best"]["sha256"]
        spot = sweet_spot_for_sha256(best, records)
        self.assertEqual((spot.experiment_id, spot.conf, spot.imgsz), ("E027", 0.269, 640))


class EnhancedImagesTest(unittest.TestCase):
    """Enhanced copies are found by filename; each image type is paired with its model."""

    def test_index_finds_images_by_filename_across_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            for folder, name in (("train", "a.png"), ("test", "b.png")):
                (Path(tmp) / "images" / folder).mkdir(parents=True, exist_ok=True)
                (Path(tmp) / "images" / folder / name).write_bytes(b"")
            (Path(tmp) / "images" / "train" / "notes.txt").write_text("x")
            index = index_enhanced(Path(tmp))
        self.assertEqual(sorted(index), ["a.png", "b.png"])
        self.assertEqual(index["b.png"].parent.name, "test")

    def test_duplicate_filenames_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for folder in ("train", "val"):
                (Path(tmp) / "images" / folder).mkdir(parents=True)
                (Path(tmp) / "images" / folder / "same.png").write_bytes(b"")
            with self.assertRaisesRegex(DetectError, "share the name"):
                index_enhanced(Path(tmp))

    def test_missing_delivery_explains_where_it_should_be(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(DetectError, "UWCT_ENHANCED_ROOT"):
                index_enhanced(Path(tmp))

    def _delivery(self, root, folder, *names):
        (Path(root) / "images" / folder).mkdir(parents=True, exist_ok=True)
        for n in names:
            (Path(root) / "images" / folder / n).write_bytes(b"")
        return Path(root)

    def test_all_deliveries_are_merged_and_the_zextra_one_is_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            v1 = self._delivery(Path(tmp) / "v1", "train", "a.png")
            zx = self._delivery(Path(tmp) / "zx", "zExtra", "z.png")
            both = index_all_enhanced({ENHANCED_SOURCE: v1, ENHANCED_ZEXTRA_SOURCE: zx})
            only_v1 = index_all_enhanced({ENHANCED_SOURCE: v1, ENHANCED_ZEXTRA_SOURCE: Path(tmp) / "absent"})
        self.assertEqual(sorted(both), ["a.png", "z.png"])
        self.assertEqual(sorted(only_v1), ["a.png"])

    def test_the_v1_delivery_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(DetectError, "UWCT_ENHANCED_ROOT"):
                index_all_enhanced({ENHANCED_SOURCE: Path(tmp) / "absent"})

    def test_a_name_in_two_deliveries_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            v1 = self._delivery(Path(tmp) / "v1", "train", "same.png")
            zx = self._delivery(Path(tmp) / "zx", "zExtra", "same.png")
            with self.assertRaisesRegex(DetectError, "two enhanced deliveries"):
                index_all_enhanced({ENHANCED_SOURCE: v1, ENHANCED_ZEXTRA_SOURCE: zx})

    def test_committed_records_give_each_model_its_image_type(self):
        self.assertEqual(checkpoint_image_source("E001"), RAW_SOURCE)
        self.assertEqual(checkpoint_image_source("E004"), ENHANCED_SOURCE)
        self.assertEqual(checkpoint_image_source("E027"), RAW_SOURCE)  # v4 "mixed": evaluated on raw frames
        self.assertIsNone(checkpoint_image_source(None))
        self.assertIsNone(checkpoint_image_source("E999"))

    def test_each_model_is_shown_the_image_type_of_its_records(self):
        for model in MODELS:
            self.assertEqual(checkpoint_image_source(model.experiment_id), model.image_source, model.experiment_id)


class ModelCatalogueTest(unittest.TestCase):
    """The GUI's model list and the scores it shows, read from the committed records."""

    def test_the_final_model_is_the_default(self):
        self.assertEqual(MODELS[0].experiment_id, "E027")
        self.assertEqual(len({m.experiment_id for m in MODELS}), len(MODELS))

    def test_unknown_model_is_refused_by_name(self):
        with self.assertRaisesRegex(KeyError, "E027"):
            model_by_id("E999")

    def test_scores_come_from_the_records(self):
        import json

        records = ROOT / "experiments" / "records"
        for exp in ("E001", "E004", "E027"):
            scores = recorded_scores(exp)
            run = json.loads((records / exp / "run_record.json").read_text(encoding="utf-8"))
            test = json.loads((records / exp / "final_test_record.json").read_text(encoding="utf-8"))
            self.assertEqual(scores.val, run["val_metrics"]["map50_95"])
            self.assertEqual(scores.test, test["metrics"]["ultralytics_reported"]["map50_95"])

    def test_only_e027_has_a_zenodo_sk_score(self):
        self.assertIsNotNone(recorded_scores("E027").zenodo_sk)
        for exp in ("E001", "E004"):
            self.assertIsNone(recorded_scores(exp).zenodo_sk)
            self.assertNotIn("Zenodo sk", recorded_scores(exp).caption())

    def test_a_model_without_records_shows_no_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            scores = recorded_scores("E027", Path(tmp))
        self.assertEqual((scores.val, scores.test, scores.zenodo_sk), (None, None, None))
        self.assertTrue(scores.caption().endswith("none"))


if __name__ == "__main__":
    unittest.main()
