import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.image_manifest import label_status_for, validate_label_file
from dataset_audit.scan import extract_video_id, video_id_to_session_id


class TestIds(unittest.TestCase):
    def test_underscore_frame_separator(self):
        vid = extract_video_id("06_HERR_02_CW_20200901_GP299683_00451")
        self.assertEqual(vid, "06_HERR_02_CW_20200901_GP299683")
        self.assertEqual(video_id_to_session_id(vid), "06_HERR_02_CW_20200901")

    def test_scene_frame_separator(self):
        vid = extract_video_id("16_HERR_01_CW_20220509_GP141792scene00301")
        self.assertEqual(vid, "16_HERR_01_CW_20220509_GP141792")
        self.assertEqual(video_id_to_session_id(vid), "16_HERR_01_CW_20220509")

    def test_multi_file_session_shares_session_id(self):
        a = video_id_to_session_id(extract_video_id("06_HERR_02_CW_20200901_GP299683_00001"))
        b = video_id_to_session_id(extract_video_id("06_HERR_02_CW_20200901_GP309683_00001"))
        self.assertEqual(a, b)

    def test_unparseable_name_raises(self):
        with self.assertRaises(ValueError):
            extract_video_id("no_frame_number_here")


class TestLabelValidation(unittest.TestCase):
    def _check(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.txt"
            p.write_text(text)
            return validate_label_file(p)

    def test_valid_multi_object(self):
        v = self._check("0 0.5 0.5 0.1 0.2\n0 0.2 0.3 0.05 0.05\n")
        self.assertEqual((v.n_instances, v.issues), (2, []))

    def test_empty_file_is_flagged(self):
        self.assertIn("empty_label_file", self._check("").issues)

    def test_malformed_line(self):
        self.assertIn("malformed_line_field_count_4", self._check("0 0.5 0.5 0.1\n").issues)

    def test_unexpected_class(self):
        self.assertIn("unexpected_class_id_1", self._check("1 0.5 0.5 0.1 0.1\n").issues)

    def test_out_of_range_and_outside_image(self):
        issues = self._check("0 1.2 0.5 0.1 0.1\n0 0.99 0.5 0.1 0.1\n").issues
        self.assertIn("center_out_of_range", issues)
        self.assertIn("box_extends_outside_image", issues)


class TestLabelPolicy(unittest.TestCase):
    def test_labeled_image(self):
        self.assertEqual(label_status_for("Set2000", True, []), ("positive", []))
        self.assertEqual(label_status_for("Set2000", True, ["x"])[0], "invalid")

    def test_missing_label_follows_explicit_policy(self):
        self.assertEqual(label_status_for("Set2000", False, [])[0], "negative_confirmed")
        self.assertEqual(label_status_for("zExtra", False, [])[0], "negative_pending")
        status, issues = label_status_for("Set100", False, [])
        self.assertEqual(status, "invalid")
        self.assertIn("missing_label_not_allowed_for_source", issues)

    def test_unknown_source_is_never_implicitly_negative(self):
        with self.assertRaises(ValueError):
            label_status_for("NewDelivery", False, [])


if __name__ == "__main__":
    unittest.main()
