import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit import leakage as L

GROUPS = {
    ("ours", "S_TRAIN"): "train",
    ("ours", "S_TRAIN2"): "train",
    ("ours", "S_VAL"): "val",
    ("ours", "S_TEST"): "test",
    ("zenodo", "Z_TRAIN"): "train",
    ("zenodo", "Z_HOLD"): "test",
}


def row(image_id, kind="original", split="train", group="S_TRAIN", origin="ours", ancestors="", **extra):
    return {"image_id": image_id, "origin": origin, "kind": kind, "group_id": group, "split": split,
            "ancestors": ancestors, **extra}


def originals():
    return [
        row("tr1"), row("tr2", group="S_TRAIN2"),
        row("va1", split="val", group="S_VAL"),
        row("te1", split="test", group="S_TEST"),
        row("te_x", split="excluded", group="S_TEST"),  # excluded by the label review, still held out
        row("z1", origin="zenodo", group="Z_TRAIN"),
        row("zh", split="test", origin="zenodo", group="Z_HOLD"),
    ]


class ManifestRulesTest(unittest.TestCase):
    def check(self, extra_rows):
        return L.check_manifest(originals() + extra_rows, GROUPS)

    def test_clean_manifest_passes(self):
        derived = [
            row("c1", kind="colour_change", ancestors="tr1"),
            row("p1", kind="copy_paste", ancestors="tr1;tr2"),
            row("d1", kind="diffusion", ancestors="tr2", group="S_TRAIN2"),
            row("zp", kind="copy_paste", origin="zenodo", group="Z_TRAIN", ancestors="z1"),
        ]
        self.assertEqual(self.check(derived), [])

    def test_pasted_lobster_from_test_session_is_leakage(self):
        # The background is train, but one pasted lobster comes from a test session.
        errors = self.check([row("p1", kind="copy_paste", ancestors="tr1;te1")])
        self.assertEqual(len(errors), 1)
        self.assertIn("held-out group(s) ours:S_TEST (test)", errors[0])

    def test_excluded_original_from_test_session_still_counts_as_held_out(self):
        errors = self.check([row("d1", kind="diffusion", ancestors="te_x")])
        self.assertTrue(any("held-out" in e for e in errors))

    def test_derived_image_from_held_out_session_may_stay_excluded(self):
        self.assertEqual(self.check([row("d1", kind="diffusion", split="excluded", ancestors="va1")]), [])

    def test_zenodo_holdout_protects_its_derived_images(self):
        errors = self.check([row("zc", kind="colour_change", origin="zenodo", group="Z_TRAIN", ancestors="zh")])
        self.assertTrue(any("zenodo:Z_HOLD" in e for e in errors))

    def test_derived_image_never_in_val_or_test(self):
        for split in ("val", "test"):
            errors = self.check([row("c1", kind="colour_change", split=split, ancestors="tr1")])
            self.assertTrue(any("originals only" in e for e in errors), split)

    def test_derived_image_needs_original_ancestors(self):
        self.assertTrue(any("no ancestors" in e for e in self.check([row("c1", kind="colour_change")])))
        self.assertTrue(any("not in the manifest" in e
                            for e in self.check([row("c1", kind="colour_change", ancestors="nope")])))
        chained = [row("c1", kind="colour_change", ancestors="tr1"), row("c2", kind="colour_change", ancestors="c1")]
        self.assertTrue(any("not an original" in e for e in self.check(chained)))

    def test_original_must_follow_its_group(self):
        errors = self.check([row("bad", split="train", group="S_TEST")])
        self.assertTrue(any("its group S_TEST is test" in e for e in errors))
        self.assertTrue(any("not in the group table" in e for e in self.check([row("u", group="S_UNKNOWN")])))
        self.assertTrue(any("must not list ancestors" in e for e in self.check([row("o", ancestors="tr1")])))

    def test_vocabulary_and_unique_ids(self):
        self.assertTrue(any("unknown kind" in e for e in self.check([row("x", kind="gan")])))
        self.assertTrue(any("appears 2 times" in e for e in self.check([row("tr1")])))

    def test_declared_heldout_must_agree(self):
        ok = [row("p1", kind="copy_paste", split="excluded", ancestors="tr1;te1", declared_heldout="yes"),
              row("c1", kind="colour_change", ancestors="tr1", declared_heldout="no")]
        self.assertEqual(self.check(ok), [])
        wrong = [row("c1", kind="colour_change", ancestors="tr1", declared_heldout="yes")]
        self.assertTrue(any("delivery says held-out=yes, computed no" in e for e in self.check(wrong)))

    def test_errors_are_deterministic(self):
        bad = [row("p1", kind="copy_paste", ancestors="te1;va1"), row("c1", kind="colour_change", split="val",
                                                                      ancestors="tr1")]
        self.assertEqual(self.check(bad), self.check(list(reversed(bad))))

    def test_groups_from_split_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "split_manifest.csv"
            p.write_text("session_id,source_set,split\nA,Set2000,train\nB,Set2000,test\n", encoding="utf-8")
            self.assertEqual(L.groups_from_split_manifest(p), {("ours", "A"): "train", ("ours", "B"): "test"})


class DuplicateRulesTest(unittest.TestCase):
    def test_identical_file_across_splits(self):
        rows = originals()
        errors = L.exact_duplicates(rows, {"tr1": "aa", "te1": "aa", "va1": "bb", "te_x": "bb"})
        self.assertEqual(len(errors), 1)  # te_x is excluded, so its copy of va1 does not count
        self.assertIn("tr1 (train)", errors[0])
        self.assertIn("te1 (test)", errors[0])

    def test_near_duplicates_across_splits_only(self):
        rows = originals()
        hashes = {"tr1": 0b1111, "tr2": 0b1111, "va1": 0b0111, "te1": 0xFFFF0000}
        errors = L.near_duplicates(rows, hashes, max_distance=1)
        self.assertEqual(errors, [
            "near-duplicate (dHash distance 1): tr1 (train) ~ va1 (val)",
            "near-duplicate (dHash distance 1): tr2 (train) ~ va1 (val)",
        ])

    def test_dhash_survives_a_colour_change(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        with tempfile.TemporaryDirectory() as t:
            a, b, c = (Path(t) / n for n in ("a.png", "b.png", "c.png"))
            img = Image.new("RGB", (90, 80))
            img.putdata([((x * 3) % 256, (y * 5) % 256, (x * y) % 256) for y in range(80) for x in range(90)])
            img.save(a)
            img.point(lambda v: int(v * 0.7) + 30).save(b)  # a uniform "water colour" change
            img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(c)
            ha, hb, hc = L.dhash(a), L.dhash(b), L.dhash(c)
            self.assertLessEqual((ha ^ hb).bit_count(), 4)
            self.assertGreater((ha ^ hc).bit_count(), 4)

    def test_image_paths_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "a.png").write_bytes(b"x")
            rows = [row("tr1", path="a.png"), row("te1", split="test", group="S_TEST", path="gone.png")]
            with self.assertRaises(L.LeakageError):
                L.image_paths(rows, Path(t))
            self.assertEqual(set(L.image_paths(rows[:1], Path(t))), {"tr1"})


if __name__ == "__main__":
    unittest.main()
