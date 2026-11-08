import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit import leakage as L
from dataset_audit import model_a as M

BOX = "0 0.5 0.5 0.1 0.1\n"
BOX2 = "0 0.2 0.2 0.1 0.1\n"
Z_MPA = "kso__mpa_cam1_110822_rec_001_frame_0001_jpg.rf.aa"
Z_SK = "kso__sk_cam2_120824_rec001_frame_0002_jpg.rf.bb"
Z_REC = "kso__REC_004_frame_0037_jpeg.rf.cc"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["derived_image", "folder"])
        w.writeheader()
        w.writerows(rows)


def put(root, folder, stem, ext, label):
    for sub, name, text in (("images", f"{stem}{ext}", "img"), ("labels", f"{stem}.txt", label)):
        p = root / folder / sub / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def v1(name, session, split, status):
    return {"filename": f"{name}.png", "source_set": "Set2000", "session_id": session, "split": split,
            "label_status": status, "n_instances": "1" if status == "positive" else "0"}


class Fixture:
    """A tiny copy of the four raw roots: biology, zip 1, zip 2 and the colour redo."""

    def __init__(self, tmp: Path):
        self.bio = tmp / "biology"
        self.roots = {k: tmp / k for k in ("harsh_model_a_2026-11", "harsh_v1_extra_2026-11", "harsh_v1_colour_2026-11")}
        self.img_dir, self.lbl_dir = self.bio / "img", self.bio / "lbl"
        self.img_dir.mkdir(parents=True)
        self.lbl_dir.mkdir(parents=True)
        self.v1 = [v1("tr_pos", "S_TR", "train", "positive"), v1("tr_neg", "S_TR", "train", "negative_confirmed"),
                   v1("tr_low", "S_TR", "train", "negative_confirmed"), v1("va_pos", "S_VA", "val", "positive"),
                   v1("va_neg", "S_VA", "val", "negative_confirmed"), v1("te_pos", "S_TE", "test", "positive"),
                   {"filename": "zx.png", "source_set": "zExtra", "session_id": "S_ZX", "split": "excluded",
                    "label_status": "negative_pending", "n_instances": "0"}]
        for r in self.v1:
            (self.img_dir / r["filename"]).write_text("img", encoding="utf-8")
            if r["label_status"] == "positive":
                (self.lbl_dir / r["filename"].replace(".png", ".txt")).write_text(BOX, encoding="utf-8")
        za = self.roots["harsh_model_a_2026-11"]
        write_csv(za / "label_review_verdicts.csv", [
            {"image": "tr_low", "outcome_in_canonical_v2": "kept as empty frame", "confidence": "low"},
            {"image": "va_neg", "outcome_in_canonical_v2": "excluded (needs boxes)", "confidence": "medium"}])
        put(za, M.ZENODO_FOLDER, Z_MPA, ".jpg", BOX)
        put(za, M.ZENODO_FOLDER, Z_SK, ".jpg", BOX)
        put(za, M.ZENODO_FOLDER, Z_REC, ".jpg", "")
        for base, folder, _o, _k in M.DERIVED_FOLDERS:
            for sub in ("images", "labels"):
                (self.roots[base] / folder / sub).mkdir(parents=True, exist_ok=True)
        self.index = {k: [] for k in self.roots}
        self.pastes = {k: [] for k in self.roots}

    def derived(self, base, folder, stem, label, base_image, pasted=(), ext=".png", **extra):
        put(self.roots[base], folder, stem, ext, label)
        self.index[base].append({"derived_image": stem, "folder": folder, "base_image": base_image,
                                 "pasted_lobsters": str(len(pasted)), **extra})
        for n, src in enumerate(pasted, 1):
            self.pastes[base].append({"derived_image": stem, "paste_no": str(n), "source_image": src})

    def build(self):
        for k, root in self.roots.items():
            write_csv(root / "derived_images_index.csv", self.index[k])
            write_csv(root / "pasted_lobsters_index.csv", self.pastes[k])
        dirs = {"Set2000": (self.img_dir, self.lbl_dir), "zExtra": (self.img_dir, None)}
        return M.build(self.v1, dirs, self.bio, self.roots)


DIFF = "harsh_model_a_2026-11", "5 Diffusion - DM_underwater (train only)"
PASTE = "harsh_model_a_2026-11", "3 Synthetic - copy-paste/ours"
COLOUR_Z = "harsh_v1_colour_2026-11", "zenodo"


class BuildTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def by_id(self, rows):
        return {r["image_id"]: r for r in rows}

    def test_splits_and_exclusions(self):
        fx = self.fx
        fx.derived(*DIFF, "tr_pos", BOX, "tr_pos", uses_uwct_v1_val_or_test_session="no")
        fx.derived(*DIFF, "va_pos", BOX, "va_pos", uses_uwct_v1_val_or_test_session="yes")
        fx.derived(*DIFF, "tr_low", "", "tr_low", uses_uwct_v1_val_or_test_session="no")
        fx.derived(*PASTE, "paste_0", BOX + BOX2, "tr_pos", pasted=["tr_pos"], uses_uwct_v1_val_or_test_session="no")
        fx.derived(*PASTE, "paste_1", BOX2, "tr_neg", pasted=["va_pos"], uses_uwct_v1_val_or_test_session="yes")
        fx.derived(*COLOUR_Z, "water__" + Z_SK, BOX, Z_SK, ext=".jpg")
        fx.derived(*COLOUR_Z, "water__" + Z_MPA, BOX, Z_MPA, ext=".jpg")
        rows, groups, report = fx.build()
        r = self.by_id(rows)
        split = {k: (v["split"], v["exclude_reason"]) for k, v in r.items()}
        self.assertEqual(split["ours/original/tr_pos.png"], ("train", ""))
        self.assertEqual(split["ours/original/tr_low.png"], ("excluded", "label_review_low_confidence_empty"))
        self.assertEqual(split["ours/original/va_neg.png"], ("val", ""))  # val keeps every v1 frame
        self.assertNotIn("ours/original/zx.png", r)
        self.assertEqual(split[f"zenodo/original/{Z_MPA}.jpg"], ("train", ""))
        self.assertEqual(split[f"zenodo/original/{Z_SK}.jpg"], ("test", ""))
        self.assertEqual(split[f"zenodo/original/{Z_REC}.jpg"], ("excluded", "zenodo_no_site"))
        self.assertEqual(split["ours/diffusion/tr_pos.png"], ("train", ""))
        self.assertEqual(split["ours/diffusion/va_pos.png"], ("excluded", "ancestor_not_train"))
        self.assertEqual(split["ours/diffusion/tr_low.png"], ("excluded", "ancestor_label_excluded"))
        self.assertEqual(split["ours/copy_paste/paste_0.png"], ("train", ""))
        self.assertEqual(split["ours/copy_paste/paste_1.png"], ("excluded", "ancestor_not_train"))
        self.assertEqual(r["ours/copy_paste/paste_1.png"]["ancestors"],
                         "ours/original/tr_neg.png;ours/original/va_pos.png")
        self.assertEqual(split[f"zenodo/colour_change/water__{Z_SK}.jpg"], ("excluded", "ancestor_not_train"))
        self.assertEqual(split[f"zenodo/colour_change/water__{Z_MPA}.jpg"], ("train", ""))
        self.assertEqual(r["ours/original/tr_pos.png"]["path"], "biology/img/tr_pos.png")
        self.assertEqual(r["ours/original/tr_neg.png"]["label_path"], "")
        self.assertIn({"origin": "zenodo", "group_id": "sk_cam2_120824", "split": "test"}, groups)
        # the output passes the leakage rules with v1-style groups for ours
        table = {("ours", s): sp for s, sp in (("S_TR", "train"), ("S_VA", "val"), ("S_TE", "test"))}
        table.update({("zenodo", g["group_id"]): g["split"] for g in groups})
        self.assertEqual(L.check_manifest(rows, table), [])
        self.assertEqual(report["excluded_by_reason"]["ancestor_not_train"], 3)

    def test_diffusion_label_must_equal_original(self):
        self.fx.derived(*DIFF, "tr_pos", BOX2, "tr_pos")
        with self.assertRaisesRegex(M.ModelAError, "differ from the original"):
            self.fx.build()

    def test_copy_paste_needs_background_plus_pasted_boxes(self):
        self.fx.derived(*PASTE, "paste_0", BOX, "tr_pos", pasted=["tr_pos"])
        with self.assertRaisesRegex(M.ModelAError, "expected background 1 \\+ pasted 1"):
            self.fx.build()

    def test_unlisted_file_is_refused(self):
        put(self.fx.roots[DIFF[0]], DIFF[1], "stray", ".png", "")
        with self.assertRaisesRegex(M.ModelAError, "unlisted"):
            self.fx.build()

    def test_unknown_ancestor_is_refused(self):
        self.fx.derived(*DIFF, "nope", "", "not_an_original")
        with self.assertRaisesRegex(M.ModelAError, "not among the originals"):
            self.fx.build()


class HelpersTest(unittest.TestCase):
    def test_zenodo_group(self):
        self.assertEqual(M.zenodo_group(Z_MPA + ".jpg"), ("mpa", "mpa_cam1_110822"))
        self.assertEqual(M.zenodo_group(Z_SK + ".jpg"), ("sk", "sk_cam2_120824"))
        self.assertEqual(M.zenodo_group(Z_REC + ".jpg"), ("", "REC"))

    def test_same_boxes_ignores_order_and_rounding(self):
        a = [(0, 0.5, 0.5, 0.1, 0.1), (0, 0.2, 0.2, 0.1, 0.1)]
        b = [(0, 0.2, 0.2, 0.1, 0.1), (0, 0.5000004, 0.5, 0.1, 0.1)]
        self.assertTrue(M.same_boxes(a, b))
        self.assertFalse(M.same_boxes(a, b[:1]))
        self.assertTrue(M.contains_boxes(a, b[:1]))

    def test_read_label_rejects_bad_lines(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.txt"
            for bad in ("1 0.5 0.5 0.1 0.1", "0 0.5 0.5 0.1", "0 1.5 0.5 0.1 0.1", "0 a b c d"):
                p.write_text(bad + "\n", encoding="utf-8")
                with self.assertRaises(M.ModelAError):
                    M.read_label(p)
            p.write_text("\n" + BOX, encoding="utf-8")
            self.assertEqual(len(M.read_label(p)), 1)


if __name__ == "__main__":
    unittest.main()
