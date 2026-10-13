import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_audit.versioning import sha256_file
from training import experiment as X
from yolo_prep import build as B

CONFIG_DIR = ROOT / "configs" / "experiments"
E001 = CONFIG_DIR / "E001_yolo_baseline.yaml"
E002 = CONFIG_DIR / "E002_yolo_imgsz1280.yaml"
E003 = CONFIG_DIR / "E003_rtdetr_baseline.yaml"
E005 = CONFIG_DIR / "E005_rtdetr_amp_off.yaml"


def valid_cfg(**overrides):
    cfg = X.load_config(E001)
    cfg["train_args"] = dict(cfg["train_args"])
    cfg.update(overrides)
    return cfg


class ParserTest(unittest.TestCase):
    def test_types(self):
        cfg = X.parse_config_text(
            "# comment\na: 1\nb: -0.5\nc: true\nd: null\ne: \"8.4.155\"\nf: 'x y'\ng: plain_v1.pt  # c\n"
            "h:\n  x: 2\n  y: false\n")
        self.assertEqual(cfg, {"a": 1, "b": -0.5, "c": True, "d": None, "e": "8.4.155", "f": "x y",
                               "g": "plain_v1.pt", "h": {"x": 2, "y": False}})

    def test_rejects_yaml_ambiguities(self):
        for value in ("yes", "No", "on", "OFF", "True", "1e-3", "010", "1_000", "8.4.155", "[1, 2]", "{a: 1}", "*ref"):
            with self.subTest(value=value), self.assertRaises(X.ConfigError):
                X.parse_config_text(f"k: {value}\n")

    def test_rejects_structure_outside_subset(self):
        for text in ("a:\n\tb: 1\n", "a:\n  b:\n    c: 1\n", "a: 1\na: 2\n", "  a: 1\n", "a:\n", "- a\n",
                     "a:\n   b: 1\n", "a:\n  b: 1\n  b: 2\n"):
            with self.subTest(text=text), self.assertRaises(X.ConfigError):
                X.parse_config_text(text)


class CommittedConfigsTest(unittest.TestCase):
    def test_e001_is_valid_and_matches_approved_settings(self):
        cfg = X.load_and_validate(E001)
        self.assertEqual((cfg["id"], cfg["parent"], cfg["model"], cfg["dataset_version"], cfg["seed"]),
                         ("E001", "none", "yolo26s.pt", "v1", 0))
        self.assertEqual(cfg["ultralytics_version"], B.TARGET_ULTRALYTICS_VERSION)
        self.assertEqual(cfg["train_args"], {"imgsz": 640, "epochs": 100, "patience": 30, "batch": 16,
                                             "deterministic": True})

    def test_e002_changes_only_imgsz(self):
        e1, e2 = X.load_and_validate(E001), X.load_and_validate(E002)
        self.assertEqual(e2["parent"], "E001")
        diff = {k for k in e1 if k in ("id", "slug", "parent", "changed") or e1[k] != e2[k]}
        self.assertEqual(diff, {"id", "slug", "parent", "changed", "train_args"})
        changed_args = {k for k in e1["train_args"] if e1["train_args"][k] != e2["train_args"][k]}
        self.assertEqual(changed_args, {"imgsz"})
        self.assertEqual(e2["train_args"]["imgsz"], 1280)

    def test_e003_changes_only_the_model(self):
        e1, e3 = X.load_and_validate(E001), X.load_and_validate(E003)
        self.assertEqual(e3["parent"], "E001")
        diff = {k for k in e1 if k in ("id", "slug", "parent", "changed") or e1[k] != e3[k]}
        self.assertEqual(diff, {"id", "slug", "parent", "changed", "model"})
        self.assertEqual(e3["model"], "rtdetr-l.pt")
        self.assertEqual(X.model_family(e3["model"]), "rtdetr")
        self.assertEqual(X.model_family(e1["model"]), "yolo")

    def test_e005_is_e003_with_amp_off(self):
        e1, e3, e5 = X.load_and_validate(E001), X.load_and_validate(E003), X.load_and_validate(E005)
        self.assertEqual(e5["parent"], "E001")  # E003 diverged and is not registered (D-025)
        diff = {k for k in e3 if k in ("id", "slug", "parent", "changed") or e3[k] != e5[k]}
        self.assertEqual(diff, {"id", "slug", "parent", "changed", "train_args"})
        self.assertEqual({**e3["train_args"], "amp": False}, e5["train_args"])
        self.assertNotIn("amp", e1["train_args"])  # E001/E003 ran with the Ultralytics default amp=True

    def test_model_family_is_required(self):
        cfg = X.load_and_validate(E003)
        for name in ("detr-l.pt", "fasterrcnn.pt", "rtdetr.pt"):
            with self.subTest(name=name), self.assertRaises(X.ConfigError):
                X.validate_config({**cfg, "model": name})

    def test_requirements_pin_matches_config(self):
        pin = (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")
        self.assertIn(f"ultralytics=={X.load_config(E001)['ultralytics_version']}", pin)


class ValidationTest(unittest.TestCase):
    def assertInvalid(self, cfg, fragment, **kw):
        with self.assertRaises(X.ConfigError) as ctx:
            X.validate_config(cfg, **kw)
        self.assertIn(fragment, str(ctx.exception))

    def test_test_split_cannot_be_selected(self):
        for key, value in (("data", "data.yaml"), ("split", "test"), ("data", "images/test"),
                           ("val", False), ("cfg", "other.yaml"), ("fraction", 0.5), ("resume", True)):
            cfg = valid_cfg()
            cfg["train_args"][key] = value
            with self.subTest(key=key):
                self.assertInvalid(cfg, "set by the runner")
        for value in ("test", "images/test", "data/yolo/lobster_v1/images/test/", "test.yaml", "C:\\x\\test\\y"):
            cfg = valid_cfg()
            cfg["train_args"]["mosaic_source"] = value
            with self.subTest(value=value):
                self.assertInvalid(cfg, "refers to the test split")
        cfg = valid_cfg()
        cfg["train_args"]["test"] = 1
        self.assertInvalid(cfg, "may not name a split")
        self.assertFalse(X.mentions_test_split("latest"))
        self.assertFalse(X.mentions_test_split("contest_v2"))

    def test_field_rules(self):
        self.assertInvalid(valid_cfg(dataset_version="v9"), "frozen dataset version")
        self.assertInvalid(valid_cfg(ultralytics_version="8.3.0"), "pinned version")
        self.assertInvalid(valid_cfg(model="weights/yolo26s.pt"), "official checkpoint name")
        self.assertInvalid(valid_cfg(seed=True), "non-negative integer")
        self.assertInvalid(valid_cfg(parent="E001"), "own parent")
        self.assertInvalid(valid_cfg(parent="E077"), "no config E077_*.yaml")
        self.assertInvalid(valid_cfg(dataset_fingerprint="abc"), "64 lowercase hex")
        self.assertInvalid(valid_cfg(extra=1), "unknown keys")
        cfg = valid_cfg()
        del cfg["train_args"]["epochs"]
        self.assertInvalid(cfg, "train_args.epochs")
        self.assertInvalid(valid_cfg(), "file name must be E001_yolo_baseline.yaml", config_file="E001_other.yaml")


class TrainArgsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.cfg = X.load_and_validate(E001)

    def tearDown(self):
        self._tmp.cleanup()

    def test_trainval_yaml_has_no_test_split(self):
        path = X.write_trainval_yaml(self.tmp / "ds", self.tmp / "data_trainval.yaml")
        self.assertEqual(X.data_yaml_keys(path), {"path", "train", "val", "names"})
        self.assertNotIn("images/test", path.read_text(encoding="utf-8"))

    def test_full_and_smoke_args(self):
        data = X.write_trainval_yaml(self.tmp / "ds", self.tmp / "data_trainval.yaml")
        full = X.build_train_args(self.cfg, data, self.tmp / "run", "train", smoke=False)
        self.assertEqual(full["data"], str(data))
        self.assertEqual((full["epochs"], full["imgsz"], full["patience"], full["seed"], full["deterministic"]),
                         (100, 640, 30, 0, True))
        self.assertEqual((full["exist_ok"], full["val"], full["save"]), (False, True, True))
        self.assertNotIn("split", full)
        self.assertNotIn("fraction", full)
        smoke = X.build_train_args(self.cfg, data, self.tmp / "run", "train", smoke=True)
        self.assertEqual((smoke["epochs"], smoke["fraction"]), (1, X.SMOKE_FRACTION))
        self.assertEqual(self.cfg["train_args"]["epochs"], 100)  # config not mutated

    def test_full_data_yaml_with_test_is_refused(self):
        full_yaml = self.tmp / "data.yaml"
        full_yaml.write_text("path: '/x'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: lobster\n")
        with self.assertRaises(X.ConfigError):
            X.build_train_args(self.cfg, full_yaml, self.tmp / "run", "train", smoke=False)
        data = X.write_trainval_yaml(self.tmp / "ds", self.tmp / "tv.yaml")
        with self.assertRaises(X.ConfigError):
            X.assert_no_test_access({"data": str(data), "split": "test"})
        with self.assertRaises(X.ConfigError):
            X.assert_no_test_access({"data": str(data), "name": "images/test"})

    def test_output_locations_outside_repo_and_raw(self):
        raw = self.tmp / "raw"
        self.assertEqual(X.check_output_location(self.tmp / "runs", ROOT, raw), (self.tmp / "runs").resolve())
        for bad in (ROOT / "runs", ROOT, raw / "x", Path("relative/runs")):
            with self.subTest(bad=bad), self.assertRaises(X.ConfigError):
                X.check_output_location(bad, ROOT, raw)


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


class BuiltDatasetTest(unittest.TestCase):
    """A tiny frozen version 'v7' in a temporary processed root, built with the real builder."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        t = Path(self._tmp.name)
        raw = t / "raw"
        imgs, lbls, extra = raw / "imgs", raw / "lbls", raw / "extra"
        for d in (imgs, lbls, extra):
            d.mkdir(parents=True)
        for name in ("pos.png", "neg.png", "vpos.png", "tpos.png"):
            (imgs / name).write_bytes(b"png-" + name.encode())
        for name in ("pos", "vpos", "tpos"):
            (lbls / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (extra / "pending.png").write_bytes(b"png-pending")
        self._saved = (B.DATASET_SOURCES, B.BIOLOGY_ROOT)
        B.DATASET_SOURCES = [("Set2000", imgs, lbls), ("zExtra", extra, None)]
        B.BIOLOGY_ROOT = raw

        self.processed = t / "processed"
        vdir = self.processed / "v7"
        vdir.mkdir(parents=True)
        _write_csv(vdir / "split_manifest.csv", [{"session_id": s, "split": sp} for s, sp in
                                                (("S1", "train"), ("S2", "val"), ("S3", "test"), ("S9", "excluded"))])
        row = lambda f, src, s, sp, st, n: {"filename": f, "source_set": src, "session_id": s, "split": sp,
                                            "label_status": st, "n_instances": n, "issues": ""}
        _write_csv(vdir / "image_manifest.csv", [
            row("pos.png", "Set2000", "S1", "train", "positive", 1),
            row("neg.png", "Set2000", "S1", "train", "negative_confirmed", 0),
            row("vpos.png", "Set2000", "S2", "val", "positive", 1),
            row("tpos.png", "Set2000", "S3", "test", "positive", 1),
            row("pending.png", "zExtra", "S9", "excluded", "negative_pending", 0),
        ])
        files = {f: sha256_file(vdir / f) for f in ("split_manifest.csv", "image_manifest.csv")}
        (self.processed / "FROZEN.json").write_text(json.dumps({"versions": {"v7": {"dir": "v7", "files": files}}}))
        self.ds = t / "yolo" / "lobster_v7"
        summary = B.build_yolo_dataset(vdir / "image_manifest.csv", vdir / "split_manifest.csv", self.ds, "copy",
                                       version="v7", version_status="frozen")
        self.cfg = {"dataset_version": "v7", "dataset_fingerprint": summary["dataset_fingerprint"],
                    "ultralytics_version": B.TARGET_ULTRALYTICS_VERSION}

    def tearDown(self):
        B.DATASET_SOURCES, B.BIOLOGY_ROOT = self._saved
        self._tmp.cleanup()

    def verify(self):
        return X.verify_built_dataset(self.ds, self.cfg, processed_root=self.processed)

    def assertRefused(self, fragment):
        with self.assertRaises(X.ConfigError) as ctx:
            self.verify()
        self.assertIn(fragment, str(ctx.exception))

    def test_valid_view_reports_train_and_val_only(self):
        info = self.verify()
        self.assertEqual(info["splits_used_for_training"], ["train", "val"])
        self.assertEqual(set(info["counts"]), {"train", "val"})
        self.assertEqual(info["counts"]["train"], {"images": 2, "positive": 1, "negative": 1, "instances": 1})

    def test_config_fingerprint_must_match(self):
        self.cfg["dataset_fingerprint"] = "0" * 64
        self.assertRefused("config dataset_fingerprint")

    def test_edited_label_is_detected(self):
        (self.ds / "labels" / "train" / "pos.txt").write_text("0 0.4 0.4 0.1 0.1\n")
        self.assertRefused("fingerprint mismatch")

    def test_stray_image_is_detected(self):
        (self.ds / "images" / "train" / "pending.png").write_bytes(b"x")
        self.assertRefused("images/train")

    def test_zextra_in_view_is_refused(self):
        with open(self.ds / "build_manifest.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows.append({**rows[0], "filename": "pending.png", "session_id": "S9", "label_status": "negative_pending"})
        _write_csv(self.ds / "build_manifest.csv", rows)
        self.assertRefused("excluded (pending labels)")

    def test_modified_frozen_manifest_is_refused(self):
        with open(self.processed / "v7" / "split_manifest.csv", "a", encoding="utf-8") as f:
            f.write("S4,train\n")
        self.assertRefused("failed verification")

    def test_draft_build_is_refused(self):
        summary = json.loads((self.ds / "build_summary.json").read_text())
        summary["version_status"] = "draft"
        (self.ds / "build_summary.json").write_text(json.dumps(summary))
        self.assertRefused("frozen versions only")


@unittest.skipUnless((ROOT / "data" / "yolo" / "lobster_v1" / "build_summary.json").is_file(),
                     "local data/yolo/lobster_v1 view not built on this machine")
class LocalV1ViewTest(unittest.TestCase):
    def test_local_v1_view_matches_e001(self):
        info = X.verify_built_dataset(ROOT / "data" / "yolo" / "lobster_v1", X.load_and_validate(E001))
        self.assertEqual(info["counts"]["train"], {"images": 1433, "positive": 1274, "negative": 159, "instances": 4000})
        self.assertEqual(info["counts"]["val"], {"images": 324, "positive": 269, "negative": 55, "instances": 1010})


class TrainRunnerGuardTest(unittest.TestCase):
    def test_refuses_windows_before_anything_else(self):
        from training import train as T
        with mock.patch("platform.system", return_value="Windows"):
            with self.assertRaises(T.TrainError) as ctx:
                T.run(E001, Path("/tmp/runs"), Path("/tmp/w"), "full", env_collect=dict)
        self.assertIn("Windows", str(ctx.exception))

    def test_refuses_without_gpu_packages(self):
        from training import train as T
        with mock.patch("platform.system", return_value="Linux"), \
                mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(T.TrainError):
                T.require_gpu_environment()


class NotebookTest(unittest.TestCase):
    def test_notebook_has_no_outputs_or_secrets_and_guards_training(self):
        import ast
        nb = json.loads((ROOT / "notebooks" / "colab_train_baseline.ipynb").read_text(encoding="utf-8"))
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertTrue(all(not c.get("outputs") and c.get("execution_count") is None
                            for c in nb["cells"] if c["cell_type"] == "code"))
        for src in code:
            ast.parse(src)
        text = "\n".join(code)
        for secret in ("ghp_", "github_pat_", "gho_"):
            self.assertNotIn(secret, text)
        # Raw data is reconstructed from the shared ZIP and includes zExtra Photos (D-014).
        # What must hold is that the raw check covers the training sources and that zExtra never
        # reaches the dataset view, training or the final test evaluation.
        self.assertIn('"--subset", "training"', text)
        for src in code:
            if any(s in src for s in ("prepare_yolo_dataset.py", "train_yolo.py", "final_test_eval.py")):
                self.assertNotIn("zExtra", src)
        for flag in ("RUN_SMOKE = False", "RUN_FULL = False", "RESUME = False"):
            self.assertIn(flag, text)


if __name__ == "__main__":
    unittest.main()
