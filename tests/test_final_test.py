import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from dataset_audit.versioning import sha256_file
from training import experiment as X
from training import final_test as F
from test_run_record import make_record


class DeclarationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "final_candidates.csv"

    def tearDown(self):
        self._tmp.cleanup()

    def test_committed_file_has_expected_header(self):
        header = F.FINAL_CANDIDATES_PATH.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(tuple(header.split(",")), F.FINAL_CANDIDATE_FIELDS)

    def test_undeclared_experiment_is_refused(self):
        self.path.write_text(",".join(F.FINAL_CANDIDATE_FIELDS) + "\n", encoding="utf-8")
        with self.assertRaises(F.FinalTestError):
            F.require_declared("E001", self.path)
        with self.assertRaises(F.FinalTestError):
            F.require_declared("E001", self.path.with_name("missing.csv"))

    def test_declared_experiment_is_accepted(self):
        self.path.write_text(",".join(F.FINAL_CANDIDATE_FIELDS) + "\nE001,2026-10-01T00:00:00+00:00,D-013,final baseline\n",
                             encoding="utf-8")
        self.assertEqual(F.require_declared("E001", self.path)["decision"], "D-013")
        self.path.write_text(",".join(F.FINAL_CANDIDATE_FIELDS) + "\nE001,,,\n", encoding="utf-8")
        with self.assertRaises(F.FinalTestError):
            F.require_declared("E001", self.path)

    def test_bad_header_is_refused(self):
        self.path.write_text("experiment_id\nE001\n", encoding="utf-8")
        with self.assertRaises(F.FinalTestError):
            F.declared_candidates(self.path)


class RunReadinessTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run = Path(self._tmp.name) / "E001"
        (self.run / "train" / "weights").mkdir(parents=True)
        self.best = self.run / "train" / "weights" / "best.pt"
        self.best.write_bytes(b"fake weights")
        self.record = make_record()
        self.record["outputs"]["best"]["sha256"] = sha256_file(self.best)
        self.write(self.record)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, record):
        (self.run / "run_record.json").write_text(json.dumps(record), encoding="utf-8")

    def test_finished_full_run_is_accepted(self):
        self.assertEqual(F.check_run_for_final_test(self.run)["run_name"], "E001")

    def test_refusals(self):
        smoke = copy.deepcopy(self.record)
        smoke["kind"] = "smoke"
        self.write(smoke)
        with self.assertRaises(F.FinalTestError):
            F.check_run_for_final_test(self.run)
        self.write(self.record)
        self.best.write_bytes(b"other weights")
        with self.assertRaises(F.FinalTestError):
            F.check_run_for_final_test(self.run)
        with self.assertRaises(F.FinalTestError):
            F.check_run_for_final_test(self.run.with_name("E999"))

    def test_second_final_test_is_refused(self):
        (self.run / F.FINAL_DIR).mkdir()
        (self.run / F.FINAL_DIR / F.FINAL_RECORD).write_text("{}", encoding="utf-8")
        with self.assertRaises(F.FinalTestError):
            F.check_run_for_final_test(self.run)


class DataFileTest(unittest.TestCase):
    def test_final_test_needs_full_data_yaml_and_training_refuses_it(self):
        with tempfile.TemporaryDirectory() as t:
            full = Path(t) / "data.yaml"
            full.write_text("path: '/x'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: lobster\n")
            trainval = X.write_trainval_yaml(Path(t), Path(t) / "data_trainval.yaml")
            F.assert_full_data_yaml(full)
            with self.assertRaises(F.FinalTestError):
                F.assert_full_data_yaml(trainval)
            with self.assertRaises(X.ConfigError):
                X.assert_trainval_only(full)


class OperatingPointTest(unittest.TestCase):
    def test_conf_from_f1_curve_and_values_at_conf(self):
        px = [0.0, 0.25, 0.5, 0.75, 1.0]
        f1 = [[0.1, 0.4, 0.7, 0.5, 0.0]]
        self.assertEqual(F.operating_conf(px, f1), 0.5)
        self.assertEqual(F.operating_conf(px, f1[0]), 0.5)  # flat curve
        self.assertEqual(F.value_at(px, [[0.2, 0.4, 0.6, 0.8, 1.0]], 0.49), 0.6)
        with self.assertRaises(F.FinalTestError):
            F.operating_conf(px, [[0.1, 0.2]])

    def test_prediction_examples_one_per_test_session(self):
        rows = [{"split": s, "session_id": sid, "filename": f} for s, sid, f in (
            ("test", "B", "b3.png"), ("test", "B", "b1.png"), ("test", "B", "b2.png"),
            ("test", "A", "a1.png"), ("val", "V", "v1.png"), ("train", "T", "t1.png"))]
        self.assertEqual(F.select_prediction_examples(rows), ["a1.png", "b2.png"])

    def test_final_record_requires_metrics(self):
        kw = dict(run_record=make_record(), declaration={"experiment_id": "E001"}, git={"commit": "a" * 40},
                  dataset={"fingerprint": "f" * 64}, test_counts={}, conf={"value": 0.4}, artifacts=[],
                  attempts=1, environment={})
        rec = F.build_final_record(metrics={"ultralytics_reported": {"precision": 1.0, "recall": 1.0, "map50": 1.0,
                                                                     "map50_95": 1.0}}, **kw)
        self.assertEqual((rec["label"], rec["split"]), ("FINAL TEST EVALUATION", "test"))
        with self.assertRaises(F.FinalTestError):
            F.build_final_record(metrics={"ultralytics_reported": {}}, **kw)


class IsolationTest(unittest.TestCase):
    """The test split is reachable from final_test.py only; training never imports it."""

    SOURCES = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))

    def test_only_final_test_passes_the_test_split(self):
        offenders = []
        for path in self.SOURCES:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                for kw in node.keywords:
                    if kw.arg != "split":
                        continue
                    value = ast.unparse(kw.value)
                    rel = path.relative_to(ROOT).as_posix()
                    if value in ("X.VAL_SPLIT",):
                        continue
                    if rel == "src/training/final_test.py" and value == "TEST_SPLIT":
                        continue
                    offenders.append(f"{rel}: split={value}")
        self.assertEqual(offenders, [])
        self.assertEqual(F.TEST_SPLIT, "test")

    def test_training_code_does_not_import_final_test(self):
        for rel in ("src/training/train.py", "src/training/experiment.py", "src/training/run_record.py",
                    "scripts/train_yolo.py", "scripts/register_run.py"):
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
            names = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
            modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
            self.assertFalse({"final_test", "training.final_test"} & (names | modules), rel)

    def test_final_test_refuses_on_windows(self):
        with mock.patch("platform.system", return_value="Windows"):
            from training.train import TrainError
            with self.assertRaises(TrainError):
                F.run_final_test(Path("/tmp/runs"), "E001", env_collect=dict)

    def test_cli_requires_explicit_confirmation(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import final_test_eval
        with mock.patch.object(sys, "argv", ["final_test_eval.py", "--experiment", "E001", "--runs-root", "/tmp/r"]), \
                mock.patch("sys.stderr"), mock.patch.object(final_test_eval, "run_final_test") as run:
            with self.assertRaises(SystemExit) as ctx:
                final_test_eval.main()
        self.assertEqual(ctx.exception.code, 2)
        run.assert_not_called()


class NotebookAuthAndFinalTestTest(unittest.TestCase):
    def setUp(self):
        nb = json.loads((ROOT / "notebooks" / "colab_train_baseline.ipynb").read_text(encoding="utf-8"))
        self.code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        self.text = "\n".join(self.code)

    def test_auth_uses_askpass_and_never_puts_token_in_arguments(self):
        for forbidden in ("extraHeader", "extraheader", "Authorization", "base64", "x-access-token:{",
                          "@github.com", "credential.helper=store", "git config --global"):
            self.assertNotIn(forbidden, self.text, forbidden)
        self.assertIn('"GIT_ASKPASS": ASKPASS', self.text)
        self.assertIn('"-c", "credential.helper="', self.text)
        self.assertEqual(self.text.count("userdata.get("), 1)  # read once, inside git_remote()
        git_remote = self.text.split("def git_remote", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("userdata.get(TOKEN_SECRET_NAME)", git_remote)
        self.assertIn("env=env", git_remote)
        self.assertIn('git_remote("clone"', self.text)
        self.assertIn('git_remote("fetch"', self.text)

    def test_final_test_cells_are_guarded_and_labelled(self):
        final_cells = [c for c in self.code if "final_test_eval.py" in c]
        self.assertEqual(len(final_cells), 1)
        self.assertIn("RUN_FINAL_TEST = False", final_cells[0])
        self.assertIn("--confirm-final-test", final_cells[0])
        self.assertIn("FINAL TEST EVALUATION", final_cells[0])
        # no other cell evaluates on test or points training at a test data file
        for c in self.code:
            if c is not final_cells[0]:
                self.assertNotIn('split="test"', c)
                self.assertNotIn("--confirm-final-test", c)


if __name__ == "__main__":
    unittest.main()
