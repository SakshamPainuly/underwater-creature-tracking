import copy
import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training import run_record as R


def make_record(kind="full"):
    return R.build_record(
        kind=kind,
        experiment={"id": "E001", "slug": "yolo_baseline", "parent": "none", "changed": "initial",
                    "config_path": "configs/experiments/E001_yolo_baseline.yaml", "config_sha256": "c" * 64,
                    "config": {"seed": 0}},
        run_name="E001", run_dir=Path("/drive/runs/E001"),
        git={"commit": "a" * 40, "branch": "p4/colab-baseline", "dirty": False},
        dataset={"version": "v1", "status": "frozen", "fingerprint": "f" * 64,
                 "splits_used_for_training": ["train", "val"]},
        model={"name": "yolo26s.pt", "checkpoint_path": "/content/weights/yolo26s.pt", "checkpoint_sha256": "b" * 64},
        ultralytics_version="8.4.155", train_args={"imgsz": 640, "epochs": 100, "seed": 0},
        ultralytics_args=None,
        environment={"torch": "2.x", "cuda_version": "12.x", "gpus": [{"name": "Tesla T4", "vram_gb": 15.0}]},
        launches=R.launch_summary([]), epochs_completed=57,
        outputs={"best": {"path": "/drive/runs/E001/train/weights/best.pt", "size": 1, "sha256": "d" * 64}},
        val_metrics={"split": "val", "precision": 0.5, "recall": 0.5, "map50": 0.5, "map50_95": 0.3},
    )


class RunRecordTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.registry = Path(self._tmp.name) / "registry.csv"

    def tearDown(self):
        self._tmp.cleanup()

    def test_record_contains_required_provenance(self):
        rec = make_record()
        for key in ("git", "dataset", "model", "ultralytics_version", "seed", "train_args", "environment",
                    "duration", "outputs", "val_metrics", "limitations"):
            self.assertIn(key, rec)
        self.assertIsNone(rec["test_metrics"])
        self.assertEqual(R.validate_record(rec), [])

    def test_incomplete_or_test_records_are_rejected(self):
        rec = make_record()
        for mutate, fragment in (
            (lambda r: r.update(test_metrics={"map50": 0.9}), "test metrics"),
            (lambda r: r["val_metrics"].update(split="test"), "val metrics"),
            (lambda r: r["val_metrics"].pop("map50"), "val metrics"),
            (lambda r: r["git"].update(commit=None), "git commit"),
            (lambda r: r["dataset"].update(status="draft"), "frozen"),
            (lambda r: r["dataset"].update(splits_used_for_training=["train", "val", "test"]), "splits_used"),
            (lambda r: r["outputs"].pop("best"), "best checkpoint"),
            (lambda r: r["model"].pop("checkpoint_sha256"), "pretrained checkpoint"),
        ):
            bad = copy.deepcopy(rec)
            mutate(bad)
            self.assertTrue(any(fragment in p for p in R.validate_record(bad)), fragment)

    def test_registry_append_and_refusals(self):
        row = R.append_to_registry(self.registry, make_record())
        self.assertEqual(row["val_map50"], 0.5)
        self.assertEqual(row["gpu"], "Tesla T4")
        with open(self.registry, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(tuple(rows[0]), R.REGISTRY_FIELDS)
        self.assertEqual(R.registered_experiments(self.registry), {"E001"})
        with self.assertRaises(R.RecordError):
            R.append_to_registry(self.registry, make_record())  # duplicate
        smoke = make_record(kind="smoke")
        smoke["experiment"]["id"], smoke["run_name"] = "E009", "E009_smoke"
        with self.assertRaises(R.RecordError):
            R.append_to_registry(self.registry, smoke)

    def test_registry_header_mismatch_is_refused(self):
        self.registry.write_text("experiment_id,run_name\n", encoding="utf-8")
        with self.assertRaises(R.RecordError):
            R.append_to_registry(self.registry, make_record())

    def test_committed_registry_has_the_expected_header(self):
        with open(ROOT / "experiments" / "registry.csv", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertEqual(tuple(header), R.REGISTRY_FIELDS)

    def test_launch_summary_marks_interruptions(self):
        s = R.launch_summary([
            {"event": "start", "mode": "full", "time": "2026-09-18T10:00:00+00:00"},
            {"event": "start", "mode": "resume", "time": "2026-09-18T12:00:00+00:00"},
            {"event": "end", "mode": "resume", "time": "2026-09-18T12:30:00+00:00"},
        ])
        self.assertEqual([l["outcome"] for l in s["launches"]], ["interrupted", "completed"])
        self.assertEqual(s["train_seconds_recorded"], 1800)


if __name__ == "__main__":
    unittest.main()
