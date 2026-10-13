"""Stdlib-only tests for src/reporting/e001_e004.py (no matplotlib needed)."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reporting.e001_e004 import (  # noqa: E402
    FIGURES,
    ReportError,
    confusion_counts,
    headline,
    load_records,
    render_markdown,
)

RECORDS = ROOT / "experiments" / "records"


class ReportFromCommittedRecords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = load_records(RECORDS)

    def test_headline_test_matches_final_test_record(self):
        for exp in ("E001", "E004"):
            rec = json.loads((RECORDS / exp / "final_test_record.json").read_text(encoding="utf-8"))
            h = headline(self.recs[exp], "test")
            self.assertEqual(h["mAP50"], rec["metrics"]["ultralytics_reported"]["map50"])
            self.assertEqual(h["F1"], rec["metrics"]["at_val_conf"]["f1"])
            self.assertEqual(h["conf"], rec["metrics"]["at_val_conf"]["conf"])

    def test_val_confusion_counts_match_overall(self):
        for exp in ("E001", "E004"):
            c = confusion_counts(self.recs[exp], "val")
            o = self.recs[exp]["val"]["overall"]
            self.assertEqual((c["TP"], c["FP"], c["FN"]), (o["tp"], o["fp"], o["fn"]))

    def test_markdown_is_deterministic_and_references_every_figure(self):
        md = render_markdown(self.recs)
        self.assertEqual(md, render_markdown(self.recs))
        for name in FIGURES:
            self.assertIn(f"figures/{name}", md)
        self.assertIn("## 7. Limitations", md)


class ReportRefusals(unittest.TestCase):
    def test_missing_record_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            for exp in ("E001", "E004"):
                shutil.copytree(RECORDS / exp, Path(tmp) / exp)
            (Path(tmp) / "E004" / "final_test_record.json").unlink()
            with self.assertRaisesRegex(ReportError, "final_test_record.json"):
                load_records(Path(tmp))

    def test_record_for_wrong_experiment_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for exp in ("E001", "E004"):
                shutil.copytree(RECORDS / exp, Path(tmp) / exp)
            shutil.copy(RECORDS / "E001" / "val_eval_record.json", Path(tmp) / "E004" / "val_eval_record.json")
            with self.assertRaisesRegex(ReportError, "expected E004"):
                load_records(Path(tmp))


if __name__ == "__main__":
    unittest.main()
