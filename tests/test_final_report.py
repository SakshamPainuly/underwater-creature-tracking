"""Stdlib-only tests for src/reporting/final_report.py (no matplotlib needed)."""

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reporting.e001_e004 import ReportError  # noqa: E402
from reporting.final_report import (  # noqa: E402
    FIGURES,
    FINAL,
    QUOTED,
    ROUND1,
    ROUND2,
    experiments_rows,
    load_inputs,
    phase_hours,
    render_markdown,
    round1_summary,
    round2_bar,
    test_confusion,
    test_headline,
    v4_composition,
)


class FinalReportFromCommittedFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = load_inputs(ROOT)

    def test_round1_reproduces_the_d040_bar(self):
        # D-040 fixed the bar at 0.62542 = yolo26m mean 0.61127 + range 0.01415 before round 2 ran.
        summ = round1_summary(self.inputs)
        self.assertAlmostEqual(summ["yolo26m"]["mean"], 0.61127, places=5)
        self.assertAlmostEqual(summ["yolo26m"]["range"], 0.01415, places=5)
        self.assertAlmostEqual(round2_bar(self.inputs), 0.62542, places=5)
        self.assertEqual(summ["yolo26l"]["n"], 2)  # E018 not run

    def test_only_the_declared_final_model_clears_the_bar(self):
        bar = round2_bar(self.inputs)
        above = [e for row in ROUND2 for e in row[2:]
                 if self.inputs["runs"][e]["run"]["val_metrics"]["map50_95"] > bar]
        self.assertEqual(above, [FINAL])
        with open(ROOT / "experiments" / "final_candidates.csv", newline="", encoding="utf-8") as f:
            declared = [r["experiment_id"] for r in csv.DictReader(f)]
        self.assertIn(FINAL, declared)

    def test_test_numbers_are_the_records(self):
        rec = json.loads((ROOT / "experiments" / "records" / FINAL / "final_test_record.json").read_text(encoding="utf-8"))
        h = test_headline(rec)
        self.assertEqual(h["mAP50-95"], rec["metrics"]["ultralytics_reported"]["map50_95"])
        self.assertEqual(h["F1"], rec["metrics"]["at_val_conf"]["f1"])
        c = test_confusion(self.inputs["sk"])
        self.assertEqual(c["TP"] + c["FN"], self.inputs["sk"]["dataset"]["test_counts"]["instances"])

    def test_v4_composition_matches_the_training_counts(self):
        comp = v4_composition(self.inputs)
        train = sum(v for (split, _o, _k), v in comp.items() if split == "train")
        self.assertEqual(train, self.inputs["tests"][FINAL]["dataset"]["counts"]["train"]["images"])
        self.assertEqual(comp[("test", "zenodo", "original")], self.inputs["sk"]["dataset"]["test_counts"]["images"])

    def test_every_covered_run_is_in_the_export_once(self):
        rows = experiments_rows(self.inputs)
        ids = [r["experiment_id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        for runs in ROUND1.values():
            self.assertTrue(set(runs) <= set(ids))
        self.assertEqual(sum(p["runs"] for p in phase_hours(self.inputs)), len(ids))
        self.assertTrue(all(p["gpu_hours"] > 0 for p in phase_hours(self.inputs)))

    def test_markdown_is_deterministic_and_references_every_figure(self):
        md = render_markdown(self.inputs)
        self.assertEqual(md, render_markdown(self.inputs))
        for name in FIGURES:
            self.assertIn(f"figures/{name}", md)
        for heading in ("## 4. How the model was tuned", "## 6. The GUI model: before and after", "## 7. Compute"):
            self.assertIn(heading, md)
        self.assertNotRegex(md, r"\d\s?(cm|mm)\b|\d\s(m|m/s)\b")  # rule 9: no physical units ("yolo26m" is fine)

    def test_lobster_only_copy_leaves_out_the_other_species(self):
        md = render_markdown(self.inputs, lobster_only=True)
        rows = experiments_rows(self.inputs, lobster_only=True)
        self.assertNotIn("tritonia", md.lower())
        self.assertNotIn("Tritonia", {r["group"] for r in rows})
        self.assertNotIn("## 10.", md)
        phases = phase_hours(self.inputs, lobster_only=True)
        self.assertEqual(sum(p["runs"] for p in phases), len(rows))
        self.assertIn(f"**{sum(p['gpu_hours'] for p in phases):.2f}**", md)  # total recomputed, not edited
        self.assertEqual(render_markdown(self.inputs), render_markdown(self.inputs, lobster_only=False))
        for name in FIGURES:
            self.assertIn(f"figures/{name}", md)

    def test_quoted_numbers_are_still_in_their_sources(self):
        for key, quote in QUOTED.items():
            path, text = quote["check"]
            self.assertIn(text, (ROOT / path).read_text(encoding="utf-8"), f"{key}: source no longer says it")


class FinalReportRefusals(unittest.TestCase):
    def test_missing_final_test_record_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "experiments", root / "experiments")
            (root / "data" / "processed" / "v4").mkdir(parents=True)
            shutil.copy(ROOT / "data" / "processed" / "v4" / "mixed_manifest.csv", root / "data" / "processed" / "v4")
            (root / "experiments" / "records" / FINAL / "final_test_zenodo_sk_record.json").unlink()
            with self.assertRaisesRegex(ReportError, "final_test_zenodo_sk_record.json"):
                load_inputs(root)


if __name__ == "__main__":
    unittest.main()
