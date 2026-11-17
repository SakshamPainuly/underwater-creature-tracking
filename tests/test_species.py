"""CPU tests for the GUI's species file, folder sources and owner-declared settings (stdlib only)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detect import (  # noqa: E402
    DEFAULT_CONF,
    DetectError,
    list_folder_images,
    load_species,
    parse_species,
    sweet_spot_for_sha256,
)

LOBSTER = {"key": "lobster", "label": "Lobster", "models": [{"id": "E027"}]}
OTHER = {"key": "slug", "label": "Sea slug", "animal": "sea slug",
         "models": [{"id": "S1", "label": "S1", "weights": "weights/slug/best.pt", "description": "d",
                     "declared_conf": 0.16, "declared_imgsz": 640, "declared_source": "owner's val run"}],
         "sources": [{"label": "frames", "kind": "folder", "path": "samples/slug/images", "held_out": True}],
         "limitations": ["only one limitation"]}


class SpeciesFileTest(unittest.TestCase):
    def test_committed_files_parse_and_lobster_comes_first(self):
        for name in ("gui_species.json", "gui_species_bundle.json"):
            species = load_species(ROOT / "configs" / name)
            self.assertEqual([s.key for s in species], ["lobster"], name)
            self.assertEqual(species[0].models[0].experiment_id, "E027", name)  # final model is the default
            self.assertTrue(species[0].limitations, name)
            self.assertNotIn("tritonia", json.dumps([s.key for s in species]).lower(), name)  # sent separately

    def test_bundle_file_uses_folder_sources_only(self):
        for s in load_species(ROOT / "configs" / "gui_species_bundle.json"):
            self.assertTrue(all(src.kind == "folder" for src in s.sources), s.key)
            self.assertNotIn("E004", [m.experiment_id for m in s.models])  # needs the enhanced deliveries

    def test_builtin_and_declared_models(self):
        lobster, other = parse_species({"species": [LOBSTER, OTHER]})
        self.assertEqual(str(lobster.models[0].weights).replace("\\", "/"), "weights/E027_best.pt")
        self.assertIsNone(lobster.models[0].declared_conf)
        m = other.models[0]
        self.assertEqual((m.declared_conf, m.declared_imgsz, m.declared_source), (0.16, 640, "owner's val run"))
        self.assertEqual(other.animal, "sea slug")
        self.assertEqual(other.limitations, ("only one limitation",))
        self.assertEqual((other.sources[0].kind, other.sources[0].name, other.sources[0].held_out),
                         ("folder", "samples/slug/images", True))

    def test_split_source_takes_its_status_from_the_project(self):
        entry = dict(LOBSTER, sources=[{"label": "t", "kind": "split", "name": "test"},
                                       {"label": "v", "kind": "split", "name": "val"}])
        test, val = parse_species({"species": [entry]})[0].sources
        self.assertTrue(test.held_out and "HELD OUT" in test.note)
        self.assertFalse(val.held_out)

    def test_bad_files_are_refused_with_the_reason(self):
        bad = [
            ({}, "at least one species"),
            ({"species": [{"key": "x"}]}, "label"),
            ({"species": [LOBSTER, LOBSTER]}, "twice"),
            ({"species": [dict(LOBSTER, models=[])]}, "at least one model"),
            ({"species": [dict(LOBSTER, models=[{"id": "NEW"}])]}, "label, weights, description"),
            ({"species": [dict(LOBSTER, models=[{"id": "E027", "declared_conf": 1.5}])]}, "declared_conf"),
            ({"species": [dict(LOBSTER, sources=[{"label": "a", "kind": "video"}])]}, "source kind"),
            ({"species": [dict(LOBSTER, sources=[{"label": "a", "kind": "folder"}])]}, "path"),
            ({"species": [dict(LOBSTER, sources=[{"label": "a", "kind": "split", "name": "nope"}])]}, "unknown split"),
        ]
        for data, reason in bad:
            with self.assertRaises(DetectError) as ctx:
                parse_species(data)
            self.assertIn(reason, str(ctx.exception))

    def test_missing_or_broken_file(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(DetectError):
                load_species(Path(d) / "none.json")
            (Path(d) / "bad.json").write_text("{", encoding="utf-8")
            with self.assertRaises(DetectError):
                load_species(Path(d) / "bad.json")


class DeclaredSettingsTest(unittest.TestCase):
    def test_unknown_checkpoint_uses_declared_values_and_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            spot = sweet_spot_for_sha256("0" * 64, Path(d), declared_conf=0.16, declared_imgsz=960,
                                         declared_source="owner's val run")
            self.assertEqual((spot.conf, spot.imgsz, spot.known, spot.conf_declared), (0.16, 960, False, True))
            self.assertIn("owner's val run", spot.conf_source)
            self.assertIn("not read from a project record", spot.conf_source)
            plain = sweet_spot_for_sha256("0" * 64, Path(d))
            self.assertEqual((plain.conf, plain.conf_declared), (DEFAULT_CONF, False))

    def test_record_wins_over_declared_and_declared_fills_a_gap(self):
        with tempfile.TemporaryDirectory() as d:
            for exp, with_conf in (("E900", True), ("E901", False)):
                folder = Path(d) / exp
                folder.mkdir()
                (folder / "run_record.json").write_text(json.dumps(
                    {"experiment": {"id": exp}, "outputs": {"best": {"sha256": exp * 16}},
                     "train_args": {"imgsz": 640}}), encoding="utf-8")
                if with_conf:
                    (folder / "val_eval_record.json").write_text(json.dumps({"operating_conf": 0.3}),
                                                                 encoding="utf-8")
            recorded = sweet_spot_for_sha256("E900" * 16, Path(d), declared_conf=0.9)
            self.assertEqual((recorded.conf, recorded.conf_declared, recorded.experiment_id), (0.3, False, "E900"))
            gap = sweet_spot_for_sha256("E901" * 16, Path(d), declared_conf=0.16, declared_source="s")
            self.assertEqual((gap.conf, gap.conf_declared, gap.experiment_id, gap.imgsz), (0.16, True, "E901", 640))


class FolderSourceTest(unittest.TestCase):
    def test_lists_images_sorted_and_ignores_other_files(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("b.png", "a.JPG", "notes.txt"):
                (Path(d) / name).write_bytes(b"x")
            self.assertEqual([p.name for p in list_folder_images(d)], ["a.JPG", "b.png"])
            self.assertEqual(len(list_folder_images(d, limit=1)), 1)

    def test_missing_or_empty_folder_is_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(DetectError) as ctx:
                list_folder_images(Path(d) / "missing")
            self.assertIn("gui_species.json", str(ctx.exception))
            with self.assertRaises(DetectError):
                list_folder_images(d)


if __name__ == "__main__":
    unittest.main()
