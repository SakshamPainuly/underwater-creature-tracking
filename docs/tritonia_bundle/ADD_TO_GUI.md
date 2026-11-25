# Adding E010 (Tritonia sea slug) to the GUI

The GUI repository config stays lobster only (Saksham's design: `tests/test_species.py`). Tritonia is added in
**the GUI folder you run**, as `docs/gui_bundle/CLAUDE.md` section b describes. Nothing in the repository changes.

## Steps
In the folder below, `GUI` means the GUI folder you run, e.g. `D:/DeepSight_GUI_for_Harsh` or the repository itself.

1. Unzip this bundle.
2. Copy these into `GUI`, keeping their paths:
   - `weights/E010_best.pt` -> `GUI/weights/E010_best.pt`
   - `samples/tritonia/` -> `GUI/samples/tritonia/`
   - `experiments/records/E010/` -> `GUI/experiments/records/E010/` (lets the GUI recognise the checkpoint by its
     sha256 and read its validated settings)
3. Open `GUI/configs/gui_species.json`. Inside `"species": [ ... ]`, after the lobster entry, add a comma and paste the
   whole content of `species_entry.json`. Keep lobster first: E027 stays the default model.
4. Run `python scripts/gui_check.py`. It must end with `GUI CHECK PASSED` and list E010 with its checkpoint present.
5. Start the GUI as usual. A species selector appears, and Tritonia sea slug is one of its choices.

## What to expect
- Model: E010, YOLO26s, `weights/E010_best.pt`, sha256 `48d72b79770bf8085bf70530eae4c71648a0b6c88c1e3580b90ca84e98909ca5`.
- Settings: conf 0.16 (max point F1 on val), imgsz 640, declared in `species_entry.json` and in
  `experiments/records/E010/val_eval_record.json`.
- Sources: 9 val frames, 9 test frames (held out: look, do not tune), 10 unlabelled frames from 2013-05-18.
- Class 0 (`lobster`) is an empty slot in this checkpoint. The current GUI ignores `classes_to_show`; none of the 28
  sample frames got a class-0 box (`check_predictions.json`).
- Read `REPORT.md` first: E010 finds about half the slugs (test recall 0.24) and is the best Tritonia model the project
  has (D-041, D-042 and D-045 found nothing better).

Checked 2026-11-24: this entry, added after the lobster entry of `configs/gui_species_bundle.json`, parses with
`src/detect/species.py` (`parse_species`), and every file it names is in this bundle.
