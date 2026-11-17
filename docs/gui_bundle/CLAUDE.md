# CLAUDE.md — DeepSight GUI folder (context for the assistant on Harsh's MacBook)

This file is the full context for continuing the GUI work on this machine. Read it before changing
anything. `README.md` has the setup commands; `HOW_THINGS_ARE_STORED.txt` lists every file;
`PACKAGING.md` is the plan for turning this prototype into a software package.

## 1. What this folder is

A portable copy of the DeepSight detection GUI, exported from Saksham's project repository on
2026-11-16 by `scripts/export_gui_bundle.py`. It contains the GUI code, the lobster models with the
records that hold their validated settings, a few sample images, and reference documents. It does
**not** contain the repository, the raw dataset or the training code.

**Lobster only.** Everything in this folder is about the lobster detector. Teammates' other species
are not included: each teammate sends their own work separately.

Goal on this machine: run the lobster GUI for the team demo, with every function listed in
section 3 working.

State when exported (checked on Windows with the same code): E027 (final model) and E001 (first
baseline) are in `weights/` and work.

## 2. The project in one page

DeepSight is a university team project (StFX, CSCI 555 Data Mining & Machine Learning, Dr. Jacob
Levman): detect underwater animals in images, then track them. Final report due 2026-12-15.

| Person | Species | Work |
|---|---|---|
| Saksham Painuly | Lobster | detection pipeline, experiments E001/E004/E007/E011–E029, this GUI |
| Anu Rani | own species (not in this folder) | RT-DETR lobster baseline (E005) |
| Harsh Patel | own species (not in this folder) | image enhancement and augmented/synthetic lobster data |

What was done for lobsters, in order (numbers are from the committed records; the full pack is
`docs/final_report_data.md`; the decision log and the experiment log stay in the project repository):

1. **Data foundation (Sept).** 2,105 GoPro frames with 6,522 labelled lobster boxes, plus 436
   unlabelled "zExtra" frames. Split by recording session, never by image: 1,433 train / 324 val /
   348 test images (dataset v1, frozen). The val and test images never changed afterwards.
2. **Baselines on v1 (Sept–Oct).** E001 = YOLO26s on raw frames (test mAP50-95 0.683).
   E004 = the same on Harsh's enhanced images (0.656: enhancement did not help detection).
   E005 = RT-DETR-L (0.660).
3. **Val-only evaluation (Oct).** A second evaluation on validation that keeps per-image results:
   false positives on lobster-free frames, results per session and per lighting.
4. **Model A data (Nov), dataset v4.** Harsh's combined data (originals, colour-change, copy-paste
   and diffusion copies) plus public Zenodo frames, rebuilt on *our* session split: the delivered
   split had 5 of our 6 test sessions in its train/val part. A leakage check removed 992 derived
   images whose source frame or pasted lobster came from a val/test session. Training: 4,111 images.
   Zenodo site `sk` (428 frames) is held out as the only new-site test.
5. **Tuning (Nov 11–13), on Kaggle 2×T4.** Rules fixed before any run: one selection metric
   (val mAP50-95), three seeds per model size to measure noise, test never used to choose.
   Round 1 (size × seed): YOLO26m chosen, YOLO26l kept. Round 2 (five setting combinations × two
   sizes): only **E027** (YOLO26l; AdamW lr 0.001, weight decay 0.001, mixup 0.1; 30 epochs) cleared
   the pre-declared bar (val mAP50-95 0.6280 > 0.62542).
6. **Final test of E027 (Nov 13), once per target.** v1 test: mAP50-95 0.697, mAP50 0.910,
   precision 0.848, recall 0.843 at the val-chosen confidence 0.269 (E001: 0.683 / 0.897 / 0.845 /
   0.813). New site Zenodo sk: mAP50-95 0.389, precision 0.434: it does not transfer well to a new site.
7. **E027 val-only evaluation (Nov 16).** On validation: 838 found / 239 false / 172 missed
   (E001: 809 / 208 / 201); a false box on 19 of the 55 lobster-free frames (E001: 16).

Limitations that must be stated with these results: test sessions share sites with training
sessions (new sessions, not new sites); one seed for the final model; val numbers are optimistic
because val chose the model, checkpoint and confidence; pixels only, no physical units.

## 3. What the GUI does (all of this must keep working)

1. **Species selector** (sidebar, top): shown only when `configs/gui_species.json` lists more than
   one species. This folder lists lobster only, so it is hidden.
2. **Model selector** per species, with the model's description and its recorded scores.
3. **Checkpoint box**: the weights path; the file is recognised by its sha256 against
   `experiments/records/*/run_record.json`.
4. **Sweet-spot sliders**: confidence, NMS IoU, inference size, match IoU. They start at the
   model's validated values, a "Reset to sweet spot" button restores them, and a line says when a
   slider is off the validated setting.
5. **Side-by-side comparison**: a second model of the same species on the same image, at its own
   validated settings (lobster: E027 next to E001).
6. **Two ways to pick an image**: browse a configured source, or upload any image.
7. **Ground-truth overlay** where label files exist, with per-image counts Agree / Extra / Missed.
8. **Held-out warnings**: on test sources, a warning if the confidence slider is moved.
9. **Detections table**, **CSV download** and **YOLO .txt download** (same format as training labels).
10. **Confidence chart** per detection, and a **"What this cannot tell you"** panel per species.

Requested by the professor and not implemented: drawing the animal's outline instead of a box
(it needs a segmentation model; boxes are what the detectors produce).

## 4. How it is built

```
scripts/gui_app.py        Streamlit screen only: widgets, layout. No logic.
src/detect/predict.py     detection, drawing, label reading, per-image matching, image listing
src/detect/species.py     reads configs/gui_species.json -> species, models, sources
src/detect/models.py      the built-in lobster models (E027, E001, E004) and their recorded scores
src/detect/sweet_spot.py  validated slider values for a checkpoint (from records, else declared)
src/detect/enhanced.py    raw vs enhanced images (E004 only; not used in this folder)
configs/gui_species.json  THE file to edit: species, models, image sources, limitations
experiments/records/<id>/ the records the settings and scores are read from
```

- Detection runs on CPU through Ultralytics (`model.predict(..., device="cpu")`), so results match
  the validated settings on any machine. Any Ultralytics **detection** checkpoint (`.pt`) works
  without code changes. A segmentation, pose or non-Ultralytics model would need changes in
  `predict.py` (`load_model`, `detect`).
- **Where slider values come from** (`sweet_spot.py`): the checkpoint's sha256 is looked up in
  `experiments/records/*/run_record.json`. If found, confidence = the val-chosen operating
  confidence in that run's `val_eval_record.json` or `final_test_record.json`, and inference size
  = the training size. If the records have no confidence, or the checkpoint is not in the records,
  the values **declared** for the model in `configs/gui_species.json` are used, and the GUI says
  they are declared. With neither, the lobster baseline's values are used and the GUI warns that
  they are not validated.
- **Image sources** in the species file: `{"kind": "folder", "path": "..."}` is any folder of
  images, relative to this folder or absolute. Labels are read only when the folder follows the
  convention `.../images/<name>/x.png` with `.../labels/<name>/x.txt` (YOLO format). `"held_out":
  true` turns on the do-not-tune warning. (`raw` and `split` kinds need the full repository.)
- The app writes nothing to disk. Downloads are made in memory by the browser.

## 5. Tasks on this machine

### a. Set up and check
Follow `README.md`. `python scripts/gui_check.py` must end with `GUI CHECK PASSED`; it lists each
model, whether its checkpoint is present, and where its settings come from.

### b. Adding another species (only if its owner asks for it)
This folder ships lobster only. If a teammate wants their own model in the same app, they add it
themselves:
1. Put the checkpoint under `weights/<species>/` (for example `weights/<species>/best.pt`).
2. Add a species entry to `configs/gui_species.json` (`key`, `label`, `animal`, `icon`, `models`,
   `sources`, `limitations`; the lobster entry and `src/detect/species.py` show the format). Each
   model entry needs:
   - `id`: a short name for the model, `label`: what the selector shows, `description`: what it
     was trained on (data, image count, classes, who trained it).
   - `declared_conf`: the confidence threshold chosen **on validation data** for this model;
     `declared_imgsz`: the training image size; `declared_source`: one sentence saying where those
     two values come from (which run, which file). If no validated threshold exists, leave
     `declared_conf` out: the GUI will then say the settings are not validated. Do not invent one.
   - `scores_note`: one sentence with the model's recorded scores and their source, or leave it out.
   - `limitations`: what the model cannot tell you (data size, sites, how it was evaluated).
3. Add a `folder` source pointing at a few of that species' images (YOLO labels in a matching
   `labels/` folder if there are any).
4. Run `python scripts/gui_check.py`, then start the app and test every function in section 3.
   With two or more species the species selector appears by itself.
5. To compare two models side by side, add a second model entry to the same species.

### c. Anything else
New lobster image folders: add a `folder` source to the lobster entry. Code changes are only needed
for a model that is not an Ultralytics detector.

## 6. Rules to keep

These come from the project's own rules and are why the numbers in the report can be trusted.

1. **Never invent a number.** Every score or threshold shown in the GUI names the record or the
   person it comes from. Write "UNKNOWN" rather than guess.
2. **Thresholds are chosen on validation, never on test images.** Looking at test images is fine;
   keeping a slider value found on them is not.
3. **The GUI computes no accuracy metric.** Per-image Agree / Extra / Missed counts are
   observations. Do not add precision, recall or mAP over a set of images to the app.
4. **A missing label file is not a negative.** Only an empty label file means "no animal here".
5. **Pixels only.** No centimetres, metres or speeds: there is no calibration. Image sets that are
   a sparse timelapse are not "video tracking".
6. **Keep the split of labour:** logic in `src/detect/`, widgets in `scripts/gui_app.py`,
   configuration in `configs/gui_species.json`. Prefer editing the species file over editing code.
7. Do not edit files under `experiments/records/` or the sample labels: they are copies of records.
8. Changes made here do not flow back to the repository by themselves. Tell Saksham what changed
   (ideally send the changed files) so the repository copy stays the source of truth.

## 7. macOS notes

- Python 3.10–3.12. Apple Silicon and Intel both work; inference is on CPU by design.
- Box labels use Arial or Helvetica, found automatically on macOS.
- First start is slow (Ultralytics and torch load); each later detection takes well under a second
  to a few seconds per image for E027 (a large model: 52.9 MB, 26.2 M parameters).
- If the browser does not open, go to http://localhost:8501.

## 8. Prototype, not a package

This folder is a **prototype** demo GUI, not a deployed software package: it runs from this folder
for one local user, finds its files by their position in the folder, and has only been checked on
Windows. That is enough for the demo. If it should become a package (installable, one start
command, weights downloaded and checked), follow `PACKAGING.md`: what is prototype-level today, the
steps, a `pyproject.toml` starting point, and what must be confirmed before sharing it outside the
team (lab data rights, the Ultralytics AGPL licence). Do not start that work unless asked.

## 9. Reference documents in `docs/`

- `final_report_data.md` (+ `figures/`, `final_report_experiments.csv`): every lobster number, with
  how it was computed and its limitations. Start here for anything about results. This is the
  lobster-only copy of the report pack.
- `DATASET.md`: the lobster data card.
- Decision IDs (D-0xx) and the full experiment log are in the project repository
  (`docs/DECISIONS.md`, `docs/EXPERIMENTS.md`), not in this folder: ask Saksham if one is needed.
- `MANIFEST.json` (folder root): every file in this folder with its sha256, and the repository
  commit it was exported from.
