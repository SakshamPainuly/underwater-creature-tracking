# DeepSight GUI: portable folder

A self-contained copy of the DeepSight detection GUI. It runs on macOS, Windows or Linux without
the project repository or the raw dataset. It carries the lobster detectors only (teammates'
other species are sent separately):

| Species | Model in this folder | Status |
|---|---|---|
| Lobster | E027 (final model) and E001 (first baseline, for side-by-side comparison) | works out of the box |

Part 1 is how to run the app. Part 2 explains the data (training, validation and test sets) and
everything the lobster work did, with the results. All numbers come from the records and documents
in this folder (`experiments/records/`, `docs/final_report_data.md`, `docs/DATASET.md`). Claims are
marked OBSERVED (read from files or runs), INFERRED, ASSUMPTION, LIMITATION or UNKNOWN.

---

# Part 1: run the app

## Start (macOS)

1. Open Terminal in this folder. Python 3.10, 3.11 or 3.12 is needed (`python3 --version`).
2. Create the environment and install (about 5 minutes, about 1 GB):

   ```
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.14.0 torchvision==0.29.0
   python -m pip install -r requirements-gui.txt
   ```

   If pip cannot find those exact torch versions, install the newest: `python -m pip install torch torchvision`.
3. Check the setup (about 30 seconds). It must end with `GUI CHECK PASSED`:

   ```
   python scripts/gui_check.py
   ```

4. Start the app. It opens in the browser at http://localhost:8501:

   ```
   streamlit run scripts/gui_app.py
   ```

Next time, only `source .venv/bin/activate` and step 4 are needed.

## What the app does

- Detects lobsters in an uploaded image or in the sample images (`samples/lobster/`), with E027 by
  default, each model at its own confidence chosen on validation (E027 0.269, E001 0.298).
- Runs E027 and E001 side by side on the same image.
- Shows each model's recorded scores (read from `experiments/records/`).
- Draws the ground-truth boxes next to the predictions where a label file exists.
- It computes no metric: per-image counts in the app are observations, not scores.

## Where things are

`HOW_THINGS_ARE_STORED.txt` lists every folder and file. `CLAUDE.md` is the full context for an
AI assistant (or a person) continuing the work: what the project did, how the GUI is built and
the rules it keeps.

This is a **prototype** demo GUI, not a deployed software package. `PACKAGING.md` describes how
to turn it into one, if that is ever wanted.

---

# Part 2: the data and what we did

## 1. Pipeline in one line

Raw lab frames → audit → split by recording session → image manifest → YOLO dataset → training on
GPU (Colab, later Kaggle) → model choice on validation only → one test evaluation of the declared
final model → this GUI. Every step reads and writes files that are kept in the project repository;
every choice has a numbered decision (D-###) and every training run an experiment ID (E###) in the
project repository's `docs/DECISIONS.md` and `docs/EXPERIMENTS.md` (not shipped in this folder).

## 2. The raw data

| Source | Images | Labels | How it is used |
|---|---:|---:|---|
| Lab delivery, Set 100 | 105 | 105 | one session, train only (frames too close in time) |
| Lab delivery, Set 2000 | 2,000 | 1,759 | a missing label file means "no lobster" (confirmed negative) |
| Lab delivery, zExtra | 436 | 0 | not labelled: never used for training or scoring; shown in the GUI only |
| Zenodo record 15755673 (CC BY 4.0, Fiorina, Obst, Sundelof, Ljungberg, Sande 2025) | 908 used | yes | site `mpa` for training, site `sk` as a held-out new-site test |
| Harsh's derived images (colour change, copy-paste, diffusion) | see section 4 | copied from their source | training only, after a leakage check |

- OBSERVED: lab frames are PNGs exported from GoPro videos (1920×1080 or 1920×1088), labels in
  YOLO format (`class x_center y_center width height`, normalised), one class `0 = lobster`.
  No videos were delivered.
- OBSERVED: frames are 300–900 frames apart (roughly 10–30 s; exact fps UNKNOWN), so they are
  sparse frames, not video. That is why there are no tracking results (no ground-truth track IDs).
- The raw files were never changed: every data step checks them against a recorded fingerprint.

## 3. How the training, validation and test sets were made (dataset v1)

**Rule: the split unit is the recording session** (the deployment code without the GoPro file
number, e.g. `21_SUB6_03_BW_20220810`). All frames and all recording files of a session sit in one
split, so the model is never scored on frames that are near-copies of training frames. Images were
never split one by one. The v1 split is frozen (never changed afterwards).

| Split | Sessions | Images | With lobster | Confirmed empty | Boxes |
|---|---:|---:|---:|---:|---:|
| train | 32 | 1,433 | 1,274 | 159 | 4,000 |
| val | 7 | 324 | 269 | 55 | 1,010 |
| test | 6 | 348 | 321 | 27 | 1,512 |
| excluded (zExtra) | 11 | 436 | not labelled | not labelled | none |

**Validation sessions (7):** used for every choice (best epoch, best run, confidence threshold).

| Session | Images | With lobster | Boxes | Red light |
|---|---:|---:|---:|---|
| 18_ALRA_08_CW_20220616 | 36 | 33 | 60 | no |
| 19_SUB1_04_CW_20220621 | 36 | 3 | 3 | yes |
| 19_SUB2_07_CW_20220622 | 36 | 25 | 28 | yes |
| 21_MACK_10_BW_20220815 | 54 | 54 | 465 | no |
| 21_SUB6_07_BW_20220812 | 54 | 47 | 95 | no |
| 21_SUB6_09_BW_20220815 | 54 | 53 | 125 | no |
| 21_SUB6_10_BW_20220815 | 54 | 54 | 234 | no |

**Test sessions (6):** used once, at the end, only for the declared final model.

| Session | Images | With lobster | Boxes | Red light |
|---|---:|---:|---:|---|
| 06_HERR_04_CW_20200902 | 57 | 57 | 148 | no |
| 06_HERR_08_CW_20200903 | 55 | 45 | 49 | no |
| 06_HERR_11_CW_20200905 | 54 | 45 | 66 | no |
| 13_SHRI_12_BW_20210811 | 72 | 72 | 577 | yes |
| 21_MACK_09_BW_20220815 | 56 | 48 | 114 | no |
| 21_SUB6_03_BW_20220810 | 54 | 54 | 558 | no |

The other 32 labelled sessions are train. Nine sessions have two GoPro files: OBSERVED that they
share one session code and one split; INFERRED that they are sequential chapters of one recording;
UNKNOWN whether any were simultaneous cameras.

**The test set is never used for training or tuning.** Nothing was chosen by looking at a test
score: not the model, the epoch, the settings or the confidence threshold.

LIMITATIONS of the v1 split (state them with every result):
- Test sessions share sites (mostly also dates) with train sessions, so the v1 test measures
  **new sessions at known sites, not new sites**. The Zenodo `sk` test (below) is the only new-site check.
- Substrate and visibility are missing for most sessions; only the red-light flag is complete.
- Concentrated sets: one val session (`21_MACK_10`) holds 465 of 1,010 val boxes; two test sessions
  (`13_SHRI_12`, `21_SUB6_03`) hold 1,135 of 1,512 test boxes.
- Some "empty" frames in very turbid or red-light sessions could not be visually confirmed.

## 4. Dataset versions

| Version | What it is | Used by |
|---|---|---|
| v1 (frozen) | lab frames, the split above | E001, E005 |
| v2 | the same frames as Harsh's enhanced copies (same split and labels) | E004 |
| v4 (frozen) | v1 + Zenodo + Harsh's derived images, on the v1 split ("Model A" data) | E007, tuning, E027 |

**Val and v1 test are identical in every version** (the same 324 and 348 real, unmodified frames),
so all models are scored on the same images.

**How v4 was built (D-031, D-032):**
- Harsh's own split was **not used**: it put 5 of our 6 test sessions in his train or val. Our
  v1 split decides for every image.
- **Leakage check:** a derived image (colour change, copy-paste, diffusion) enters train only if
  every image it was made from (the background and every pasted lobster) comes from a train
  session. 992 derived images failed that and were left out. The check also confirmed no identical
  files across splits; the 2 near-duplicate flags were checked by eye and are different scenes.
- Colour-change images were redone by Harsh with colours taken from v1 train frames only.
- Harsh's AI label review (no person checked a frame): its 90 excluded train frames and 31
  low-confidence "empty" train frames were left out of train. Val and test keep every v1 frame.
- Zenodo: site `mpa` in train, site `sk` held out as a new-site test, the 24 no-site `REC_` frames left out.

**v4 images per split:**

| Split | Origin | Original | Colour change | Copy-paste | Diffusion | Total |
|---|---|---:|---:|---:|---:|---:|
| train | ours | 1,312 | 192 | 516 | 1,312 | 3,332 |
| train | Zenodo mpa | 480 | 93 | 206 | 0 | 779 |
| val | ours | 324 | 0 | 0 | 0 | 324 |
| test (v1) | ours | 348 | 0 | 0 | 0 | 348 |
| test_zenodo_sk | Zenodo sk | 428 | 0 | 0 | 0 | 428 |

Train: 4,111 images (3,721 with lobsters, 390 confirmed empty, 11,784 boxes). Zenodo sk: 428
images (285 empty), 143 boxes.

LIMITATION: train labels rest partly on an unverified AI review; up to 29 train frames have a
lobster without a box (kept, as in the earlier runs).

## 5. Experiments (training runs)

All runs: Ultralytics YOLO26 (COCO-pretrained), 640 px, except E005 (RT-DETR-l).

| Run | Model | Data | What changed | Val mAP50-95 |
|---|---|---|---|---:|
| E001 | YOLO26s | v1 | first baseline | 0.6115 |
| E004 | YOLO26s | v2 | enhanced images instead of raw | 0.5864 |
| E005 | RT-DETR-l | v1 | other architecture (E003 diverged, rerun without AMP) | 0.5884 |
| E007 | YOLO26s | v4 | all safe Model A data at once | 0.5993 |
| E011–E019 | YOLO26 s / m / l × 3 seeds | v4 | tuning round 1: model size (E018 not run, time) | 0.5965–0.6172 |
| E020–E029 | YOLO26 m / l × 5 setting combinations | v4 | tuning round 2: training settings | 0.5799–0.6280 |
| **E027** | **YOLO26l** | **v4** | **final model** (round 2, combination C4) | **0.6280** |

**Selection metric: val mAP50-95**, declared before the tuning comparisons (D-038). mAP50 asks
"was the lobster found?" (box overlap IoU ≥ 0.5); mAP50-95 averages over IoU 0.50–0.95, so it also
rewards tight boxes, which matters because trajectories are built from box positions.

**Tuning, in two rounds (Kaggle, 2×T4):**
- Round 1: YOLO26s / m / l, seeds 0 / 1 / 2. Seed ranges measured the noise. YOLO26m was chosen
  (YOLO26l's gain was within the noise), and YOLO26l was kept for round 2.
- Round 2: a 30-epoch schedule that always completes, × 5 combinations (schedule only; AdamW;
  mixup; AdamW + weight decay + mixup; frozen backbone) on YOLO26m and YOLO26l. A run had to beat
  the pre-declared bar 0.62542 (YOLO26m mean + range). **Only E027 cleared it** and was declared
  the final model before any test evaluation.
- E027 settings: `yolo26l.pt`, 30 epochs, AdamW lr0 0.001, weight decay 0.001, mixup 0.1,
  mosaic 1.0 (off for the last 10 epochs), batch 16, seed 0. Best epoch 25, 1.82 h on one T4.

LIMITATION: 19 configurations were compared on the same 324 val images, so the val score is
optimistic, and E027's margin over the bar (0.0025) is smaller than seed noise. Only the test
results below are unbiased.

Compute: 22 completed training runs, 39.6 GPU-hours recorded (Colab and Kaggle T4).

## 6. Results

**Held-out test, once per model** (P / R / F1 at each model's own val-chosen confidence):

| Model | Test set | mAP50 | mAP50-95 | P | R | F1 | Found / false / missed boxes |
|---|---|---:|---:|---:|---:|---:|---|
| E001 (YOLO26s, v1) | v1 test | 0.897 | 0.683 | 0.845 | 0.813 | 0.828 | 1261 / 263 / 251 |
| E004 (YOLO26s, v2) | v1 test | 0.873 | 0.656 | 0.861 | 0.752 | 0.803 | 1165 / 190 / 347 |
| E005 (RT-DETR-l, v1) | v1 test | 0.879 | 0.660 | 0.841 | 0.771 | 0.805 | not comparable |
| **E027 (YOLO26l, v4)** | v1 test | **0.910** | **0.697** | 0.848 | 0.843 | 0.845 | 1289 / 232 / 223 |
| **E027** | Zenodo sk (new site) | 0.566 | 0.389 | 0.434 | 0.622 | 0.511 | 90 / 119 / 53 |

- OBSERVED: on new sessions at known sites, E027 finds about 84% of lobster boxes and about 85% of
  its boxes are correct. Against E001 it finds 28 more lobsters with 31 fewer false boxes.
- OBSERVED: at a new site (Zenodo sk) quality drops sharply: expect many false boxes and missed
  lobsters on footage from a new location until the model is fine-tuned there.
- INFERRED, not tested: the gain over E001 comes from more data and a larger model together; no
  run separates the two.
- OBSERVED: test scores are higher than val for E001, E004 and E027 (E027 0.697 vs 0.628). INFERRED:
  the test frames are easier than the val frames, so compare models within one set only.

**Validation details for E027** (own matcher, IoU ≥ 0.5, conf 0.269): 838 found / 239 false /
172 missed, P 0.778, R 0.830; a detection on 19 of the 55 lobster-free val images (E001: 16).

**UNKNOWN:** false positives on the empty test images and per-session test results (the test
records keep no per-image predictions); a second seed of E027; leave-site-out cross-validation.

## 7. What the model cannot do

- One class (lobster). Boxes, not outlines.
- Pixels only: no sizes or speeds in cm, because there is no camera calibration.
- No tracking metrics (IDF1, HOTA, MOTA): there are no ground-truth track IDs, and the frames are
  too far apart for video tracking.
- Weak at new sites (see Zenodo sk above).

## 8. More detail

- `docs/final_report_data.md`: every lobster number, how it was computed, its limitations, figures.
- `docs/final_report_experiments.csv`: one row per lobster run with its settings and scores.
- `docs/DATASET.md`: the lobster data card (raw data, labels, v1 split, v2).
- `experiments/records/E027/` and `E001/`: the training, validation and test records the numbers come from.
