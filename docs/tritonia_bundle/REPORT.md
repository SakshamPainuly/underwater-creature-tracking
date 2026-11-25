# E010: Tritonia sea slug detector (YOLO26s)

Tags: **OBSERVED** = read from a project file or a run, **INFERRED** = reasoned from observed facts,
**UNKNOWN** = not known; the reason is given.

## What it detects
- OBSERVED: Ultralytics YOLO detection checkpoint `weights/E010_best.pt` (sha256 `48d72b79…`, 20,291,141 B),
  trained from `yolo26s.pt` with Ultralytics 8.4.155. `model.names = {0: 'lobster', 1: 'sea_slug'}`.
- OBSERVED: class 0 `lobster` is an empty slot kept so class IDs match the lobster project. The training data has
  0 lobster boxes, and none of the 28 sample images got a class-0 prediction. **Show class 1 only.**
- Species: common name *Tritonia* sea slug, a nudibranch. Latin name: **UNKNOWN**. The source only says
  "Tritonia" (genus) and states no species (docs/DATASET_TRITONIA.md).

## Data and split
- OBSERVED: the frames come from grayscale TIFF timelapse stacks provided by Dr. Wyeth. Each frame is 1056 × 480 px,
  a 3 × 2 montage of 6 cameras, one frame every 15 s, one recording per day from 2013-05-18 to 2013-06-03.
  INFERRED: all days come from one setup. Its site and lighting are UNKNOWN (not recorded in the source).
- OBSERVED: the ground truth is hand-clicked MTrackJ points (crosses), not boxes. Dataset v7 (frozen, D-036/D-037)
  puts a **fixed 20 × 17 px box centred on every cross**, so box size carries no information.
- OBSERVED: the split unit is the **recording day** (seed 0). The 150 frames were sampled at least 40 frames
  (10 min) apart.

| split | days | frames | boxes (= crosses) |
|---|---|---:|---:|
| train | 10 (0519, 0522, 0523, 0525, 0526, 0527, 0530, 0531, 0601, 0603) | 96 | 572 |
| val | 3 (0524, 0529, 0602) | 27 | 153 |
| test | 3 (0520, 0521, 0528) | 27 | 127 |

- OBSERVED: no day appears in two splits, so no frame from the same minute or day is shared. But every split
  comes from the same 6 cameras and the same scene. INFERRED: the backgrounds look near-identical across splits,
  so the scores measure **new days, not new sites or cameras**.

## Training
- OBSERVED (run_record.json): imgsz 640, 100 epochs (patience 30, all 100 ran), batch 16, seed 0,
  deterministic, all other settings at Ultralytics defaults. Trained on a Kaggle Tesla T4 on 2026-11-12 at git
  commit `2a8daf3` (clean). `best.pt` is the epoch with the highest val fitness; results.csv peaks at epoch 81.

## Scores
The project scores this model with **point metrics**. A predicted box counts as correct if its centre is within
10 px of a hand-clicked cross, matched one to one. The Ultralytics box mAP is measured against the fixed-size
boxes, so it is **not an accuracy measure** (D-033). It is shown only because it was asked for.

| metric | val (27 frames, 153 crosses) | test (27 frames, 127 crosses) |
|---|---:|---:|
| **point AP** (selection metric) | **0.496** | **0.2145** |
| point precision @ conf 0.16 | 0.614 | 0.3563 |
| point recall @ conf 0.16 | 0.4575 | 0.2441 |
| point F1 @ conf 0.16 | 0.5243 | 0.2897 |
| Ultralytics mAP50 / mAP50-95 (fixed boxes) | 0.4082 / 0.1978 | UNKNOWN: never run on test |
| Ultralytics P / R (fixed boxes) | 0.5100 / 0.4248 | UNKNOWN: never run on test |

- OBSERVED: the test was run **once**, on 2026-11-12, after E010 was declared final candidate on val. Neither the
  model nor the threshold was changed afterwards. The same test days were also used once for E006 (another model)
  and once for the tracker T002, which uses E010 detections.
- OBSERVED: val scores by day (F1): 0524 **0.148**, 0529 0.704, 0602 0.367. Test scores by day (F1): 0520 0.359,
  0521 0.274, 0528 0.275 (0528 has 80 crosses but recall is 0.175).
- INFERRED: the gap from val (0.496) to test (0.215) shows that 27 frames per split give a noisy estimate. A
  repeat of the same setup with a different seed scored 0.478 to 0.496 on val (D-042). Use the test numbers as the
  realistic expectation.

## Confidence threshold
- OBSERVED: **conf 0.16** gives the highest point F1 on val (`val_eval_record.json`). It was chosen on 2026-11-12,
  before the test. Nothing was recomputed today. Use imgsz 640 and NMS IoU 0.7.

## Later checks: no better model found (updated 2026-11-24)
- OBSERVED: after this bundle was made, D-045 (2026-11-24) tried larger models (yolo26m, yolo26l) and
  background-contrast images (bgc1, dataset v9), 5 arms x 3 seeds on the same val days. Mean val point AP:
  yolo26l 0.425, yolo26s + bgc1 0.412, yolo26l + bgc1 0.393, yolo26m 0.385, yolo26m + bgc1 0.350, all below the
  E010 baseline mean of 0.489. Earlier, D-042 (four training settings) and D-041 (28x more training frames) also
  gave no gain. **E010 stays the Tritonia reference**, so this bundle's model is unchanged.
- OBSERVED: none of these runs touched the test days (rule 7); the test numbers above are still E010's single test run.
- Records: docs/DECISIONS.md D-041, D-042, D-045; docs/EXPERIMENTS.md; PR #40.

## Strengths and weaknesses (from the bundle samples)
`check_predictions.json` has the CPU predictions. Sample labels are the fixed boxes, whose centres are the crosses.
- Good: `20130520all_15spf_t0895.png` (test) finds all 3 slugs, and `20130602all_15spf_t0307.png` (val): its 3
  predictions all land on clicked slugs.
- Misses in crowded frames: `20130528all_15spf_t0661.png` (test) finds 2 of 13 slugs, and
  `20130602all_15spf_t1211.png` (val) finds 2 of 11.
- False alarms on a difficult day: `20130524all_15spf_t0045.png` (val) makes 5 detections, none on the 1 clicked
  slug.
- INFERRED: the model misses more than half the slugs, and the error rate changes a lot from day to day. Slugs are
  about 12–20 px, faint and blurred, and the cameras are at an angle (parallax).
- UNKNOWN: whether every slug was clicked (D-029), so precision is a lower bound. Also unknown: performance on
  other sites, on other cameras, in colour or in real video. I wanted to show behaviour on "dark" frames, but the
  mean grey level of the samples only ranges from 164 to 171, so dark versus bright frames could not be shown.
- UNKNOWN: accuracy on the 10 unlabelled 2013-05-18 frames. They have no labels; they only show how the model
  behaves on an unseen day of the same setup.
- No physical units: the frames are uncalibrated. This is sparse-frame/timelapse data (15 s per frame), not video.
