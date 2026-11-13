# Experiments

Declared **before** any comparison (CLAUDE.md §5). E001 has been completed and evaluated on the held-out test split under the declared test policy.

## Declarations

- **Primary selection metric:** val mAP50.

- **Secondary:** val mAP50-95, recall at the chosen confidence threshold, false-positive rate on
  confirmed-negative images, per-condition results (red-light/night vs. daylight, per session).

- **Comparison rule:** change one variable relative to a named parent experiment; same seed, same
  dataset version and fingerprint. If several variables change, the config states why.

- **Thresholds** (confidence, NMS IoU) are chosen on val and frozen before any test evaluation.

- **Test policy:**
  - The test split is evaluated only for pre-declared final candidate(s).
  - Every test evaluation is logged below (date, experiment, reason).
  - Test results never feed back into model or threshold choice.
  - Mechanism (D-012): a final candidate is declared by a committed row in
    `experiments/final_candidates.csv` (E001 declared; E005 declared 2026-10-15 while still training,
    tested only after it is registered and reviewed; E006 declared 2026-11-10 after registration,
    tested by `scripts/tritonia_point_eval.py --final`; E010 declared 2026-11-12 after registration; E027 declared 2026-11-13 after registration, D-043). `scripts/final_test_eval.py` refuses
    undeclared experiments, fixes the operating confidence on val, then evaluates once on test
    (docs/COLAB.md).

- **Every reported result carries the v1 limitations** in docs/DATASET.md (new sessions, not new
  sites; concentrated val/test sessions; sparse metadata).

## Declared experiments (D-012)

| ID | Config | Parent | Changed | Status |
|---|---|---|---|---|
| E001 | `configs/experiments/E001_yolo_baseline.yaml` | none | initial baseline: yolo26s.pt, imgsz 640, ≤100 epochs, patience 30, batch 16, seed 0, deterministic | completed; 100 epochs; final candidate evaluated on test |
| E002 | `configs/experiments/E002_yolo_imgsz1280.yaml` | E001 | imgsz 640 → 1280 only | planned; not run |
| E003 | `configs/experiments/E003_rtdetr_baseline.yaml` | E001 | model yolo26s.pt → rtdetr-l.pt only (D-015) | diverged: NaN losses by epoch 46 with AMP on; not registered (D-025) |
| E004 | `configs/experiments/E004_yolo_enhanced.yaml` | E001 | images raw → enhanced only: dataset v1 → v2 (D-017) | completed; 81 epochs (early stop); registered; declared final candidate (D-022); evaluated on test once, 2026-10-12 |
| E005 | `configs/experiments/E005_rtdetr_amp_off.yaml` | E001 | as E003 (model yolo26s.pt → rtdetr-l.pt) plus amp true → false; replaces the diverged E003 (D-025) | training finished: 100 epochs at commit 292d9ab, run_record.json finalized 2026-10-15 on Drive (OBSERVED in the RT-DETR notebook output); effective batch 8 after automatic OOM reduction (D-026); declared final candidate 2026-10-15; registered 2026-10-17 |
| E006 | `configs/experiments/E006_yolo_seaslug.yaml` | E001 | dataset v1 (lobster) → v3 (Tritonia sea slugs; classes lobster + sea_slug, automatic boxes, D-033) only; model and settings as E001 (D-030) | completed 2026-11-10 on Kaggle (Tesla T4); 100 epochs; registered; val point-scored (D-033); declared final candidate 2026-11-10; evaluated on test once, 2026-11-10 (point eval, D-033) |
| E007 | `configs/experiments/E007_yolo_model_a_all.yaml` | E001 | dataset v1 → v4 only: all safe Model A training data (4,111 images: our originals, Zenodo `mpa`, colour change, copy-paste, diffusion; D-031, D-032); model and settings as E001 | completed 2026-11-08 on Kaggle (Tesla T4); 58 epochs (early stop); registered; not a final candidate |
| E008 | `configs/experiments/E008_yolo_seaslug_imgsz1280.yaml` | E006 | train_args.imgsz 640 → 1280 only (frames 1056 × 480 px, slugs about 12–20 px: at 640 they are shrunk to 0.61×); dataset v3, model and settings as E006 | completed 2026-11-11 on Kaggle (Tesla T4); 100 epochs; registered; val point AP 0.317 < E006's 0.403: not a final candidate, no test |
| E009 | `configs/experiments/E009_yolo_seaslug_v5.yaml` | E006 | dataset v3 → v5 only (D-035: batch 1 unchanged plus the frames of batch 2, from the 10 training days only, where every cross got an automatic box); model and settings as E006 (imgsz 640) | trained 2026-11-10 on Kaggle; val point AP 0.274 < E006's 0.403: not a final candidate, no test; not registered: its run_record.json was not saved and is lost (only the cell 17 val point scores remain) |
| E010 | `configs/experiments/E010_yolo_seaslug_fixedbox.yaml` | E006 | dataset v3 → v7 only (D-036, D-037: the same 150 frames, images and day split, with a fixed 20 × 17 px box centred on every cross instead of the automatic boxes; train 86 → 96 frames, 292 → 572 boxes; v6 had one duplicate box and is not used); model and settings as E006 (imgsz 640) | completed 2026-11-12 on Kaggle (Tesla T4); 100 epochs; registered; val point AP 0.496 > E006's 0.403: declared final candidate 2026-11-12; evaluated on test once, 2026-11-12 (point eval, D-033) |
| E011 | `configs/experiments/E011_yolos_ep60_seed0.yaml` | E007 | train_args.epochs 100 → 60 and patience 30 → 10 (D-038 round 1: together the stopping rule, up to 60 epochs while val mAP50-95 improves) | completed 2026-11-12 on Kaggle (2× Tesla T4, two runs in parallel); 41 epochs (early stop, best epoch 31); registered (D-040) |
| E012 | `configs/experiments/E012_yolom_ep60_seed0.yaml` | E007 | epochs 100 → 60, patience 30 → 10 and model yolo26s.pt → yolo26m.pt (D-038 round-1 grid point yolo26m seed 0; parent E007 per D-039) | completed 2026-11-12 on Kaggle (2× Tesla T4, two runs in parallel); 27 epochs (early stop, best epoch 17); registered (D-040) |
| E013 | `configs/experiments/E013_yolol_ep60_seed0.yaml` | E007 | epochs 100 → 60, patience 30 → 10 and model yolo26s.pt → yolo26l.pt (D-038 round-1 grid point yolo26l seed 0; parent E007 per D-039) | completed 2026-11-12 on Kaggle (2× Tesla T4, two runs in parallel); 34 epochs (early stop, best epoch 24); registered (D-040) |
| E014 | `configs/experiments/E014_yolom_ep60_seed1.yaml` | E007 | epochs 100 → 60, patience 30 → 10, model yolo26s.pt → yolo26m.pt and seed 0 → 1 (D-038 round-1 grid point yolo26m seed 1; parent E007 per D-039) | completed 2026-11-12 on Kaggle (2× Tesla T4, two runs in parallel); 26 epochs (early stop, best epoch 16); registered (D-040) |
| E015 | `configs/experiments/E015_yolol_ep60_seed1.yaml` | E007 | epochs 100 → 60, patience 30 → 10, model yolo26s.pt → yolo26l.pt and seed 0 → 1 (D-038 round-1 grid point yolo26l seed 1; parent E007 per D-039) | completed 2026-11-12 on Kaggle (2× Tesla T4, two runs in parallel); 27 epochs (early stop, best epoch 17); registered (D-040) |
| E016 | `configs/experiments/E016_yolos_ep60_seed1.yaml` | E007 | epochs 100 → 60, patience 30 → 10 and seed 0 → 1 (D-038 round-1 grid point yolo26s seed 1; parent E007 per D-039) | completed 2026-11-12 on Kaggle (2× Tesla T4, two runs in parallel); 27 epochs (early stop, best epoch 17); registered (D-040) |
| E017 | `configs/experiments/E017_yolom_ep60_seed2.yaml` | E007 | epochs 100 → 60, patience 30 → 10, model yolo26s.pt → yolo26m.pt and seed 0 → 2 (D-038 round-1 grid point yolo26m seed 2; parent E007 per D-039) | completed 2026-11-12 on Kaggle (2× Tesla T4, two runs in parallel); 40 epochs (early stop, best epoch 30); registered (D-040) |
| E018 | `configs/experiments/E018_yolol_ep60_seed2.yaml` | E007 | epochs 100 → 60, patience 30 → 10, model yolo26s.pt → yolo26l.pt and seed 0 → 2 (D-038 round-1 grid point yolo26l seed 2; parent E007 per D-039) | not run (time): the round-1 deadline rule did not start it (D-038 point 3); not re-run (D-040) |
| E019 | `configs/experiments/E019_yolos_ep60_seed2.yaml` | E007 | epochs 100 → 60, patience 30 → 10 and seed 0 → 2 (D-038 round-1 grid point yolo26s seed 2; parent E007 per D-039) | completed 2026-11-12 on Kaggle (2× Tesla T4, two runs in parallel); 26 epochs (early stop, best epoch 16); registered (D-040) |
| E020 | `configs/experiments/E020_yolom_r2_sched.yaml` | E012 | epochs 60 → 30, patience 10 → 30 (round-2 schedule: no early stop, LR decay and the 10 no-mosaic epochs complete); C1, nothing else (control) (D-040 round 2, seed 0; several variables on purpose) | completed 2026-11-13 on Kaggle (2× Tesla T4, two runs in parallel); 30 epochs (schedule complete, best epoch 29); registered; not selected (D-043) |
| E021 | `configs/experiments/E021_yolol_r2_sched.yaml` | E013 | epochs 60 → 30, patience 10 → 30; C1, nothing else (control) (D-040 round 2, seed 0; several variables on purpose) | completed 2026-11-13 on Kaggle (2× Tesla T4, two runs in parallel); 30 epochs (schedule complete, best epoch 26); registered; not selected (D-043) |
| E022 | `configs/experiments/E022_yolom_r2_lr001.yaml` | E012 | epochs 60 → 30, patience 10 → 30; C2: optimizer auto (AdamW lr 0.002) → AdamW lr0 0.001, momentum 0.9 (D-040 round 2, seed 0; several variables on purpose) | completed 2026-11-13 on Kaggle (2× Tesla T4, two runs in parallel); 30 epochs (schedule complete, best epoch 23); registered; not selected (D-043) |
| E023 | `configs/experiments/E023_yolol_r2_lr001.yaml` | E013 | epochs 60 → 30, patience 10 → 30; C2: optimizer auto (AdamW lr 0.002) → AdamW lr0 0.001, momentum 0.9 (D-040 round 2, seed 0; several variables on purpose) | completed 2026-11-13 on Kaggle (2× Tesla T4, two runs in parallel); 30 epochs (schedule complete, best epoch 25); registered; not selected (D-043) |
| E024 | `configs/experiments/E024_yolom_r2_mixup.yaml` | E012 | epochs 60 → 30, patience 10 → 30; C3: mixup 0.0 → 0.1 (D-040 round 2, seed 0; several variables on purpose) | completed 2026-11-13 on Kaggle (2× Tesla T4, two runs in parallel); 30 epochs (schedule complete, best epoch 23); registered; not selected (D-043) |
| E025 | `configs/experiments/E025_yolol_r2_mixup.yaml` | E013 | epochs 60 → 30, patience 10 → 30; C3: mixup 0.0 → 0.1 (D-040 round 2, seed 0; several variables on purpose) | completed 2026-11-13 on Kaggle (2× Tesla T4, two runs in parallel); 30 epochs (schedule complete, best epoch 25); registered; not selected (D-043) |
| E026 | `configs/experiments/E026_yolom_r2_lr001_wd_mixup.yaml` | E012 | epochs 60 → 30, patience 10 → 30; C4: AdamW lr0 0.001, momentum 0.9, weight_decay 0.0005 → 0.001, mixup 0.0 → 0.1 (D-040 round 2, seed 0; several variables on purpose) | completed 2026-11-13 on Kaggle (2× Tesla T4, two runs in parallel); 30 epochs (schedule complete, best epoch 20); registered; not selected (D-043) |
| E027 | `configs/experiments/E027_yolol_r2_lr001_wd_mixup.yaml` | E013 | epochs 60 → 30, patience 10 → 30; C4: AdamW lr0 0.001, momentum 0.9, weight_decay 0.0005 → 0.001, mixup 0.0 → 0.1 (D-040 round 2, seed 0; several variables on purpose) | completed 2026-11-13 on Kaggle (2× Tesla T4, two runs in parallel); 30 epochs (schedule complete, best epoch 25); registered; **selected and declared final candidate (D-043)** |
| E028 | `configs/experiments/E028_yolom_r2_freeze10.yaml` | E012 | epochs 60 → 30, patience 10 → 30; C5: freeze none → 10 (backbone frozen) (D-040 round 2, seed 0; several variables on purpose) | completed 2026-11-13 on Kaggle (2× Tesla T4, two runs in parallel); 30 epochs (schedule complete, best epoch 25); registered; not selected (D-043) |
| E029 | `configs/experiments/E029_yolol_r2_freeze10.yaml` | E013 | epochs 60 → 30, patience 10 → 30; C5: freeze none → 10 (backbone frozen) (D-040 round 2, seed 0; several variables on purpose) | completed 2026-11-13 on Kaggle (2× Tesla T4, two runs in parallel); 30 epochs (schedule complete, best epoch 25); registered; not selected (D-043) |
| E030 | `configs/experiments/E030_seaslug_batch8_seed0.yaml` | E010 | train_args.batch 16 → 8 (D-042 arm B); dataset v7, otherwise as E010 | completed 2026-11-12 on Kaggle (Tesla T4, one of two in parallel); 100 epochs; registered; val point AP 0.472; not a final candidate (D-042: no arm wins) |
| E031 | `configs/experiments/E031_seaslug_batch8_seed1.yaml` | E010 | train_args.batch 16 → 8 (D-042 arm B); seed 0 → 1; dataset v7, otherwise as E010 | completed 2026-11-12 on Kaggle (Tesla T4, one of two in parallel); 100 epochs; registered; val point AP 0.448; not a final candidate (D-042: no arm wins) |
| E032 | `configs/experiments/E032_seaslug_batch8_seed2.yaml` | E010 | train_args.batch 16 → 8 (D-042 arm B); seed 0 → 2; dataset v7, otherwise as E010 | completed 2026-11-12 on Kaggle (Tesla T4, one of two in parallel); 100 epochs; registered; val point AP 0.496; not a final candidate (D-042: no arm wins) |
| E033 | `configs/experiments/E033_seaslug_scale02_seed0.yaml` | E010 | train_args.scale 0.5 → 0.2 (D-042 arm C); dataset v7, otherwise as E010 | completed 2026-11-12 on Kaggle (Tesla T4, one of two in parallel); 82 epochs; registered; val point AP 0.472; not a final candidate (D-042: no arm wins) |
| E034 | `configs/experiments/E034_seaslug_scale02_seed1.yaml` | E010 | train_args.scale 0.5 → 0.2 (D-042 arm C); seed 0 → 1; dataset v7, otherwise as E010 | completed 2026-11-12 on Kaggle (Tesla T4, one of two in parallel); 100 epochs; registered; val point AP 0.471; not a final candidate (D-042: no arm wins) |
| E035 | `configs/experiments/E035_seaslug_scale02_seed2.yaml` | E010 | train_args.scale 0.5 → 0.2 (D-042 arm C); seed 0 → 2; dataset v7, otherwise as E010 | completed 2026-11-12 on Kaggle (Tesla T4, one of two in parallel); 78 epochs; registered; val point AP 0.491; not a final candidate (D-042: no arm wins) |
| E036 | `configs/experiments/E036_seaslug_imgsz1056_seed0.yaml` | E010 | train_args.imgsz 640 → 1056 (D-042 arm D); dataset v7, otherwise as E010 | completed 2026-11-12 on Kaggle (Tesla T4, one of two in parallel); 100 epochs; registered; val point AP 0.482; not a final candidate (D-042: no arm wins) |
| E037 | `configs/experiments/E037_seaslug_imgsz1056_seed1.yaml` | E010 | train_args.imgsz 640 → 1056 (D-042 arm D); seed 0 → 1; dataset v7, otherwise as E010 | completed 2026-11-12 on Kaggle (Tesla T4, one of two in parallel); 100 epochs; registered; val point AP 0.481; not a final candidate (D-042: no arm wins) |
| E038 | `configs/experiments/E038_seaslug_imgsz1056_seed2.yaml` | E010 | train_args.imgsz 640 → 1056 (D-042 arm D); seed 0 → 2; dataset v7, otherwise as E010 | completed 2026-11-12 on Kaggle (Tesla T4, one of two in parallel); 100 epochs; registered; val point AP 0.471; not a final candidate (D-042: no arm wins) |
| E039 | `configs/experiments/E039_seaslug_v8_seed0.yaml` | E010 | dataset v7 → v8 only (D-041: v7 plus 2,662 batch-2 training-day frames with fixed boxes; val and test unchanged); otherwise as E010 | completed 2026-11-13 on Kaggle (Tesla T4, one of two in parallel); 35 epochs (early stop, patience 30); registered; val point AP 0.509; not a final candidate (D-041: v8 does not win) |
| E040 | `configs/experiments/E040_seaslug_v8_seed1.yaml` | E010 | dataset v7 → v8 (D-041); seed 0 → 1; otherwise as E010 | completed 2026-11-13 on Kaggle (Tesla T4, one of two in parallel); 43 epochs (early stop, patience 30); registered; val point AP 0.461; not a final candidate (D-041: v8 does not win) |
| E041 | `configs/experiments/E041_seaslug_v8_seed2.yaml` | E010 | dataset v7 → v8 (D-041); seed 0 → 2; otherwise as E010 | completed 2026-11-13 on Kaggle (Tesla T4, one of two in parallel); 59 epochs (early stop, patience 30); registered; val point AP 0.502; not a final candidate (D-041: v8 does not win) |
| E042 | `configs/experiments/E042_seaslug_base_seed1.yaml` | E010 | seed 0 → 1 only (E010 repeated, D-042 baseline); dataset v7, otherwise as E010 | completed 2026-11-12 on Kaggle (Tesla T4, one of two in parallel); 100 epochs; registered; val point AP 0.478; not a final candidate (D-042: no arm wins) |
| E043 | `configs/experiments/E043_seaslug_base_seed2.yaml` | E010 | seed 0 → 2 only (E010 repeated, D-042 baseline); dataset v7, otherwise as E010 | completed 2026-11-12 on Kaggle (Tesla T4, one of two in parallel); 100 epochs; registered; val point AP 0.493; not a final candidate (D-042: no arm wins) |
| E044 | `configs/experiments/E044_seaslug_ep200_seed0.yaml` | E010 | train_args.epochs 100 → 200 and patience 30 → 50 (D-042 arm A, together the stopping rule); dataset v7, otherwise as E010 | completed 2026-11-12 on Kaggle (Tesla T4, one of two in parallel); 176 epochs; registered; val point AP 0.435; not a final candidate (D-042: no arm wins) |
| E045 | `configs/experiments/E045_seaslug_ep200_seed1.yaml` | E010 | train_args.epochs 100 → 200 and patience 30 → 50 (D-042 arm A, together the stopping rule); seed 0 → 1; dataset v7, otherwise as E010 | completed 2026-11-12 on Kaggle (Tesla T4, one of two in parallel); 97 epochs; registered; val point AP 0.494; not a final candidate (D-042: no arm wins) |
| E046 | `configs/experiments/E046_seaslug_ep200_seed2.yaml` | E010 | train_args.epochs 100 → 200 and patience 30 → 50 (D-042 arm A, together the stopping rule); seed 0 → 2; dataset v7, otherwise as E010 | completed 2026-11-12 on Kaggle (Tesla T4, one of two in parallel); 165 epochs; registered; val point AP 0.463; not a final candidate (D-042: no arm wins) |

**Model A tuning E011–E024 (D-038), declared 2026-11-11 before any of these runs:** the selection metric
is **val mAP50-95** (class `lobster`), at both levels: Ultralytics picks each run's `best.pt` on it (in
8.4.155 fitness = mAP50-95, OBSERVED in E007's run_record), and runs/configurations are compared on the
seed mean of it (D-038 points 4 and 6). Val mAP50, P and R are reported next to it. This replaces val
mAP50 as the primary metric for these runs only; earlier experiments keep their declared metrics.
Comparisons with E001/E007 show both (mAP50-95 / mAP50: E001 0.6115 / 0.8527, E007 0.5993 / 0.8423).
Why (for the report, D-038 point 8): trajectories are built from box positions, so box tightness matters
as well as finding the lobster; mAP50-95 measures both. Neither is a tracking metric (CLAUDE.md rule 9).

**Model A round 2 E020–E029 (D-040), declared 2026-11-12 before any of these runs:** round 1 selected
**yolo26m** (E012/E014/E017, val mAP50-95 mean 0.61127, range 0.01415; D-038 point 4 read as "the top model
against each smaller one", D-040 point 1). Round 2 runs five combinations C1–C5 on yolo26m and yolo26l, all
with a 30-epoch schedule that is not stopped early (D-040 point 2). Selection metric unchanged (val mAP50-95). A
round-2 run replaces the yolo26m round-1 configuration only if its val mAP50-95 is above **0.62542** (mean +
range); among runs above it the highest is selected and declared alone. Otherwise E012, E014 and E017 are the
final candidates (demo model E017). Val mAP50, P, R and Ultralytics' confusion matrix counts are reported next
to it and do not select. LIMITATION: up to 19 configurations selected on 324 val images; round 2 is single-seed.

**Model A round 2 result (D-043, 2026-11-13, OBSERVED):** all 10 runs finished 30 epochs at commit 97810a5 (clean); val mAP50-95 of best.pt:

| Combination | yolo26m | yolo26l |
|---|---|---|
| C1 schedule only (control) | E020 0.6150 | E021 0.6171 |
| C2 + AdamW lr0 0.001 | E022 0.6197 | E023 0.6172 |
| C3 + mixup 0.1 | E024 0.6179 | E025 0.6227 |
| C4 + AdamW lr0 0.001, weight_decay 0.001, mixup 0.1 | E026 0.6219 | **E027 0.6280** |
| C5 + freeze 10 | E028 0.5799 | E029 0.5899 |

Only **E027** (yolo26l, C4) is above the bar 0.62542 (0.6280, margin 0.0026): it is selected and declared alone (val mAP50 0.8537, P 0.7792, R 0.8307; confusion matrix 847 found / 163 missed / 239 false alarms of 1010 val boxes, round-1 yolo26m mean 829 / 181 / 271). It replaces E012/E014/E017 as the final candidate and the demo model. LIMITATION: single seed, small margin, and one of 19 configurations compared on the same 324 val images, so its val score is optimistic; only its test evaluation is unbiased. Final test (2026-11-13, once per target, Test-evaluation log): v1 test mAP50-95 0.6966 (mAP50 0.9096); Zenodo `sk` mAP50-95 0.3892 (mAP50 0.5663).

**E006 (Tritonia sea slugs, D-030), declared before any result:** selection metric val mAP50 of class
`sea_slug`; also reported per class: P, R, mAP50-95 on val, and false positives on the confirmed-negative
val frames if there are any. E006 is not compared with E001–E005: different data and classes. Its 3 test
days are evaluated only after a sea slug final-test procedure is declared here. LIMITATION: 150 frames,
3 val and 3 test days from one setup; results measure new days, not new sites.
Updated by D-033 (2026-11-04, declared before any E006 run): the selection metric is **val point AP**
against the MTrackJ crosses (10 px, one to one), with point precision, recall and F1 at the val-best-F1
threshold, overall and per day. Ultralytics' P, R and mAP are against the automatic boxes and are not
accuracy. Test only for a declared final candidate, with the val threshold (`--final`, `--conf`).

**E006 final-test procedure, declared 2026-11-10 before any test result:** E006 is declared in
`experiments/final_candidates.csv`. The test is run once, by cell 18 of
`notebooks/colab_train_tritonia.ipynb` (GPU, no training): it checks the declaration, the registered
`best.pt` (sha256 `6f6710d1…`) and the val threshold **conf 0.06**, then runs
`scripts/tritonia_point_eval.py --split test --final E006 --conf 0.06` on all 27 frames of the 3 test
days (127 crosses), with the same 10 px one-to-one matching. Reported: point P, R, F1 at conf 0.06 and
point AP, overall and per day, logged in the test-evaluation log below. The test result changes
neither the model nor the threshold.

**E008 vs E006 (Tritonia, imgsz 1280), declared 2026-11-10 before any E008 result:** E008 is E006
with only `imgsz` 640 → 1280, so both see the same v3 frames, automatic boxes, day split and seed. They
are compared on **val point AP** (D-033), then on val point P / R / F1 at each run's own val-best-F1
threshold, overall and per day; the val point eval of E008 uses imgsz 1280 for inference, as trained.
Ultralytics' val mAP of both is shown, labelled not accuracy (automatic boxes). E006 stays the declared
final candidate. E008 is declared as a second final candidate only after it is registered and only if
its val point AP is higher than E006's 0.403; its threshold is then its own val-best-F1 one. Test
numbers of E006 and E008 are reported side by side and never used to choose. LIMITATION: one run per
arm (seed 0) on 27 val frames: a small difference may be noise.

**E009 vs E006 (Tritonia, more training frames), declared 2026-11-10 before any E009 result:** E009 is E006
with only the dataset changed, v3 → v5 (D-035). v5 holds every v3 frame unchanged (checked by
`make_tritonia_version.py --extends v3`) plus complete frames of batch 2 from the 10 training days, so the
27 val and 27 test frames are the same as E006's. Comparison as for E008: **val point AP**, then val point
P / R / F1 at each run's own val-best-F1 threshold, overall and per day. E009 is declared as a final
candidate only after registration and only if its val point AP is higher than E006's 0.403; then its test
is run once with its own val threshold, and E006's and E009's test numbers are reported side by side
(test never chooses). E008 and E009 each change one variable from E006; combining them would be a new,
separately declared experiment. LIMITATION: one run per arm (seed 0), 27 val frames.

**E010 vs E006 (Tritonia, fixed boxes on every cross), declared 2026-11-11 before any E010 result:** E010 is
E006 with only the dataset changed, v3 → v7 (D-036; v7 replaces v6, D-037): same frames, images and day split, so the 27 val and 27
test frames and their 153 / 127 crosses are E006's. Comparison as for E008 and E009: **val point AP**, then
val point P / R / F1 at each run's own val-best-F1 threshold, overall and per day. Ultralytics' val mAP of
E010 is against the fixed boxes and is not comparable with E006's (different val boxes), nor accuracy.
E010 is declared as a final candidate only after registration and only if its val point AP is higher than
E006's 0.403; then its test is run once with its own val threshold and reported beside E006's (test never
chooses). LIMITATION: one run per arm (seed 0), 27 val frames.

**Tritonia tuning round 1 E030–E038 and E042–E046 (D-042), declared 2026-11-12 before any of these runs:** baseline = E010's
settings on v7 (seed 0 = E010, seeds 1 / 2 = E042 / E043); arms A (epochs 200, patience 50), B (batch 8),
C (scale 0.2), D (imgsz 1056), each seeds 0 / 1 / 2 and one setting changed from E010. Selection metric:
**val point AP** (D-033) on all 27 val frames, 153 crosses, 10 px one to one, each run at its own best-val-F1
threshold; per arm the **mean over 3 seeds** and the seed range. An arm wins only if its mean exceeds the
baseline mean by more than the larger of the two seed ranges; fewer than 3 finished seeds cannot win. Point
P / R / F1, per day, and Ultralytics' val metrics (not accuracy) are reported next to it. Round 2 (winning
settings combined, 3 seeds, configs committed first) only if an arm wins; a test of its result only after a
separate user approval; no test in round 1. LIMITATION: four arms compared on 27 val frames from 3 days.

**Tritonia v8, more training frames E039–E041 (D-041), declared 2026-11-12 before any of these runs:** arm v8 =
E010's settings with only the dataset changed, v7 → v8 (train 96 → 2,758 frames, 572 → 18,950 fixed boxes; the
27 val and 27 test frames are v7's, unchanged, so their 153 / 127 crosses are the same), seeds 0 / 1 / 2 =
E039 / E040 / E041. Reference = the D-042 baseline (E010, E042, E043: mean val point AP 0.489, seed range
0.018), not re-run. Selection metric and rule as D-042: **val point AP** on all 27 val frames, 153 crosses,
10 px one to one, each run at its own best-val-F1 threshold; **mean over 3 seeds** and the seed range; v8 wins
only if its mean exceeds 0.489 by more than the larger of the two seed ranges; fewer than 3 finished seeds cannot
win. Point P / R / F1, per day, and Ultralytics' val metrics (not accuracy) are reported next to it. If v8 wins,
the seed with the highest val point AP is the candidate; a test of it only after a separate user approval, run
once with its own val threshold and reported beside E006's and E010's (test never chooses). LIMITATIONS: 27 val
frames from 3 days; added frames are 75 s apart and partly redundant; unclicked slugs in added frames are
background examples (D-029).

E001–E003 and E005 use frozen dataset v1 (fingerprint `9f2bf8573df4354e1373786967a1155802e4cdbfe127e1f07fd3fdb1983ef810`).
E004 uses dataset v2 (fingerprint `8a0d4194d1f7fb3972ad33b09cf570988aebb5350c5d856851056c961528626e`):
v1's sessions, split and labels byte for byte, with Harsh's enhanced images. All use ultralytics
8.4.155. Colab procedure: docs/COLAB.md.

**YOLO vs RT-DETR (D-015, D-025):** the RT-DETR side is now E005 (E003 diverged). E001 and E005 are compared on val mAP50, then on the secondary
metrics above, with parameter count, GFLOPs and inference speed on the same GPU type reported
alongside, and E005's AMP-off training is stated. E005 is evaluated on test only after it is registered, reviewed and declared in
`experiments/final_candidates.csv`; test numbers are reported, never used to choose.

**Raw vs enhanced images (D-017), declared before any E004 result:**
- The question is whether Harsh's enhancement changes detection quality for the E001 detector.
  E004 is E001 with only the images changed, so both use the same sessions, split, labels,
  model, settings and seed.
- **Comparison:** E001 and E004 are compared on val mAP50, then on the secondary metrics above.
  Both are measured on the same 324 val frames and boxes, each seen through its own image
  version (raw vs enhanced).
- **Operating point:** E004's confidence is chosen on E004's own val, never copied from E001.
- **Test:** E004 is evaluated on test (the same 348 v1 test frames, enhanced) only after it is
  registered, reviewed and declared in `experiments/final_candidates.csv`. E001's and E004's test
  numbers are reported side by side and are never used to choose between them.
- **LIMITATION:** one run per arm with seed 0, so a small difference may be run-to-run noise.
  No claim of "better" beyond what val shows consistently across the declared metrics.
- **zExtra** (raw or enhanced) gives qualitative observations only: it has no labels.
- **Reporting:** E004 results are not reported before Harsh's answers are recorded in D-017.
  Updated by D-018 (2026-10-01): the answers are recorded. Val results and this val comparison
  may now be reported with D-018's limitations. The E004 test evaluation waits for the
  review-list check in D-018.
  Updated by D-022 (2026-10-12): the check is done. E004 is declared in
  `experiments/final_candidates.csv` and may be evaluated on test once. LIMITATION (D-022): one
  val image (`21_SUB6_10_BW_20220815_GP080026_13777`, 7 boxes) and one test image
  (`06_HERR_08_CW_20200903_GP120003_00001`, 1 box) were among the images Harsh reviewed while
  choosing the enhancement profile; whether others were viewed is UNKNOWN.

## Run record requirements

Each run writes `run_record.json` (src/training/run_record.py) in its off-repo run folder; full runs are
added to `experiments/registry.csv` with `scripts/register_run.py` on the local PC:

- git commit + dirty flag, dataset version + fingerprint,
- model + pretrained checkpoint, Ultralytics args, seed,
- env report (GPU, VRAM, CUDA, torch, ultralytics), duration,
- val metrics, and weights path + sha256.

## Run log

| ID | Date | Parent | Changed | Dataset | Val mAP50 | Val mAP50-95 | Notes |
|---|---|---|---|---|---:|---:|---|
| E001 | 2026-09-19 | none | initial baseline: yolo26s.pt, imgsz 640, ≤100 epochs, patience 30, batch 16, seed 0, deterministic | v1 | 0.8527 | 0.6115 | 100 epochs completed; best.pt recorded; final candidate |
| E004 | 2026-09-30 | E001 | images raw → enhanced only (dataset v1 → v2, D-017) | v2 | 0.8385 | 0.5864 | 81 epochs (early stop, patience 30); best.pt recorded; not a final candidate (D-018) |
| E005 | 2026-10-15 | E001 | rtdetr-l.pt, amp off (D-025); effective batch 8 (D-026) | v1 | 0.8359 | 0.5884 | 100 epochs completed over 5 launches; best.pt recorded, later lost on Drive (final test uses the recovered epoch-41 best.pt, D-027); final candidate |
| E007 | 2026-11-08 | E001 | dataset v1 → v4 only: all 4,111 safe Model A training images (D-032) | v4 | 0.8423 | 0.5993 | 58 epochs (early stop at 28 + 30); best.pt recorded; Kaggle; not a final candidate |
| E006 | 2026-11-10 | E001 | dataset v1 → v3 only: Tritonia sea slugs, automatic boxes (D-030, D-033) | v3 | — | — | not mAP (D-033): val point AP 0.403; point P / R / F1 0.504 / 0.451 / 0.476 at conf 0.06; 100 epochs; best.pt recorded; Kaggle; final candidate |
| E008 | 2026-11-10 | E006 | imgsz 640 → 1280 only | v3 | — | — | not mAP (D-033): val point AP 0.317; point P / R / F1 0.436 / 0.379 / 0.406 at conf 0.12; 100 epochs; best.pt recorded; Kaggle; registered; not a final candidate (below 0.403) |
| E009 | 2026-11-10 | E006 | dataset v3 → v5 only (D-035) | v5 | — | — | not mAP (D-033): val point AP 0.274; point P / R / F1 0.340 / 0.333 / 0.337 at conf 0.06; Kaggle; not registered (run record lost); not a final candidate (below 0.403) |
| E010 | 2026-11-12 | E006 | dataset v3 → v7 only: fixed 20 × 17 px box on every cross (D-036, D-037) | v7 | — | — | not mAP (D-033): val point AP 0.496; point P / R / F1 0.614 / 0.458 / 0.524 at conf 0.16; 100 epochs; best.pt recorded; Kaggle; registered; final candidate; tested once 2026-11-12 |
| E011 | 2026-11-11 | E007 | yolo26s seed 0, ≤60 epochs, patience 10 (D-038 round 1) | v4 | 0.8320 | 0.5999 | 41 epochs (early stop, best epoch 31); best.pt recorded; Kaggle, two runs in parallel; registered; not selected (D-040) |
| E012 | 2026-11-12 | E007 | yolo26m seed 0, ≤60 epochs, patience 10 (D-038 round 1) | v4 | 0.8408 | 0.6030 | 27 epochs (early stop, best epoch 17); best.pt recorded; Kaggle, two runs in parallel; registered; round-1 selected configuration (yolo26m, D-040) |
| E013 | 2026-11-11 | E007 | yolo26l seed 0, ≤60 epochs, patience 10 (D-038 round 1) | v4 | 0.8495 | 0.6124 | 34 epochs (early stop, best epoch 24); best.pt recorded; Kaggle, two runs in parallel; registered; not selected (D-040) |
| E014 | 2026-11-12 | E007 | yolo26m seed 1, ≤60 epochs, patience 10 (D-038 round 1) | v4 | 0.8481 | 0.6136 | 26 epochs (early stop, best epoch 16); best.pt recorded; Kaggle, two runs in parallel; registered; round-1 selected configuration (yolo26m, D-040) |
| E015 | 2026-11-12 | E007 | yolo26l seed 1, ≤60 epochs, patience 10 (D-038 round 1) | v4 | 0.8538 | 0.6160 | 27 epochs (early stop, best epoch 17); best.pt recorded; Kaggle, two runs in parallel; registered; not selected (D-040) |
| E016 | 2026-11-12 | E007 | yolo26s seed 1, ≤60 epochs, patience 10 (D-038 round 1) | v4 | 0.8325 | 0.5965 | 27 epochs (early stop, best epoch 17); best.pt recorded; Kaggle, two runs in parallel; registered; not selected (D-040) |
| E017 | 2026-11-12 | E007 | yolo26m seed 2, ≤60 epochs, patience 10 (D-038 round 1) | v4 | 0.8509 | 0.6172 | 40 epochs (early stop, best epoch 30); best.pt recorded; Kaggle, two runs in parallel; registered; round-1 selected configuration (yolo26m, D-040) |
| E019 | 2026-11-12 | E007 | yolo26s seed 2, ≤60 epochs, patience 10 (D-038 round 1) | v4 | 0.8410 | 0.5965 | 26 epochs (early stop, best epoch 16); best.pt recorded; Kaggle, two runs in parallel; registered; not selected (D-040) |
| E020 | 2026-11-13 | E012 | yolo26m seed 0, 30 epochs, patience 30, C1 schedule only (control) (D-040 round 2) | v4 | 0.8409 | 0.6150 | 30 epochs (schedule complete, best epoch 29); best.pt recorded; Kaggle, two runs in parallel; registered; not selected (D-043) |
| E021 | 2026-11-13 | E013 | yolo26l seed 0, 30 epochs, patience 30, C1 schedule only (control) (D-040 round 2) | v4 | 0.8474 | 0.6171 | 30 epochs (schedule complete, best epoch 26); best.pt recorded; Kaggle, two runs in parallel; registered; not selected (D-043) |
| E022 | 2026-11-13 | E012 | yolo26m seed 0, 30 epochs, patience 30, C2 + AdamW lr0 0.001 (D-040 round 2) | v4 | 0.8459 | 0.6197 | 30 epochs (schedule complete, best epoch 23); best.pt recorded; Kaggle, two runs in parallel; registered; not selected (D-043) |
| E023 | 2026-11-13 | E013 | yolo26l seed 0, 30 epochs, patience 30, C2 + AdamW lr0 0.001 (D-040 round 2) | v4 | 0.8457 | 0.6172 | 30 epochs (schedule complete, best epoch 25); best.pt recorded; Kaggle, two runs in parallel; registered; not selected (D-043) |
| E024 | 2026-11-13 | E012 | yolo26m seed 0, 30 epochs, patience 30, C3 + mixup 0.1 (D-040 round 2) | v4 | 0.8494 | 0.6179 | 30 epochs (schedule complete, best epoch 23); best.pt recorded; Kaggle, two runs in parallel; registered; not selected (D-043) |
| E025 | 2026-11-13 | E013 | yolo26l seed 0, 30 epochs, patience 30, C3 + mixup 0.1 (D-040 round 2) | v4 | 0.8602 | 0.6227 | 30 epochs (schedule complete, best epoch 25); best.pt recorded; Kaggle, two runs in parallel; registered; not selected (D-043) |
| E026 | 2026-11-13 | E012 | yolo26m seed 0, 30 epochs, patience 30, C4 + AdamW lr0 0.001, weight_decay 0.001, mixup 0.1 (D-040 round 2) | v4 | 0.8515 | 0.6219 | 30 epochs (schedule complete, best epoch 20); best.pt recorded; Kaggle, two runs in parallel; registered; not selected (D-043) |
| E027 | 2026-11-13 | E013 | yolo26l seed 0, 30 epochs, patience 30, C4 + AdamW lr0 0.001, weight_decay 0.001, mixup 0.1 (D-040 round 2) | v4 | 0.8537 | 0.6280 | 30 epochs (schedule complete, best epoch 25); best.pt recorded; Kaggle, two runs in parallel; registered; **selected; declared final candidate (D-043)**; val confusion matrix 847 found / 163 missed / 239 false alarms of 1010; final test 2026-11-13: v1 test mAP50-95 0.6966, Zenodo `sk` 0.3892 (Test-evaluation log) |
| E028 | 2026-11-13 | E012 | yolo26m seed 0, 30 epochs, patience 30, C5 + freeze 10 (D-040 round 2) | v4 | 0.8313 | 0.5799 | 30 epochs (schedule complete, best epoch 25); best.pt recorded; Kaggle, two runs in parallel; registered; not selected (D-043) |
| E029 | 2026-11-13 | E013 | yolo26l seed 0, 30 epochs, patience 30, C5 + freeze 10 (D-040 round 2) | v4 | 0.8428 | 0.5899 | 30 epochs (schedule complete, best epoch 25); best.pt recorded; Kaggle, two runs in parallel; registered; not selected (D-043) |
| E030 | 2026-11-12 | E010 | D-042 arm B, seed 0 | v7 | — | — | not mAP (D-033): val point AP 0.472; point P / R / F1 0.516 / 0.523 / 0.519 at conf 0.12; 100 epochs; best.pt recorded; Kaggle; registered |
| E031 | 2026-11-12 | E010 | D-042 arm B, seed 1 | v7 | — | — | not mAP (D-033): val point AP 0.448; point P / R / F1 0.458 / 0.503 / 0.480 at conf 0.14; 100 epochs; best.pt recorded; Kaggle; registered |
| E032 | 2026-11-12 | E010 | D-042 arm B, seed 2 | v7 | — | — | not mAP (D-033): val point AP 0.496; point P / R / F1 0.535 / 0.542 / 0.539 at conf 0.16; 100 epochs; best.pt recorded; Kaggle; registered |
| E033 | 2026-11-12 | E010 | D-042 arm C, seed 0 | v7 | — | — | not mAP (D-033): val point AP 0.472; point P / R / F1 0.559 / 0.464 / 0.507 at conf 0.12; 82 epochs; best.pt recorded; Kaggle; registered |
| E034 | 2026-11-12 | E010 | D-042 arm C, seed 1 | v7 | — | — | not mAP (D-033): val point AP 0.471; point P / R / F1 0.723 / 0.392 / 0.508 at conf 0.26; 100 epochs; best.pt recorded; Kaggle; registered |
| E035 | 2026-11-12 | E010 | D-042 arm C, seed 2 | v7 | — | — | not mAP (D-033): val point AP 0.491; point P / R / F1 0.575 / 0.451 / 0.505 at conf 0.13; 78 epochs; best.pt recorded; Kaggle; registered |
| E036 | 2026-11-12 | E010 | D-042 arm D, seed 0 | v7 | — | — | not mAP (D-033): val point AP 0.482; point P / R / F1 0.532 / 0.484 / 0.507 at conf 0.1; 100 epochs; best.pt recorded; Kaggle; registered |
| E037 | 2026-11-12 | E010 | D-042 arm D, seed 1 | v7 | — | — | not mAP (D-033): val point AP 0.481; point P / R / F1 0.523 / 0.582 / 0.551 at conf 0.18; 100 epochs; best.pt recorded; Kaggle; registered |
| E038 | 2026-11-12 | E010 | D-042 arm D, seed 2 | v7 | — | — | not mAP (D-033): val point AP 0.471; point P / R / F1 0.608 / 0.477 / 0.535 at conf 0.2; 100 epochs; best.pt recorded; Kaggle; registered |
| E039 | 2026-11-13 | E010 | D-041 v8, seed 0 | v8 | — | — | not mAP (D-033): val point AP 0.509; point P / R / F1 0.677 / 0.451 / 0.541 at conf 0.23; 35 epochs (early stop); best.pt recorded; Kaggle; registered |
| E040 | 2026-11-13 | E010 | D-041 v8, seed 1 | v8 | — | — | not mAP (D-033): val point AP 0.461; point P / R / F1 0.596 / 0.425 / 0.496 at conf 0.15; 43 epochs (early stop); best.pt recorded; Kaggle; registered |
| E041 | 2026-11-13 | E010 | D-041 v8, seed 2 | v8 | — | — | not mAP (D-033): val point AP 0.502; point P / R / F1 0.597 / 0.523 / 0.558 at conf 0.18; 59 epochs (early stop); best.pt recorded; Kaggle; registered |
| E042 | 2026-11-12 | E010 | D-042 baseline seed 1 | v7 | — | — | not mAP (D-033): val point AP 0.478; point P / R / F1 0.698 / 0.392 / 0.502 at conf 0.28; 100 epochs; best.pt recorded; Kaggle; registered |
| E043 | 2026-11-12 | E010 | D-042 baseline seed 2 | v7 | — | — | not mAP (D-033): val point AP 0.493; point P / R / F1 0.552 / 0.516 / 0.534 at conf 0.17; 100 epochs; best.pt recorded; Kaggle; registered |
| E044 | 2026-11-12 | E010 | D-042 arm A, seed 0 | v7 | — | — | not mAP (D-033): val point AP 0.435; point P / R / F1 0.574 / 0.405 / 0.475 at conf 0.14; 176 epochs; best.pt recorded; Kaggle; registered |
| E045 | 2026-11-12 | E010 | D-042 arm A, seed 1 | v7 | — | — | not mAP (D-033): val point AP 0.494; point P / R / F1 0.547 / 0.497 / 0.520 at conf 0.15; 97 epochs; best.pt recorded; Kaggle; registered |
| E046 | 2026-11-12 | E010 | D-042 arm A, seed 2 | v7 | — | — | not mAP (D-033): val point AP 0.463; point P / R / F1 0.728 / 0.386 / 0.504 at conf 0.25; 165 epochs; best.pt recorded; Kaggle; registered |

E005 is registered in `experiments/registry.csv`; its run record is archived in `experiments/records/E005/`.
The archived `run_record.json` was saved from the record printed in the Colab output (2026-10-17), not
downloaded. Its output sha256 values, sizes and val metrics match the notebook outputs in Git, and its
config sha256 matches `E005_rtdetr_amp_off.yaml`. Byte-identity with the Drive file is checked by
notebook cell 15a: UNKNOWN until that cell prints MATCH.

### E001 training observations

Registered in `experiments/registry.csv`; metadata archived in `experiments/records/E001/` (D-013).
From `results.csv` (OBSERVED, 2026-09-21):

- The best epoch is **61**, on both val mAP50 (0.8529) and val mAP50-95 (0.6110). Because epoch 61
  maximizes both metrics, it maximizes any weighted combination of them, so it is the fitness peak
  whichever weights 8.4.155 uses — D-012 records those weights as UNKNOWN. The registered
  val mAP50 0.8527 is the post-training validation of `best.pt`, against 0.8529 measured in-training.
- Training ran all **100 epochs, 39 of them after the peak**, although `E001_yolo_baseline.yaml`
  sets `patience: 30`. Early stopping should have triggered near epoch 91.
  INFERRED: patience is either counted differently in 8.4.155 than the config implies, or was not
  applied. UNKNOWN — requires confirmation from the Colab training log.
- No effect on any reported metric: `best.pt` is the epoch-61 checkpoint and every val and test
  figure above comes from it. The cost was GPU time only.
- LIMITATION for E002: at imgsz 1280 on a Tesla T4, budget for a full 100 epochs rather than
  relying on an early stop.

### E004 training observations

Registered in `experiments/registry.csv`; metadata archived in `experiments/records/E004/` (D-013):
`run_record.json` (9,115 B, sha256 `a6209080…`) and `results.csv` (9,840 B, sha256 `eaaa929b…`,
matching the record). From these files (OBSERVED, 2026-10-01):

- The run used one uninterrupted launch on a Tesla T4: 9,566 s (2.66 h), about 117 s per epoch.
  Commit `97ac710`, dataset v2 `8a0d4194…`, the same `yolo26s.pt` checkpoint as E001
  (`646f8bc3…`), and the same torch 2.11.0+cu128 / CUDA 12.8 / ultralytics 8.4.155 as E001.
- The best epoch is **51**, on both val mAP50 (0.8383) and val mAP50-95 (0.5846), so it is the
  fitness peak whatever the weights. The registered val mAP50 0.8385 is the post-training
  validation of `best.pt`.
- Training **stopped early at epoch 81**, which is 51 + 30 (`patience: 30`). Early stopping worked
  as configured. INFERRED: E001 ran all 100 epochs (see above) because it was resumed twice, and
  the patience count may not survive a resume in 8.4.155; E004 was never resumed. UNKNOWN —
  requires confirmation.
- Val mAP50 swung widely in the early epochs (0.226 at epoch 3, 0.484 at epoch 11) and settled
  later. The train losses fell steadily throughout.
- **Val comparison with E001 (declared rule above; D-018 limitations apply):** mAP50 0.8385 vs
  0.8527 (−0.014); mAP50-95 0.5864 vs 0.6115 (−0.025); precision 0.8056 vs 0.7998; recall
  0.7465 vs 0.7901 (all at Ultralytics val defaults). One run per arm, so these gaps may include
  run-to-run noise. The secondary metrics (recall at a val-chosen confidence, false positives on
  negative images, per-condition results) are not computed yet (computed on 2026-10-11, see the
  next section). LIMITATION: E004 trained 81
  epochs and E001 trained 100. It is UNKNOWN whether E004 would have improved after epoch 81;
  E001 did not improve after its epoch-61 peak.

### E007 training observations

Registered in `experiments/registry.csv`; metadata archived in `experiments/records/E007/` (D-013):
`run_record.json` (9,820 B, sha256 `b678ded4…`) and `results.csv` (7,096 B, sha256 `02768a40…`,
matching the record). From these files and the Kaggle notebook output (OBSERVED, 2026-11-08 UTC):

- One uninterrupted launch in an interactive Kaggle session on one Tesla T4 (2 × T4 attached, one
  used): 6,498 s (1.81 h), about 112 s per epoch. Commit `428f5bd`, dataset v4 `bbd4a8b1…` (frozen,
  D-032), the same `yolo26s.pt` checkpoint as E001 (`646f8bc3…`), torch 2.11.0+cu130, CUDA 13.0,
  Ultralytics 8.4.155.
- Before training, in the same notebook: archive sha256 checks and raw checks of all four raw folders
  passed; the built view's fingerprint matched the config; `verify_yolo_dataset.py` printed VERIFIED
  for train, val, test and test_zenodo_sk; `check_leakage.py` flagged only the near-duplicate pair
  reviewed in D-031 (different scenes); smoke test passed.
- The best epoch is **28**, on both val mAP50 (0.8420 in training) and val mAP50-95 (0.5985), so
  it is the fitness peak whatever the weights. The registered val mAP50 0.8423 is the post-training
  validation of `best.pt`. Training **stopped early at epoch 58** = 28 + 30 (`patience: 30`), never
  resumed, as in E004.
- From epoch 9 to 58, val mAP50 stayed between 0.776 and 0.842. Train losses fell throughout
  (box 1.298 → 0.834, cls 1.768 → 0.469), while val losses were lowest near epoch 28 and rose after
  (val box 1.158 at epoch 28 → 1.213 at epoch 58; val cls 1.039 → 1.085). INFERRED: the model
  overfits the training data after about epoch 28.
- **Val comparison with E001** (the same 324 frames; Ultralytics val defaults): mAP50 0.8423 vs
  0.8527 (−0.010); mAP50-95 0.5993 vs 0.6115 (−0.012); precision 0.8021 vs 0.7998; recall
  0.7783 vs 0.7901. OBSERVED: about 2.9 times the training images (4,111 vs E001's 1,433) did not raise any
  val metric except precision (+0.002).
- LIMITATIONS: one run per arm (seed 0), so gaps of about 0.01 may be run-to-run noise; E001 ran on
  Colab and E007 on Kaggle (both Tesla T4; torch/CUDA builds differ: cu128 vs cu130); without the
  data ladder (D-032) no single source can be blamed or credited. Val holds only UWCT v1 sessions,
  while 2,319 of E007's train images are derived and 779 come from Zenodo: HYPOTHESIS, not tested,
  that the added data helps on new sites (Zenodo `sk`) without helping on our own sessions; `sk` is
  a test split, evaluated only for a declared final candidate (rule 7).
- Off-repo outputs (referenced by path + sha256, CLAUDE.md §5): `best.pt` 20,289,541 B sha256
  `f5f5a249…`, `last.pt` sha256 `9cbaefdf…`, in `E007_package.zip` (86 files: run folders of E007
  and its smoke test, and every notebook log; 86,522,254 B, sha256 `8d57c662…`), downloaded to
  `D:\uwct_runs\E007\` on the user's PC and checked (all four sha256 match). The Kaggle
  `/kaggle/working` copy ends with the session.
- The secondary metrics (recall at a val-chosen confidence, false positives on negative images,
  per-condition results, D-020) are not computed yet for E007.

### E006 training observations (Tritonia sea slugs)

Registered in `experiments/registry.csv`; metadata archived in `experiments/records/E006/` (D-013):
`run_record.json` (10,037 B, sha256 `6b1b7083…`), `results.csv` (12,443 B, sha256 `4c7c8126…`,
matching the record) and `val_point_scores.json` (1,266 B, sha256 `c23952b5…`), all downloaded from
the saved Kaggle notebook version. From these files and the Kaggle log (OBSERVED, 2026-11-10 UTC):

- **Two identical training runs.** E006 was first trained in the interactive Kaggle session
  (2026-11-09 23:32–23:36 UTC, 219 s). Saving the notebook version re-ran every cell, so E006 was
  trained again (2026-11-10 00:08–00:12 UTC, 231 s), with the smoke test before it. Both at commit
  `2d3b6cb`, on one Tesla T4 (2 × T4 attached), seed 0, deterministic. Both gave the same val metrics
  to every printed digit and the same point scores, per day too. Their checkpoint files differ
  (`best.pt` `4123869b…` vs `6f6710d1…`); INFERRED: Ultralytics stores a save timestamp in each
  checkpoint. The first run's weights were not downloaded and are gone; **the registered run is the
  second one**, the only one whose weights exist.
- **Resume launch with no effect on E006.** The saved version also ran the resume cell (`RESUME =
  True`) on the finished run. Ultralytics refused to resume ("not a resumable training checkpoint")
  and started a new training on its own `coco8.yaml` sample in `/tmp/weights/runs/detect/train`
  (OBSERVED in the log). `best.pt`, `last.pt`, `results.csv` and the val metrics of `runs/E006` have
  the same sha256 and values before and after that launch. The record's launch list still shows it as
  "resume, completed, 137 s"; that entry is wrong and is not training time of E006. DEFECT in
  `train_yolo.py --resume`: it does not refuse a run that has already finished (fix pending).
- Commit `2d3b6cb`, dataset v3 `c1d42afa…` (frozen, D-030), the same `yolo26s.pt` checkpoint as
  E001 (`646f8bc3…`), torch 2.11.0+cu130, CUDA 13.0, Ultralytics 8.4.155. Train 86 frames / 292
  automatic boxes, val 20 frames / 73 boxes in the YOLO view (frames without a box are excluded,
  D-033); no negative frames, so no false-positive-on-negatives figure.
- All **100 epochs** ran. The best epoch by Ultralytics' fitness is **79** (in-training val mAP50
  0.6162, mAP50-95 0.4191), 21 epochs before the end, so patience 30 never triggered. Val mAP50 was 0
  until epoch 6 and varied widely later (0.150–0.616 from epoch 40 on). Train losses fell throughout
  (box 3.05 → 1.23, cls 10.55 → 1.06).
- **Ultralytics val of `best.pt` (against the automatic boxes, NOT accuracy, D-033):** P 0.731,
  R 0.558, mAP50 0.619, mAP50-95 0.417.
- **Accuracy: val point scores against the MTrackJ crosses** (`scripts/tritonia_point_eval.py`,
  all 27 val frames incl. the 7 the automatic labels excluded, 153 crosses, 10 px one to one;
  threshold = best val F1):

  | val day | crosses | predictions | matched | point P | point R | F1 |
  |---|---:|---:|---:|---:|---:|---:|
  | 2013-05-24 | 15 | 29 | 2 | 0.069 | 0.133 | 0.091 |
  | 2013-05-29 | 77 | 40 | 33 | 0.825 | 0.429 | 0.564 |
  | 2013-06-02 | 61 | 68 | 34 | 0.500 | 0.557 | 0.527 |
  | **all val** | 153 | 137 | 69 | 0.504 | 0.451 | 0.476 |

  Point AP (all val) **0.403**; operating confidence **0.06** (val-chosen; frozen for any test run).
- OBSERVED: results differ strongly between the three val days; on 2013-05-24, 27 of 29 predictions
  match no cross. HYPOTHESIS, not tested: that day looks different (light, visibility, slug size),
  or it has slugs nobody clicked. Checking needs the per-frame predictions
  (`val_point_predictions/`, off-repo) drawn on the frames.
- LIMITATIONS (D-033): point precision is a lower bound (it is UNKNOWN whether every slug was
  clicked); about half the crosses have no training box, so many slugs were background in training;
  point scores are not mAP and cannot be compared with E001–E007; 3 val days / 27 frames from one
  setup, so the scores are noisy and measure new days, not new sites; val also chose the checkpoint
  and the threshold, so they are optimistic. Frames are 15 s apart: sparse-frame/timelapse analysis.
- Off-repo outputs (CLAUDE.md §5): `best.pt` 20,291,141 B sha256 `6f6710d1…`, `last.pt` 20,291,141 B
  sha256 `6dfd76fb…`, downloaded with the saved version's output (`kaggle kernels output`) to
  `D:\uwct_tritonia\runs\E006\` on the user's PC and checked (both sha256 match the record).

### E008 and E009 vs E006: val point scores (Tritonia)

Source (OBSERVED): the cell 17 output of `notebooks/colab_train_tritonia.ipynb` for each run, pasted by
Anu on 2026-11-10 (Kaggle notebook `anusidhu7/notebookc47ea7c413`, commit `caaf18d`).
- **E008 is registered** in `experiments/registry.csv`; `run_record.json` (9,646 B, sha256 `2673bf74…`,
  downloaded by Anu from the Kaggle session output) is archived in `experiments/records/E008/`. From it
  (OBSERVED): commit `caaf18d`, clean; dataset v3 `c1d42afa…`; imgsz 1280; all 100 epochs, one launch,
  727 s on a Tesla T4; `best.pt` 20,362,885 B sha256 `a3a1ea47…` (off-repo, not downloaded).
  Ultralytics val of `best.pt` against the automatic boxes (NOT accuracy, D-033): P 0.685, R 0.507,
  mAP50 0.594, mAP50-95 0.426 (E006: 0.731, 0.558, 0.619, 0.417). Its per-day point scores
  (`val_point_scores.json`) were not downloaded.
- **E009 is not registered and cannot be:** its Kaggle session ended without a saved version with
  output, so `run_record.json`, `best.pt` and the per-day point scores are lost. The cell 17 summary
  below is the only record. It is not re-run: it is not a final candidate, and a re-run would be a
  new run, not a recovery. All 27 val frames, 153 crosses, 10 px one to
one, threshold = each run's own best val F1 (D-033).

| run | changed from E006 | point AP | conf | predictions | matched | point P | point R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **E006** | — | **0.403** | 0.06 | 137 | 69 | **0.504** | **0.451** | **0.476** |
| E008 | imgsz 640 → 1280 | 0.317 | 0.12 | 133 | 58 | 0.436 | 0.379 | 0.406 |
| E009 | dataset v3 → v5 (D-035) | 0.274 | 0.06 | 150 | 51 | 0.340 | 0.333 | 0.337 |

- Outcome by the declared rule: neither val point AP is higher than E006's 0.403, so **E006 stays the only
  final candidate**; E008 and E009 are not declared final candidates and are **not evaluated on test**.
- HYPOTHESIS, not tested: v5's added frames are those where every cross got an automatic box, which
  favours sparse days with clear, bright slugs (D-035 limitations), so more such boxes did not teach the
  harder cases; a larger input size does not fix boxes that are missing for about half the crosses. Both
  point to the automatic labels, not the amount of data or the image size, as the limit.
- LIMITATION: one run per arm (seed 0) on 27 val frames; the differences may be partly noise. These runs
  show that E006 was not beaten, not that imgsz 1280 or v5 are worse in general.

### E010 vs E006: val point scores (Tritonia, fixed boxes)

Source (OBSERVED): `val_point_scores.json` (1,270 B) and `run_record.json` (9,779 B, sha256 `3e1d8c2d…`)
from the E010 Kaggle session (notebook `anusidhu7/notebooka3e73b4a4d`, commit `2a8daf3`), downloaded by
Anu on 2026-11-12 as a zip from the live session (two saved versions had no output) and kept in
`D:\uwct_tritonia\runs\E010\` with `best.pt`; the record is archived in `experiments/records/E010/`.
- From the record (OBSERVED): commit `2a8daf3`, clean; dataset v7 `8ea6503c…`, frozen and verified by
  Ultralytics (train 96 images / 572 boxes, val 27 / 153); all 100 epochs, one launch, 239 s on a Tesla T4;
  `best.pt` 20,291,141 B sha256 `48d72b79…`, checked against the downloaded file.
- Ultralytics val of `best.pt` against the **fixed** val boxes (NOT accuracy, D-033, and not comparable with
  E006's, which are against other boxes): P 0.510, R 0.425, mAP50 0.408, mAP50-95 0.198.
- Point scores: all 27 val frames, 153 crosses, 10 px one to one, threshold = each run's own best val F1.

| run | changed from E006 | point AP | conf | predictions | matched | point P | point R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E006 | — | 0.403 | 0.06 | 137 | 69 | 0.504 | 0.451 | 0.476 |
| **E010** | dataset v3 → v7 (fixed boxes) | **0.496** | 0.16 | 114 | 70 | **0.614** | **0.458** | **0.524** |

Per val day (point P / R / F1, matched / crosses; E006 at conf 0.06, E010 at conf 0.16):

| day | crosses | E006 | E010 |
|---|---:|---|---|
| 20130524 | 15 | 0.069 / 0.133 / 0.091 (2 / 15, 29 predictions) | 0.167 / 0.133 / 0.148 (2 / 15, 12 predictions) |
| 20130529 | 77 | 0.825 / 0.429 / 0.564 (33 / 77) | 0.769 / 0.649 / 0.704 (50 / 77) |
| 20130602 | 61 | 0.500 / 0.557 / 0.527 (34 / 61) | 0.486 / 0.295 / 0.367 (18 / 61) |

- Outcome by the declared rule: E010's val point AP 0.496 is higher than E006's 0.403, so **E010 is declared
  a final candidate** (`experiments/final_candidates.csv`, 2026-11-12) with its own val threshold conf 0.16
  and `best.pt` sha256 `48d72b79…`. Its test was run once on 2026-11-12 (test-evaluation log below):
  test point AP 0.215, below E006's 0.248; test never chooses.
- OBSERVED: the gain is not uniform. E010 finds many more slugs on 20130529 (50 vs 33 matched) and fewer
  on 20130602 (18 vs 34); overall matched crosses are almost equal (70 vs 69), and the higher point AP
  and precision come mostly from fewer false boxes (114 vs 137 predictions).
- HYPOTHESIS, not tested: boxes on every cross (including frames the automatic labels excluded) give
  fewer slugs counted as background in training, hence fewer false boxes; the fixed 20 × 17 px size may
  fit some days' slugs worse than others (20130602).
- LIMITATION: one run per arm (seed 0) on 27 val frames from 3 days; the difference may be partly noise.
  Point precision is a lower bound (UNKNOWN whether every slug was clicked, D-029).

### ID renumbering (2026-11-13)

`p4/colab-baseline` (#25) used D-040 and E020–E029 for Model A while this branch, not yet merged, used the same numbers for Tritonia. Following CLAUDE.md §10 the unmerged branch renumbered (only the colliding IDs):

| original (Kaggle run, records) | now | what |
|---|---|---|
| D-040 | D-042 | Tritonia tuning round 1 (decision) |
| E025 / E026 | E042 / E043 | D-042 baseline, seeds 1 / 2 |
| E027 / E028 / E029 | E044 / E045 / E046 | D-042 arm A, seeds 0 / 1 / 2 |

E030–E041 and D-041 are unchanged. The archived `run_record.json` / `val_point_scores.json` in `experiments/records/E042`–`E046` are byte-identical to the downloaded ones and still name E025–E029 and D-040 (each folder has a `RENUMBERED.md`); in `experiments/registry.csv` those rows carry the new ID and `(ran as E0xx …)` in `changed`, and `run_dir` keeps the Kaggle path. Off-repo files keep their names (`d040_results.zip`, `D:\uwct_tritonia\runs\d040\`).

### D-042 round 1: Tritonia tuning, val point scores (E030–E038 and E042–E046)

Source (OBSERVED): `d040_results.zip` (sha256 `9ba80567…`) and `d040_weights.zip` (`58c26d97…`), downloaded by Anu on
2026-11-12 from an interactive Kaggle session (2 × Tesla T4, `notebooks/kaggle_tritonia_tuning.ipynb`, commit
`1f84325`, clean), kept in `D:\uwct_tritonia\runs\d040\`. Every run record names commit `1f84325`, dataset v7
`8ea6503c…` and its config's settings; every `best.pt` matches its recorded sha256; all 14 are registered and their
`run_record.json` + `val_point_scores.json` are archived in `experiments/records/E0xx/`. All 5 smoke runs passed
(27–35 s each). Recomputed from the files: identical to the notebook's cell 9 table.
- **Procedure note (OBSERVED):** cell 8 was run twice. The first run trained and scored all 14; the second stopped
  every run at once ("already exists", nothing overwritten) but replaced `queue_status.json` (kept as
  `queue_status_second_cell8_attempt.json`) and the 14 `train_E0xx.log` files with that one-line message, so the
  original training logs are lost. `queue_status.json` was rebuilt from the run folders before cell 9. Records,
  `results.csv`, weights and point scores are from the first run.
- Val point AP (27 val frames, 153 crosses, 10 px one to one, each run at its own best-val-F1 threshold); seed 0 of
  the baseline is E010:

| arm | change from E010 | runs (val point AP) | mean | seed range | mean P / R / F1 | rule |
|---|---|---|---:|---:|---|---|
| baseline | — | E010 0.496, E042 0.478, E043 0.493 | **0.489** | 0.018 | 0.621 / 0.455 / 0.520 | reference |
| A | epochs 200, patience 50 | E044 0.435, E045 0.494, E046 0.463 | 0.464 | 0.059 | 0.616 / 0.429 / 0.500 | no evidence of a gain |
| B | batch 8 | E030 0.472, E031 0.448, E032 0.496 | 0.472 | 0.049 | 0.503 / 0.523 / 0.513 | no evidence of a gain |
| C | scale 0.2 | E033 0.472, E034 0.471, E035 0.491 | 0.478 | 0.020 | 0.619 / 0.436 / 0.507 | no evidence of a gain |
| D | imgsz 1056 | E036 0.482, E037 0.481, E038 0.471 | 0.478 | 0.011 | 0.555 / 0.514 / 0.531 | no evidence of a gain |

- Mean val point F1 per day (20130524 / 20130529 / 20130602; 15 / 77 / 61 crosses): baseline 0.174 / 0.663 / 0.420;
  A 0.038 / 0.681 / 0.373; B 0.164 / 0.670 / 0.429; C 0.143 / 0.705 / 0.314; D 0.131 / 0.692 / 0.420.
- **Outcome by the declared rule (D-042 point 4): no arm wins.** Every arm's mean is below the baseline mean
  (−0.011 to −0.025), so there is no round 2 and no test; E010 stays the Tritonia reference.
- OBSERVED: the baseline is stable across seeds (0.478–0.496); E010 was a typical seed, not a lucky one. Arm A
  varies most (two seeds trained to 176 / 165 epochs, one stopped at 97). Arms B and D find more crosses (mean
  80.0 and 78.7 matched vs 69.7) with more false boxes (159 and 143 predictions vs 114): higher recall, lower
  precision, about the same F1. Training time per run: 223–309 s at imgsz 640, 647–654 s at 1056.
- INFERRED: within these four settings the training configuration is not what limits Tritonia accuracy; the
  differences are about the size of the seed spread. HYPOTHESIS, not tested: the limit is the data (96 training
  frames from 10 days, 12–20 px slugs, crosses instead of outlines, D-036).
- LIMITATION: 27 val frames from 3 days of one setup; 3 seeds per arm; the 20130524 day has only 15 crosses, so
  its per-day numbers are very noisy. These are val numbers only (rule 7).

### D-041: Tritonia v8, val point scores (E039–E041)

Source (OBSERVED): `d041_results.zip` (sha256 `5cc51f45…`) and `d041_weights.zip` (`747205ef…`), downloaded by Anu on
2026-11-13 (UTC) from an interactive Kaggle session (2 × Tesla T4, `notebooks/kaggle_tritonia_v8.ipynb`, commit
`4021eef`, clean; one cell 8 attempt, `20261112T234910Z`), kept in `D:\uwct_tritonia\runs\d041_results\` and
`d041_weights\`. Every run record names commit `4021eef`, dataset v8 `52cd95c3…` and its config's settings; every
`best.pt` matches its recorded sha256; all 3 are registered and their `run_record.json` + `val_point_scores.json` are
archived in `experiments/records/E0xx/`. The smoke run (E039) passed in 57 s. Recomputed from the files: identical to
the notebook's cell 9 summary (`d041_summary.json`).
- Val point AP (27 val frames, 153 crosses, 10 px one to one, each run at its own best-val-F1 threshold):

| arm | change from E010 | runs (val point AP) | mean | seed range | mean P / R / F1 | rule |
|---|---|---|---:|---:|---|---|
| baseline (v7, D-042) | — | E010 0.496, E042 0.478, E043 0.493 | **0.489** | 0.018 | 0.621 / 0.455 / 0.520 | reference |
| v8 | dataset v7 → v8 | E039 0.509, E040 0.461, E041 0.502 | 0.491 | 0.048 | 0.623 / 0.466 / 0.532 | no evidence of a gain |

- Mean val point F1 per day (20130524 / 20130529 / 20130602; 15 / 77 / 61 crosses): baseline 0.174 / 0.663 / 0.420;
  v8 0.018 / 0.689 / 0.440.
- **Outcome by the declared rule (D-041): v8 does not win.** Its mean is +0.002 above the baseline mean, but the
  rule needs more than 0.489 + 0.048 (the larger seed range) = 0.537. No final candidate, no test; E010 stays the
  Tritonia reference.
- OBSERVED: all three runs stopped early (patience 30) at 35 / 43 / 59 of 100 epochs; training took 34 / 42 / 48 min
  per run (queue 83 min in total, well under the 4–5 h estimate). Mean matched crosses 71.3 with 115.0 predictions
  (baseline 69.7 / 114). The seed range grew from 0.018 to 0.048. On 20130524 v8 finds almost none of the 15
  crosses (matched 0 / 0 / 1 across seeds).
- INFERRED: 28× more training frames (96 → 2,758) from the same training days did not measurably change val point
  AP; together with D-042, neither the training settings tried nor more frames from these days is what limits it.
  HYPOTHESIS, not tested: added frames 75 s apart add little new appearance (same days, same slugs); unclicked slugs
  in them are trained as background (D-029); the val days differ from the training days.
- LIMITATION: 27 val frames from 3 days of one setup; 3 seeds; 20130524 has only 15 crosses. Val numbers only
  (rule 7).

### Val secondary metrics: E001 vs E004 (D-020)

Val only; test not used. Both runs evaluated on 2026-10-11 in one Colab Tesla T4 session at commit
`d84a664` (`scripts/val_eval.py`). Records: `experiments/records/E001/val_eval_record.json`
(16,107 B, sha256 `3db3ecb4…`) and `experiments/records/E004/val_eval_record.json` (16,203 B,
sha256 `9e68b2c8…`). Full tables, per session included: `experiments/comparisons/E001_vs_E004_val.md`.
Matching: IoU ≥ 0.5, each run at its own val-chosen operating confidence (OBSERVED):

| metric | E001 (v1) | E004 (v2) |
|---|---:|---:|
| operating conf | 0.298 | 0.276 |
| TP / FP / FN | 809 / 208 / 201 | 780 / 215 / 230 |
| precision / recall / F1 | 0.795 / 0.801 / 0.798 | 0.784 / 0.772 / 0.778 |
| AP50 (own matcher) | 0.848 | 0.836 |
| negative val images with a detection (of 55) | 16 (22 detections) | 14 (19 detections) |
| red-light/night, 31 boxes: AP50 / P / R | 0.741 / 0.706 / 0.774 | 0.713 / 0.759 / 0.710 |
| not flagged, 979 boxes: AP50 / P / R | 0.851 / 0.799 / 0.802 | 0.841 / 0.785 / 0.774 |
| parameters / GFLOPs (640) | 9.95 M / 22.7 | 9.95 M / 22.7 |
| inference, batch 1, median (T4) | 16.3 ms | 16.3 ms |

- OBSERVED: E001's operating confidence 0.2983 equals the one in its `final_test_record.json`,
  which checks the new code against `final_test.py`.
- OBSERVED: E004 is lower than E001 on AP50, recall and F1 overall and in the not-flagged
  condition; it has slightly fewer false positives on negative images (14 vs 16 images).
- OBSERVED: under red light E004 has higher precision and lower recall and AP50; the recall gap is
  2 of 31 boxes. Per session the results are mixed (E004 lower on `18_ALRA_08`, AP50 0.814 vs
  0.644, and more false positives on `21_SUB6_07`, 55 vs 29; slightly higher on `21_SUB6_10`).
- INFERRED: on this val set the enhanced images did not improve detection. HYPOTHESIS, not
  tested: the gap is run-to-run noise or the shorter E004 training (81 vs 100 epochs).
- LIMITATION: val also chose each run's checkpoint and operating confidence, so the numbers are
  optimistic; one seed per run; red-light val is 2 sessions / 31 boxes; `21_MACK_10` holds 465 of
  1,010 val boxes; the enhancement step's own cost is not timed. The two records carry the earlier
  limitation wording "Val chose E001's checkpoint", which means each run's checkpoint.

### Model A tuning round 1 (E011–E019, D-038 / D-040)

Kaggle version "D-039 round 1 E011-E019 (fix)", commit cf7bea9, 2× Tesla T4 with one run per GPU; 8 of 9
runs finished, E018 not run (time). Val (324 v1 frames), best.pt, OBSERVED from the archived run records:

| Model | Runs (seed 0 / 1 / 2) | Val mAP50-95 mean (range) | Val mAP50 mean | P mean | R mean |
|---|---|---|---|---|---|
| yolo26s | E011 / E016 / E019 | 0.5976 (0.0034) | 0.8352 | 0.7790 | 0.7794 |
| yolo26m | E012 / E014 / E017 | 0.6113 (0.0142) | 0.8466 | 0.7730 | 0.8040 |
| yolo26l | E013 / E015 / — | 0.6142 (0.0036) | 0.8517 | 0.7824 | 0.8206 |

Selected: **yolo26m** (D-040 point 1). E001 0.6115 / 0.8527, E007 0.5993 / 0.8423 (mAP50-95 / mAP50).
Ultralytics' val confusion matrices (best.pt; counts read from `val_best/confusion_matrix.png`, its internal
operating point; 1010 val boxes), found / missed / false alarms: E011 793/217/231, E016 847/163/384,
E019 830/180/249; E012 800/210/226, E014 849/161/332, E017 839/171/256; E013 859/151/272, E015 853/157/275.
Mean per model: s 823/187/288, m 829/181/271, l 856/154/274 (yolo26l finds about 27 more lobsters than yolo26m
at a similar false-alarm count; reported, not used to select).
No run completed its schedule: all stopped by patience at epoch 26–41 of 60, best epochs 16–31 with the
learning rate still 0.0010–0.0015 (of 0.002), and mosaic was never switched off (close_mosaic from epoch 51).
This is the reason for the round-2 schedule (D-040 point 2). Seconds per epoch with two runs in parallel:
m 189–201, l 202–206, s 188–201 (E017 ran alone at the end: 168). Kaggle quota: 12:00 of 30 h after round 1 (two GPUs counted once).

## Test-evaluation log

| Date | Experiment | Reason | Result |
|---|---|---|---|
| 2026-09-19 | E001 | Final held-out test evaluation after completed validation run and candidate declaration | mAP50 0.8974; mAP50-95 0.6832; precision @ val-chosen conf 0.298 = 0.8447; recall = 0.8127; F1 = 0.8284 |
| 2026-10-12 | E004 | Final held-out test evaluation of the declared candidate (D-022), raw vs enhanced comparison (D-017) | mAP50 0.8734; mAP50-95 0.6555; precision @ val-chosen conf 0.276 = 0.8614; recall = 0.7521; F1 = 0.8031 |
| 2026-10-18 | E005 | Final held-out test evaluation of the declared candidate after registration, with the recovered epoch-41 best.pt (D-027) | mAP50 0.8788; mAP50-95 0.6604; precision @ val-chosen conf 0.619 = 0.8411; recall = 0.7712; F1 = 0.8046 |
| 2026-11-10 | E006 | Final held-out test evaluation of the declared Tritonia candidate (D-033): point eval against the MTrackJ crosses, 27 frames of the 3 test days, val threshold | not mAP (D-033): test point AP 0.248; point P / R / F1 @ val-chosen conf 0.06 = 0.3594 / 0.3622 / 0.3608 (128 predictions, 127 crosses, 46 matched) |
| 2026-11-12 | E010 | Final held-out test evaluation of the second declared Tritonia candidate (val point AP 0.496 > E006's 0.403): point eval against the MTrackJ crosses, the same 27 test frames, val threshold | not mAP (D-033): test point AP 0.2145; point P / R / F1 @ val-chosen conf 0.16 = 0.3563 / 0.2441 / 0.2897 (87 predictions, 127 crosses, 31 matched) |
| 2026-11-13 | E027 | Final held-out test evaluation of the declared Model A candidate (D-043), v1 test (`--target test`) | mAP50 0.9096; mAP50-95 0.6966; precision @ val-chosen conf 0.269 = 0.8478; recall = 0.8426; F1 = 0.8452 |
| 2026-11-13 | E027 | Final held-out evaluation of the same candidate on the new-site Zenodo `sk` frames (`--target test_zenodo_sk`, D-031, D-043) | mAP50 0.5663; mAP50-95 0.3892; precision @ val-chosen conf 0.269 = 0.4341; recall = 0.6224; F1 = 0.5115 |

E006 test evaluation (2026-11-10, OBSERVED in the Kaggle cell 18 output pasted by Anu): Kaggle interactive session (notebook `anusidhu7/notebookc47ea7c413`), evaluation commit `b1738df`, registered `best.pt` sha256 `6f6710d1…` (checked by cell 18; attached from the private Kaggle dataset `anusidhu7/uwct-e006-weights`, a copy of the local registered run folder, because the latest output of the training notebook holds a later re-run's `best.pt`), val threshold conf 0.06, attempt 1. Per-day test scores are in `test_point_scores.json` in the session output; not yet archived in the repo. LIMITATIONS: crosses are the ground truth and point precision is a lower bound (whether every slug was clicked is UNKNOWN, D-029); 27 frames from 3 days of one setup (new days, not new sites); training boxes are automatic and cover about half of the crosses (D-033). Test is lower than val (F1 0.361 vs 0.476, point AP 0.248 vs 0.403). The result changes neither the model nor the threshold.

E010 test evaluation (2026-11-12, OBSERVED in the Kaggle cell 18 output pasted by Anu and in `test_point_scores.json`, 1,123 B, sha256 `ca1a80a2…`, downloaded from the session as a zip and archived in `experiments/records/E010/`; prediction files in `D:\uwct_tritonia\runs\E010_final_test\`): Kaggle interactive session (notebook `anusidhu7/notebooka3e73b4a4d`), evaluation commit `59de837`, registered `best.pt` sha256 `48d72b79…` (checked by cell 18; attached from the private Kaggle dataset `anusidhu7/uwct-e010-weights`), val threshold conf 0.16.
- Procedure deviations (OBSERVED): (1) version 1 of `uwct-e010-weights` has no top `E010/` folder, so the cell's search for `**/E010/train/weights/best.pt` found nothing; the search line was edited in the session to `**/train/weights/best.pt` (the sha256 check still selects only the registered file), and the repo cell was changed to match afterwards. (2) A first attempt at about 00:51 UTC created `runs/E010_final_test/` and stopped before any prediction (the folder was empty, no `test_point_predictions/`, no scores; its output was not kept, reason UNKNOWN); the empty folder was removed with `os.rmdir` and the cell run again. No test frame was predicted or scored before the reported run, so it is attempt 1 of the test.
- Per test day (point P / R / F1, matched / crosses, predictions): 20130520 0.368 / 0.350 / 0.359 (7 / 20, 19); 20130521 0.217 / 0.370 / 0.274 (10 / 27, 46); 20130528 0.636 / 0.175 / 0.275 (14 / 80, 22). E006's per-day test scores were not archived, so the comparison is overall only.
- E006 vs E010 on test (OBSERVED): point AP 0.248 vs 0.215; F1 0.361 vs 0.290; precision about equal (0.359 vs 0.356), recall lower for E010 (0.362 vs 0.244, 46 vs 31 matched). On val E010 was higher (0.496 vs 0.403). Test never chooses: both results are reported, and neither the models nor the thresholds change.
- INFERRED: the val gain of E010 did not carry over to the test days; with 27 frames per split and one run per arm, the val difference was partly day-specific or noise (on val E010 was better on one day and worse on another). LIMITATIONS as for E006: crosses are the ground truth and point precision is a lower bound (D-029); 27 frames from 3 days of one setup (new days, not new sites); the fixed 20 × 17 px boxes are not drawn slug outlines (D-036).

E027 test evaluations (2026-11-13, OBSERVED in the two final-test records and the notebook log): Kaggle
notebook `sakshampainuly007/notebookc7645e72e4` (`notebooks/kaggle_final_test_E027.ipynb`), evaluation commit
`a060c70` (clean; training commit `97810a5`), dataset v4 fingerprint `bbd4a8b1…` (frozen registry verified; all
4 raw checks passed; `verify_yolo_dataset.py` printed VERIFIED, including `test_zenodo_sk` 428 / 285 / 143),
`best.pt` sha256 `821bb767…` (equal to `run_record.json`; copied from the round-2 notebook output), operating
confidence 0.269 (max mean F1 on val), one T4, attempt 1 for each target (`attempts.jsonl`: one start and one end
each). Records: `experiments/records/E027/final_test_record.json` (8,814 B, sha256 `ac0e7aa2…`, v1 test) and
`experiments/records/E027/final_test_zenodo_sk_record.json` (10,000 B, sha256 `f7252c70…`, Zenodo `sk`). Plots,
prediction examples and the package `E027_final_test_package.zip` (sha256 `c72921ca…`, matches the notebook) are
off-repo in `D:/uwct_runs/final_test_E027/`.

| metric | E001 v1 test | E027 v1 test | E027 Zenodo `sk` |
|---|---:|---:|---:|
| images (negative) / boxes | 348 (27) / 1,512 | 348 (27) / 1,512 | 428 (285) / 143 |
| mAP50 | 0.8974 | 0.9096 | 0.5663 |
| mAP50-95 | 0.6832 | 0.6966 | 0.3892 |
| operating conf (chosen on val) | 0.298 | 0.269 | 0.269 |
| precision / recall / F1 @ operating conf | 0.8447 / 0.8127 / 0.8284 | 0.8478 / 0.8426 / 0.8452 | 0.4341 / 0.6224 / 0.5115 |
| Ultralytics P / R (max-F1 conf on the evaluated split) | 0.8309 / 0.8208 | 0.8434 / 0.8477 | 0.7322 / 0.4895 |
| confusion matrix TP / FP / FN (Ultralytics) | 1261 / 263 / 251 | 1289 / 232 / 223 | 90 / 119 / 53 |
| E027 val mAP50-95 (for reference) | 0.6115 | 0.6280 | 0.6280 |

- OBSERVED: on the v1 test E027 is higher than E001 on every metric in the table (mAP50-95 +0.0134), with fewer
  false positives and fewer missed boxes. Test is higher than val for both models (E027 0.6966 vs 0.6280).
- OBSERVED: on Zenodo `sk` mAP50-95 falls to 0.3892 and precision at the val-chosen confidence to 0.434; the
  Ultralytics confusion matrix shows 119 false positives against 90 found lobsters, and 53 of 143 lobsters missed.
- INFERRED: the detector does not transfer well to a new site, and the val-chosen operating confidence is too low
  for `sk` (the max-F1 confidence on `sk` itself gives P 0.732 / R 0.490, but choosing it on `sk` is not allowed).
- UNKNOWN — requires the per-image predictions, which the records do not keep: how many of the false positives fall
  on the 285 lobster-free `sk` images and on the 27 negative v1 test images.
- Declared before the run: these numbers are reported and change neither the model, the threshold nor the
  selection (D-043). The comparison with E001 is not a selection: E001 and E027 differ in dataset (v1 → v4), model
  size and tuning, and each is one seed.

E027 test-evaluation limitations:

- v1 test: sessions share sites (mostly also dates) with train sessions, so it measures new-session, not new-site
  generalization (D-007); 13_SHRI_12 + 21_SUB6_03 hold 1,135 of 1,512 test boxes (D-008); substrate/visibility
  metadata is mostly missing; zExtra is excluded.
- Zenodo `sk` is the only new-site measurement: one site (other country, camera setup and image size, D-031),
  labels from the Zenodo dataset, not our annotators; 143 boxes on 143 images, so false positives on the 285
  lobster-free images weigh heavily (D-032). Training used Zenodo `mpa`, but no `sk` or `sk`-derived image.
- Single seed; E027 was chosen among 19 configurations on the same 324 val images, so its val score is optimistic
  (D-043); the test scores are the unbiased estimate.
- Ultralytics-reported P / R use the confidence that maximizes F1 on the evaluated split; the operating point uses
  the val-chosen 0.269.
- Test results were not used for training, threshold selection, or model choice.

E001 test-evaluation limitations:

- v1 test sessions share sites (mostly also dates) with train sessions: results measure new-session generalization, not new-site generalization (D-007).
- Test boxes are concentrated: 13_SHRI_12 + 21_SUB6_03 hold 1,135 of 1,512 test boxes (D-008).
- Substrate/visibility metadata is mostly missing; zExtra is excluded.
- Ultralytics-reported precision/recall use the confidence that maximizes F1 on the test split; the reported operating point uses confidence 0.298 selected on val.
- Test results were not used for training, threshold selection, or model choice.

E004 test evaluation (2026-10-12): Colab Tesla T4, evaluation commit `cbeb99a` (training commit
`97ac710`), dataset v2 fingerprint `8a0d4194…` (verified equal to training), `best.pt` sha256
`0f5470c6…` (equal to `run_record.json`), attempt 1. Record:
`experiments/records/E004/final_test_record.json` (8,450 B, sha256 `8d157e00…`); plots and
prediction examples stay on Drive in `runs/E004/final_test/`.

| test metric | E001 (v1, raw) | E004 (v2, enhanced) |
|---|---:|---:|
| mAP50 | 0.8974 | 0.8734 |
| mAP50-95 | 0.6832 | 0.6555 |
| operating conf (chosen on val) | 0.298 | 0.276 |
| precision / recall / F1 @ operating conf | 0.8447 / 0.8127 / 0.8284 | 0.8614 / 0.7521 / 0.8031 |
| Ultralytics P / R (max-F1 conf on test) | 0.8309 / 0.8208 | 0.8344 / 0.7799 |
| confusion matrix TP / FP / FN (Ultralytics) | 1261 / 263 / 251 | 1165 / 190 / 347 |

- OBSERVED: on test, E004 is lower than E001 on mAP50, mAP50-95, recall and F1 and higher on
  precision, with fewer false positives and more missed boxes. Val (D-020) agrees on mAP, recall
  and F1 but not on precision: at the operating confidence E004's val precision is slightly lower
  (0.784 vs 0.795, 215 vs 208 false positives), so the precision difference is not consistent
  across splits.
- INFERRED: on this data, Harsh's enhancement did not improve detection with the E001 detector.
  HYPOTHESIS, not tested: run-to-run noise or E004's shorter training (81 vs 100 epochs) explain
  part of the gap.
- Declared before the run: these test numbers are reported and are not used to choose between
  E001 and E004 (D-017, D-022).

E004 test-evaluation limitations (in addition to E001's above, which apply unchanged):

- One run per arm (seed 0); E001 trained 100 epochs, E004 stopped early at 81.
- One val image and one test image (`06_HERR_08_CW_20200903_GP120003_00001`, 1 box) were among
  the images Harsh reviewed while choosing the enhancement profile; whether others were viewed is
  UNKNOWN (D-022). The delivery's provenance is INFERRED; five images re-run near-identically (D-023).
- The enhancement adapts to each image's statistics; it is not a fixed per-pixel mapping (D-018).
- Environment differs from training and from E001's test: torch 2.11.0+cu130 / CUDA 13.0
  (E004's training and E001's test: 2.11.0+cu128 / 12.8); same GPU type, Ultralytics 8.4.155 and
  Python 3.13.15.
- Test results were not used for training, threshold selection, or model choice.

E005 test evaluation (OBSERVED in the RT-DETR notebook output, cells R3, 16 and 17; attempt 1; record
`runs/E005/final_test/final_test_record.json` on Drive, outside Git):

- Weights: `train/weights/best.pt` sha256 `b2fc006a…` (263,628,970 B, checkpoint epoch 40 = results.csv
  epoch 41, EMA present; cell R3), not the registered `ae5f2356…`, which is lost (D-027).
- D-027 val check before the test split was touched: val P 0.771, R 0.814, mAP50 0.836, mAP50-95 0.588
  (324 images, 1,010 boxes); absolute differences from the registered val metrics printed as 0.0 on all
  four (tolerance 0.002).
- Test split: 348 images (27 backgrounds), 1,512 boxes. Ultralytics-reported P 0.8105, R 0.7996 (at the
  confidence that maximizes F1 on test). Confusion matrix as printed by cell 17 (rows = predicted,
  cols = true; lobster, background): [[1443, 1458], [69, 0]], i.e. TP / FP / FN 1443 / 1458 / 69
  (TP + FN = 1,512 test boxes). The FP count is far above E001's 263 and E004's 190 (table above).
  UNKNOWN — requires confirmation: whether Ultralytics builds the RT-DETR confusion matrix at the same
  confidence threshold as for YOLO; until then these confusion-matrix counts are not compared across
  model families.
- Ultralytics 8.4.155, torch 2.11.0+cu130, Tesla T4; notebook COMMIT a2b9f99.

E005 test-evaluation limitations:

- v1 test sessions share sites (mostly also dates) with train sessions: results measure new-session generalization, not new-site generalization (D-007).
- Test boxes are concentrated: 13_SHRI_12 + 21_SUB6_03 hold 1,135 of 1,512 test boxes (D-008).
- Substrate/visibility metadata is mostly missing; zExtra is excluded.
- Ultralytics-reported precision/recall use the confidence that maximizes F1 on the test split; the reported operating point uses confidence 0.619 selected on val.
- Evaluated weights are the recovered epoch-41 best.pt, not the registered file; E005's final `last.pt` and the 100-epoch `results.csv` are lost (98 epochs survive) (D-027).
- False positives on negative test images were not reported separately: UNKNOWN — requires the per-image predictions in the final-test record.
- Test results were not used for training, threshold selection, or model choice.
