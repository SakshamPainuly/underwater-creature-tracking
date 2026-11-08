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
    tested only after it is registered and reviewed). `scripts/final_test_eval.py` refuses
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
| E006 | `configs/experiments/E006_yolo_seaslug.yaml` (committed once v3 is built; template in D-030) | E001 | dataset v1 (lobster) → v3 (Tritonia sea slugs; classes lobster + sea_slug) only; model and settings as E001 (D-030) | planned; waits for labelling batch 1 and the frozen v3 |
| E007 | `configs/experiments/E007_yolo_model_a_all.yaml` | E001 | dataset v1 → v4 only: all safe Model A training data (4,111 images: our originals, Zenodo `mpa`, colour change, copy-paste, diffusion; D-031, D-032); model and settings as E001 | completed 2026-11-08 on Kaggle (Tesla T4); 58 epochs (early stop); registered; not a final candidate |

**E006 (Tritonia sea slugs, D-030), declared before any result:** selection metric val mAP50 of class
`sea_slug`; also reported per class: P, R, mAP50-95 on val, and false positives on the confirmed-negative
val frames if there are any. E006 is not compared with E001–E005: different data and classes. Its 3 test
days are evaluated only after a sea slug final-test procedure is declared here. LIMITATION: 150 frames,
3 val and 3 test days from one setup; results measure new days, not new sites.

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

## Test-evaluation log

| Date | Experiment | Reason | Result |
|---|---|---|---|
| 2026-09-19 | E001 | Final held-out test evaluation after completed validation run and candidate declaration | mAP50 0.8974; mAP50-95 0.6832; precision @ val-chosen conf 0.298 = 0.8447; recall = 0.8127; F1 = 0.8284 |
| 2026-10-12 | E004 | Final held-out test evaluation of the declared candidate (D-022), raw vs enhanced comparison (D-017) | mAP50 0.8734; mAP50-95 0.6555; precision @ val-chosen conf 0.276 = 0.8614; recall = 0.7521; F1 = 0.8031 |
| 2026-10-18 | E005 | Final held-out test evaluation of the declared candidate after registration, with the recovered epoch-41 best.pt (D-027) | mAP50 0.8788; mAP50-95 0.6604; precision @ val-chosen conf 0.619 = 0.8411; recall = 0.7712; F1 = 0.8046 |

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
