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
    `experiments/final_candidates.csv` (E001 declared). `scripts/final_test_eval.py` refuses
    undeclared experiments, fixes the operating confidence on val, then evaluates once on test
    (docs/COLAB.md).

- **Every reported result carries the v1 limitations** in docs/DATASET.md (new sessions, not new
  sites; concentrated val/test sessions; sparse metadata).

## Declared experiments (D-012)

| ID | Config | Parent | Changed | Status |
|---|---|---|---|---|
| E001 | `configs/experiments/E001_yolo_baseline.yaml` | none | initial baseline: yolo26s.pt, imgsz 640, ≤100 epochs, patience 30, batch 16, seed 0, deterministic | completed; 100 epochs; final candidate evaluated on test |
| E002 | `configs/experiments/E002_yolo_imgsz1280.yaml` | E001 | imgsz 640 → 1280 only | planned; not run |
| E003 | `configs/experiments/E003_rtdetr_baseline.yaml` | E001 | model yolo26s.pt → rtdetr-l.pt only (D-015) | planned; not run |
| E004 | `configs/experiments/E004_yolo_enhanced.yaml` | E001 | images raw → enhanced only: dataset v1 → v2 (D-017) | completed; 81 epochs (early stop); registered; test evaluation on hold (D-018) |

E001–E003 use frozen dataset v1 (fingerprint `9f2bf8573df4354e1373786967a1155802e4cdbfe127e1f07fd3fdb1983ef810`).
E004 uses dataset v2 (fingerprint `8a0d4194d1f7fb3972ad33b09cf570988aebb5350c5d856851056c961528626e`):
v1's sessions, split and labels byte for byte, with Harsh's enhanced images. All use ultralytics
8.4.155. Colab procedure: docs/COLAB.md.

**YOLO vs RT-DETR (D-015):** E001 and E003 are compared on val mAP50, then on the secondary
metrics above, with parameter count, GFLOPs and inference speed on the same GPU type reported
alongside. E003 is evaluated on test only after it is registered, reviewed and declared in
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

E001 test-evaluation limitations:

- v1 test sessions share sites (mostly also dates) with train sessions: results measure new-session generalization, not new-site generalization (D-007).
- Test boxes are concentrated: 13_SHRI_12 + 21_SUB6_03 hold 1,135 of 1,512 test boxes (D-008).
- Substrate/visibility metadata is mostly missing; zExtra is excluded.
- Ultralytics-reported precision/recall use the confidence that maximizes F1 on the test split; the reported operating point uses confidence 0.298 selected on val.
- Test results were not used for training, threshold selection, or model choice.