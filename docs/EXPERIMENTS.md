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
| E005 | `configs/experiments/E005_rtdetr_amp_off.yaml` | E001 | as E003 (model yolo26s.pt → rtdetr-l.pt) plus amp true → false; replaces the diverged E003 (D-025) | running; effective batch 8 after automatic OOM reduction (D-026); resumed from epoch 58 of 100; declared final candidate 2026-10-15 |

All use frozen dataset v1 (fingerprint `9f2bf8573df4354e1373786967a1155802e4cdbfe127e1f07fd3fdb1983ef810`)
and ultralytics 8.4.155. Colab procedure: docs/COLAB.md.

**YOLO vs RT-DETR (D-015, D-025):** the RT-DETR side is now E005 (E003 diverged). E001 and E005 are compared on val mAP50, then on the secondary
metrics above, with parameter count, GFLOPs and inference speed on the same GPU type reported
alongside, and E005's AMP-off training is stated. E005 is evaluated on test only after it is registered, reviewed and declared in
`experiments/final_candidates.csv`; test numbers are reported, never used to choose.

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