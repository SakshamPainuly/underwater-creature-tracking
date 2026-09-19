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

Both use frozen dataset v1 (fingerprint `9f2bf8573df4354e1373786967a1155802e4cdbfe127e1f07fd3fdb1983ef810`)
and ultralytics 8.4.155. Colab procedure: docs/COLAB.md.

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