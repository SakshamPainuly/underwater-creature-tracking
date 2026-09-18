# Experiments

Declared **before** any comparison (CLAUDE.md §5). No experiments have been run yet.

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
- **Every reported result carries the v1 limitations** in docs/DATASET.md (new sessions, not new
  sites; concentrated val/test sessions; sparse metadata).

## Run record requirements
Each run writes `run_record.json` and one row in `experiments/registry.csv` (created in P4):
- git commit + dirty flag, dataset version + fingerprint,
- model + pretrained checkpoint, Ultralytics args, seed,
- env report (GPU, VRAM, CUDA, torch, ultralytics), duration,
- val metrics, and weights path + sha256.

## Run log
| ID | Date | Parent | Changed | Dataset | Val mAP50 | Val mAP50-95 | Notes |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | none yet |

## Test-evaluation log
| Date | Experiment | Reason | Result |
|---|---|---|---|
| — | — | — | none yet |
