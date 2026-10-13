# my_model — RT-DETR (E005, AMP off) for the YOLO vs RT-DETR comparison

Decisions: docs/DECISIONS.md **D-015** (RT-DETR, E003) and **D-025** (E003 diverged to NaN with AMP on;
rerun as E005 with AMP off). Rules: CLAUDE.md. Runbook: docs/COLAB.md.

## What is here

| File | Purpose |
|---|---|
| `colab_train_rtdetr.ipynb` | Google Colab notebook: builds the v1 dataset view, then preflight → smoke test → full training → final test for **E005** (`rtdetr-l.pt`, AMP off) |

## Where everything else lives (shared with the YOLO pipeline, on purpose)

| What | Where |
|---|---|
| Experiment config | `configs/experiments/E005_rtdetr_amp_off.yaml` (parent E001; model and AMP change, D-025) |
| Training code | `scripts/train_yolo.py` → `src/training/train.py` (picks `RTDETR` or `YOLO` from the checkpoint name) |
| Final test code | `scripts/final_test_eval.py` → `src/training/final_test.py` |
| Dataset | Not copied here. Colab downloads the shared ZIP (cell 7) and builds `data/yolo/lobster_v1/` from the frozen v1 manifests, the same view E001 used. Labels are YOLO txt, one class `0: lobster`; RT-DETR reads them as they are. |
| Runs, weights, plots | Google Drive: `MyDrive/uwct/runs/E005*/` (never in Git; `E003/` is the diverged run, never resume it) |
| Run records after training | `experiments/records/E005/` + `experiments/registry.csv` (D-013) |

## How to run on Colab

1. On the local PC: commit this work (after approval) and push branch `anu/model-development` to
   `SakshamPainuly/underwater-creature-tracking`, then copy `git rev-parse HEAD`.
2. Open `colab_train_rtdetr.ipynb` in Colab (badge at the top, or File → Open notebook → GitHub),
   set the runtime to **GPU**, add the `UWCT_GITHUB_TOKEN` secret (docs/COLAB.md).
3. Cell 1: paste the sha into `COMMIT`. Run cells 1–10; cell 10 must print `VERIFIED`.
4. Cell 11 (preflight) → set `RUN_SMOKE = True`, run cell 12 → run cell 12b (losses finite).
   If batch 16 runs out of memory in fp32 or losses are NaN: stop, do not edit the config silently.
5. Set `RUN_FULL = True`, run cell 13. After a disconnect: cells 1–10, then cell 14.
6. Cell 15, then download `run_record.json` and register it on the local PC
   (`python scripts/register_run.py <path>/run_record.json`).
7. Compare with E001 on **val mAP50** first. Test (cells 16–17) only after E005 is declared in
   `experiments/final_candidates.csv`.
