# Underwater Creature Video Tracking Software (CSCI 555)

Detection-first pipeline for underwater creature video analysis. Initial use case:
lobster detection (`0: lobster`) on a biology-department-provided dataset. Dataset-specific
paths, naming and label policies live in `src/dataset_audit/config.py`.

Project rules for contributors and Claude Code: `CLAUDE.md`. Data card: `docs/DATASET.md`.
Decisions: `docs/DECISIONS.md`. Experiments: `docs/EXPERIMENTS.md`.

## Status

Phases 1–2 (dataset audit, session-level split, negative-image audit, YOLO dataset view)
and P3 (versioning, provenance, tests) are implemented. P4 (Colab training, experiment configs,
run records; docs/COLAB.md) is complete on `p4/colab-baseline`: **E001, the first baseline lobster
detector, was trained on Colab (100 epochs, val mAP50 0.8527) and evaluated once on the held-out
test split (mAP50 0.8974)**. Every result must be reported with the v1 limitations in
docs/DATASET.md — the test split measures generalization to new *sessions*, not new *sites*.
E002 (imgsz 1280) is configured but not run. No tracking or video work has started: the delivery
contains no video (docs/DATASET.md).
`data/raw/biology/` is read-only and never written to by any script.

## Project layout

```
data/
  raw/biology/         read-only provided dataset (git-ignored, ~5.5 GB)
  processed/           dataset version v1 manifests (frozen, see FROZEN.json) + raw fingerprints
                       (raw_fingerprint.json = historical v1; raw_fingerprint_d011_post_cleanup.json = current)
  processed/<vX.Y>/    later dataset versions (drafts until frozen)
  yolo/                per-machine YOLO dataset views (git-ignored, rebuilt on each machine)
src/dataset_audit/     scan, xlsx metadata, split, image manifest, versioning (stdlib only)
src/yolo_prep/         YOLO dataset view builder (stdlib only); verify.py (Ultralytics, GPU only)
src/training/          experiment configs + run records (stdlib); train.py (GPU only)
configs/experiments/   E###_<slug>.yaml experiment configs
experiments/           registry.csv (registered full runs), final_candidates.csv (declared test candidates)
  records/<E###>/      committed run metadata: run_record.json, results.csv, final_test_record.json (D-013)
notebooks/             Colab notebook (calls scripts only)
scripts/               CLI entry points
tests/                 CPU unit tests (stdlib unittest)
docs/                  data card, decision log, experiment log
```

## Dependencies

- Local development PC: Python ≥ 3.10, standard library only (`requirements.txt` is empty on purpose).
- Colab / lab GPU: `pip install -r requirements-gpu.txt` (pins `ultralytics==8.4.155`).
  Never install it on the local PC.

## Dataset versions

Dataset **v1** (`data/processed/*.csv|json`) is frozen: its sha256 hashes are in
`data/processed/FROZEN.json`, and the generator scripts refuse to write to it (there is no
override). Corrections or new data create a new draft version from a frozen parent with a
recorded decision, e.g.:

```
python scripts/run_audit.py            --version v1.1 --from-version v1 --decision D-XXX
python scripts/run_split.py            --version v1.1
python scripts/build_image_manifest.py --version v1.1
python scripts/freeze_dataset_version.py v1.1      # only with explicit approval
```

A draft must keep every parent session's split and every parent image's label status
(unless listed in its `VERSION.json` as approved changes) before it can be frozen.

## Building the YOLO dataset view (seconds, no training)

```
python scripts/prepare_yolo_dataset.py                 # v1 -> data/yolo/lobster_v1/
python scripts/verify_yolo_dataset.py                  # Colab/lab only: Ultralytics reads it correctly
```

`prepare_yolo_dataset.py` verifies the frozen v1 hashes first. The view holds `images/`
links to the raw PNGs (symlinks, or hardlinks on Windows without Developer Mode; linked
images share bytes with the raw files and must never be written to), `labels/` with copied
positive labels and empty files for confirmed negatives, and `data.yaml`.
`build_summary.json` records the dataset version, a dataset fingerprint, input manifest hashes
and the git commit. zExtra is never included. Set `UWCT_RAW_ROOT` if the raw data lives
elsewhere (the folder containing `Set 100`, `Set 2000`, `zExtra Photos`).

## Checks

```
python -m unittest discover tests                  # CPU unit tests, seconds
python scripts/check_raw_unchanged.py              # structural raw-data check
python scripts/check_raw_unchanged.py --subset training --root <copy>   # e.g. Colab copy
python scripts/env_report.py                       # OS/Python and, if present, GPU/CUDA/torch
```

The raw-data check is a **structural/completeness** check by default: every file's path and
size, plus sha256 of `.txt`/`.xlsx`/`.docx` contents. PNG contents are *not* hashed by default,
so it does not cryptographically prove PNG pixels are unchanged; `--full` hashes every file
(requires a fingerprint recorded with `--write --full`).

## Multi-file sessions

Nine sessions contain two GoPro recording files. VERIFIED: both files share one session code
and always stay in the same split. INFERRED: the file IDs look like sequential GoPro chapters
of one recording. UNKNOWN — requires confirmation: whether any were simultaneous cameras.
(The flag text `multi_camera_session` in the frozen v1 `split_manifest.csv` is a historical
name for "multi-file session".)
