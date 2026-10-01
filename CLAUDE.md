# CLAUDE.md — Underwater Creature Video Tracking Software

Authoritative project rules for Claude Code. If a task conflicts with a rule here, stop and ask.
Details: README.md (usage), docs/DATASET.md (data card), docs/EXPERIMENTS.md (experiments),
docs/DECISIONS.md (decisions). Task template: PROMPT.md.

## 1. Project
Underwater video pipeline: detection → multi-object tracking → trajectories/analytics →
visualization/export → optional Streamlit GUI. First target class: `0: lobster`. University
project with weekly demos: work with the data available now; new data becomes a new dataset
version and never blocks progress. Must-haves beat optional features.

## 2. Hard rules
1. `data/raw/**` is read-only. Never rename, move, delete, edit or re-save anything there; never
   write outputs or caches inside it. Scripts that write must refuse output paths inside the raw root.
   Sole recorded exception: the user-approved D-011 removal of two redundant ZIP archives
   (2026-09-18). Any further exception needs explicit user approval and its own DECISIONS.md entry.
2. `data/yolo/*/images/**` may be hardlinks/symlinks to raw files: never open them for writing, never
   run in-place image tools on them. Image processing writes to a new directory. Non-PNG sources
   must be built with `--link-mode copy` (Ultralytics rewrites corrupt JPEGs in place).
3. Local Windows PC: code, metadata, git, CPU tests of about a minute or less. No training, no
   large-scale video inference, no sweeps. Do not install torch/ultralytics/opencv or other heavy
   packages locally without explicit user approval. GPU work runs on Colab or the lab GPU only.
4. Frozen dataset versions (registered in `data/processed/FROZEN.json`, currently v1) are immutable.
   Never edit, regenerate or overwrite their files; there is no override. Corrections or new data
   create a new draft version (e.g. v1.1, v2) from a parent with a DECISIONS.md entry; freezing a
   version (`scripts/freeze_dataset_version.py`) requires explicit user approval.
5. Split unit = session (deployment code without the GoPro file ID). All recording files of a
   session stay in one split. Never split by image. Multi-file sessions: VERIFIED to share one
   session grouping; INFERRED to be sequential GoPro chapters; UNKNOWN whether any were simultaneous
   cameras. Do not describe them as multi-camera.
6. "Missing label = negative" only where a source's `missing_label_policy` is `negative` (currently
   Set 2000 only). zExtra stays excluded. Never invent boxes or give negatives a class.
7. The test split is never used for training, tuning, threshold choice, early stopping or model
   selection. Only pre-declared final candidates are evaluated on test, and every test evaluation is
   logged in docs/EXPERIMENTS.md.
8. Never fabricate data, metrics, files or results. Report only what exists in files or runs.
9. No tracking metrics (IDF1, HOTA, MOTA) without ground-truth track IDs. No physical units
   (cm, m, cm/s) without documented calibration: use px, frames, and seconds only when fps is known.
   Sparse-frame (non-video) sequences are "sparse-frame/timelapse analysis", never "video tracking".

## 3. Scientific reporting
Mark claims OBSERVED / INFERRED / ASSUMPTION / HYPOTHESIS / LIMITATION; write
"UNKNOWN — requires confirmation" when appropriate. v1 results must state: test sessions share
sites (mostly also dates) with train sessions, so they measure generalization to new sessions, not new
sites; substrate/visibility metadata is mostly missing; zExtra is excluded.

## 4. Data pipeline and versioning
raw → `run_audit.py` → `run_split.py` → `build_image_manifest.py` (all `--version`-scoped) →
`prepare_yolo_dataset.py --version` (per machine) → `verify_yolo_dataset.py` (GPU env).
- New data: new folder under `data/raw/`, new `DATASET_SOURCES` entry with an explicit label policy,
  new version extending its parent (parent sessions keep their splits unless the decision says otherwise).
- Class IDs are never renumbered; new species append IDs.
- `data/yolo/` is rebuilt per machine; never copy it between machines or commit it.
- Raw check: `scripts/check_raw_unchanged.py` must pass after any data-touching change. By default it
  is a structural/completeness check (paths + sizes, content hashes of txt/xlsx/docx only);
  `--full` also hashes PNG contents. Describe results accordingly.
  Default fingerprint: `raw_fingerprint_d011_post_cleanup.json`. `raw_fingerprint.json` is the
  historical v1 (pre-cleanup) record: never overwritten, and against it the two removed ZIPs show as missing.

## 5. Experiments
- Every training run = committed config `configs/experiments/E###_<slug>.yaml` naming its parent
  experiment and the single variable changed (or why several changed).
- Each run writes `run_record.json`: git commit + dirty flag, dataset version + fingerprint, model +
  pretrained checkpoint, Ultralytics args, seed, env report, duration, val metrics (test metrics only
  for declared final evaluations). One row per run in `experiments/registry.csv`.
- Weights and run folders live off-repo (Drive/lab storage), referenced by path + sha256.
- Report P, R, mAP50, mAP50-95, per-condition results and false positives on negative images, never a
  single number. Selection metric is declared in docs/EXPERIMENTS.md before comparisons.

## 6. GPU procedure
Before any long job: `env_report.py` → raw check → `prepare_yolo_dataset.py` →
`verify_yolo_dataset.py` (must print VERIFIED) → smoke test (≤1 epoch, small fraction) → launch with
checkpoints on persistent storage and a resume path.

## 7. Code conventions
- Python ≥3.10. Standard library first; heavy imports only inside GPU-side modules.
- Logic in `src/<package>/`; thin CLIs in `scripts/`; notebooks only call scripts.
- No absolute paths in code. Raw root via `UWCT_RAW_ROOT`; other locations via CLI args with
  repo-relative defaults. No hidden global state.
- Stages communicate through files with documented schemas (docs/PIPELINE.md once created).
  No abstraction layer until a second implementation exists. Don't rewrite working code without a
  demonstrated defect or requirement.
- Type hints on public functions, deterministic output, actionable errors.
- Tests: `python -m unittest discover tests` (CPU, seconds) before proposing any commit.

## 8. Git
- Never commit without the user's explicit approval of a commit plan (files, grouping, messages).
- Coherent phase-level commits; the message says what and why.
- Never commit: `data/raw/`, `data/yolo/`, `runs/`, weights (`*.pt`, `*.onnx`) unless explicitly
  chosen, videos, secrets, `.env`, caches.
- Before proposing a commit: `git status`, diff summary, tests pass, raw check passes if data code changed.
- No push, remotes, force operations or history rewrites unless the user asks.

## 9. Working method
Inspect repo, manifests and configs → plan → smallest change → test → report exactly what changed
(files created/modified, checks run, raw-data status). When uncertain, inspect or ask; never guess.
Stop at the task's stop condition and wait for approval before starting the next phase.

## 10. Team workflow
- Owners: Saksham, `p4/colab-baseline` (integration branch, YOLO E001/E002); Anu,
  `anu/model-development` (RT-DETR E003, D-015); Harsh, image preprocessing (delivers data, not code).
- `p4/colab-baseline` is protected: changes land only by PR and squash merge, with 1 approving review.
  Owner override (D-019): Saksham may merge his own PRs without a review; such a PR says "Owner merge,
  no review". Teammates' PRs still need his approving review, and no one else bypasses the protection.
  The override covers merge review only, not the §2 hard rules. Before starting work, `git fetch` and
  merge `origin/p4/colab-baseline` into your branch. Every PR is checked against this file before merging.
- Layout (D-016): notebooks in `notebooks/`, configs in `configs/experiments/`, code in `src/<package>/`,
  CLIs in `scripts/`, docs in `docs/`. No per-person or per-model top-level folders. Ask before adding
  a new top-level folder.
- Decision/experiment IDs: take the next free ID from `origin/p4/colab-baseline`, then re-check just
  before opening the PR and renumber if there is a collision. Write "user-approved" only for text that
  person approved. Never edit an accepted decision; add one that supersedes it.
- Delivered data (e.g. preprocessed images) is new data (§4): a new raw source and a new dataset version
  on the parent's session split. Never use a split shipped with the data.
- A PR states what and why, the files changed, the test result, the raw check if data code changed, and
  any GPU runs (run ids). It contains only what it is about.
