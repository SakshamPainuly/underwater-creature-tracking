# Decision log

Short, dated records of project decisions. New dataset versions must reference an entry here
(`--decision D-XXX`). Entries are appended, never rewritten; a reversed decision gets a new entry.

---

## D-001 — Split unit is the session (2026-09-17, accepted)
- **Context:** Images are frames sampled from GoPro videos. Frames of one session share lobsters,
  background, lighting and camera position; image-level random splitting would leak.
- **Decision:** Split by `session_id` = deployment code without the GoPro file ID. All recording
  files of a session stay in one split.
- **Consequence:** 56 sessions; train/val/test/excluded are disjoint by session (verified).

## D-002 — v1 split method (2026-09-17, accepted)
- **Decision:** Seeded (`SPLIT_SEED = 20260917`), about 70/15/15 by session, stratified by the
  red-light/night flag only (substrate/visibility are missing for 39 of 44 Set 2000 sessions, too
  sparse to stratify on). Set 100 (a single session) is train-only. The 11 zExtra sessions are excluded.
- **Consequence:** train 32 / val 7 / test 6 / excluded 11 sessions.

## D-003 — Missing-label semantics per source (2026-09-17, accepted)
- **Decision:**
  - Set 2000: missing label = confirmed negative (stated in the dataset documentation).
  - zExtra: pending, excluded. Spreadsheet QC notes mention lobsters on unlabeled zExtra images.
  - Set 100: a missing label is an error (documented as fully labeled).
  - Encoded as an explicit allowlist, `MISSING_LABEL_POLICY` in `config.py`; unknown sources are
    an error, never "negative".
- **Consequence:** 241 confirmed-negative images (train 159 / val 55 / test 27).

## D-004 — Ultralytics dataset view (2026-09-17, accepted)
- **Decision:**
  - Build a derived `images/` + `labels/` tree outside the raw data: images linked, positive labels
    copied, confirmed negatives as empty label files.
  - Pin `ultralytics==8.4.155`.
- **Reason (from the 8.4.155 source):** Ultralytics looks for labels by swapping the last
  `/images/` for `/labels/` and writes a `.cache` next to the labels. Pointing it at the raw folders
  would silently drop all Set 2000 boxes and write into the raw data. Missing and empty label files
  are both treated as background. Corrupt JPEGs are re-saved in place, so non-PNG sources are built
  by copying.

## D-005 — Freeze dataset v1; immutable versions (2026-09-17, accepted)
- **Decision:** Register v1 in `data/processed/FROZEN.json` (sha256 of `deployment_metadata.csv`,
  `deployment_metadata.json`, `split_manifest.csv`, `image_manifest.csv`,
  `dataset_consistency_report.json`).
  - Frozen versions can never be written; there is no override.
  - Changes go into a draft version (from a frozen parent, with a decision entry) that must keep
    parent split assignments and label statuses unless explicitly listed as approved changes.
  - Freezing a draft requires explicit user approval.
- **Consequence:** The v1 generators were verified to regenerate all five v1 files byte-identically.

## D-006 — "Multi-file session" terminology (2026-09-17, accepted)
- **Decision:** Nine sessions contain two GoPro recording files.
  - VERIFIED: they share one session code and one split.
  - INFERRED: the IDs (same 4-digit file number, consecutive 2-digit prefix, frame numbering
    restarting per file) look like sequential chapters of one recording.
  - UNKNOWN — requires confirmation: whether any were simultaneous cameras.
  - Call them "multi-file sessions". The v1 flag text `multi_camera_session` is kept unchanged as a
    historical name, so v1 regenerates byte-identically.

## D-007 — Site/date overlap is a documented v1 limitation (2026-09-17, accepted)
- **Context:**
  - 8 of 29 site-date groups span more than one split.
  - 5 of the 6 test sessions share site and date with a train session.
  - Every test site also appears in train.
- **Decision:** Keep the v1 split unchanged; report that v1 test results measure generalization to
  new sessions, not new sites. A new-site test is possible only in a future version (open question Q4).

## D-008 — Known v1 split caveats retained (2026-09-17, accepted)
- **Decision:** The split stays frozen despite:
  - val being dominated by `21_MACK_10` (465 of 1,010 boxes);
  - `19_SUB1_04` in val contributing 3 boxes (likely one lobster over 3 frames);
  - test boxes concentrated in `13_SHRI_12` + `21_SUB6_03` (1,135 of 1,512);
  - sparse-label sessions `06_HERR_02`, `19_BESU_01` (low visibility; negatives less certain).
  These are reported as caveats, with per-session results.

## D-009 — Raw-data fingerprint (2026-09-17, accepted)
- **Decision:** `data/processed/raw_fingerprint.json` is recorded once and never overwritten. By
  default it is structural/completeness: every file's path and size plus sha256 of
  `.txt`/`.xlsx`/`.docx`; PNG contents are not hashed. `--full` hashes every file and is needed for a
  cryptographic PNG content check.

## D-010 — Governance and compute rules (2026-09-17, accepted)
- **Decision:**
  - CLAUDE.md holds the permanent rules; PROMPT.md is the task template.
  - No training or heavy installs on the local PC (`requirements.txt` is stdlib-only;
    `requirements-gpu.txt` is for Colab/lab).
  - Commits only after explicit user approval of a commit plan.
  - Detection work (P4–P6) does not wait for MP4s. Only video inference and tracking (P7/P8) do;
    without video, only clearly labeled sparse-frame/timelapse analysis is allowed.

## D-011 — Raw-data storage cleanup: remove redundant ZIP archives (2026-09-18, accepted)
- **Context:** `data/raw/biology/Set 2000/` holds two ZIP archives that duplicate folders already
  extracted next to them:
  - `Set 2000/batch_2 (2k - imgOnly).zip` (4,029,491,083 bytes) ↔ `Set 2000/batch_2 (2k - imgOnly)/`
  - `Set 2000/batch_2_2k_labels.zip` (610,869 bytes) ↔ `Set 2000/batch_2_2k_labels/`

  They are not used by any code, manifest or frozen v1 artifact (`FROZEN.json`, split/image
  manifests, consistency report, deployment metadata). They appear only as two entries in the v1
  raw fingerprint (`data/processed/raw_fingerprint.json`, D-009).
- **Verification before deletion (OBSERVED, 2026-09-18, read-only):**
  - Image ZIP: 2,000 entries vs 2,000 extracted files; all names and uncompressed sizes match,
    nothing only in the ZIP or only on disk. PNG contents were not compared byte-for-byte
    (LIMITATION).
  - Label ZIP: 1,759 entries vs 1,759 extracted files; all names and sizes match, and CRC32 of all
    1,759 extracted label files matches the ZIP's recorded CRC32.
  - `scripts/check_raw_unchanged.py` (default structural check) against the v1 fingerprint:
    4,409 expected / 4,409 present, 0 missing / added / size-changed / content-changed.
- **Decision:**
  - Explicit, user-approved, one-time exception to CLAUDE.md hard rule 1 for storage cleanup: the
    two ZIP archives above are deleted from `data/raw/biology/` by the user. No other raw file is
    removed, moved, renamed or modified, and the extracted folders are kept unchanged.
  - The v1 raw fingerprint `data/processed/raw_fingerprint.json` stays unchanged and preserved in
    git as the historical, pre-cleanup record of the raw directory (4,409 files). It is not
    overwritten, renamed or edited; D-009 is not reversed.
  - A separately named post-cleanup fingerprint,
    `data/processed/raw_fingerprint_d011_post_cleanup.json`, is recorded once after deletion. It
    represents the raw directory after removal of the redundant ZIP archives, and must equal the v1
    fingerprint minus exactly the two ZIP entries, with all other entries identical.
  - `FROZEN.json` and all frozen v1 manifest/split artifacts are unchanged; dataset v1 content,
    splits and labels are unaffected, so no new dataset version is created.
  - Expected retained raw size (computed before deletion, to be confirmed by measurement after
    deletion): 4,407 files, 5,477,325,614 bytes. Documentation uses the measured value.
- **Outcome (OBSERVED, 2026-09-18, read-only verification after the user deleted the ZIPs):**
  - Search for `*.zip` under `data/raw/`: none found.
  - `scripts/check_raw_unchanged.py` (default structural check) against the v1 fingerprint
    `data/processed/raw_fingerprint.json`: 4,409 expected / 4,407 present; missing 2 (exactly
    `Set 2000/batch_2 (2k - imgOnly).zip` and `Set 2000/batch_2_2k_labels.zip`); added 0; size
    changed 0; content changed 0. Exit 1 is expected, because the v1 fingerprint is the pre-cleanup
    record.
  - Retained raw directory (measured): 4,407 files, 5,477,325,614 bytes (5.48 GB / 5.10 GiB),
    identical to the value computed before deletion.
  - `verify_frozen('v1')` (hashes against `FROZEN.json`): VERIFIED. `raw_fingerprint.json` is
    unchanged vs HEAD.

## D-012 — P4 Colab training setup and E001/E002 baseline (2026-09-18, accepted)
- **Context:** P4 sets up reproducible GPU training on Google Colab for frozen dataset v1. The
  local PC stays standard-library only (D-010).
- **Decision (user-approved choices):**
  - **Code on Colab:** the private GitHub repo, read with a dedicated fine-grained token (this
    repository only, Contents read-only) stored in Colab Secrets as `UWCT_GITHUB_TOKEN`.
    - The token is read from Colab Secrets at each clone/fetch and handed to that git process
      only, through its environment and a `GIT_ASKPASS` helper that holds no secret.
    - It is never in a command-line argument, URL, file or `.git/config`, and credential helpers
      are disabled.
    - The notebook checks out one exact commit and never reuses the general GitHub CLI credential.
  - **Data on Colab:** only the v1 training sources (Set 100; Set 2000 images and labels) are copied
    to `MyDrive/uwct/raw/biology/`, then to Colab local disk, where the structural raw check
    (`--subset training`) runs. zExtra is never copied.
  - **Runs:** `MyDrive/uwct/runs/<run>/` (off-repo). Weights are referenced by path and sha256 and
    never committed.
  - **E001:** `yolo26s.pt`, imgsz 640, max 100 epochs, patience 30, batch 16, seed 0,
    `deterministic=True`, ultralytics 8.4.155, dataset v1 (fingerprint `9f2bf857…`). If
    `yolo26s.pt` is unavailable in 8.4.155, stop and report; no substitute model.
  - **E002 (prepared, not run):** identical to E001 except imgsz 1280. Its full run is blocked
    until E001 is registered in `experiments/registry.csv`.
  - **Configs:** YAML in a restricted subset, validated locally with the standard library and
    cross-checked against PyYAML on Colab (no local PyYAML install).
  - **Test split:** the training process gets a data file with only train and val. Configs cannot
    name a split or a data file. Validation after training uses `split="val"`.
  - **Final test evaluation:** a separate stage (`src/training/final_test.py`,
    `scripts/final_test_eval.py`, notebook cells 16–17), never imported by training.
    - It runs only for experiments declared in `experiments/final_candidates.csv` at the
      checked-out commit, once per finished full run.
    - The operating confidence is chosen on val first; then `best.pt` is evaluated with the full
      `data.yaml` and `split="test"`.
    - Metrics, plots and prediction examples go to `runs/<id>/final_test/` on Drive, and each
      evaluation is logged in docs/EXPERIMENTS.md.
  - **Checkpoint selection:** `best.pt` within a run uses Ultralytics' built-in fitness on val
    (exact 8.4.155 weights UNKNOWN — requires confirmation). Experiments are compared on val mAP50
    (docs/EXPERIMENTS.md).
- **Consequence:** no dataset version, split or frozen file changes. The run records and registry
  implement CLAUDE.md §5. Runbook: docs/COLAB.md.
