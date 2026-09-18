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
