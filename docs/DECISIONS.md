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

## D-013 — Run records are archived in the repository (2026-09-21, accepted)
- **Context:** CLAUDE.md §5 keeps run folders off-repo (Drive/lab storage), referenced by path and
  sha256. E001's run folder lives at `/content/drive/MyDrive/uwct/runs/E001`. Two of the three
  collaborators have no access to that Drive, so at commit `f83c8f3` the repository recorded E001's
  metrics in docs/EXPERIMENTS.md with no in-repo artifact behind them, and
  `experiments/registry.csv` held only its header.
- **Decision (user-approved):** the small metadata artifacts of a finished run are committed under
  `experiments/records/<experiment_id>/`:
  - `run_record.json` — the record `scripts/register_run.py` consumes (CLAUDE.md §5 fields);
  - `results.csv` — Ultralytics' per-epoch training curve;
  - `final_test_record.json` — present only for a declared final test evaluation.

  Everything else stays off-repo exactly as before: weights (`best.pt`, `last.pt`), plots,
  prediction images and the run folder itself, referenced by path + sha256. The folder name
  `records` is deliberate: `.gitignore` line 37 (`runs/`) matches at any depth, so
  `experiments/runs/` would be silently ignored.
- **Scope:** this narrows, and does not reverse, CLAUDE.md §5. Bulk artifacts are still off-repo;
  only metadata under ~50 KB per run is tracked. A run's records are committed once, after
  registration, and are never edited afterwards.
- **Consequence (OBSERVED, 2026-09-21):** E001 registered in `experiments/registry.csv` from
  `experiments/records/E001/run_record.json`; the per-run archive is
  `run_record.json` (9,634 B, sha256 `f8b1ac9a…`), `results.csv` (12,096 B, sha256 `cf3378a7…`),
  `final_test_record.json` (8,391 B, sha256 `a39189f0…`). E002's full run is no longer blocked
  (D-012). No dataset version, split, frozen file or raw file is affected.

## D-014 — Colab raw data comes from the shared ZIP; zExtra is copied but unused (2026-09-21, accepted)
- **Context:** D-012 specified that only the v1 training sources (Set 100, Set 2000 images and
  labels) are uploaded to `MyDrive/uwct/raw/biology/` and copied to Colab local disk, and that
  `zExtra Photos/` is never copied. In practice the delivery is kept as a single ZIP on Drive and
  fetched with `gdown`, because uploading the loose folders (~5.5 GB) is impractical to repeat.
  Commit `f83c8f3` replaced the copy cell accordingly but did not update D-012, docs/COLAB.md or
  the notebook guard test, which searched for the removed `SUBSET` list and errored.
- **Decision (user-approved):** the documents follow the notebook. The shared ZIP is the supported
  way to provision raw data on Colab:
  - the notebook downloads one ZIP by hard-coded Drive file ID, extracts the two inner `Set 2000`
    archives and copies `Set 100`, `zExtra Photos` and the project documents into
    `/content/raw/biology/`, then sets `UWCT_RAW_ROOT`;
  - nothing is ever written back to Drive, and the raw data on the local PC is untouched;
  - `zExtra Photos/` is present in the Colab copy. This supersedes D-012's "zExtra is never
    copied" **for the Colab provisioning step only**.
- **What is unchanged:** CLAUDE.md hard rule 6 still holds — zExtra is excluded from every dataset
  version, the YOLO view and all training and evaluation. `prepare_yolo_dataset.py` builds from the
  frozen v1 image manifest, which contains no zExtra image, and `train_yolo.py` refuses to start if
  the built view contains one. Dataset v1, its splits, `FROZEN.json` and the raw fingerprints are
  unaffected.
- **Limitations (OBSERVED, 2026-09-21):**
  - The notebook's comment says zExtra is kept "for fingerprint verification", but cell 8 runs
    `check_raw_unchanged.py --subset training`, so the zExtra files on Colab are neither verified
    nor used. Dropping `--subset training` would close this gap; not done here.
  - The ZIP's inner archives are the ones removed locally under D-011, where PNG contents were
    never byte-compared. Training sources are still covered by the structural raw check (paths and
    sizes) and by the frozen v1 hash verification before the view is built.
  - `gdown` is installed inside the cell and is not pinned in `requirements-gpu.txt`.
  - The Drive file ID is committed in the notebook, so anyone who can read the repository can fetch
    the dataset if the link's sharing scope allows it. The scope is UNKNOWN — requires confirmation.
- **Consequence:** docs/COLAB.md rewritten (setup step 2, the cell-7 row, a new section on what is
  and is not verified); the notebook guard in `tests/test_training_config.py` now asserts what
  still matters — the raw check runs on the training subset, and no cell that invokes
  `prepare_yolo_dataset.py`, `train_yolo.py` or `final_test_eval.py` mentions zExtra. E001's
  results are unaffected: its run record shows the frozen v1 registry verified and Ultralytics
  seeing exactly 1,433 train / 324 val images.

## D-015 — RT-DETR baseline E003 for a YOLO vs RT-DETR comparison (2026-09-24, accepted)
- **Context:** a second detector family is trained independently on the same data so it can be
  compared fairly with the YOLO baseline E001. Inspection (OBSERVED, 2026-09-24): the v1 labels are
  YOLO txt, one class `0: lobster`, and the splits come from the frozen v1 manifests. Ultralytics
  8.4.155 (the pinned version) ships `RTDETR` with its own trainer, NMS-free validator and
  predictor, lists `rtdetr-l.pt` / `rtdetr-x.pt` as official assets, and reads the same YOLO-format
  `data.yaml`, empty negative label files included.
- **Decision (user-approved):**
  - Use Ultralytics RT-DETR on the existing v1 YOLO dataset view. **No annotation conversion**, no
    new dataset version, and no change to `prepare_yolo_dataset.py`; COCO JSON would only be needed
    for a non-Ultralytics RT-DETR implementation, whose metrics would not be directly comparable.
  - **E003** (`configs/experiments/E003_rtdetr_baseline.yaml`): parent E001, single variable changed
    is the model `yolo26s.pt -> rtdetr-l.pt`. Dataset v1 (same fingerprint), imgsz 640, ≤100
    epochs, patience 30, batch 16, seed 0, `deterministic=True`, ultralytics 8.4.155 are all as in
    E001. `rtdetr-l.pt` is the smaller of the two official RT-DETR checkpoints.
  - The runner picks the Ultralytics model class from the checkpoint name
    (`experiment.model_family`: `rtdetr-*` -> `RTDETR`, `yolo*` -> `YOLO`, anything else is refused).
    Every load goes through it: the pretrained checkpoint, val of `best.pt`, resume, and the final
    test stage. The run record's `model` block gains a `family` field.
  - **Comparison rule:** the same run-record, registry and final-test machinery as E001. Model
    choice (if any) is made on **val mAP50** (docs/EXPERIMENTS.md); secondary metrics as declared
    there. Parameter count, GFLOPs and inference speed on the same GPU type are reported alongside,
    because `rtdetr-l` is a considerably larger model than `yolo26s`.
  - **Test split:** E003 is evaluated on test only after its full run is registered and reviewed and
    it is declared in `experiments/final_candidates.csv` (CLAUDE.md rule 7). E001 and E003 test
    numbers are then reported side by side and are never used to choose between them.
- **Risks to check in the smoke test (UNKNOWN until run on Colab):**
  - batch 16 at imgsz 640 may not fit on a Tesla T4 with RT-DETR-L;
  - Ultralytics notes that AMP can produce NaN losses for RT-DETR;
  - `F.grid_sample` has no deterministic implementation; Ultralytics sets deterministic mode with
    `warn_only=True` (OBSERVED in 8.4.155 source), so it warns, and E003 is less reproducible than
    E001.

  If batch or AMP has to change, stop and record a new decision (a second changed variable).
- **Known wording gap:** the val-metrics note in `run_record.json` says "conf=0.001, iou=0.7" for
  every family; RT-DETR's validator does not apply NMS, so the IoU part does not apply to E003.
- **Consequence:** no dataset version, split, frozen file or raw file changes. The shared YOLO
  notebook is unchanged. E003 runs from `my_model/colab_train_rtdetr.ipynb` (user-requested
  location): a copy of it for branch `anu/model-development` and E003, with an added smoke-test
  cell that stops on non-finite losses (docs/COLAB.md). `scripts/train_yolo.py` keeps its name and trains either family.

## D-016 — Repository layout: all Colab notebooks live in `notebooks/` (2026-09-29, accepted)
- **Context:** D-015 placed the E003 notebook and a README in a new top-level `my_model/` folder.
  The documented layout (README.md) keeps notebooks in `notebooks/`, experiment configs in
  `configs/experiments/` and runbooks in `docs/`. A per-person model folder would split the
  shared pipeline by author instead of by stage.
- **Decision (user-requested):** `my_model/colab_train_rtdetr.ipynb` moves to
  `notebooks/colab_train_rtdetr.ipynb` (content unchanged except the Colab badge path).
  `my_model/README.md` is removed; its runbook steps are in docs/COLAB.md ("Other experiments") and
  its file table duplicated README.md. `my_model/` no longer exists. This supersedes the notebook
  location stated in D-015; nothing else in D-015 changes.
- **Rule going forward:** new models are added as experiment configs plus code in `src/`, never
  as a per-person or per-model top-level folder. Notebooks go in `notebooks/`, docs in `docs/`.
- **Consequence:** no code, config, dataset, split, frozen or raw file changes. The Colab badge
  points at `notebooks/` on branch `anu/model-development`, so it works once that branch includes
  this commit. `COMMIT` in cell 1 stays empty on this branch (it is set per run).

## D-017 — Enhanced images (Harsh, 2026-09) as image source; dataset v2 and experiment E004 (2026-09-29, accepted)
- **Context:** a teammate (Harsh) delivered preprocessed ("enhanced") copies of the Set 100 +
  Set 2000 images. The goal is to train and test the E001 baseline on them, the same way as E001,
  for a fair raw vs enhanced comparison. Inspection of the delivery (OBSERVED, 2026-09-29):
  - 2,105 PNGs in `images/{train,val,test}/` plus macOS metadata (`__MACOSX/`, `.DS_Store`).
  - Every filename is in the v1 image manifest. They are exactly the v1 train + val + test
    images; no zExtra.
  - Every image has the same width × height as its raw original, and 0 of 40 sampled images
    are byte-identical to the raw file.
  - The delivered folders are **not the v1 split**. They are session-grouped, but assigned
    differently: 234 of the delivered `test/` images are v1 train, and 221 of the delivered
    `train/` images are v1 test.
- **Problem found in the pipeline:** the dataset fingerprint covers `build_manifest.csv` + label
  files, not image bytes. Swapping in other images would have given E001's fingerprint, so the
  records could not tell raw and enhanced runs apart.
- **Decision (user-requested):**
  - **Raw source:** the delivery is stored as a new read-only raw root,
    `data/raw/enhanced_2026-09/images/{train,val,test}/`, as delivered minus the macOS files.
    Content fingerprint (`--full`): `data/processed/raw_fingerprint_enhanced_2026-09.json`.
    Colab root: `UWCT_ENHANCED_ROOT`.
  - **Images found by filename:** the delivered folders, and the split they imply, are never used.
  - **Image sources:** `config.IMAGE_SOURCES` names each image root (`biology` = raw,
    `enhanced_2026-09`). A version's `VERSION.json` may name its `image_source` (default `biology`).
    Labels always come from the raw data.
  - **Builder rules** for a non-raw source:
    - every image must be a PNG with the same size as its raw original;
    - `build_manifest.csv` gains `image_source` and `image_sha256`, so the fingerprint identifies
      the image content;
    - `verify_built_dataset` (preflight, final test) re-hashes the images and checks the image
      source.
    - v1's build is byte-identical to before: rebuilt with this code, its fingerprint is still
      `9f2bf857…ef810` (OBSERVED).
  - **Dataset v2** = draft from v1 (`scripts/create_dataset_version.py`): the v1 manifests byte
    for byte (same sessions, split, labels, label policies; zExtra excluded) plus
    `image_source: enhanced_2026-09`. Freezing it needs the user's explicit approval (rule 4).
  - **E004** (`configs/experiments/E004_yolo_enhanced.yaml`): parent E001. The single variable
    changed is the images (raw → enhanced). `dataset_version`/`dataset_fingerprint` change only
    because they identify those images. Everything else is as in E001.
  - **Procedure:** same code path, notebook and machinery as E001 (preflight → smoke → full →
    register → declare → final test). The operating confidence is chosen on E004's own val.
- **Comparison rule:** declared in docs/EXPERIMENTS.md before any E004 result exists.
- **zExtra:** still excluded and unlabelled (D-014). Enhanced zExtra images, when delivered, are a
  separate raw root and are for qualitative inspection only. No metric is possible without labels.
- **UNKNOWN — requires confirmation (Harsh):**
  1. the method and its parameters, and whether they were identical for every image;
  2. whether anything was tuned, fitted or chosen while looking at images (in particular any
     v1 val/test image);
  3. whether any geometric change (crop, shift, warp, rotation) was applied. The sizes match,
     but that does not rule out a shift.

  Results are not reported before these are answered and recorded here.

## D-018 — Harsh's answers on the enhanced images; E004 val results reported, test after a review-list check (2026-10-01, accepted)
- **Context:** D-017 left three questions about the enhanced images UNKNOWN and held back E004
  results until they were answered. E004 trained on 2026-09-30 (commit `97ac710`, one launch,
  81 epochs) and is registered; its records are in `experiments/records/E004/` (D-013). Harsh
  answered in writing on 2026-10-01. The points below are his statements, summarized; we have not
  verified them unless marked OBSERVED.
- **Answers:**
  1. **Method:** profile `fusion_mild`, a classical pipeline with no learned model and no physical
     dehazing model:
     - red/blue compensation from green (strength 0.5);
     - channel balancing (maximum gain 4.0, common 99.5th-percentile highlight normalization);
     - LAB luminance stretch (1st–99th percentiles, blend 0.5, if normalized range > 0.03);
     - CLAHE on luminance (clip 1.5, 8×8 tiles, blend 0.3);
     - gamma branch (0.85) and sharpening branch (Gaussian σ 1, unsharp 0.2);
     - multiscale fusion (≤ 5 pyramid levels; contrast, saturation and well-exposedness
       weights); final blend 1.0; denoising off; near-black border threshold 8/255.
  2. **Same for every image:** one configuration for all images. Nothing depends on split, folder,
     video or annotations. Per-image statistics (channel means, percentiles, CLAHE histograms,
     border mask, fusion weights) are computed from each image, so the correction is adaptive
     while the settings are fixed. No random operations.
  3. **Tuning:** no training and no learned checkpoint. The settings were not re-tuned for this
     delivery. However, the profile was originally chosen by **visually reviewing lobster images**
     under several profiles; seven review images are documented in Harsh's
     `previously_reviewed_images.json` (not in this repository). That review excluded Harsh's own
     test split, which is not v1's: D-017 OBSERVED that 221 of his `train/` images are v1 test.
  4. **Geometry:** no crop, rotation, flip, translation, warp or resize; the original coordinate
     grid and dimensions are kept, so labels need no transformation (consistent with the matching
     sizes OBSERVED in D-017). Caveats: pyramid fusion resamples internally and filters mix
     neighbouring pixels, so edges can look sharper or softer; edge-connected near-black pixels
     can be set to black.
  5. **Output:** 8-bit, three-channel PNGs. The zExtra export uses the same definitions and
     settings.
- **Provenance caveat:** the answers describe Harsh's *updated* notebook, which was not run on our
  images. He verified that it reproduces the original definitions pixel for pixel on three inputs.
  INFERRED: the answers also describe the delivered images.
- **Decision (user-approved, 2026-10-01; option chosen: "val now, test later"):**
  - The D-017 reporting hold is lifted for **val**. E004 val results and the declared E001 vs E004
    val comparison (docs/EXPERIMENTS.md) may be reported, each with the limitations below.
  - The E004 **test** evaluation (declaring E004 in `experiments/final_candidates.csv`) waits
    until Harsh's review list has been checked against the v1 val and test filenames and the
    result is recorded in a new decision. If a reviewed image is in v1 val or test, that decision
    states how E004's results are qualified.
  - D-017 is not edited. This decision replaces only its last line ("Results are not reported
    before these are answered and recorded here").
- **Limitations every E004 result carries:**
  - UNKNOWN — requires confirmation: whether any v1 val/test image was among the images reviewed
    while the profile was chosen.
  - The enhancement adapts to each image's statistics; it is not a fixed per-pixel mapping.
  - One run per arm (seed 0); E001 trained 100 epochs, E004 stopped early at 81.
  - The v1 limitations (new sessions, not new sites; concentrated val/test sessions; sparse
    metadata; zExtra excluded).
- **Open:** obtain `previously_reviewed_images.json` from Harsh and check it. Optional: ask Harsh to
  run the updated notebook on a few v1 raw images, and compare the result with the delivered images
  pixel for pixel, to replace the INFERRED provenance above with an OBSERVED one.

## D-019 — Owner override: Saksham may merge his own PRs without a review (2026-10-01, accepted)
- **Context:** CLAUDE.md §10 required 1 approving review for every change to the protected
  `p4/colab-baseline` and said never to bypass it. GitHub does not let an author approve their own
  PR, so the owner's PRs waited on a teammate. PR #5 (E004 records, D-018) was merged by the owner
  on 2026-10-01 as `0446faa` with no review, through the repository admin's bypass.
- **Decision (user-approved, 2026-10-01; option chosen: "process rules only"):**
  - Saksham, the repository owner, may merge his own PRs into `p4/colab-baseline` without an
    approving review. Each such PR says "Owner merge, no review", so the history shows it. PR #5
    is the first such merge (recorded here after the fact).
  - Teammates' PRs still need Saksham's approving review. No one else bypasses the protection.
  - Everything else is unchanged: PR + squash merge, the commit-plan approval in §8, and every PR
    is still checked against CLAUDE.md before merging.
  - The override covers **merge review only**. The §2 hard rules (raw data, frozen versions,
    splits, test split, no fabrication) keep their existing exception routes, which need
    explicit user approval and their own DECISIONS.md entry. Never fabricating data, metrics,
    files or results has no exception.
- **Consequence:** CLAUDE.md §10 is updated to match. The GitHub branch protection stays as it is;
  the owner's admin bypass is how an owner merge is done.

## D-020 — Val-only secondary-metrics evaluation; IoU 0.5 matching fixed before results (2026-10-10, accepted 2026-10-12)
- **Context:** the selection metric and secondary metrics are declared in docs/EXPERIMENTS.md, but
  the repository had no val evaluation of the secondary metrics. `src/training/final_test.py`
  chooses the operating confidence on val (max mean F1) and then evaluates on test. Comparing E001
  (v1) with E004 (v2, D-017/D-018) needs these numbers on val, without touching test (rule 7).
- **Decision (user-approved; plan approved on 2026-10-01, this text approved on 2026-10-12):**
  - New stage `src/training/val_eval.py` with CLIs `scripts/val_eval.py` (GPU env) and
    `scripts/compare_val_eval.py` (local, standard library). It reads only the run's train/val data
    file and manifest rows whose split is `val`, and refuses any image path outside `images/val/`.
  - Operating confidence: the same rule as `final_test.py` (max mean F1 on Ultralytics' val curves);
    E001 must reproduce 0.298 as a check of the code.
  - Matching rule, **fixed here before any result exists**: greedy by confidence, one ground-truth
    box per prediction, IoU ≥ 0.5 (`MATCH_IOU`). It is not tuned afterwards.
  - Reported at the operating confidence: TP/FP/FN, P, R, F1; false positives on the confirmed-negative
    val images (images with ≥ 1 detection, total detections); the same per session and per
    `lighting_flag` (from the version's `split_manifest.csv`), with AP50 per group (101-point).
  - Cost, for the comparisons that declare it (D-015): parameters and GFLOPs of the unfused model
    at the training imgsz, and speed as the median ms per image over the batch-1 val predictions
    (first image dropped as warm-up) plus Ultralytics' batched val speed, with the GPU name.
    Speeds are compared only between runs measured on the same GPU type.
  - `val_eval_record.json` (schema `uwct.val_eval_record/1`) joins the D-013 archive in
    `experiments/records/<experiment_id>/`. Predictions, per-image CSV and plots stay off-repo.
- **Limitations (stated with every result):** val chose each run's checkpoint and every operating
  confidence, so these are not unbiased estimates; one seed per run; E001 trained 100 epochs, E004
  81 (early stop); red-light val is 2 sessions / 31 boxes; `21_MACK_10` holds 465 of 1,010 val boxes.
- **Run (2026-10-11):** E001 and E004 evaluated on Colab in one Tesla T4 session at commit `d84a664`.
  Two GPU-only defects were fixed on the way: `c45b341` (the val path check followed the view's
  symlinks to the raw files) and `d84a664` (`model.info(verbose=False)` returns None). E001
  reproduced the operating confidence 0.2983 of its `final_test_record.json`. Records archived in
  `experiments/records/E001/` and `E004/`; the side-by-side tables are in
  `experiments/comparisons/E001_vs_E004_val.md` (`scripts/compare_val_eval.py`). The two records
  carry the earlier limitation wording "Val chose E001's checkpoint"; it means each run's checkpoint.

## D-021 — Collaborators use a short-lived classic token on Colab (2026-10-11, accepted 2026-10-12)
- **Context:** D-012 has the Colab notebook clone the private repo with a fine-grained token
  (this repository only, Contents read-only). On 2026-10-11 Anu's clone in cell 4 failed with
  `403 Write access to repository not granted`, using a fine-grained read-only token stored as
  `UWCT_GITHUB_TOKEN`. OBSERVED: `anusidhu9` is a collaborator with write permission and no pending
  invite. OBSERVED (GitHub docs, checked 2026-10-11): fine-grained tokens do not support access to
  repositories where the user is a collaborator. The repo belongs to a personal account, so only
  the owner can use the D-012 token. The same applies to Harsh.
- **Decision (user-approved; option "classic token now" chosen on 2026-10-11, this text approved on 2026-10-12):**
  - The owner keeps the D-012 fine-grained, read-only token.
  - Collaborators use a classic token with scope `repo` and an expiration of 30 days or less,
    stored only in Colab Secrets as `UWCT_GITHUB_TOKEN`. This supersedes the token rule of D-012 for
    collaborators only. The notebook code (GIT_ASKPASS, no credential storage, clone/fetch only)
    is unchanged.
  - LIMITATION: a classic `repo` token can write to every repository its holder can access,
    including this one (collaborators already have write access). Branch protection on
    `p4/colab-baseline` still applies. A leaked token is revoked on GitHub at once.
- **Rejected for now:** moving the repo to an organization with Anu and Harsh as members, so
  everyone can use fine-grained read-only tokens. It changes the repo URL, the notebooks' `REPO`,
  every remote and the branch protection. It can be revisited in its own decision.
- **Consequence:** docs/COLAB.md (one-time setup, authentication limitation) is updated. No code,
  data, dataset version or experiment changes.

## D-022 — Harsh's green-enhancement follow-up; E004 declared for test with limitations (2026-10-12, accepted)
- **Context:** D-018 held E004's test evaluation until Harsh's review list was checked against the
  v1 val and test filenames. Harsh answered in writing on 2026-10-12. The points below are his
  statements, summarized; we have not verified them unless marked OBSERVED.
- **Answers:**
  1. **Review list:** seven documented green-profile review images, six from Set 2000 and one from
     Set 100. None is from zExtra.
  2. **Other images viewed while the profile was chosen:** "not sure". There is no complete
     viewing or decision log. Later verification runs with fixed settings also processed the
     listed Set 100 image and one zExtra image.
  3. **Version that produced the 2,105 delivered images:** UNKNOWN. No execution record links a
     notebook version, date or environment to the delivery.
  4. **Five-image pixel check:** the current notebook was re-run on five raw images (Set 100 and
     Set 2000) with unchanged `fusion_mild` settings. The outputs were not compared with the
     delivery.
- **Check (OBSERVED, `data/processed/image_manifest.csv`, 2026-10-12):** of the seven images,
  five are in v1 train. `21_SUB6_10_BW_20220815_GP080026_13777` is in **val** (7 boxes) and
  `06_HERR_08_CW_20200903_GP120003_00001` is in **test** (1 box; 348 test images). All five
  pixel-check filenames are present in `data/raw/enhanced_2026-09/`.
- **Decision (user-approved, 2026-10-12; option chosen: "declare E004 for test, with limitations"):**
  - E004 is declared a final candidate by a committed row in `experiments/final_candidates.csv`.
    Its test evaluation may run once and is logged in docs/EXPERIMENTS.md. E004's test numbers are
    reported whatever they are and are never used to choose between E001 and E004.
  - This replaces D-018's test hold. D-018 is not edited.
- **Limitations every E004 result carries, in addition to D-018's:**
  - one val image and one test image (above) were among the documented review images;
  - whether other val or test images were viewed while the profile was chosen is UNKNOWN;
  - the delivery's provenance stays INFERRED, not OBSERVED.
- **Open:** Harsh's handoff ZIP (requested 2026-10-12). When it arrives, the five delivered images
  are compared with his outputs on decoded pixel arrays, and the result is recorded in a new
  decision. A match would be observed equivalence for those five pairs only.

## D-023 — Five-image pixel check of the enhanced delivery: near-identical, not bit-identical (2026-10-12, accepted)
- **Context:** D-022 left open a comparison of Harsh's five-image re-run with the delivered
  enhanced images. Harsh's handoff ZIP arrived on 2026-10-12: `harsh_green_followup_2026-10-12.zip`,
  15 files, SHA-256 `f7bd3f0c79be9fa5b0f5516a45527097cead87a037887e9dcce8ee635af7f5f6` (computed
  on receipt; Harsh's own hash was not received, so arrival intact is not confirmed). The ZIP is
  kept off-repo on the shared Drive. It was extracted outside the repository and read only.
- **Checks (OBSERVED, 2026-10-12):**
  - Harsh's seven raw inputs match our `data/raw/biology` files by SHA-256 (7/7). His relative
    paths differ: our Set 2000 images sit in a nested `batch_2 (2k - imgOnly)/` folder.
  - His five outputs match the file and decoded-pixel hashes in his `pixel_check_manifest.json` (5/5).
  - Against the delivered images in `data/raw/enhanced_2026-09/` (decoded RGB arrays, same shapes):
    not bit-identical. Per image 96–210 pixels differ (0.005–0.010%), maximum absolute difference
    2–3 of 255, mean absolute difference about 0.00005.
  - `scripts/check_raw_unchanged.py` (default structural check) afterwards: RAW DATA MATCHES
    FINGERPRINT.
- **Interpretation:**
  - INFERRED: the delivery was produced with the `fusion_mild` method and settings described in
    D-018. A different formula or setting would change most pixels, not about 0.01% by 1–3 levels.
  - HYPOTHESIS, not tested: the differences come from the runtime (the re-run used OpenCV 5.0.0;
    Harsh's earlier recorded review used 4.12.0).
  - LIMITATION: near-equivalence for five images only. It does not identify the notebook version
    or date that produced all 2,105 images, and it does not bear on which images were viewed.
- **Decision (user-approved, 2026-10-12):** recorded as above. D-022's limitations and E004's
  declaration are unchanged; no image, dataset version or experiment changes. The comparison
  script was a one-off outside the repository and is not committed.

## D-024 — Enhanced zExtra (Harsh, 2026-10) as a separate read-only image source, for qualitative inspection only (2026-10-12, accepted)
- **Context:** D-017 said enhanced zExtra images, when delivered, form a separate raw root for
  qualitative inspection only. Harsh shared `zExtra_Photos_green_fusion_mild.zip` via Google Drive
  on 2026-10-12: 1,601,390,788 B, SHA-256
  `94ef7084332d7c8195dc0c9fff35e8ea989aa7c7784de5f1c673fb00ff87df8b` (computed on receipt; Harsh's
  own hash was not received).
- **Inspection (OBSERVED, 2026-10-12, read from the ZIP before extraction):**
  - 436 entries, all PNG, in one folder `zExtra Photos/`; no unsafe paths, no macOS files;
  - the filenames are exactly the 436 raw zExtra names: none missing, none extra, no duplicates;
  - every image has the same width × height as its raw original, and 0 of 436 are byte-identical
    to it; all RGB.
- **Decision (user-approved, 2026-10-12):**
  - **Raw source:** `data/raw/enhanced_zextra_2026-10/images/zExtra/<name>.png`, extracted byte for
    byte (ZIP CRC verified), registered as image source `enhanced_zextra_2026-10` (env
    `UWCT_ENHANCED_ZEXTRA_ROOT`). Full content fingerprint:
    `data/processed/raw_fingerprint_enhanced_zextra_2026-10.json` (436 files, verified MATCH).
    Read-only like every raw root (CLAUDE.md rule 1). The ZIP's layout (`zExtra Photos/<name>.png`)
    differs from the first delivery, so the extraction was a one-off outside the repository rather
    than `scripts/unpack_image_source.py`.
  - **Use:** qualitative inspection only, e.g. the GUI's raw/enhanced switch showing E004 on
    enhanced zExtra. zExtra stays unlabelled and excluded from every dataset version (D-003,
    D-014). No metric is computed on it; no training, tuning or threshold choice uses it.
- **LIMITATIONS:** the profile (`fusion_mild`, same settings as D-018) is Harsh's statement, from the
  filename and D-018 answer 5: INFERRED, not verified. No record of the notebook version, runtime or
  date (the same provenance gap as D-022). D-022 noted that one zExtra image was processed in Harsh's
  later verification runs; that does not affect qualitative use.
- **Consequence:** no dataset version, split, label, frozen file or experiment changes.

## D-025 — E003 diverged to NaN losses; RT-DETR rerun as E005 with AMP off (2026-10-13, accepted)
- **Context (OBSERVED, Colab session 2026-10-13, Tesla T4, ultralytics 8.4.155, torch 2.11.0+cu130;
  notebook outputs saved in commit `f186f4e`):** the E003 smoke test (`E003_smoke_20261013T154708`)
  finished with finite losses (train giou 1.313, cls 7.450, l1 0.530; val mAP50 0.004 after 1 epoch
  on 10% of train). The E003 full run trained with `amp=True` (Ultralytics default). When it was
  resumed, every training loss was `nan` from the first batch of epoch 46, i.e. the checkpoint saved
  after epoch 45 already produced non-finite losses. UNKNOWN — requires confirmation: the first
  epoch with non-finite losses (in `RUNS_ROOT/E003/train/results.csv` on Drive, not read here).
  Ultralytics notes that AMP can produce NaN losses for RT-DETR; D-015 anticipated this and required
  a decision before changing AMP.
- **Decision (user-approved):**
  - **E003 is abandoned as diverged.** Not registered, not a final-test candidate, never compared
    with E001. Its Drive folder `RUNS_ROOT/E003/` is kept as evidence and not resumed.
  - **E005** (`configs/experiments/E005_rtdetr_amp_off.yaml`) reruns E003 from the pretrained
    `rtdetr-l.pt` with `amp: false` (fp32 training); everything else is identical to E003.
  - **Parent is E001, not E003:** `train.py` requires a registered parent and E003 will not be
    registered. Relative to E001, E005 therefore changes two variables: the model (as in E003) and
    AMP. HYPOTHESIS: AMP caused the divergence; E005 tests it. A YOLO vs RT-DETR comparison with
    E005 must state that E001 trained with AMP on and E005 with AMP off.
  - Fresh run under a new ID (no resume from any E003 checkpoint). The smoke test and the
    non-finite-loss check (notebook cell 12b) run first, and notebook cell 15 now reports
    non-finite losses in the full run.
- **Risks (UNKNOWN until run):** fp32 needs more GPU memory than AMP and is slower; E003 used
  ~12.7 GB at batch 16 with AMP on a 15 GB T4, so batch 16 may not fit. If it runs out of memory, or
  losses are still non-finite, stop and record a new decision before changing batch or lr.
- **Consequence:** no dataset, split, frozen or raw file changes. `my_model/colab_train_rtdetr.ipynb`
  now runs E005 (switches reset to `False`, the E003 outputs cleared; they remain in Git history).
  E003 stays in docs/EXPERIMENTS.md with status "diverged".

## D-026 — E005 trained at batch 8 after an automatic out-of-memory reduction (2026-10-13, accepted)
- **Context (OBSERVED, Colab 2026-10-13, Tesla T4 15 GB, ultralytics 8.4.155):** the E005 smoke test
  (`E005_smoke_20261013T163151`, `amp=False`, 10% of train) fitted at batch 16. The E005 full run
  then printed `CUDA out of memory with batch=16. Reducing to batch=8 and retrying (1/3)` before
  epoch 1, and trained at batch 8 (180 iterations per epoch instead of 90; ~10.2 GB GPU memory).
  Ultralytics made this change itself; the config still says `batch: 16`. Losses stayed finite
  through epoch 21; best val mAP50 so far 0.832 / mAP50-95 0.585 (epoch 15). The session stopped
  during epoch 22. UNKNOWN — requires confirmation: the cause of the stop (output ended without a
  visible error; a Colab disconnect is likely).
- **Decision (user-approved):** keep the E005 run and resume it from `train/weights/last.pt` rather
  than restarting. E005's effective batch size is **8**. D-025 required a decision before batch
  changed; this entry records it after the fact.
- **Consequences:**
  - Compared with E001, E005 now differs in three ways: model (rtdetr-l.pt), AMP (off) and batch
    (8 instead of 16). Every E001 vs E005 comparison states all three; the YOLO vs RT-DETR
    difference cannot be attributed to the model alone.
  - E005's `run_record.json` and `experiments/registry.csv` row show the configured batch 16
    (`train_args`, and Ultralytics' `args.yaml` written before the reduction). The effective batch
    is 8 per this entry; the registry row gets a note when registered.
  - The validation of `best.pt` after training uses the configured batch 16 (inference only, lower
    memory); this does not change the val metrics' meaning.
  - If the resumed run runs out of memory again or shows non-finite losses: stop and record a new
    decision. A clean rerun would be a new experiment with `batch: 8` in its config.

## D-027 — E005's final test uses the epoch-41 best.pt recovered from Drive (2026-10-18, accepted)
- **Context (OBSERVED):** the E005 log `runs/E005_logs/20261014T213006Z.log` shows epoch 100/100,
  `Optimizer stripped ... best.pt, 66.2MB` and the run record (best.pt sha256 `ae5f2356…`, 66,225,149
  B). Drive never received these final files: `runs/E005` (and the user's download
  `uwct-20261018T225917Z-1-001`) holds `results.csv` with 98 epochs (11,843 B), unstripped
  `best.pt` 263,628,970 B sha256 `b2fc006a…`, `last.pt` sha256 `5a52fc26…`, no `run_record.json`.
  No 66,225,149 B file was found in Drive (incl. Trash and old versions, cell R1d).
- **Reasoning:** by the results.csv fitness (0.1·mAP50 + 0.9·mAP50-95), epoch 41 is the best of
  epochs 1–98 (P 0.772, R 0.811, mAP50 0.8358, mAP50-95 0.5889); the log shows epochs 99–100 at val
  mAP50 0.797 / 0.799, so they did not replace it. Ultralytics writes best.pt only at a new best.
  INFERRED: the recovered best.pt holds the same epoch-41 EMA weights as the registered file, which
  differs only by `strip_optimizer` (optimizer removed, fp16, new date) — hence a different sha256.
- **Decision (user-approved 2026-10-18):** E005's final test evaluates the recovered best.pt,
  named in `experiments/records/E005/weights_recovery.json`. It is accepted only if (1) its sha256
  is `b2fc006a…`, (2) its checkpoint epoch is 40 (0-based, = results.csv epoch 41; notebook cell R3),
  and (3) before the test split is touched, its val metrics reproduce the registered ones within
  0.002 absolute on P, R, mAP50 and mAP50-95 (`src/training/final_test.py`; the tolerance allows
  for fp32 vs fp16 weights). Otherwise the test is refused and E005 needs a retrain decision.
- **Consequences:** the registered run record and registry row are unchanged. The final test record
  stores both the registered and the evaluated sha256 and the val differences. E005's `last.pt` and
  100-epoch `results.csv` are lost (only 98 epochs survive); this is a LIMITATION of every E005
  report. Training code is unchanged.

## D-029 — Tritonia (sea slug) timelapse data as a new source; class 1 and label approach (2026-11-03, proposed)
- **Status:** proposed by Anu on `anu/sea-slug-detection` (draft PR #16). The choices marked
  "user-approved" below were made by Anu on 2026-11-03 (Saksham did not know the data). The entry is
  accepted when the PR is reviewed and merged. D-028 is intentionally unused: it was a tracking
  proposal (PR #15) that Anu withdrew on 2026-11-03 before merge; the PR is closed and its branches
  deleted.
- **Context (OBSERVED, 2026-11-03; details in docs/DATASET_TRITONIA.md):** Dr. Wyeth's Tritonia data
  is on an external drive (`E:\Tritonia\`): 17 grayscale TIFF stacks (one per day, 2013-05-18 to
  2013-06-03, 37,170 frames of 1056 × 480 px, 15 s per frame, each frame a 6-camera montage) and 16
  MTrackJ `.mdf` files with 956 manually clicked tracks (117,969 points, no boxes, no class field).
  `20130518` has no MDF. The lobster pipeline accepts only class `0` (`CLASS_NAMES` in
  `src/yolo_prep/build.py`; `image_manifest.py` flags any other class ID).
- **Proposed decision:**
  - **Class (user-approved 2026-11-03):** `1: sea_slug` (appended; `0: lobster` unchanged, CLAUDE.md
    §4). A general name: other sea slug species may join this class later, each with its own decision.
  - **Raw source:** a new read-only source `tritonia` holding the 17 TIFF stacks, the 16 MDF files and
    `Tritonia Info.txt` byte for byte, with a content fingerprint (all files hashed) before use.
    **Location (user-approved 2026-11-03):** `D:\uwct_raw\tritonia\` on the PC's internal D: drive
    (917 GB free; C: had 17 GB, too little for a ~18.9 GB copy), copied from the external drive with
    SHA-256 checked on both sides, set via `UWCT_TRITONIA_ROOT`. Read-only (rule 1); nothing writes
    there. Colab never receives the stacks, only extracted frames and labels.
  - **Missing-label policy:** `pending`. A frame without points is excluded, never treated as negative,
    until Dr. Wyeth confirms every visible slug was clicked (rule 6).
  - **Split unit:** the recording day (one stack = one session), so all frames of a day stay in one
    split (rule 5). The split itself is a later step with its own seed and decision.
  - **Labels for detection (user-approved 2026-11-03):** boxes drawn by hand. Batch 1 = 150 frames
    (seed 0), spread evenly over the 16 labelled days, only frames with at least one MDF point, at
    least 40 frames (10 min) apart within a day (`scripts/make_tritonia_label_batch.py`), labelled by
    Anu under the rules in docs/TRITONIA_LABELLING.md. The slug of every MDF cross in a labelled frame
    is boxed, and nothing without a cross (amended 2026-11-03 before any frame was kept: the first
    trial frame showed crisp bright specks without crosses, while tracked slugs are faint 10-25 px
    smudges; the Info file says the whole set was tracked by hand, so untracked specks are INFERRED
    not to be slugs). Within batch 1 a frame's missing box therefore means "no tracked slug"
    (policy `negative` for the labelled frames only; unlabelled frames stay `pending`). MDF points
    locate the slugs; their boxes are still drawn by hand around the visible smudge.
    Rejected: a point-based detector (new model code) and postponing detection. Not allowed: boxes of
    an assumed size placed around points (rule 8).
  - **Model:** YOLO26s (`yolo26s.pt`, E001's settings) trained on a separate sea-slug-only dataset
    version; one config per run (§5). A combined lobster + sea slug model is a later decision: the
    lobster images were never checked for sea slugs.
  - **Reporting:** frames are 15 s apart, so all results are "sparse-frame/timelapse analysis", never
    "video tracking"; px and frames only (seconds via the stated 15 s); no physical units.
    The MDF tracks carry hand-assigned track IDs, so IDF1/HOTA/MOTA become possible on this data
    once a detector and tracker produce tracks (rule 9), per camera, since slugs are tracked per camera.
- **LIMITATIONS:** the 6-camera layout (INFERRED 3 × 2 tiles of 352 × 240 px) is not recorded in the
  files read; the MDF track flag (`true`/`false`) and the one `Reference` line are not yet understood;
  2 points sit on or just past the frame edge; site and setting are UNKNOWN; the 17 days are
  INFERRED to be one setup, so a day split measures new-day, not new-site, generalization.
- **Consequences:** an MTrackJ reader, a TIFF-stack frame extractor and the batch builder in `src/`
  and `scripts/` with tests (no pixel library needed: the stacks are uncompressed). After labelling:
  the box ZIP becomes a new read-only label source, a new draft dataset version with its own
  decision and split, and a Colab notebook plus experiment config for YOLO26s. No change to v1, v2,
  their splits, labels or experiments. A local 50-frame inspection sample exists at
  `C:\sample_data\Tritonia\` (outside the repository, not a dataset).

## D-030 — Tritonia dataset v3, day split and the E006 YOLO26s run (2026-11-03, proposed)
- **Status:** proposed by Anu on `anu/sea-slug-detection` (PR #16). The split counts and building the
  whole chain before labelling are user-approved (Anu, 2026-11-03). v3 does not exist yet: it is
  created after labelling batch 1 is ingested, and freezing it needs Anu's explicit approval.
- **Context:** D-029 chose hand-drawn boxes on a 150-frame batch (16 labelled days, 9–10 frames each)
  and YOLO26s on a sea-slug-only dataset. The lobster pipeline (version registry, YOLO view, training
  checks) assumed one dataset with one class; this entry extends it without changing any lobster
  version, view, config or run.
- **Decision:**
  - **Label source:** `scripts/ingest_tritonia_boxes.py` checks the makesense.ai YOLO export against
    the batch manifest and writes `D:\uwct_raw\tritonia_boxes_batch1\` (labels byte for byte, the ZIP,
    `notes.txt`, `ingest_report.json` with every label's sha256). Read-only (rule 1;
    `UWCT_TRITONIA_BOXES_ROOT`). Refused on any problem: a frame with no label file and no
    `exclude` note, a class other than 0/1, a box outside the frame or across a camera border
    (3 px tolerance).
  - **Dataset v3:** a new lineage with no parent (`dataset_kind: tritonia`, classes `0: lobster`,
    `1: sea_slug`), made by `scripts/make_tritonia_version.py`. Frames marked `exclude` are
    `excluded`; a label file with boxes is `positive`; an empty one is `negative_confirmed` (D-029:
    every visible slug in a labelled frame is boxed). Images and labels are named by sha256.
  - **Split (user-approved counts):** unit = recording day (rule 5); the 16 days, sorted and shuffled
    with seed 0, give 3 val and 3 test days, the other 10 train. Computed now from the day names
    alone, before any box exists (checked by `tests/test_tritonia_dataset.py`):
    - val: 2013-05-24, 2013-05-29, 2013-06-02
    - test: 2013-05-20, 2013-05-21, 2013-05-28
    - train: 2013-05-19, 05-22, 05-23, 05-25, 05-26, 05-27, 05-30, 05-31, 06-01, 06-03
  - **YOLO view:** `scripts/prepare_tritonia_yolo.py` (Colab) builds `data/yolo/tritonia_v3/` from the
    frozen v3, copying each frame and label only if its sha256 matches; same layout and fingerprint
    rule as the lobster views. Training (`train.py`) verifies it through `tritonia_dataset.verify_view`
    and reads the class names from the dataset; lobster views are checked exactly as before.
  - **E006** (`configs/experiments/E006_yolo_seaslug.yaml`, committed once v3 is built, with the
    fingerprint the notebook prints):
    ```
    id: E006
    slug: yolo_seaslug
    parent: E001
    changed: "dataset v1 (lobster) -> v3 (Tritonia sea slugs, classes lobster + sea_slug; D-030)"
    dataset_version: v3
    dataset_fingerprint: "<printed by notebooks/colab_train_tritonia.ipynb cell 9>"
    model: yolo26s.pt
    ultralytics_version: "8.4.155"
    seed: 0
    train_args:
      imgsz: 640
      epochs: 100
      patience: 30
      batch: 16
      deterministic: true
    ```
    Everything except the dataset is E001's. Selection metric: val mAP50 of class `sea_slug`.
  - **Notebook:** `notebooks/colab_train_tritonia.ipynb` copies the label source and the 150 clean
    frames from Drive, builds and verifies the view, then preflight, smoke test and full run like the
    lobster notebook. No val-eval or final-test cell yet: those scripts are lobster-specific, and the
    3 test days stay untouched until a sea slug final-test procedure is declared (rule 7).
- **LIMITATIONS:** 150 frames from one setup; 3 val and 3 test days, so val/test numbers are noisy
  and measure new days, not new sites. One labeller. Run records of v3 carry these Tritonia
  limitations instead of v1's (`run_record.TRITONIA_LIMITATIONS`).
- **Consequences:** new `src/dataset_audit/tritonia_dataset.py`, three scripts, tests;
  `versioning.open_new_lineage` (a draft with no parent; `freeze` skips the parent check only for a
  non-lobster `dataset_kind`); `experiment.default_dataset_dir` / `dataset_kind`; class names and
  limitations taken from the dataset in `train.py`, `run_record.py`, `verify_yolo_dataset.py`.
  No change to v1, v2, E001–E005 or their records.

## D-031 — Model A: Harsh's mixed training data on the UWCT v1 split, with a mandatory leakage check (2026-11-03, accepted 2026-11-04)
- **Numbering note (2026-11-04):** drafted as D-029 / dataset v3 / E006 onwards; renumbered to
  D-031 / v4 / E007 onwards because PR #16 (merged first) uses D-029, D-030, v3 and E006 for the
  Tritonia sea slug work. Content otherwise unchanged.
- **Status: accepted as a whole by the user on 2026-11-04 ("accpeted").** Points 1, 5 and 6 were
  user-approved 2026-11-03 (the split and evaluation plan: "yes"; label handling: no hand check);
  the draft manifest of v4 was built for the leakage check (user: "yes go ahead with the next
  phase", 2026-11-03). Acceptance allows building v4's YOLO view and training E007 onwards on Kaggle.
  It does not freeze v4: freezing needs its own explicit user approval (CLAUDE.md rule 4).
- **Context (stated by Harsh in `data/raw/harsh_readme_2026-11/README.md` and his answers of
  2026-11-03; checked against the final delivery below):** a Kaggle delivery (~9.5 GB, 11,356 files listed in
  `SHA256SUMS.txt`) for a final lobster model ("Model A"): Set 100 + Set 2000 originals with
  AI-reviewed labels, 400 colour-change images (200 ours, 200 Zenodo), 986 copy-paste images
  (587 ours, 399 Zenodo), 932 Zenodo originals (record 15755673, western Sweden, CC BY 4.0, credit
  required: Fiorina, Obst, Sundelöf, Ljungberg and Sande 2025) and 1,371 DM_underwater diffusion
  copies. `derived_images_index.csv` (2,757 derived images) and `pasted_lobsters_index.csv`
  (1,903 pasted lobsters) name the source image and session of every background and pasted lobster.
- **Split conflict (OBSERVED, comparison of the README's session lists with
  `data/processed/split_manifest.csv`):** Harsh's own split ("canonical-v2", seed 42) puts 21 of
  the 45 labelled sessions in a different split from UWCT v1. 9 sessions that are train for him
  are val or test in v1, including 5 of the 6 v1 test sessions (06_HERR_04, 06_HERR_11, 21_MACK_09,
  21_SUB6_03 in his train; 06_HERR_08 in his val). Harsh counts 757 derived images that use one of
  those 9 sessions (65 colour change, 229 copy-paste, 463 diffusion). 10 v1 train sessions are
  val/test for him and have no colour-change or copy-paste images yet.
- **Label review (stated by Harsh):** 331 frames reviewed by an AI model (Claude), with a 5-epoch
  YOLO model's detections drawn on as hints; no person checked any frame. 123 frames excluded
  (86 "empty" Set 2000 frames showing lobsters, 33 unclear, 4 wrong boxes); of 122 frames kept as
  "confirmed empty", 15 are high, 76 medium and 31 low confidence.
- **ZIP check (OBSERVED, 2026-11-03, read without extracting):** `DeepSight Model A data.zip.zip`
  (9,549,815,408 B, sha256 `b70823f8a722509eb0c6234e83d3a3efdf63a69dea743a941b564245ac75f547`),
  11,352 files, no unsafe paths, symlinks or macOS files. It is an earlier package than Harsh's
  answers describe: its README is 2,263 B, and `SHA256SUMS.txt`, `derived_images_index.csv`,
  `pasted_lobsters_index.csv`, `originals_index.csv` and `data_original_only.yaml` are missing.
  - Ancestors from `sources - ours.csv` (background and every pasted lobster) and diffusion file
    names, mapped through the v1 `image_manifest.csv`: 65 of 200 colour-change, 229 of 587
    copy-paste and 463 of 1,371 diffusion images use a v1 val/test session, which reproduces
    Harsh's 757. Usable for v1 train: 135, 358 and 908.
  - All 1,982 original label files equal the raw labels up to 6-decimal rounding (largest
    coordinate change ≤ 1e-6, all class 0): no box was edited. The cleaning is the exclusion of
    123 frames: 119 negatives (87 train, 20 val, 12 test) and 4 positives (3 train, 1 val).
  - Zenodo: two sites, `mpa` (2022; 478 images, 226 with a lobster) and `sk` (2024; 428 images,
    143 with a lobster), in 17 site-camera-date groups; 26 images (`REC_00NN` and two odd names)
    have no site in their name.
- **Lab data (OBSERVED, `split_manifest.csv`):** 45 labelled sessions in 29 site-date groups and
  11 site codes (2020–2022); 6 site codes have a single session. ASSUMPTION (the data provider
  cannot be asked): the site is the second name field, so `06_HERR`, `16_HERR` and `19_HERR` are one
  site. If they are different places, this is only stricter than needed.
- **Final delivery (OBSERVED, 2026-11-03):** three ZIPs, each matching the sha256 Harsh stated
  (`DeepSight Model A data.zip` `92d4d51c…`, `DeepSight UWCT v1 extra - 10 sessions.zip`
  `828a330c…`, `DeepSight UWCT v1 colour change.zip` `1fa5cf0f…`), unpacked to
  `data/raw/harsh_model_a_2026-11/` (11,358 files), `data/raw/harsh_v1_extra_2026-11/` (1,374) and
  `data/raw/harsh_v1_colour_2026-11/` (676). Every file matches its `SHA256SUMS.txt` on disk; full
  fingerprints: `data/processed/raw_fingerprint_harsh_*_2026-11.json`. The ZIPs were deleted after
  the check (user-approved; Harsh's Kaggle copy remains).
  - **Colour change redone (Harsh's option (a)):** zip 1's colour images took water colours from
    canonical-v2 train frames, which include v1 val/test sessions. `harsh_v1_colour_2026-11`
    replaces that folder: the same 135 + 200 base frames, colours from the 1,343 v1 train frames
    (stated by Harsh). Zip 2's 63 colour images already use v1 train colours (its README).
  - Zenodo, from file names: `mpa` 480 images (237 boxes, 10 groups), `sk` 428 (143 boxes,
    8 groups), 24 `REC_` frames with no boxes, as Harsh stated.
  - The 123 label-review exclusions are exactly the v1 frames missing from Harsh's originals
    (v1 has 2,105 labelled frames, Harsh 1,982).
- **v4 draft (OBSERVED, `scripts/build_model_a_manifest.py`):** `data/processed/v4/` = a copy of
  v1 plus `mixed_manifest.csv` (6,413 rows), `zenodo_groups.csv` and `model_a_report.json`. Our
  originals and their labels come from the raw biology files and v1; Harsh's original copies are
  not used. Label checks passed for every derived image: colour change and diffusion boxes equal
  the original's, copy-paste boxes are the background's plus one per pasted lobster (tolerance 1e-6).

  | Split | Ours: originals | colour | copy-paste | diffusion | Zenodo: originals | colour | copy-paste |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | train | 1,312 | 192 | 516 | 1,312 | 480 (`mpa`) | 93 | 206 |
  | val | 324 | – | – | – | – | – | – |
  | test | 348 (v1) | – | – | – | 428 (`sk`) | – | – |
  | excluded | 121 | 6 | 257 | 494 | 24 | 107 | 193 |

  Excluded: 90 train frames Harsh's review excluded and 31 low-confidence "empty" train frames;
  992 derived images with a val/test or no-site ancestor; 65 derived images built on one of the
  31 low-confidence frames. `check_leakage.py`, rules 1–5: PASSED (Harsh's flag agrees on every row).
- **Proposed decision:**
  1. **Split:** UWCT v1 sessions keep their v1 splits (CLAUDE.md §10: a split shipped with the data
     is never used). Harsh's split, `data.yaml` and `data_original_only.yaml` are not used.
  2. **Raw sources:** the delivery is unpacked into new read-only folders under `data/raw/`
     (Zenodo JPGs built with `--link-mode copy`, rule 2), checked against `SHA256SUMS.txt` and the
     ZIP hash, and fingerprinted. They are listed in `MODEL_A_ROOTS` (`src/dataset_audit/config.py`),
     not `DATASET_SOURCES`, which drives the v1 audit and the default raw check. Label policy: every
     image has a label file of the same stem; an empty file is a negative, a missing one an error.
  3. **Dataset version v4** (draft, parent v1) with one mixed manifest: one row per image with its
     origin, kind, group, split and ancestors (schema: `src/dataset_audit/leakage.py`).
  4. **Leakage rules**, enforced by `scripts/check_leakage.py`, which must print
     `LEAKAGE CHECK: PASSED` (with `--image-root`, on the training machine) before any Model A run,
     next to the raw check and `verify_yolo_dataset.py`:
     - a derived image is in train only if every ancestor (background and every pasted lobster's
       source) is in a train group; an original excluded by the label review still counts as held
       out; all other derived images are left out;
     - val and test hold real, unmodified originals only;
     - Harsh's `uses_uwct_v1_val_or_test_session` flag must agree with the computed answer;
     - no identical file (sha256) and no near-duplicate (dHash distance ≤ 4) across train, val and
       test. A flagged pair is reviewed by a person and the outcome recorded before training.
  5. **Labels:** boxes come from the raw label files (unchanged, see the ZIP check). Harsh's 123
     exclusions apply to train only. No frame is checked by hand (user choice), so val and test keep
     every v1 frame and label: they stay identical to the frames E001–E005 were scored on. In train,
     the "confirmed empty" frames marked low confidence in `label_review_verdicts.csv` are also left
     out, never used as negatives.
  6. **Evaluation:** three results, each reported with its own meaning:
     - **v1 test (locked):** new sessions at known sites. Only pre-declared final candidates are
       evaluated there, once, and logged (rule 7). Its frames and labels are unchanged from v1
       (point 5), so E001, E004 and E005 results stay directly comparable.
     - **Leave-site-out cross-validation** over the train + val sessions (about 4 folds, whole
       sites per fold, derived images filtered per fold), for the final configuration: new sites.
       The test sessions are never part of it.
     - **Zenodo `sk` (held out):** an external new site (other country, camera setup and image
       size). Training uses `mpa` only; derived Zenodo images with an `sk` ancestor and the 24
       `REC_` images without a site (no lobsters, per Harsh) are left out.
     A new site-held-out test split is not made: with 11 sites, mostly one session each, the score
     would depend on which sites were picked, and E001–E005 could not be compared on it.
  7. **Runs** go on Kaggle GPU notebooks (the lab GPU is dropped, see the platform note), one at a time, in stages: (1) a data ladder (originals only, then
     one new source at a time, then all) with E001's settings; (2) model choice on the best data
     mix; (3) hyperparameter tuning in a pre-declared search space and budget; (4) the final
     configuration with 3 seeds; (5) leave-site-out cross-validation (point 6). Each stage gets its
     own experiment configs (E007 onwards) and, where it changes more than one variable (tuning), a
     decision.
  8. **Delivered model:** the evaluated model (trained on train + val) is the one the report
     describes. Retraining on all lab data including test, for a product, would break rule 7 and
     needs explicit user approval and its own decision; such a model would have no held-out score.
- **Points settled before acceptance (two items still measured in the first Kaggle session):**
  - rule 6 (identical files and near-duplicates across splits): run 2026-11-03 on 5,211 images: no identical files across splits; 2 near-duplicate flags at
    the threshold (dHash distance 4), both the same Zenodo `mpa` frame
    (`mpa_cam2_120822_rec_001_frame_0021`, train) and its copy-paste image against one `sk` test frame
    (`sk_cam3_120824_rec087_frame_0038`). Checked 2026-11-04 by Claude at the user's request
    (visual comparison of the two originals): **different scenes**. OBSERVED: the burnt-in timestamps
    are 2022/08/11 10:19:23 vs 2024/08/11 13:19:04, and the trap shape and angle, seabed, rope and
    lobster position all differ; both are top-down views of a trap on green water, which explains the
    hash similarity. All three images are kept; rule 6 counts as passed;
  - **decided 2026-11-04, user-approved ("i aggree for both"):** the 29 train frames the review
    kept as `positive_incomplete` (a lobster without a box; 108 boxes, no derived image) **stay in
    train**, as in E001–E005, so the first Model A run changes only the data. LIMITATION: up to
    29 unboxed lobsters are trained as background;
  - **decided 2026-11-04, user-approved:** the selection metric stays **val mAP50**
    (docs/EXPERIMENTS.md); mAP50-95, P, R and false positives on negative images are still reported;
  - **platform, decided 2026-11-04 by the user** ("lets drop the plan of using the lab's gpu
    altogether and use kaggle's gpu"): runs go on **Kaggle GPU notebooks**, not the lab PC (an RTX
    A2000 12 GB, reported the same day, no longer used). Harsh's three deliveries are already Kaggle
    datasets and are attached read-only under `/kaggle/input` (rule 1 holds by construction); the
    lab-PC copy and its download/fingerprint steps are dropped. The Kaggle copy is still checked
    against the committed fingerprints before use. ASSUMPTION — requires confirmation in the first
    Kaggle session: GPU type, VRAM, session time limit and weekly GPU quota (these set batch size,
    whether a run needs a resume across sessions, and how many tuning trials and folds fit). OPEN:
    how `biology` (our originals, 5.2 GB) reaches Kaggle (a private Kaggle dataset uploaded by the
    user, or Harsh's `1 Original - our lobsters` copy if it proves byte-identical). Time per run:
    UNKNOWN — measured by the smoke test.
- **LIMITATIONS (expected):** train labels rest on an unverified AI review; the README states
  that other frames may also be missing boxes; test sessions share sites with train (§3); we have
  seen E001/E004 test results, so choices are made from validation numbers only; Zenodo images are
  640 × 360 and show pots and timestamps, unlike ours; the site definition is an ASSUMPTION; the
  v1 test is concentrated (2 sessions hold 1,135 of 1,512 boxes); zExtra stays unlabelled and
  excluded.
- **Consequence so far:** `src/dataset_audit/leakage.py`, `scripts/check_leakage.py`,
  `src/dataset_audit/model_a.py`, `scripts/build_model_a_manifest.py` and their tests added;
  `MODEL_A_ROOTS` in `config.py` (also write-protected through `RAW_ROOTS`); three raw folders and
  their fingerprints; the v4 draft. No v1/v2, split, label or biology raw file changes.

## D-032 — Model A, first run E007: all safe v4 training data at once (no data ladder); v4 frozen; YOLO view of v4 (2026-11-07)
- **Status: accepted by the user on 2026-11-07 ("i approve").** The two choices below are the
  user's (quoted).
- **Context:** D-031 point 7 planned a data ladder as stage (1): originals only, then one new source
  at a time, then all, with E001's settings. In the first Kaggle session (2026-11-07; Kaggle clock 2026-11-08 UTC) the four
  archives (`biology`, `harsh_model_a_2026-11`, `harsh_v1_extra_2026-11`, `harsh_v1_colour_2026-11`)
  matched their recorded sha256 and the raw checks against the committed fingerprints passed
  (`biology` structural, Harsh's three full). OBSERVED, Kaggle environment: 2 × Tesla T4 (14.6 GB
  each), Python 3.13, torch 2.11.0+cu130, Ultralytics 8.4.155, 20 GB `/kaggle/working`.
- **User choices (2026-11-07):**
  1. Data for the first run: "we'll use every thing we have and which is safe from the start and
     we'll be tuning it afterwards".
  2. Freezing v4 once the builder passes its tests: "Yes, freeze v4".
- **Decision:**
  1. **E007** (parent E001) trains on the whole v4 train split: 4,111 images (ours 1,312 originals,
     192 colour change, 516 copy-paste, 1,312 diffusion; Zenodo `mpa` 480 originals, 93 colour
     change, 206 copy-paste), validated on the 324 v1 val frames. Model and settings as E001; the
     one variable changed is the dataset (v1 → v4). Stage (1) of D-031 point 7 is dropped; stages
     (2)–(5) (model choice, tuning, final configuration with 3 seeds, leave-site-out CV) are unchanged
     and start from E007.
  2. **YOLO view of v4** (`src/yolo_prep/mixed.py`, `scripts/prepare_yolo_dataset.py --version v4`):
     built from `mixed_manifest.csv`; images always copied (rule 2), named
     `<origin>__<kind>__<file>` (a diffusion copy keeps its source's file name); splits `train`,
     `val`, `test` (the 348 v1 test frames) and `test_zenodo_sk` (the 428 held-out Zenodo `sk`
     frames, in its own data file `data_test_zenodo_sk.yaml`, so a v1 test score never mixes the
     two). Each image's sha256 is in `build_manifest.csv`, so the dataset fingerprint covers image
     bytes and labels. Training verifies the view through `verify_built_dataset` (dataset_kind
     `lobster_mixed`, added to v4's `VERSION.json` before freezing).
  3. **v4 frozen** with `scripts/freeze_dataset_version.py v4` after the builder's tests passed.
- **Trial build (OBSERVED, local, draft, 2026-11-07):** train 4,111 images (3,721 with lobsters,
  11,784 boxes), val 324 (269; 1,010), test 348 (321; 1,512), test_zenodo_sk 428 (143; 143); not
  built: 1,202 (992 ancestor not train, 90 label review excluded, 65 ancestor excluded, 31
  low-confidence empty, 24 Zenodo without site). Val and test equal v1's frames and boxes. Raw
  checks of all four raw folders passed after the build.
- **zExtra** (436 frames, 11 sessions in no split, no ground truth) stays out of v4 (rule 6). The
  final model may be run on it for a qualitative demo only: images with predicted boxes and counts of
  frames with detections, never precision, recall or mAP.
- **LIMITATIONS:** without the ladder, a change between E001 and E007 cannot be attributed to any one
  source (Zenodo, colour change, copy-paste, diffusion). E007 vs E001 compares v1 train with all
  safe extra data, on the same 324 val frames; E001 ran on Colab, E007 on Kaggle (both Tesla T4).

## D-033 — Tritonia batch 1: automatic boxes for training, accuracy against the crosses (2026-11-04, proposed)
- **Numbering note (2026-11-09):** drafted and committed as D-031 on `anu/sea-slug-detection`;
  renumbered to D-033 because `p4/colab-baseline` (PR #17, merged first) uses D-031 for Model A.
  Content otherwise unchanged. Files written before the renumbering still say D-031 for this
  decision and cannot be edited: `notes.txt` in the read-only label source
  `D:\uwct_tritonia\raw\tritonia_boxes_batch1\` (and on Drive), and the automatic-label output
  in `D:\uwct_tritonia\work\tritonia_auto_labels\`. In Tritonia files, "D-031" from before
  2026-11-09 means this entry.
- **Status:** proposed by Anu on `anu/sea-slug-detection`. The approach (the "crosses plan": boxes
  made by code, no hand labelling, scores measured against Dr. Wyeth's crosses) and this text were
  user-approved by Anu on 2026-11-04 (the `--extra` wording under "Label source" was added just
  after, to match the ingest run). Accepted when the PR is reviewed and merged. Supersedes D-029's "boxes drawn by hand" and D-030's selection metric (val mAP50 of
  `sea_slug`) for the Tritonia work; D-029's class, source and label rules and D-030's day split,
  YOLO view and E006 setup stay.
- **Context (OBSERVED, 2026-11-04):** hand labelling did not work out: 34 frames drawn by hand had
  loose boxes (median 41 px wide, slugs about 12–20 px); in a review session with suggested boxes,
  36 val/test frames were checked but the boxes drawn there were again loose (median 40 × 31 px).
  Anu decided not to label further. The MTrackJ crosses are human labels (clicked by Dr. Wyeth's
  group) but points, not boxes; turning them into boxes of an assumed size stays rejected (rule 8).
- **Proposed decision:**
  - **Training boxes from the image:** `scripts/auto_label_tritonia.py` boxes the bright blob next
    to each cross (`src/dataset_audit/tritonia_prelabel.py`; standard library, deterministic, method
    and parameters in its docstring and `PARAMS`). The box size comes from the pixels only. A cross
    gets no box, and a flag in `suggestions.csv`, when the blob is too faint, not compact, a
    duplicate, or looks like the on-screen timestamp. A frame where no cross got a box is marked
    `exclude` (an empty label would claim "no slug"). Nobody reviews these boxes.
  - **Label source:** the output (`labels.zip`, `notes.txt`, `suggestions.csv`,
    `auto_label_report.json`) is ingested unchanged by `scripts/ingest_tritonia_boxes.py` into
    `D:\uwct_raw\tritonia_boxes_batch1` (D-030; the last two via `--extra`, stored as `extra/<name>`
    with their sha256 in `ingest_report.json`), and v3 is built from it. All 150 frames use the
    automatic boxes, the 34 hand-labelled ones too, so the boxes follow one rule; the hand labels and
    review records stay unused in `D:\uwct_work\tritonia_batch1_labels`.
  - **Scores against the crosses:** `scripts/tritonia_point_eval.py`
    (`src/dataset_audit/tritonia_point_eval.py`). A prediction matches the nearest unmatched cross
    within 10 px of its box, one to one, highest confidence first. Reported: point precision, point
    recall, F1 and point AP (all-point interpolated), overall and per day, on every frame of the
    split (frames the automatic labels excluded too: their crosses are still ground truth).
    Matching rule and 10 px declared here, before any run.
  - **Threshold and selection:** the confidence threshold is the one with the best val F1 (grid
    0.01–0.99); the selection metric for comparing Tritonia models is **val point AP**. Test is scored
    only for a declared final candidate, with the val threshold, logged in docs/EXPERIMENTS.md; the
    script refuses test without `--final` and `--conf` (rule 7).
  - **Ultralytics' own val numbers** (P, R, mAP printed during training and used to keep `best.pt`)
    are computed against the automatic boxes of the val days. They are not accuracy and are never
    reported as such.
- **Counts (OBSERVED, 2026-11-04, `auto_label_report.json`):** 853 crosses, 436 boxes (51%), 18 of
  150 frames excluded. Train: 96 frames (10 excluded), 573 crosses, 292 boxes. Val: 27 (7), 153, 73.
  Test: 27 (1), 127, 71. The real ingest accepted the ZIP in a dry run into a scratch folder
  (132 label files, 436 boxes, 0 problems). Flags: no clear blob 365, same blob as another cross 25,
  timestamp text 16, blob not compact 11; 11 boxes clipped at a camera border.
- **Check of the method on the 34 hand-labelled frames (OBSERVED, zoomed crops):** boxes are tight
  on clear blobs (median 21 × 17 px); some fall on faint texture in washed-out camera views; some
  clear slugs get no box.
- **LIMITATIONS:**
  - About half of the crosses have no training box, so many real slugs are background in training;
    some boxes are on texture, not slugs. The model learns from noisy labels; results show what
    automatic labels give, not careful hand labels.
  - Point scores are not mAP: box size and fit are not measured, and they cannot be compared with
    the lobster mAP numbers (CLAUDE.md §5 asks for mAP50 and mAP50-95; not possible without boxes).
  - Point precision is a lower bound: whether every slug was clicked is UNKNOWN (D-029).
  - 3 val and 3 test days, 27 frames each: scores are noisy and measure new days, not new sites.
- **Consequences:** new `tritonia_prelabel.py`, `tritonia_point_eval.py`,
  `scripts/auto_label_tritonia.py`, `scripts/tritonia_point_eval.py`, tests; the labeller
  (`scripts/label_tritonia.py`) keeps the suggestion, resize, `s` and `--split` features from this
  work but is not needed for batch 1. Follow-up: a val point-evaluation cell in
  `notebooks/colab_train_tritonia.ipynb`. No change to v1, v2, E001–E005.

## D-034 — Tritonia data on the local PC moved into one folder (2026-11-09, proposed)
- **Numbering note (2026-11-09):** committed as D-032 on `anu/sea-slug-detection` (5377078);
  renumbered to D-034 because `p4/colab-baseline` (PR #18) uses D-032. Content otherwise unchanged.
- **Context:** the Tritonia files on the PC's D: drive were split over `D:\uwct_raw\` (raw copy
  `tritonia\`, D-029; label source `tritonia_boxes_batch1\`, D-030/D-033) and `D:\uwct_work\`
  (labelling batch, automatic labels, unused hand labels, provenance CSVs, label ZIPs). The user
  asked for one common folder. Moving raw roots is forbidden by CLAUDE.md rule 1 unless the user
  approves an exception; the user approved this move on 2026-11-09 (exception to rule 1, location
  only).
- **Decision:** new layout, same drive (a rename; no file was opened for writing):
  - `D:\uwct_raw\tritonia\` → `D:\uwct_tritonia\raw\tritonia\` (`UWCT_TRITONIA_ROOT`)
  - `D:\uwct_raw\tritonia_boxes_batch1\` → `D:\uwct_tritonia\raw\tritonia_boxes_batch1\`
    (`UWCT_TRITONIA_BOXES_ROOT`)
  - everything in `D:\uwct_work\` → `D:\uwct_tritonia\work\`
  Both `raw\` folders stay read-only (rule 1); `work\` stays writable.
- **Verification (OBSERVED, 2026-11-09):** relative path, size and SHA-256 of all 875 files
  (18,910,835,751 B) recorded before and after the move; the two records are identical.
- **Not changed:** file contents; frozen v3 (`data/processed/v3/` holds no absolute paths);
  `E:\Tritonia\` (not touched); Google Drive paths (`MyDrive/uwct/raw/…`, `MyDrive/uwct/work/…`).
  `work\tritonia_label_batch1\batch.json` still records `"root": "D:\uwct_raw\tritonia"`: a
  historical record of where the batch was made from, left as is.
- **Leftovers:** the then-empty folder `D:\uwct_raw\` was deleted on 2026-11-09 at the user's
  request, after the move was verified. `D:\uwct_work\` no longer exists (moved as a whole).
- **Consequences:** paths updated in `notebooks/colab_train_tritonia.ipynb` (plus a note on the
  layout), `docs/DATASET_TRITONIA.md`, `docs/TRITONIA_LABELLING.md` and the usage examples of the
  Tritonia scripts. Earlier decisions (D-029, D-030, D-033) keep their old paths as written; this entry
  supersedes them for location only. Any lab-PC or user environment variable that pointed at the
  old paths must be updated.

## D-035 — Tritonia batch 2: more training frames, kept only where every cross got a box; dataset v5 (2026-11-10, proposed)
- **Status:** proposed by Anu on `anu/e006-results`; text and freezing v5 user-approved by Anu on
  2026-11-10. Accepted when the PR is reviewed and merged. Extends D-033 (same automatic boxes, same scoring against the crosses); D-030's day split stays.
- **Context (OBSERVED, 2026-11-10):** E006 (v3: 86 included training frames, 292 boxes) scored val point
  F1 0.476 and test point F1 0.361. In v3 only about half of the crosses got a training box, so many
  clicked slugs are background in training (D-033 LIMITATION). The MTrackJ files hold 117,969 points on
  21,840 frames; v3 uses 150 frames. Only 35 of v3's 150 frames have a box for every cross.
- **Decision:**
  - **Batch 2** (`scripts/make_tritonia_batch2.py`, `pick_extra_frames`): the 150 batch 1 frames copied
    unchanged (sha256 checked), plus frames from the **10 v3 training days only**: up to 500 per day,
    frames with at least one cross, at least 5 frames (75 s) from each other and from every batch 1
    frame, seed 0. Result: 2,662 added frames (fewer than 500 on every day because of the gap rule).
    `D:\uwct_tritonia\work\tritonia_label_batch2\`.
  - **Boxes:** `scripts/auto_label_tritonia.py` unchanged (D-033 method and parameters) on the whole
    batch. The batch 1 frames give byte-identical label files and the same 18 exclusions as v3's label
    source (OBSERVED: 132 of 132 label files identical).
  - **Keep only complete added frames** (`scripts/tritonia_batch2_notes.py`, `incomplete_added_frames`):
    an added frame where any cross got no box (any flag) is marked `exclude`, so no clicked slug is a
    background example in an added frame. Batch 1 frames keep v3's treatment. OBSERVED: 536 of 2,662
    added frames kept (1,186 boxes); kept per day 20130519 121, 20130522 83, 20130523 62, 20130525 71,
    20130526 1, 20130527 27, 20130530 0, 20130531 85, 20130601 3, 20130603 83.
  - **Label source:** ingested unchanged by `scripts/ingest_tritonia_boxes.py` into
    `D:\uwct_tritonia\raw\tritonia_boxes_batch2\` (read-only, rule 1), with suggestions.csv,
    auto_label_report.json and batch2_filter_report.json as `--extra`.
  - **Dataset v5:** `make_tritonia_version.py --version v5 --decision D-035 --extends v3 --seed 0
    --val-days 3 --test-days 3`. `--extends v3` refuses unless every v3 frame is in v5 with identical
    manifest fields (split, status, boxes, hashes) and the split rule is v3's. So v5's val and test frames
    are exactly v3's (27 + 27); only train grows. v5 is a new Tritonia lineage (`parent: null`, like v3)
    recording `extends: v3`. Freezing needs explicit user approval.
  - **Experiment:** E009 = E006 with dataset v3 → v5 only (docs/EXPERIMENTS.md, declared before any result).
- **LIMITATIONS:**
  - "Complete" means every **cross** got a box, not every slug: whether every visible slug was clicked is
    UNKNOWN (D-029). Dr. Wyeth wrote that "the entire data set has been tracked manually"; many bright
    shapes have no cross, and whether they are slugs is UNKNOWN — requires confirmation.
  - The rule favours frames with few crosses and clear blobs: crowded days give almost nothing
    (20130526, 20130530, 20130601), so training under-represents crowded scenes. Some boxes sit on faint
    texture (D-033); nobody reviews them.
  - Added frames 75 s apart are close in time and similar; the effective amount of new data is smaller
    than the frame count.
- **Consequences:** new scripts and functions above with tests; `make_version` takes `extends` and records
  the label source folder name; the notebook picks its inputs by dataset version (cell 1). No change to
  v1–v4, E001–E008. v3 and its label source are not touched.

## D-036 — Tritonia fixed-size boxes on every cross; dataset v6 (2026-11-11, proposed)
- **Status:** proposed by Anu on `anu/e006-results`; text, freezing v6 and the private Kaggle upload
  (`anusidhu7/uwct-tritonia-v6-inputs`) user-approved by Anu on 2026-11-11. Its dataset is replaced by v7
  (D-037: one duplicate box); v6 is not used.
  Accepted when the PR is reviewed and merged. Changes D-033's box method only: accuracy is still scored
  against the crosses (point P / R / F1 / AP, 10 px one to one); D-030's day split stays.
- **Context (OBSERVED, 2026-11-10/11):** E006 (v3, automatic boxes) val point AP 0.403. E008 (imgsz 1280)
  0.317 and E009 (5× more automatic boxes, D-035) 0.274 did not beat it. In v3 only 292 of the 573 train
  crosses have a box, so about half the clicked slugs are background in training. On the v3 train days the
  automatic boxes have median size 20 × 17 px (IQR w 17–24, h 13–20) and their centres lie a median 1.0 /
  0.5 px (x / y) from the cross (IQR x −1.5 to 2.5, y −1.5 to 2.0 px): the cross marks the slug's centre
  (INFERRED for the crosses with a clear blob; the others have no box to compare).
- **Decision:**
  - **Boxes** (`scripts/fixed_label_tritonia.py`, `fixed_box`, `fixed_label_batch`): on every frame of
    batch 1, one **20 × 17 px** box (the train-day median above) centred on **every** cross, clipped to the
    camera view that holds the cross, so no box crosses a camera border. No image content is used and no
    frame is excluded. OBSERVED: 150 frames, 853 crosses, 853 boxes (51 clipped at a view edge); train 96
    frames / 573 boxes, val 27 / 153, test 27 / 127.
  - **Label source:** ingested unchanged by `scripts/ingest_tritonia_boxes.py` into
    `D:\uwct_tritonia\raw\tritonia_boxes_fixed1\` (read-only, rule 1), with fixed_label_report.json as `--extra`.
  - **Dataset v6:** `make_tritonia_version.py --version v6 --decision D-036 --boxes …tritonia_boxes_fixed1
    --batch …tritonia_label_batch1 --seed 0 --val-days 3 --test-days 3`: the same 150 frames and the same
    day split as v3 (checked: identical split_manifest), only the boxes differ. A new Tritonia lineage
    (`parent: null`, like v3). Freezing needs explicit user approval.
  - **Experiment:** E010 = E006 with dataset v3 → v6 only (the box method), declared in docs/EXPERIMENTS.md
    before any result; it replaces E006 as final candidate only if its val point AP is higher than 0.403.
- **LIMITATIONS:**
  - A box is drawn on a cross even where no slug is visible (dark, blurred), and it has one size for every
    slug, so some boxes are loose or hold mostly background. HYPOTHESIS, not tested: on crowded, faint days
    this may teach the model to fire on flat background and lower precision.
  - Train grows from 86 to 96 frames because frames the automatic labels excluded now have boxes: part of
    the same change (every cross gets a box), not a second variable chosen separately.
  - Val and test also get fixed boxes; they matter only for Ultralytics' mAP and its best-checkpoint choice
    (against these boxes, NOT accuracy, D-033). The point scores use the crosses as before.
  - Whether every slug was clicked stays UNKNOWN (D-029); bright shapes without a cross get no box.
- **Consequences:** new functions with tests in `tritonia_prelabel.py`, a new CLI, notebook inputs for v6
  (cell 1). No change to v1–v5, E001–E009 or their label sources.

## D-037 — Tritonia v7: v6 with one duplicate fixed box merged; v6 not used (2026-11-11, proposed)
- **Status:** proposed by Anu on `anu/e006-results`; text, freezing v7, pointing E010 at v7 and the private
  Kaggle upload (`anusidhu7/uwct-tritonia-v7-inputs`) user-approved by Anu on 2026-11-11. Corrects the
  dataset of D-036; D-036's box method otherwise stays.
- **Context (OBSERVED, 2026-11-11):** E010's notebook cell 10 (`verify_yolo_dataset.py`) failed on v6:
  Ultralytics read 572 train instances, the manifest says 573. In train frame
  `20130527all_15spf_t1273` tracks 28 (point 613) and 34 (point 1) have a cross at exactly (237, 473), so
  `fixed_label_batch` wrote two identical boxes and Ultralytics dropped one ("1 duplicate labels
  removed"). It is the only frame of the 150 with crosses on one spot. v6 is frozen (rule 4) and cannot be
  changed. No run used v6 (E010 stopped before training).
- **Decision:**
  - `fixed_label_batch` gives crosses at exactly the same position in one frame one box
    (`duplicate_crosses` in the report; test added). OBSERVED: 853 crosses, 852 boxes, 1 duplicate.
  - Label source `D:\uwct_tritonia\raw\tritonia_boxes_fixed2\` (read-only); **dataset v7** built and frozen
    as D-036 describes for v6. v7 differs from v6 only in that frame (train 572 boxes).
  - **v6 is not used** by any experiment; it stays frozen and registered as a record.
  - E010 (not run yet) uses v7 instead of v6; its declaration otherwise stands.
- **LIMITATION:** two tracks on one spot may be one slug clicked twice or two touching slugs; one box
  covers both either way. In the point eval the two crosses stay two crosses (one can be matched), as for
  every earlier run.
- **Consequences:** v7, E010 config and notebook input mapping updated; v1–v6 unchanged.

## D-038 — Model A tuning sprint: model size × seed, then declared combinations (2026-11-11, accepted)
- **Status:** accepted 2026-11-11. The user approved the plan ("i approve of your plan"), asked to "utilize
  the kaggle gpu to the fullest", for "hyperparameter tuning with different combinations" and "bigger models"
  (yolo26l, the third seed and round 2 were added for that), and then approved the selection metric and the
  epoch rule below ("i approve all your decisions go ahead"). The user also asked that the final report
  explain what the metric means for the project and why it was picked (point 8).
- **Context (OBSERVED):** E007 (v4, yolo26s, 100 epochs, patience 30) peaked at epoch 28 (val mAP50
  0.8420, mAP50-95 0.5985 in results.csv) and stopped at 58; train losses kept falling while val mAP50
  stayed at about 0.80–0.82. Ultralytics `optimizer=auto` chose AdamW lr 0.002 with linear decay over 100
  epochs, so the learning rate was still about 0.0015 at the peak. In the pinned Ultralytics 8.4.155,
  `best.pt` is the epoch with the highest "fitness", and fitness equals val mAP50-95: E007's run_record
  shows fitness 0.59933 = mAP50-95 0.59933. In E007, epoch 28 was the best on mAP50, mAP50-95 and fitness
  alike. E007 ran in a 2× Tesla T4 session but trained on GPU 0 only, with 2 dataloader workers
  (4 CPUs). The user must show the report on 2026-11-13 and has about 24 h. The Kaggle weekly GPU quota is
  about 30 h, of which E007 used about 2.5 h (the user reports no other GPU use).
- **Decision:**
  1. **Selection metric: val mAP50-95 (class `lobster`), at both levels.** It picks the best epoch of a run
     (Ultralytics fitness, unchanged) and the best run/configuration (points 4 and 6), so one metric is used
     throughout. Val mAP50, P and R are always reported next to it. This replaces val mAP50 as the selection
     metric for Model A runs from E011 on; it is declared before any of these runs (CLAUDE.md §5).
     Comparisons with E001 and E007 show both metrics (mAP50-95 / mAP50: E001 0.6115 / 0.8527, E007
     0.5993 / 0.8423).
  2. **Round 1 (configs E011–E019, committed before any run):** yolo26s / yolo26m / yolo26l × seeds 0 / 1 / 2,
     all v4, imgsz 640, batch 16, **epochs 60 as a ceiling and patience 10**, otherwise as E007. A run trains
     while val mAP50-95 improves, up to 60 epochs, and stops 10 epochs after its last improvement. The
     learning-rate schedule decays over the 60 epochs; an Ultralytics run cannot be extended past its
     `epochs` value, so the ceiling is set up front. Chain: E011 = E007 with epochs 100 → 60 and patience
     30 → 10 (two variables on purpose: together they are the stopping rule); E012 / E013 = E011 with the
     model changed; the seed runs change only the seed.
  3. **Execution:** one Kaggle background version on the 2×T4 session, one training process per GPU
     (`CUDA_VISIBLE_DEVICES`), taking runs from one queue in the order E011 (s0), E013 (l0), E012 (m0),
     E016 (s1), E014 (m1), E015 (l1), E019 (s2), E017 (m2), E018 (l2): every size gets seed 0 first and the
     longest run starts at once. A run starts only if its worst-case duration (60 epochs × the slowest
     measured seconds per epoch of its model size, or a default until one is measured, + 10 min) ends
     before a deadline of 9.5 h after the notebook starts, so no run is ever cut off. Each finished run is
     packaged (zip + sha256) at once. Runs not started are recorded as "not run (time)", never as results.
  4. **Round 1 selection (val only):** for each model, the mean val mAP50-95 of its finished seeds (mAP50,
     P, R and the seed range are reported too). The winner is the model with the highest mean. If the gap
     to a smaller model is not larger than the bigger of the two seed ranges, the smaller model wins
     (cheaper, no evidence of a gain). A model with one finished seed has no range: it is reported, and it
     wins only if it beats the best mean by more than the largest range among the multi-seed models.
  5. **Round 2 (Thursday 2026-11-12, only if quota allows: it starts only when the Kaggle quota meter shows
     at least round 2's deadline + 4 h left):** five combinations, fixed now, each one run (seed 0) with
     the round-1 winning model and otherwise the round-1 settings (epochs 60, patience 10). Configs
     E020–E024 are instantiated with that model before the run:
     - C1: `optimizer: AdamW`, `lr0: 0.001`, `momentum: 0.9`, `cos_lr: true`
     - C2: `cos_lr: true`, `mixup: 0.1`
     - C3: `freeze: 10` (backbone frozen)
     - C4: `optimizer: AdamW`, `lr0: 0.001`, `momentum: 0.9`, `weight_decay: 0.001`, `mixup: 0.1`
     - C5: `cos_lr: true`, `close_mosaic: 20`

     `momentum: 0.9` reproduces the AdamW beta1 that `auto` used. Several variables change per run on
     purpose (a combination search); this decision is the reason (CLAUDE.md §5). Queue order C1–C5, same
     worst-case deadline rule, with the deadline set when round 2 is launched.
  6. **Overall selection (val only):** a round-2 combination replaces the round-1 winner only if its val
     mAP50-95 beats the winner's seed mean by more than the winner's seed range (one seed must clear the
     noise).
  7. **Final candidates:** the finished runs of the selected configuration are declared in
     `experiments/final_candidates.csv` before any test evaluation. Each is evaluated once on the v1 test
     and once on Zenodo `sk`, and the mean and range are reported (mAP50-95, mAP50, P, R). The demo model
     is the declared run with the highest val mAP50-95. E001 (v1) is compared on val and on its existing
     test result. If the selected configuration does not beat E001's val mAP50-95 0.6115, the report says
     so.
  8. **Report:** the final report explains the metric for the project. mAP50 counts a detection as correct
     when its box overlaps the annotated box with IoU ≥ 0.5 ("was the lobster found?"). mAP50-95 averages
     over IoU thresholds 0.50–0.95, so it also rewards tight boxes. The tracker links detections from frame
     to frame and the trajectories are built from box positions in px, so both finding the lobster and box
     accuracy matter. mAP50-95 measures both, and Ultralytics already uses it to choose the best epoch, so
     selecting runs on it keeps one criterion throughout. LIMITATION: neither is a tracking metric (no
     ground-truth track IDs, CLAUDE.md rule 9), and the operating confidence threshold (P/R trade-off,
     false positives on negatives) is chosen later, on val.
  9. **Deferred:** leave-site-out cross-validation (D-031 stage 5) to after 2026-11-13, before 2026-12-15.
     Also deferred: imgsz > 640, RT-DETR on v4 and a run without derived images. Ultralytics `model.tune()`
     is not used (it needs dozens to hundreds of runs).
- **ASSUMPTION — requires confirmation:** Kaggle counts quota by session time regardless of the number of
  GPUs. This is checked on the quota meter after the first hour of the background run. The time per epoch
  with two runs sharing 4 CPUs is UNKNOWN; the worst-case deadline rule (point 3) bounds the session and
  may skip later seeds, especially of yolo26l.
- **LIMITATIONS:** 324 val images select among up to 14 configurations, so the selected val score is
  optimistic. Only the single test evaluation of the declared runs is unbiased. Round 2 has one seed per
  combination. Val/test measure new sessions at known sites; only Zenodo `sk` is a new site. Early
  stopping on val means val also chose each run's length (as in every earlier run).

## D-039 — Round-1 configs name E007 as parent (2026-11-11, accepted)
- **Status:** accepted 2026-11-11 (user-approved: "yes commit push and open the PR"). Supersedes the parent chain in D-038 point 2;
  everything else in D-038 is unchanged.
- **Context (OBSERVED):** the first background run of the D-038 notebook (Kaggle version "D-038 round 1
  E011-E019 (2affc59)", 2026-11-11 21:13 UTC) started E011 and then stopped E013, E012 and E016 within
  seconds: `train.py` refuses a full run whose parent is not in `experiments/registry.csv` ("parent E011
  is not in registry.csv at this commit"). D-038 made E012–E019 children of E011 / E012 / E013, which run
  in the same session and cannot be registered first. No run produced results; the user stopped the version
  (E011 had trained for about 15 min; its partial run is not used).
- **Decision:** E012–E019 name **E007** (registered, reviewed) as parent; their `changed` field lists every
  variable relative to E007 (epochs, patience, and model and/or seed). The configs differ from the D-038
  versions only in `parent` and `changed`: model, seed and train_args are the same. The parent guard is
  kept as it is. Round-2 configs (E020–E024) name the round-1 winner, which is registered before round 2.
- **Consequences:** the round-1 notebook is re-run from Cell 1 on the fixed commit; all nine runs start
  from scratch. Comparisons within round 1 are still single-variable in substance (same settings, only
  model or seed differ), as D-038 points 4 and 6 already assume.


## D-041 — Tritonia v8: fixed boxes on all batch-2 frames (v7 + 2,662 training-day frames); E039–E041 (2026-11-12, proposed)
- **Status:** proposed by Anu on `anu/e006-results`. Plan user-approved by Anu on 2026-11-12 ("ok do both", to the
  D-041 proposal: v8 = v7 plus all batch-2 frames with fixed boxes, val/test unchanged, E010's settings at 3 seeds,
  D-042's rule). Freezing v8 and the private Kaggle upload need their own approval.
- **Context (OBSERVED):** D-042: no training setting beats E010's (baseline mean val point AP 0.489 over 3 seeds,
  range 0.018). v7 trains on 96 frames; the MTrackJ files hold crosses on 21,840 frames. Batch 2 (D-035) already
  holds the 150 batch-1 frames plus 2,662 frames from the 10 training days; D-035 kept only 536 of them because
  the automatic boxes missed crosses. Fixed boxes (D-036) give every cross a box, so that filter is not needed.
- **Decision:**
  - **Labels:** `scripts/fixed_label_tritonia.py` (unchanged, D-036/D-037 method) on
    `D:\uwct_tritonia\work\tritonia_label_batch2`. OBSERVED: 2,812 frames, 19,326 crosses, 19,230 boxes
    (96 duplicate crosses merged, D-037; 1,207 clipped at a view edge), no frame excluded; the 150 batch-1 label
    files are byte-identical to v7's label source. Ingested unchanged into
    `D:\uwct_tritonia\raw\tritonia_boxes_fixed3\` (read-only, rule 1).
  - **Dataset v8:** `make_tritonia_version.py --version v8 --decision D-041 --extends v7 --seed 0 --val-days 3
    --test-days 3`: `--extends v7` refuses unless every v7 frame is in v8 unchanged, so val and test are v7's
    27 + 27 frames; only train grows. Freezing needs explicit user approval.
  - **Experiments:** E039 / E040 / E041 = E010 with dataset v7 → v8 only, seeds 0 / 1 / 2, parent E010.
    Compared with the D-042 baseline (E010, E042, E043) by D-042's rule, declared in docs/EXPERIMENTS.md before
    any run: v8 wins only if its mean val point AP exceeds 0.489 by more than the larger of the two seed ranges.
    A test only after a separate user approval.
  - **Freeze (OBSERVED):** v8 frozen on 2026-11-12 (UTC), user-approved by Anu ("yes", to freezing v8). YOLO view
    fingerprint `52cd95c302e3f4a5e13f0646b055a16cd7d4e556b3a212f21c4ce3f144261b84` (local build: train 2,758 frames
    / 18,950 boxes, val 27 / 153, test 27 / 127); val and test label files byte-identical to the v7 view's.
- **LIMITATIONS:** whether every slug was clicked is UNKNOWN (D-029): an unclicked slug in an added frame is a
  background example. Added frames are 75 s apart, so the effective amount of new data is smaller than the frame
  count. More frames make each epoch about 28× longer; 100 epochs are kept so only the dataset changes.
- **Consequences:** label source `tritonia_boxes_fixed3`, draft v8, configs E039–E041. No change to v1–v7 or
  earlier experiments.

## D-042 — Tritonia hyperparameter tuning, round 1: four single-setting arms × 3 seeds on v7 (2026-11-12, proposed)

> **Renumbered 2026-11-13** (CLAUDE.md §10, before this branch's PR): numbered **D-040** until then, and its runs E042–E046 ran on Kaggle as **E025–E029** (commit `1f84325`); `p4/colab-baseline` (#25) had meanwhile used D-040 and E020–E029. Content otherwise unchanged. Archived run records and off-repo files keep the original IDs and names (`d040_*`). D-041 kept its number (the frozen v8 records it).
- **Status:** proposed by Anu on `anu/e006-results`. Plan user-approved by Anu on 2026-11-12 ("ok proceed with
  this but dont do anything for gui"): baseline E010, arms A–D, 3 seeds, the selection rule, round 2 and a
  test only on a later approval, one Kaggle notebook. No GUI work is part of it.
- **Context (OBSERVED):** E010 (v7, yolo26s, imgsz 640, 100 epochs, patience 30, batch 16) has the best Tritonia
  val point AP (0.496; E006 0.403) and a lower test point AP than E006 (0.215 vs 0.248), so one seed on 27 val
  frames is not enough to tell settings apart. Its `results.csv`: no val detections until about epoch 20, best
  epoch 81 of 100, val mAP50-95 still about 0.18–0.19 at epochs 80–100 (against the fixed boxes, not accuracy,
  D-033). 96 training images at batch 16 = 6 optimizer steps per epoch. Augmentation is the Ultralytics default,
  including `scale 0.5`; frames are 1056 × 480 and slugs about 12–20 px, shrunk to 0.61× at imgsz 640.
  One E010 run took 239 s on a Tesla T4.
- **Decision:**
  1. **Baseline:** E010's settings on frozen dataset v7 (chosen on val; test results never choose, rule 7).
     Seed 0 is E010 itself; E042 / E043 repeat it with seeds 1 / 2.
  2. **Round 1 arms**, each one setting changed from E010, seeds 0 / 1 / 2, parent E010 (registered):
     A `epochs 100 → 200` with `patience 30 → 50` (two variables on purpose: together the stopping rule;
     E044–E046); B `batch 16 → 8` (E030–E032); C `scale 0.5 → 0.2` (E033–E035); D `imgsz 640 → 1056`
     (E036–E038; the val point eval runs at the same imgsz). Everything else as E010; `workers: 2` is pinned
     for one process per GPU and is not a changed variable. IDs started at E025 because D-038 reserves E020–E024; E025–E029 were renumbered E042–E046 on 2026-11-13 (see the note above).
  3. **Execution:** `notebooks/kaggle_tritonia_tuning.ipynb`, one Kaggle session (GPU T4 × 2, one training
     process per GPU), at one pushed commit: build and verify v7, smoke test, then the 14 runs from one queue
     with seed 0 of every arm first, then the val point eval of every finished run (`tritonia_point_eval.py
     --split val`, all 27 val frames, 10 px one to one, threshold = that run's best val F1). Each run's folder
     and the comparison are zipped for download in the session (saved versions have kept no output, E010).
  4. **Selection metric: val point AP** (D-033), per arm the **mean over its 3 seeds**, with the seed range
     (max − min). Point P / R / F1 at each run's threshold, per day, and Ultralytics' val metrics (labelled not
     accuracy) are reported next to it. An arm **wins** only if its mean is higher than the baseline mean by
     more than the larger of the two seed ranges; otherwise the result is "no evidence of a gain". An arm with
     fewer than 3 finished seeds is reported and cannot win.
  5. **Round 2** (only if at least one arm wins): one config combining the winning settings, 3 seeds, compared
     with the baseline by the same rule; its configs are committed before it runs. If it wins it may be
     declared a final candidate and tested once, only after a separate user approval (the 27 test frames were
     already used by E006 and E010). If no arm wins, E010 stays the Tritonia reference and there is no test.
- **LIMITATION:** 27 val frames from 3 days and four arms compared on them: the best arm can still be partly
  chosen by noise; the 3 seeds and the seed-range rule reduce this but do not remove it. All runs are on one
  setup (new days, not new sites); point precision is a lower bound (D-029). Arm D costs about 2.7× the compute.
- **Consequences:** 14 configs E030–E038 and E042–E046, the notebook, docs/EXPERIMENTS.md rows and the declaration below,
  all before any run. No change to v7, E006, E010 or any earlier result.

