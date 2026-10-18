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
