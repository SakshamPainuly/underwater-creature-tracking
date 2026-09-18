# Dataset card — Lobster detection dataset, version v1

Evidence labels: OBSERVED (verified from files), INFERRED, UNKNOWN — requires confirmation.

## Source
Biology department delivery, stored read-only at `data/raw/biology/` (about 8.9 GB, 4,409 files,
structural fingerprint in `data/processed/raw_fingerprint.json`). PNG frames exported from GoPro
MP4 videos (1920×1080 or 1920×1088, 8-bit RGB), YOLO labels made with LabelImg. **No video files
are included** (OBSERVED).

| Source (config name) | Folder | Images | Label files | Missing-label policy |
|---|---|---:|---:|---|
| Set100 | `Set 100/` | 105 | 105 | error (documented fully labeled) |
| Set2000 | `Set 2000/batch_2 (2k - imgOnly)/batch_2 (2k - imgOnly)/` + `batch_2_2k_labels/batch_2_2k_labels/` | 2,000 | 1,759 | negative (docs: no label = no lobster) |
| zExtra | `zExtra Photos/` | 436 | 0 | pending (excluded; not confirmed negative) |

The two `.zip` files in `Set 2000/` duplicate the extracted folders and are not used.

## Labels
- Format (OBSERVED, all 6,522 rows validated): `class x_center y_center width height`, normalized
  to [0, 1]. Single class `0 = lobster`.
- 0 malformed rows, 0 out-of-range values, 0 duplicate rows, 0 orphan labels, 0 duplicate filenames.
- 1,430 of 1,864 labeled images contain ≥2 lobsters (maximum 15).

Label statuses in `image_manifest.csv`:
- `positive`
- `negative_confirmed`: Set 2000 without a label file.
- `negative_pending`: zExtra; never used for training or evaluation.
- `invalid`: currently 0.

## Split (v1, frozen — `data/processed/FROZEN.json`)
The split unit is the session (the deployment code without the GoPro file ID); see D-001 and D-002.

| Split | Sessions | Images | Positive | Confirmed negative | Boxes |
|---|---:|---:|---:|---:|---:|
| train | 32 | 1,433 | 1,274 | 159 | 4,000 |
| val | 7 | 324 | 269 | 55 | 1,010 |
| test | 6 | 348 | 321 | 27 | 1,512 |
| excluded (zExtra) | 11 | 436 | — | — | — |

- No session appears in more than one split (verified).
- Every multi-file session sits in exactly one split.

## Multi-file sessions
- Nine sessions contain two GoPro recording files, e.g. `06_HERR_02_CW_20200901`: `GP299683` + `GP309683`.
- OBSERVED: both files share one session code and one split.
- INFERRED: same 4-digit file number, consecutive 2-digit prefix, frame numbering restarting per
  file, and the second file much shorter in 7 of 9 pairs. This matches sequential GoPro chapters of
  one recording.
- UNKNOWN — requires confirmation: whether any were simultaneous cameras.
- The flag text `multi_camera_session` in the v1 split manifest is a historical name for
  "multi-file session".

## Known limitations (state these with every v1 result)
1. **New sessions, not new sites.** 8 of 29 site-date groups span more than one split. 5 of the 6
   test sessions share site and date with a train session, and every test site also appears in
   train (D-007).
2. **Environmental metadata is sparse.** Substrate/visibility is missing for 39 of 44 Set 2000
   sessions and for Set 100. The red-light/night flag is available for all sessions (11 of 44
   Set 2000 sessions flagged).
3. **Concentrated evaluation sets.**
   - Val: `21_MACK_10` holds 465 of 1,010 boxes.
   - Test: `13_SHRI_12` + `21_SUB6_03` hold 1,135 of 1,512 boxes.
   - `19_SUB1_04` (val) has 3 boxes, likely one lobster across 3 frames.
4. **Label certainty in poor visibility.** Negatives in the very turbid or red-light sessions
   `06_HERR_02` and `19_BESU_01` could not be visually confirmed (a spot check found no missed
   lobster, but visibility is poor).
5. **Sparse frames.** Frames are sampled 300–900 frames apart, roughly 10–30 s at typical GoPro
   frame rates (UNKNOWN exact fps). They cannot support tracking.
6. Set 100 is a single session flagged by the data providers as "images too close to each other in
   time"; it is train-only.

## Other observations
- A 3-point green laser rig is visible in `19_SUB1_04` red-light footage (OBSERVED). Whether it
  gives a usable physical scale is UNKNOWN — requires confirmation.
- The spreadsheet's "Files used" cell for `06_HERR_02` lists `GP299683` twice (likely a typo for
  `GP309683`). Grouping comes from filenames, so this has no effect.
- 16 deployments listed in the spreadsheet as "in progress" (625 images) are not in this delivery.

## Raw-data fingerprint semantics
- Default check (`scripts/check_raw_unchanged.py`): **structural/completeness** — every file's path
  and size, plus sha256 of `.txt`/`.xlsx`/`.docx`. PNG contents are covered by path and size only,
  so this is not a cryptographic proof of unchanged pixels.
- `--full` hashes every file (requires a fingerprint recorded with `--write --full`).

## Open questions (for the professor)
MP4 videos and their frame rate; multi-file/chapter confirmation; zExtra annotation status;
substrate/visibility backfill; laser-rig scale; project end date.
