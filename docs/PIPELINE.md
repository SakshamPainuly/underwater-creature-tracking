# Pipeline stages and file schemas

Stages communicate through files (CLAUDE.md §7). This file documents the schemas as they are created; so far the
Tritonia tracking stage (D-043). Frames are 15 s apart: sparse-frame/timelapse analysis, never "video tracking".
Positions are in px of the 1056 × 480 montage; no physical units (rule 9).

## Tritonia tracking (D-043)

```
raw TIFF stack + MTrackJ .mdf  --make_tritonia_sequence.py-->  sequence folder (frames, manifest, ground truth)
sequence folder + E010 best.pt  --predict_tritonia_sequence.py (GPU)-->  detections folder
sequence folder + detections   --track_tritonia.py (CPU)-->  tracks.csv + scores.json
```

Cameras: 3 × 2 montage, seams at x = 351.5 / 703.5 and y = 240 (OBSERVED, D-043); ids 0 1 2 top, 3 4 5 bottom
(`tracking.sequences.camera_of`). A ground-truth track belongs to the camera holding most of its points; a
detection to the camera its centre lies in.

### Sequence folder (`scripts/make_tritonia_sequence.py`)
Off-repo frames (`D:\uwct_tritonia\sequences\<split>\`); the three small files are kept in the repo
(`data/processed/tritonia_tracking/<split>/`).

| file | content |
|---|---|
| `images/<stack>_t####.png` | every page of the day's stack, 8-bit grayscale, `t` 1-based (page index + 1) |
| `manifest.csv` | `image, day, source_tif, t, page_index, n_gt_points, png_sha256`, one row per frame |
| `gt_points.csv` | `day, camera, track_id, cluster_id, t, x, y, track_flag`, one row per MTrackJ click, sorted by `t, camera, track_id` |
| `sequence.json` | `decision, split_version, split, root, days{source_tif, tif_bytes, width, height, frames, click_cadence, tracks, gt_points, frames_with_points, tracks_per_camera, track_flags}, camera_layout, totals, manifest_sha256, gt_points_sha256` |

`click_cadence` = most common frame step inside a track (1 = every frame, 2 = every 2nd frame).

### Detections folder (`scripts/predict_tritonia_sequence.py`)

| file | content |
|---|---|
| `predictions/<stack>_t####.txt` | one line per detection, `cls x y w h conf` (normalised to the frame, conf ≥ 0.001); an empty file = no detection; one file per manifest row |
| `predictions.json` | `decision, phase, smoke, split, final, sequence{split_version, days, manifest_sha256, gt_points_sha256}, experiment, weights{path, sha256}, predict_args{conf, imgsz, batch}, env, git{commit, dirty}, seconds, summary_conf_0.001, summary_conf_0.16` |

A detection's point is its box centre.

### Tracking output (`scripts/track_tritonia.py`)

| file | content |
|---|---|
| `tracks.csv` | `day, camera, track_id, t, x, y, conf`; `track_id` 1..n per day, numbered camera by camera in order of start frame |
| `scores.json` | `decision, phase, split, final, sequence, source{mode, experiment, weights_sha256, predictions_json_sha256, predict_commit}, params{conf, gate_px, max_gap, min_len, match_px}, tracks, per_day{eval_frames, cameras{0..5}, all_cameras}, mean_over_days, all_days, note` |

Each score block holds HOTA, DetA, AssA, LocA, `HOTA_alpha` (19 alphas), MOTA, MOTP, IDSW, TP, FN, FP,
GT_tracks, MT, PT, ML, IDF1, IDP, IDR, IDTP, IDFN, IDFP (`tracking.metrics`). Scored only on evaluation frames
(frames with at least one click that day); a prediction matches a click at distance d with similarity
max(0, 1 − d / match_px) (HOTA) or when d ≤ match_px (CLEAR, Identity). Days combine cameras by summing counts;
`mean_over_days` is the unweighted mean of the per-day scores (the D-043 selection metric is its HOTA).
