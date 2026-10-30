# Pipeline stage file formats

Stages communicate through files (CLAUDE.md §7). This page documents the formats as they are added.

## Tracking (D-028, proposed)

```
video ──scripts/track_video.py (GPU)──▶ <tracks>/tracks.csv + tracks_meta.json
                                              │
synthetic ──scripts/make_synthetic_tracks.py──┤
                                              ▼
                     scripts/analyze_tracks.py ──▶ <tracks>/analytics/
                         analytics.json, track_summary.csv, trajectories.svg, occupancy.svg
```

Tracks folders live off-repo (Drive) or under the git-ignored `runs/`, never inside `data/raw/`
(the scripts refuse it).

### `tracks.csv`

One row per (frame, track) box, sorted by `frame_idx`, then `track_id`. One video per file.

| column | type | meaning |
|---|---|---|
| `video_id` | str | source video id (default: file stem); same in every row and in the meta |
| `frame_idx` | int ≥ 0 | 0-based index over every decoded frame, frames without boxes included |
| `track_id` | int ≥ 0 | tracker-assigned ID, unique within the video; at most one row per (frame, track) |
| `x1`,`y1`,`x2`,`y2` | float, 2 dp | box in original-frame pixels, top-left origin, `x2 > x1`, `y2 > y1` |
| `conf` | float in [0, 1], 4 dp | detector confidence of the tracked box |
| `class_id` | int ≥ 0 | dataset class id (`0: lobster`) |

### `tracks_meta.json` (schema `uwct.tracks_meta/1`)

Required: `schema`, `video_id`, `synthetic` (bool), `frame_width`, `frame_height`, `n_frames`
(decoded frames), `fps` (number or `null`), `detector`, `tracker`.
Written by `track_video.py` in addition: `fps_source`, `source` (video path, size, sha256),
`frame_count_reported`, detector weights path + sha256 + conf/iou/imgsz, tracker name, git state,
start/finish times. Synthetic folders add `scenario` (seed, jitter, what each track simulates).

`fps` is `null` unless known; `track_video.py` records the container-reported rate with
`fps_source`. Analytics reports seconds only when `fps` is a number.

### `analytics/`

- `analytics.json`: `overall` (n_frames, n_boxes, n_tracks, frames_with_boxes, max_boxes_in_a_frame,
  median track length, short_tracks, tracks_with_gaps), `per_track`, `boxes_per_frame`,
  `occupancy_grid`, `parameters`, `limitations`.
- `track_summary.csv`: one row per track; columns and definitions in `src/tracking/analytics.py`.
- `trajectories.svg`, `occupancy.svg`: centroid paths (gaps dashed) and a centroid-count grid.

Units: px and frames (seconds only with known fps); no calibration, so no physical units. These are
descriptive statistics of the tracker output: no IDF1/HOTA/MOTA without ground-truth track IDs.
