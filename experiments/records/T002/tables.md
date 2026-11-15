# Tritonia tracking (D-043): T002, val vs final test

Final test run once (2026-11-13), settings frozen before it (T002 = T001's val choice: conf 0.16, gate 20 px, max gap 60, min length 10) on E010 detections. Scored on clicked frames, 10 px, cameras summed per day. Sparse-frame/timelapse analysis (15 s per frame); px only. Test sessions are new days of the same setup (not new sites); UNKNOWN whether every slug was clicked, so unmatched detections are an upper bound on false positives.

## overall

| split | run | HOTA | DetA | AssA | LocA | IDF1 | MOTA | IDSW | TP | FN | FP | clicks_matched | MT | PT | ML |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | mean over 3 days | 0.168 | 0.149 | 0.191 | 0.749 | 0.238 | -0.076 | 856 | 7820 | 14570 | 7311 | 0.349 | 10 | 51 | 101 |
| test | mean over 3 days | 0.091 | 0.067 | 0.122 | 0.746 | 0.130 | -0.444 | 289 | 2542 | 12563 | 7923 | 0.168 | 2 | 53 | 132 |

## per_day

| split | day | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | eval_frames |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| val | 20130524 | 0.040 | 0.025 | 0.065 | 0.063 | -0.294 | 15 | 1683 |
| val | 20130529 | 0.342 | 0.316 | 0.372 | 0.457 | 0.259 | 426 | 1247 |
| val | 20130602 | 0.120 | 0.107 | 0.135 | 0.192 | -0.194 | 415 | 1674 |
| test | 20130520 | 0.084 | 0.061 | 0.117 | 0.121 | -0.358 | 28 | 1218 |
| test | 20130521 | 0.064 | 0.046 | 0.091 | 0.097 | -1.031 | 95 | 1903 |
| test | 20130528 | 0.123 | 0.096 | 0.160 | 0.173 | 0.057 | 166 | 872 |

## references_HOTA

| split | run | HOTA |
|---|---|---:|
| val | no linking | 0.027 |
| val | T002 | 0.168 |
| val | clicks as detections | 0.941 |
| test | no linking | 0.032 |
| test | T002 | 0.091 |
| test | clicks as detections | 0.864 |

## trajectories

| split | day | clicked_tracks | clicks_median_path_px | tracker_tracks | tracker_median_path_px | matched_half |
|---|---|---:|---:|---:|---:|---:|
| val | 20130524 | 39 | 218.120 | 53 | 16.440 | 0 |
| val | 20130529 | 65 | 320.100 | 168 | 47.110 | 10 |
| val | 20130602 | 58 | 291.025 | 106 | 38.250 | 2 |
| test | 20130520 | 43 | 159.810 | 49 | 30.420 | 2 |
| test | 20130521 | 65 | 234.450 | 98 | 52.725 | 1 |
| test | 20130528 | 79 | 286.430 | 94 | 28.235 | 2 |
