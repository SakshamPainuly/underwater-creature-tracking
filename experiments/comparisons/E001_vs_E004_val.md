### Overall (val)
| metric | E001 | E004 |
|---|---|---|
| dataset version | v1 | v2 |
| Ultralytics P (its own conf) | 0.800 | 0.806 |
| Ultralytics R (its own conf) | 0.790 | 0.747 |
| Ultralytics mAP50 | 0.853 | 0.839 |
| Ultralytics mAP50-95 | 0.611 | 0.586 |
| operating conf (chosen on val) | 0.298 | 0.276 |
| AP50, own matcher | 0.848 | 0.836 |
| precision @ conf | 0.795 | 0.784 |
| recall @ conf | 0.801 | 0.772 |
| F1 @ conf | 0.798 | 0.778 |
| TP / FP / FN @ conf | 809 / 208 / 201 | 780 / 215 / 230 |
| negative images with a detection | 16 of 55 | 14 of 55 |
| detections on negative images | 22 | 19 |
| parameters (M) | 9.95 | 9.95 |
| GFLOPs (at imgsz) | 22.7 (640) | 22.7 (640) |
| inference ms/img, batch 1 (median) | 16.3 | 16.3 |
| total ms/img, batch 1 (median) | 19.6 | 19.6 |
| GPU | Tesla T4 | Tesla T4 |

### Per lighting (val, each run at its own operating conf)
| lighting (boxes) | E001 | E004 |
|---|---|---|
| not_flagged (979) | AP50 0.851, P 0.799, R 0.802, FP 198 | AP50 0.841, P 0.785, R 0.774, FP 208 |
| red_light_or_night (31) | AP50 0.741, P 0.706, R 0.774, FP 10 | AP50 0.713, P 0.759, R 0.710, FP 7 |

### Per session (val, each run at its own operating conf)
| session (boxes) | E001 | E004 |
|---|---|---|
| 18_ALRA_08_CW_20220616 (60) | AP50 0.814, P 0.608, R 0.800, FP 31 | AP50 0.644, P 0.653, R 0.533, FP 17 |
| 19_SUB1_04_CW_20220621 (3) | AP50 1.000, P 0.500, R 1.000, FP 3 | AP50 0.916, P 0.600, R 1.000, FP 2 |
| 19_SUB2_07_CW_20220622 (28) | AP50 0.731, P 0.750, R 0.750, FP 7 | AP50 0.709, P 0.792, R 0.679, FP 5 |
| 21_MACK_10_BW_20220815 (465) | AP50 0.906, P 0.843, R 0.869, FP 75 | AP50 0.908, P 0.855, R 0.839, FP 66 |
| 21_SUB6_07_BW_20220812 (95) | AP50 0.701, P 0.698, R 0.705, FP 29 | AP50 0.640, P 0.545, R 0.695, FP 55 |
| 21_SUB6_09_BW_20220815 (125) | AP50 0.845, P 0.750, R 0.768, FP 32 | AP50 0.829, P 0.704, R 0.800, FP 42 |
| 21_SUB6_10_BW_20220815 (234) | AP50 0.826, P 0.846, R 0.726, FP 31 | AP50 0.850, P 0.859, R 0.726, FP 28 |

Limitations: Val only. Val chose E001's checkpoint (best epoch) and every operating confidence, so these numbers are optimistic for unseen data; they compare runs, they do not estimate deployment quality. One run per arm (seed 0): small differences may be run-to-run noise. Val sessions share sites with train sessions: new sessions, not new sites (D-007). Val boxes are concentrated: 21_MACK_10 holds 465 of 1,010 boxes. Red-light val is 2 sessions / 31 boxes (19_SUB1_04, 19_SUB2_07): per-lighting results are fragile. Substrate/visibility metadata is mostly missing; zExtra is excluded. Own-matcher numbers (IoU >= 0.5, greedy) can differ slightly from Ultralytics' (different matching and batching); both are recorded.
