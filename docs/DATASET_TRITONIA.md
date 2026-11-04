# Dataset card — Tritonia (sea slug) timelapse tracks, draft (D-029, proposed)

Evidence labels: OBSERVED (verified from files), INFERRED, UNKNOWN — requires confirmation.
Not yet a dataset version, not under `data/raw/`, not used by any experiment. Lobster data: DATASET.md.

## Source
- Location (OBSERVED, 2026-11-03): external drive, folder `E:\Tritonia\` (17 `.tif`, 16 `.tif.mdf`;
  18,846,191,199 B of TIFF + 4,353,299 B of MDF), plus `E:\Tritonia Info.txt` (778 B, SHA-256
  `a6ec1db19a393acc12e87ca26d8789be2787a7d7c4fbe3f41b2409c90bcd0d26`). Read only; nothing on the
  drive was changed. Project copy (D-029): `D:\uwct_raw\tritonia\` (`UWCT_TRITONIA_ROOT`), read-only,
  all 34 files with SHA-256 checked against the external drive.
- Provenance (from `Tritonia Info.txt`, comments from Dr. Wyeth; not independently verified):
  - old, low-resolution grayscale video stored as TIFF stacks, one frame every 15 s;
  - each frame is a montage of 6 cameras; slugs are tracked in each camera separately and there is
    no point tracking a slug between cameras;
  - cameras point at an angle to the substrate, not straight down: substantial parallax;
  - the entire data set was tracked manually in FIJI with the MTrackJ plugin
    ([documentation](https://imagescience.org/meijering/software/mtrackj/));
  - stated purpose: "an exercise in establishing how bad the imagery can get before the tracking fails".

## Images (OBSERVED, read with Pillow 12.3.0)
| | |
|---|---|
| Stacks | 17, one per day, `20130518all_15spf.tif` … `20130603all_15spf.tif` |
| Frames | 37,170 in total (507–2,765 per stack); 36,663 in the 16 stacks with labels |
| Frame size, mode | 1056 × 480 px, 8-bit grayscale (`L`), every stack; big-endian TIFF |
| Frame interval | 15 s (Info file and the `15spf` filename suffix; not read from TIFF metadata) |

- OBSERVED by eye in 16 trial frames: the 6 cameras are tiled 3 × 2 (about 352 × 240 px each, from
  the visible borders), numbered 1–3 on the top row and 4–6 below, each tile with a burned-in date
  and time (e.g. `13/06/02 15:39:28`). Exact tile edges in pixels are not measured yet.
- Recording dates 2013-05-18 to 2013-06-03 come from the filenames. Site, tank or field setting,
  time of day and lighting: UNKNOWN — requires confirmation.

## Labels (OBSERVED)
Frame counts, sizes and the label numbers below are reproduced by
`python scripts/inspect_tritonia.py --root <source>` (read-only, standard library; 2026-11-03 run on
`E:\Tritonia`: 17 stacks, 0 parse errors, 0 frame numbers beyond their stack, no duplicate track ids).
- Format: MTrackJ 1.5.0 data files (`<stack>.tif.mdf`, text). `Track <id> <colour> <flag>` starts a
  track; each `Point <id> <x> <y> <z> <t> <c>` is one hand-clicked slug position in montage pixels.
- `t` is the 1-based frame number: in all 16 files the largest `t` is ≤ the stack's frame count, and
  every `t` is an integer. `z` and `c` are 1 in every point.
- 956 tracks, 117,969 points; 2–1,241 points per track (median 48); 28–84 tracks per day.
- **Points only, no boxes, no class field.** Every track is a Tritonia; the species is not stated.
- Point IDs within a track skip numbers (e.g. 8 → 10): INFERRED, points deleted during annotation.
- Track `<flag>`: `true` on 780 tracks, `false` on 176. Meaning UNKNOWN — requires confirmation
  (MTrackJ documentation or Dr. Wyeth).
- `20130519…mdf` has a `Reference 1055.0 170.0 1.0` line; meaning UNKNOWN — requires confirmation.
- 2 points lie on or just outside the frame edge: `20130529` point 23 at x = −1, `20130531` point 336
  at y = 480. Kept as recorded; a converter must clip or flag them, never move them silently.
- `20130518all_15spf.tif` (507 frames) has no MDF file: unlabelled, not negative.
- Points vs pixels (OBSERVED, 2026-11-03, 150 random points per stack, 5 × 5 px mean at the point
  minus its median over frames 200–400 away): `20130523` +15.0 grey levels at the point's own frame,
  +15.6 one frame before, +3.2 eight frames after, about −4.5 at ±200; `20130602` +5.4 at d = 0,
  +3.5 at +8, +0.5 at +40. INFERRED: points sit on something brighter than usual that leaves within
  minutes, i.e. slugs appear bright, and the frame numbering lines up within about ±1 frame. By eye,
  crosses often sit beside or at the edge of a bright shape rather than at its centre, and many bright
  shapes have no cross; whether those are slugs is UNKNOWN — requires confirmation (Dr. Wyeth).

## Missing-label meaning
A frame with no point is **not** a confirmed slug-free frame. The Info file says the whole set was
tracked, but whether every visible slug in every frame was clicked is UNKNOWN — requires
confirmation. `missing_label_policy` for the MDF points: `pending` (excluded, never negative).

Hand-drawn boxes (D-029, batch 1, docs/TRITONIA_LABELLING.md): 150 frames in which every visible slug
is boxed, so in those frames only, a missing box means "no slug". Not yet delivered.

## Known limitations
- Timelapse (15 s per frame), not video: results are "sparse-frame/timelapse analysis", never
  "video tracking" (CLAUDE.md rule 9). Speeds are px per frame, or px/s only via the stated 15 s.
- No calibration: no physical units. Parallax from angled cameras makes px distances vary across
  the frame.
- Points are not boxes. A YOLO detector needs boxes drawn by hand; boxes of an assumed size around
  points would be invented labels (rule 8).
- Track IDs are per file. Whether the same slug keeps one ID across days is UNKNOWN.
- INFERRED from the consecutive dates and identical frame size: one setup recorded over 17 days.
  A split by day would then measure new-day generalization, not new sites or new cameras.

## Local inspection sample (not a dataset)
`C:\sample_data\Tritonia\`: 50 random frames (seed 0) from the 16 labelled stacks as PNG, with
per-frame point CSVs, `manifest.csv` and `stacks.csv`. 26 of the 50 frames have points (130 points in
total). For looking at the data only; outside the repository and not used for training or metrics.
