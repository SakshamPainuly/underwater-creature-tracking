# Tritonia box labelling, batch 1 (D-029)

Goal: hand-drawn boxes around every visible sea slug in ~150 frames, so YOLO26s can be trained on
class `1: sea_slug`. The MTrackJ points only help you find slugs; they are never turned into boxes.

## 1. Build the batch (local PC, ~1 min)
```
python scripts/make_tritonia_label_batch.py --root D:\uwct_raw\tritonia --out D:\uwct_work\tritonia_label_batch1
```
150 frames (seed 0), spread over the 16 labelled days, only frames with at least one MTrackJ point,
at least 40 frames (10 min) apart within a day. `images/` is what you label; `hints/` shows the same
frame with a white cross on each clicked point; `points/` has the coordinates.

## 2. Label (makesense.ai, in the browser)
1. Open https://www.makesense.ai, "Get started", drop in all PNGs from `images/` (not `hints/`),
   choose **Object Detection**.
2. Load labels from `labels.txt` (`lobster`, `sea_slug`), in that order, so `sea_slug` exports as
   class `1`. Check the order before drawing.
3. For each frame, open the matching `hints/` image beside it and draw a **rectangle** with label
   `sea_slug` around every slug.
4. Export: Actions → Export annotations → "A .zip package containing files in YOLO format".
   Save the ZIP outside `D:\uwct_raw\` (e.g. `D:\uwct_work\tritonia_label_batch1_yolo.zip`).
   Keep a second copy; makesense keeps no work after the tab closes.

makesense.ai runs in the browser; per its own description images are not uploaded to a server.
UNKNOWN — confirm before using it if Dr. Wyeth's data must not leave the PC.

## 3. Rules (the same for every frame, so the labels mean one thing)
- Slugs appear as bright shapes. A cross is where the annotator clicked, often beside or at the edge
  of the slug rather than its centre: box the slug the cross belongs to, not the cross.
- **Box every visible slug**, including ones without a cross: a frame you finish is fully labelled,
  so a missing box later means "no slug there".
- Tight box around the visible body, not the shadow or trail.
- The frame is a montage of 6 cameras. Never draw a box across a camera border; a slug cut by a
  border gets one box per camera, around the part visible in each.
- A cross with no slug you can see (too dark, blurred, hidden): no box; write the frame name and
  track number in `notes.txt`. Never draw a box just because there is a cross.
- Unsure whether something is a slug: no box, note it in `notes.txt`.
- A lobster in frame (unlikely): box it as `lobster`.
- Do not skip frames. If a frame cannot be labelled at all, list it in `notes.txt` with the reason.

## 4. Hand back
The YOLO ZIP plus `notes.txt`. They become a new read-only label source and a new dataset version
(D-029). Nothing is trained before they are checked against `manifest.csv`.
