# Tritonia box labelling, batch 1 (D-029)

Goal: hand-drawn boxes around every visible sea slug in ~150 frames, so YOLO26s can be trained on
class `1: sea_slug`. The MTrackJ points only help you find slugs; they are never turned into boxes.

## 1. Build the batch (local PC, ~1 min)
```
python scripts/make_tritonia_label_batch.py --root D:\uwct_tritonia\raw\tritonia --out D:\uwct_tritonia\work\tritonia_label_batch1
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
   Save the ZIP outside `D:\uwct_tritonia\raw\` (e.g. `D:\uwct_tritonia\work\tritonia_label_batch1_yolo.zip`).
   Keep a second copy; makesense keeps no work after the tab closes.

makesense.ai runs in the browser; per its own description images are not uploaded to a server.
UNKNOWN — confirm before using it if Dr. Wyeth's data must not leave the PC.

## 3. Rules (the same for every frame, so the labels mean one thing)
- What a slug looks like (OBSERVED 2026-11-03, crops following 4 tracks over 20 min in
  `20130519` frame 454): a **small, faint, light smudge, about 10–25 px**, right beside the cross,
  moving slowly with it. Not the dark patches (they do not move: seabed or shadow) and not a large area.
- **Box the slug of every cross, and nothing without a cross.** Dr. Wyeth's note says the whole
  data set was tracked by hand, so bright specks without a cross are INFERRED not to be slugs
  (debris, reflections). A finished frame is complete in that sense: a missing box means "no
  tracked slug there". UNKNOWN — requires confirmation: whether every slug was tracked.
- Tight box around the smudge (about 15–25 px), not the shadow, trail or surrounding seabed.
- Use the labeller (`scripts/label_tritonia.py`): **h** shows the crosses, **c** contrast (on by
  default, display only), **z** zoom. A box across a camera border is refused as you draw it.
  Drag a box's edge or corner to resize it (refused the same way if it would cross a border);
  **u** undoes the last change on the frame (a drawn, resized or deleted box).
- Suggested boxes (D-033): a frame not yet labelled opens with orange boxes suggested from the
  image next to each cross, and a red circle on each cross that got none. Check every one: delete
  wrong ones (often faint texture in washed-out views), resize loose ones, draw missing ones (look
  at every red circle). Leaving the frame saves what is on screen as your reviewed labels.
  **s** replaces a frame's boxes with fresh suggestions (**u** undoes it).
- Checking only the val and test frames (54 frames, D-030 day split):
  `python scripts/label_tritonia.py --batch D:\uwct_tritonia\work\tritonia_label_batch1 --split val,test`.
  It starts at the first frame not yet checked against suggestions; **x** still exports all frames.
- The frame is a montage of 6 cameras. Never draw a box across a camera border; a slug cut by a
  border gets one box per camera, around the part visible in each.
- A cross with no slug you can see (too dark, blurred, hidden): no box, and a `note` line in
  `notes.txt` (format below). Never draw a box just because there is a cross.
- Unsure whether something is a slug, or a frame you cannot label completely: mark the whole frame
  `exclude`. It then never reaches a dataset, so its missing boxes are never read as "no slug".
- A lobster in frame (unlikely): box it as `lobster`.
- Do not skip frames silently: every frame is either labelled or marked `exclude`.

`notes.txt` (in the batch folder), one line per issue; `#` lines are comments:
```
20130521all_15spf_t0123 | note | cross of track 12 has no visible slug
20130524all_15spf_t1450 | exclude | camera 4 too dark to tell slugs from rocks
```
`scripts/ingest_tritonia_boxes.py` refuses lines in any other form.

## 4. Hand back
The YOLO ZIP plus `notes.txt` (create it even if it holds only a comment). Then, on the local PC:
```
python scripts/ingest_tritonia_boxes.py --zip D:\uwct_tritonia\work\tritonia_batch1_yolo.zip --notes D:\uwct_tritonia\work\tritonia_label_batch1\notes.txt --batch D:\uwct_tritonia\work\tritonia_label_batch1 --out D:\uwct_tritonia\raw\tritonia_boxes_batch1
```
It checks every file against `manifest.csv` (one label file per frame unless excluded, classes 0/1,
boxes inside the frame and not across a camera border) and writes nothing if anything is wrong.
Next steps (dataset v3, Colab training): D-030.
