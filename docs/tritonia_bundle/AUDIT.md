# AUDIT: E010 Tritonia bundle (checked 2026-11-17)

- **PASS**: the test set was never used to choose the model, epoch or threshold. E010 was declared final
  candidate on val point AP (0.496 > 0.403) on 2026-11-12, before its single test run. best.pt comes from val
  fitness. Conf 0.16 = max point F1 on val (docs/EXPERIMENTS.md declarations and test-evaluation log). The bundle
  choice on 2026-11-17 was also made on val and the declaration.
- **PASS** (same video, minute or day): the split unit is the recording day; v7 split_manifest has 16 days, each in
  exactly one split.
- **FAIL** (same site and cameras): all splits come from one setup with the same 6 cameras and scene, so
  near-identical backgrounds appear in train, val and test. The scores measure new days, not new sites.
- **PASS**: every label file is valid YOLO format and its boxes sit inside the image. 18 sample label files, 90
  boxes, 5 fields each, all boxes inside [0, 1]; each image and label sha256 equals the frozen v7 image_manifest.
- **UNKNOWN**: confirmed-empty images have an empty label file. There are no confirmed-empty frames: all 150 v7
  frames are positive, and missing clicks are not negatives (missing_label_policy pending, D-029). No empty
  label files are included.
- **PASS**: the class list in the labels matches model.names. Labels use only class 1. model.names
  `{0: lobster, 1: sea_slug}` equals v7 VERSION.json class_names. Class 0 has no training data.
- **PASS**: the scores in REPORT.md match the run's own result files. A script compared 20 numbers with
  run_record.json, val_point_scores.json and test_point_scores.json: 0 mismatches. Ultralytics test mAP is
  UNKNOWN (never run).
- **PASS**: the checkpoint's sha256 in run_record.json matches the file in the bundle. `48d72b79…` is the same in
  the bundle file, the bundle run_record.json and the original registered record.
- **PASS**: the model loads and predicts on CPU with the listed Ultralytics version (8.4.155, torch 2.14.0+cpu,
  Windows). A second CPU run gave identical boxes on 28 of 28 images.
