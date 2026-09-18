"""Count what Ultralytics itself reads from a YOLO dataset view (GPU environment only).

Uses Ultralytics' own label scanner -- the code path training uses -- so the counts show what a
training run would actually see. ultralytics is imported inside the functions: this module is
importable on the local PC, but calling it there raises ImportError.
"""

from pathlib import Path

COUNT_KEYS = ("images", "negative", "instances")


def ultralytics_split_counts(data_yaml: Path, splits) -> tuple:
    """Return (data dict from check_det_dataset, {split: {images, negative, instances}})."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_yaml), autodownload=False)
    cfg = get_cfg()
    seen = {}
    for split in splits:
        ds = build_yolo_dataset(cfg, data[split], batch=1, data=data, mode="val")
        n_boxes = [len(lb["cls"]) for lb in ds.labels]
        seen[split] = {"images": len(n_boxes), "negative": sum(n == 0 for n in n_boxes),
                       "instances": sum(n_boxes)}
    return data, seen


def counts_match(expected: dict, seen: dict) -> bool:
    """expected: one split's build_summary.json counts; seen: ultralytics_split_counts() output."""
    return all(seen[k] == expected[k] for k in COUNT_KEYS)
