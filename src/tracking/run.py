"""Detector + tracker on one video -> tracks.csv + tracks_meta.json (D-028). Colab / lab GPU only.

ultralytics and cv2 are imported inside track_video(), so this module imports on the local PC but
only runs where requirements-gpu.txt is installed. Uses Ultralytics' built-in trackers
(bytetrack.yaml, botsort.yaml) with a registered YOLO or RT-DETR best.pt.

fps is the container-reported frame rate (OBSERVED from the file's metadata, not independently
confirmed); it is recorded with that source so analytics can state it.

NOT YET RUN: no project video exists (docs/DATASET.md). The first real run is a smoke test on a
short clip and must be checked by eye before any result is reported.
"""

from pathlib import Path

from dataset_audit.config import PROJECT_ROOT, RAW_ROOTS
from training.run_record import file_entry, git_state, utc_now
from tracking.schema import META_SCHEMA, check_output_dir, write_tracks_dir

TRACKERS = ("bytetrack.yaml", "botsort.yaml")
ARCHS = ("yolo", "rtdetr")


class TrackRunError(RuntimeError):
    pass


def _video_properties(video: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise TrackRunError(f"cannot open video {video}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps if fps and fps > 0 else None,
            "frame_count_reported": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()


def track_video(video: Path, weights: Path, out_dir: Path, *, arch: str = "yolo",
                tracker: str = "bytetrack.yaml", conf: float = 0.25, iou: float = 0.7, imgsz: int = 640,
                device=None, video_id: str | None = None, synthetic: bool = False) -> dict:
    """Track every frame of `video`; write the tracks folder; return its meta."""
    video, weights = Path(video), Path(weights)
    if arch not in ARCHS:
        raise TrackRunError(f"arch must be one of {ARCHS}")
    if tracker not in TRACKERS:
        raise TrackRunError(f"tracker must be one of {TRACKERS}")
    for p, what in ((video, "video"), (weights, "weights")):
        if not p.is_file():
            raise TrackRunError(f"{what} not found: {p}")
    out_dir = check_output_dir(out_dir, RAW_ROOTS)

    from ultralytics import RTDETR, YOLO

    props = _video_properties(video)
    video_id = video_id or video.stem
    model = (RTDETR if arch == "rtdetr" else YOLO)(str(weights))
    started = utc_now()
    rows = []
    n_frames = 0
    for frame_idx, result in enumerate(model.track(source=str(video), stream=True, persist=True, tracker=tracker,
                                                   conf=conf, iou=iou, imgsz=imgsz, device=device,
                                                   verbose=False)):
        n_frames = frame_idx + 1
        boxes = result.boxes
        if boxes is None or boxes.id is None:  # no confirmed tracks in this frame
            continue
        for (x1, y1, x2, y2), tid, c, cls in zip(boxes.xyxy.cpu().tolist(), boxes.id.int().cpu().tolist(),
                                                 boxes.conf.cpu().tolist(), boxes.cls.int().cpu().tolist()):
            rows.append({"video_id": video_id, "frame_idx": frame_idx, "track_id": tid, "class_id": cls,
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": c})
    if not n_frames:
        raise TrackRunError(f"no frames decoded from {video}")

    meta = {
        "schema": META_SCHEMA,
        "video_id": video_id,
        "synthetic": synthetic,
        "frame_width": props["width"],
        "frame_height": props["height"],
        "n_frames": n_frames,
        "fps": props["fps"],
        "fps_source": "container metadata (cv2.CAP_PROP_FPS)" if props["fps"] else None,
        "source": file_entry(video),
        "frame_count_reported": props["frame_count_reported"],
        "detector": {"arch": arch, "weights": file_entry(weights), "conf": conf, "iou": iou, "imgsz": imgsz},
        "tracker": {"name": tracker, "persist": True},
        "git": git_state(PROJECT_ROOT),
        "started_utc": started,
        "finished_utc": utc_now(),
    }
    write_tracks_dir(out_dir, meta, rows)
    return meta
