"""Filesystem scanning: enumerate images/labels and group them into sessions.

Read-only by design -- every function here only opens files for reading.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATASET_SOURCES, GOPRO_SUFFIX_PATTERN, VIDEO_ID_PATTERN

_VIDEO_ID_RE = re.compile(VIDEO_ID_PATTERN)
_GOPRO_SUFFIX_RE = re.compile(GOPRO_SUFFIX_PATTERN)


def extract_video_id(stem: str) -> str:
    """'06_HERR_02_CW_20200901_GP299683_00001' -> '06_HERR_02_CW_20200901_GP299683'."""
    m = _VIDEO_ID_RE.match(stem)
    if not m:
        raise ValueError(f"Filename stem does not match expected pattern: {stem!r}")
    return m.group("video_id")


def video_id_to_session_id(video_id: str) -> str:
    """Strip the trailing GoPro file ID: all recording files of one session map to one id."""
    return _GOPRO_SUFFIX_RE.sub("", video_id)


def count_instances(label_file: Path) -> int:
    """Number of non-blank annotation lines (= number of lobster boxes) in a txt file."""
    with open(label_file, "r") as f:
        return sum(1 for line in f if line.strip())


@dataclass
class SessionStats:
    session_id: str
    source_set: str
    video_ids: set = field(default_factory=set)
    n_images: int = 0
    n_labeled: int = 0
    n_instances: int = 0


def scan_all_sources() -> dict:
    """Returns {session_id: SessionStats}, scanning every configured dataset source."""
    sessions: dict[str, SessionStats] = {}

    for source_name, image_dir, label_dir in DATASET_SOURCES:
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Configured image dir does not exist: {image_dir}")

        png_files = sorted(image_dir.glob("*.png"))
        for png in png_files:
            stem = png.stem
            video_id = extract_video_id(stem)
            session_id = video_id_to_session_id(video_id)

            key = f"{source_name}::{session_id}"
            if key not in sessions:
                sessions[key] = SessionStats(session_id=session_id, source_set=source_name)
            stats = sessions[key]
            stats.video_ids.add(video_id)
            stats.n_images += 1

            if label_dir is not None:
                label_file = Path(label_dir) / f"{stem}.txt"
                if label_file.is_file():
                    stats.n_labeled += 1
                    stats.n_instances += count_instances(label_file)

    return sessions


def sessions_to_rows(sessions: dict) -> list:
    """Flatten SessionStats into plain dict rows suitable for CSV/JSON export."""
    rows = []
    for stats in sessions.values():
        rows.append(
            {
                "session_id": stats.session_id,
                "source_set": stats.source_set,
                "n_videos": len(stats.video_ids),
                "video_ids": ";".join(sorted(stats.video_ids)),
                "n_images": stats.n_images,
                "n_labeled": stats.n_labeled,
                "n_instances": stats.n_instances,
            }
        )
    rows.sort(key=lambda r: (r["source_set"], r["session_id"]))
    return rows
