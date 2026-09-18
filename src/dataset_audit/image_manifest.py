"""Build a per-image manifest joining raw-file scan results with a dataset version's
session-level split (<version dir>/split_manifest.csv).

This is read-only with respect to data/raw/biology/ -- every image and label
file is only ever opened for reading. Output goes to the same version directory.

Label status per image (never invented, only observed). A missing label file is
interpreted only through the source's explicit MISSING_LABEL_POLICY (config.py):
    "positive"          -- has a label file, parses cleanly, >=1 instance
    "negative_confirmed"-- no label file, source policy "negative" (Set2000: the
                           docs state no label file = no lobster for Set2000)
    "negative_pending"  -- no label file, source policy "pending" (zExtra: NOT a
                           confirmed negative -- annotation status unresolved)
    "invalid"           -- a label file exists but failed validation, or a label is
                           missing in a source with policy "error" (Set100); flagged,
                           never silently treated as positive or negative
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATASET_SOURCES, MISSING_LABEL_POLICY
from .scan import extract_video_id, video_id_to_session_id

_STATUS_FOR_MISSING_LABEL = {"negative": "negative_confirmed", "pending": "negative_pending", "error": "invalid"}


def label_status_for(source_name: str, has_label: bool, issues: list) -> tuple:
    """(label_status, issues) for one image; missing labels follow the source's explicit policy."""
    if has_label:
        return ("invalid" if issues else "positive"), issues
    policy = MISSING_LABEL_POLICY.get(source_name)
    if policy is None:
        raise ValueError(f"no missing-label policy configured for source {source_name!r} "
                         "(add it to MISSING_LABEL_POLICY in config.py)")
    extra = ["missing_label_not_allowed_for_source"] if policy == "error" else []
    return _STATUS_FOR_MISSING_LABEL[policy], issues + extra


@dataclass
class LabelValidation:
    n_instances: int = 0
    issues: list = field(default_factory=list)


def validate_label_file(path: Path) -> LabelValidation:
    result = LabelValidation()
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) == 0:
        result.issues.append("empty_label_file")
        return result

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            result.issues.append(f"malformed_line_field_count_{len(parts)}")
            continue

        class_id = parts[0]
        if class_id != "0":
            result.issues.append(f"unexpected_class_id_{class_id}")

        try:
            x, y, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            result.issues.append("non_numeric_coordinate")
            continue

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            result.issues.append("center_out_of_range")
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            result.issues.append("invalid_box_dimensions")
        if (x - w / 2) < -1e-6 or (x + w / 2) > 1 + 1e-6 or (y - h / 2) < -1e-6 or (y + h / 2) > 1 + 1e-6:
            result.issues.append("box_extends_outside_image")

        result.n_instances += 1

    return result


def build_image_manifest(split_map: dict) -> tuple:
    """Returns (image_rows, orphan_label_rows).

    orphan_label_rows is a list of anomaly dicts distinguished by "type":
    "orphan_label_no_matching_image" (a .txt with no matching .png in the same source)
    and "duplicate_filename" (the same image filename found in more than one source).
    """
    image_rows = []
    orphan_label_rows = []
    seen_filenames = {}  # filename -> source_set

    for source_name, image_dir, label_dir in DATASET_SOURCES:
        image_dir = Path(image_dir)
        png_stems = {p.stem for p in image_dir.glob("*.png")}

        for png in sorted(image_dir.glob("*.png")):
            stem = png.stem
            video_id = extract_video_id(stem)
            session_id = video_id_to_session_id(video_id)
            split = split_map.get(session_id, "UNKNOWN_SESSION_NOT_IN_MANIFEST")

            label_path = Path(label_dir) / f"{stem}.txt" if label_dir is not None else None
            has_label = label_path is not None and label_path.is_file()

            n_instances = 0
            issues = []
            if has_label:
                validation = validate_label_file(label_path)
                n_instances = validation.n_instances
                issues = validation.issues
            label_status, issues = label_status_for(source_name, has_label, issues)

            image_rows.append(
                {
                    "filename": png.name,
                    "source_set": source_name,
                    "session_id": session_id,
                    "video_id": video_id,
                    "split": split,
                    "has_label_file": has_label,
                    "n_instances": n_instances,
                    "label_status": label_status,
                    "issues": ";".join(issues),
                }
            )

            if png.name in seen_filenames and seen_filenames[png.name] != source_name:
                orphan_label_rows.append(  # duplicate_filename anomaly (see function docstring)
                    {"type": "duplicate_filename", "filename": png.name,
                     "sources": f"{seen_filenames[png.name]},{source_name}"}
                )
            seen_filenames[png.name] = source_name

        # orphan labels: a .txt with no matching .png in this same source
        if label_dir is not None:
            label_dir = Path(label_dir)
            for txt in sorted(label_dir.glob("*.txt")):
                if txt.stem not in png_stems:
                    orphan_label_rows.append(
                        {"type": "orphan_label_no_matching_image", "filename": txt.name,
                         "sources": source_name}
                    )

    return image_rows, orphan_label_rows


def load_split_map(split_manifest_path: Path) -> dict:
    with open(split_manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["session_id"]: row["split"] for row in reader}
