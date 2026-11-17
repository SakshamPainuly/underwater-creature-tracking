"""Run a trained detector on single images and format the results.

Scope (CLAUDE.md rules 1, 3, 9):
- Read-only with respect to `data/raw/**`: images are opened for reading, never written back,
  and `refuse_raw_output` rejects any export path inside the raw root.
- Coordinates are pixels and confidences are unitless. No physical units are produced anywhere:
  the dataset has no documented calibration (docs/DATASET.md).
- No metric is computed here. Sources without labels (zExtra) support qualitative inspection only.

Heavy imports (torch/ultralytics/PIL) are function-local so this module stays importable in the
standard-library-only environment used by the tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# E001's operating point: the confidence that maximised mean F1 on val, frozen before the single
# test evaluation (experiments/records/E001/final_test_record.json, docs/EXPERIMENTS.md).
DEFAULT_CONF = 0.298
DEFAULT_IOU = 0.7
DEFAULT_IMGSZ = 640
DEFAULT_WEIGHTS = Path("weights") / "E001_best.pt"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Shown in the GUI with every result (CLAUDE.md section 3, docs/DATASET.md).
LIMITATIONS = (
    "The v1 test measures new recording sessions, not new sites: almost every test session "
    "shares a site, and usually a date, with a training session (D-007).",
    "Only E027 was scored at a new site (Zenodo sk, one site): its mAP50-95 fell from 0.697 on "
    "the v1 test to 0.389, with many false boxes. E001 and E004 were never scored there, so their "
    "performance at an unfamiliar site is UNKNOWN.",
    "Boxes and sizes are in pixels. The dataset has no documented calibration, so no length, "
    "area or speed in centimetres or metres can be derived (CLAUDE.md rule 9).",
    "zExtra images were never labelled, trained on or evaluated. Detections on them are "
    "qualitative observations, not a measurement of accuracy: no score can be computed without "
    "ground truth.",
    "One class only (0: lobster). Anything else in frame is outside what the model was taught.",
)


class DetectError(RuntimeError):
    """Raised with an actionable message when detection cannot proceed."""


@dataclass(frozen=True)
class Detection:
    """One predicted box, in pixel coordinates of the source image."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


def resolve_weights(weights: str | os.PathLike[str] | None = None) -> Path:
    """Locate the checkpoint, preferring an explicit path, then UWCT_WEIGHTS, then the default.

    No absolute path is baked in (CLAUDE.md section 7): a relative path resolves against the
    repository root.
    """
    raw = weights or os.environ.get("UWCT_WEIGHTS") or DEFAULT_WEIGHTS
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise DetectError(
            f"Checkpoint not found: {path}\n"
            "Download the run's best.pt from the run folder on Drive (docs/COLAB.md) and place it "
            f"at {DEFAULT_WEIGHTS}, or set UWCT_WEIGHTS to its location. Weights are never "
            "committed (CLAUDE.md section 5)."
        )
    return path


def refuse_raw_output(path: str | os.PathLike[str]) -> Path:
    """Return `path`, or raise if it lies inside the read-only raw root (CLAUDE.md rule 1)."""
    from dataset_audit.config import BIOLOGY_ROOT  # local import: keeps this module standalone

    resolved = Path(path).expanduser().resolve()
    raw_root = Path(BIOLOGY_ROOT).expanduser().resolve()
    if resolved == raw_root or raw_root in resolved.parents:
        raise DetectError(
            f"Refusing to write inside the read-only raw dataset: {resolved}\n"
            "Choose a destination outside data/raw/** (CLAUDE.md rule 1)."
        )
    return resolved


def list_source_images(source: str = "zExtra", limit: int | None = None) -> list[Path]:
    """List image files of a configured dataset source, sorted by name.

    `zExtra` is the honest "unseen" set for qualitative review: it is excluded from every dataset
    version, so the model has never been trained, tuned or evaluated on it (D-003).
    """
    from dataset_audit.config import DATASET_SOURCES

    for name, image_dir, _labels in DATASET_SOURCES:
        if name.lower() == source.lower():
            directory = Path(image_dir)
            if not directory.is_dir():
                raise DetectError(
                    f"Source '{name}' is not available at {directory}\n"
                    "Set UWCT_RAW_ROOT to the folder containing 'Set 100', 'Set 2000' and "
                    "'zExtra Photos'."
                )
            files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            return files[:limit] if limit else files
    known = ", ".join(name for name, _i, _l in DATASET_SOURCES)
    raise DetectError(f"Unknown source '{source}'. Known sources: {known}.")


def list_folder_images(folder: str | os.PathLike[str], limit: int | None = None) -> list[Path]:
    """List the images of any folder, sorted by name. Read-only.

    A relative path resolves against the project root. Used for the "folder" sources of
    configs/gui_species.json, so the GUI also runs on a machine without the raw dataset.
    """
    path = Path(folder)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_dir():
        raise DetectError(
            f"Image folder not found: {path}\n"
            "Create it and put images in it, or change the path in configs/gui_species.json."
        )
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise DetectError(f"No images found in {path}")
    return files[:limit] if limit else files


def load_model(weights: str | os.PathLike[str] | None = None) -> Any:
    """Load the Ultralytics model for CPU inference. Raises DetectError with a usable message."""
    path = resolve_weights(weights)
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on the local environment
        raise DetectError(
            "ultralytics is not installed in this environment.\n"
            "The GUI needs the local CPU environment: see requirements-gui.txt. Do not install it "
            "outside a virtual environment (CLAUDE.md rule 3)."
        ) from exc
    try:
        return YOLO(str(path))
    except Exception as exc:  # pragma: no cover - corrupt or incompatible checkpoint
        raise DetectError(f"Could not load the checkpoint {path}: {exc}") from exc


def detect(
    model: Any,
    image: Any,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
) -> list[Detection]:
    """Detect on one image and return boxes sorted by descending confidence (deterministic).

    `image` is anything Ultralytics accepts for a single image: a path, a PIL image or an array.
    Nothing is written to disk.
    """
    if not 0.0 < conf <= 1.0:
        raise DetectError(f"conf must be in (0, 1], got {conf}")
    if not 0.0 < iou <= 1.0:
        raise DetectError(f"iou must be in (0, 1], got {iou}")

    results = model.predict(
        image, conf=conf, iou=iou, imgsz=imgsz, device="cpu", verbose=False, save=False
    )
    if not results:
        return []
    result = results[0]
    names = getattr(result, "names", {}) or {}

    detections: list[Detection] = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        detections.append(
            Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(box.conf[0]),
                class_id=class_id,
                class_name=str(names.get(class_id, class_id)),
            )
        )
    detections.sort(key=lambda d: (-d.confidence, d.x1, d.y1))
    return detections


def to_yolo_lines(detections: Sequence[Detection], image_width: int, image_height: int) -> list[str]:
    """Format detections as YOLO label lines: `class x_center y_center width height`, normalised.

    This is the format the project's own labels use (docs/DATASET.md), so an exported file opens
    directly in LabelImg for a human to correct. Pure function: no I/O, no heavy imports.
    """
    if image_width <= 0 or image_height <= 0:
        raise DetectError(f"image size must be positive, got {image_width}x{image_height}")

    lines: list[str] = []
    for d in detections:
        xc = ((d.x1 + d.x2) / 2) / image_width
        yc = ((d.y1 + d.y2) / 2) / image_height
        w = d.width / image_width
        h = d.height / image_height
        clamped = [min(max(v, 0.0), 1.0) for v in (xc, yc, w, h)]
        lines.append(f"{d.class_id} " + " ".join(f"{v:.6f}" for v in clamped))
    return lines


def detections_csv_rows(
    detections: Iterable[Detection], image_name: str, conf: float, imgsz: int
) -> list[dict[str, object]]:
    """One row per detection, in pixels, with the settings that produced it. Pure function."""
    rows: list[dict[str, object]] = []
    for index, d in enumerate(detections, start=1):
        rows.append(
            {
                "image": image_name,
                "detection": index,
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
                "x1_px": round(d.x1, 1),
                "y1_px": round(d.y1, 1),
                "x2_px": round(d.x2, 1),
                "y2_px": round(d.y2, 1),
                "width_px": round(d.width, 1),
                "height_px": round(d.height, 1),
                "area_px2": round(d.area, 1),
                "conf_threshold": conf,
                "imgsz": imgsz,
            }
        )
    return rows


# Windows has arial.ttf; macOS names the same font Arial.ttf (Pillow's lookup is case-sensitive)
# and always has Helvetica; most Linux systems have DejaVu.
LABEL_FONTS = ("arial.ttf", "Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf")


def _label_font(size: int) -> Any:
    """A scalable font for box labels on any platform; falls back to Pillow's built-in font."""
    from PIL import ImageFont

    for name in LABEL_FONTS:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    try:  # pragma: no cover - depends on the fonts installed
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:  # pragma: no cover
        return ImageFont.load_default()


def draw_detections(image: Any, detections: Sequence[Detection], show_confidence: bool = True) -> Any:
    """Return a copy of `image` (PIL) with boxes and labels drawn. Never modifies the original."""
    from PIL import ImageDraw

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    # Scale line and text with image size so a 1920x1080 frame and a thumbnail both stay readable.
    thickness = max(2, round(min(canvas.size) / 300))
    font = _label_font(max(14, round(min(canvas.size) / 45)))

    for d in detections:
        draw.rectangle((d.x1, d.y1, d.x2, d.y2), outline=(0, 229, 160), width=thickness)
        if not show_confidence:
            continue
        label = f"{d.class_name} {d.confidence:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = right - left, bottom - top
        pad = thickness * 2
        box_top = max(0.0, d.y1 - text_h - 2 * pad)
        draw.rectangle(
            (d.x1, box_top, d.x1 + text_w + 2 * pad, box_top + text_h + 2 * pad),
            fill=(0, 229, 160),
        )
        draw.text((d.x1 + pad, box_top + pad), label, fill=(8, 20, 16), font=font)
    return canvas


# ---------------------------------------------------------------------------
# Built dataset splits (data/yolo/**): labelled images, read-only.
# ---------------------------------------------------------------------------

# Rebuilt per machine, never committed and never copied between machines (CLAUDE.md section 4).
DEFAULT_YOLO_DATASET = Path("data") / "yolo" / "lobster_v1"
# The v4 view (D-031) holds the only copy of the held-out Zenodo sk split.
V4_YOLO_DATASET = Path("data") / "yolo" / "lobster_mixed_v4"
HELD_OUT_SPLITS = ("test", "test_zenodo_sk")

# What each split can honestly tell you. Shown in the GUI so a viewer cannot mistake one for
# another (CLAUDE.md rule 7, section 3).
SPLIT_STATUS = {
    "test": (
        "HELD OUT. Never used for training, tuning, threshold choice or model selection "
        "(CLAUDE.md rule 7). Each declared final candidate (E001, E004, E005, E027) was evaluated "
        "on it exactly once, and each evaluation is logged in docs/EXPERIMENTS.md. Look, but do "
        "not tune: changing the confidence slider while watching these images and keeping the new "
        "value is threshold selection on test, and it would invalidate the reported numbers."
    ),
    "test_zenodo_sk": (
        "HELD OUT, NEW SITE: the Zenodo sk frames (another country and camera setup, D-031), never "
        "used for training, tuning, threshold choice or model selection (CLAUDE.md rule 7). Only "
        "E027 was evaluated on it, once, on 2026-11-13; E001 and E004 were never scored here. Labels "
        "come from the Zenodo dataset; 285 of the 428 frames are lobster-free. Look, but do not "
        "tune: keeping a slider value found here is threshold selection on test."
    ),
    "val": (
        "Used for threshold choice and model selection. The frozen operating point was chosen "
        "here, so this is the legitimate split on which to explore what the sliders do."
    ),
    "train": (
        "The model was fitted on these images. Good results here are memorisation, not "
        "generalization, and say nothing about unseen data. E027 was trained on 1,312 of these "
        "1,433 v1 frames (121 were dropped by Harsh's label review) plus derived copies."
    ),
}


def dataset_for_split(split: str) -> Path:
    """Built dataset holding `split`: Zenodo sk exists only in the v4 view, everything else in v1."""
    return V4_YOLO_DATASET if split == "test_zenodo_sk" else DEFAULT_YOLO_DATASET


@dataclass(frozen=True)
class TruthBox:
    """One ground-truth box, converted to pixel coordinates of its image."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str = "lobster"

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class Match:
    """Per-image agreement between predictions and ground truth. Counts only, never a metric.

    Aggregating these across the test split would be a test evaluation, which may only be run for
    a pre-declared final candidate and must be logged (CLAUDE.md rule 7, docs/EXPERIMENTS.md).
    """

    matched: tuple[tuple[int, int, float], ...]  # (detection index, truth index, IoU)
    unmatched_detections: tuple[int, ...]  # predicted, no ground-truth partner
    unmatched_truth: tuple[int, ...]  # labelled, nothing predicted over it
    iou_threshold: float


def resolve_dataset_dir(dataset_dir: str | os.PathLike[str] | None = None) -> Path:
    """Locate the built YOLO dataset, preferring an explicit path, then UWCT_YOLO_DIR."""
    raw = dataset_dir or os.environ.get("UWCT_YOLO_DIR") or DEFAULT_YOLO_DATASET
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_dir():
        raise DetectError(
            f"Built dataset not found: {path}\n"
            "Build it on this machine with `python scripts/prepare_yolo_dataset.py --version v1` "
            "(`--version v4` for the Zenodo sk split; never copied between machines, CLAUDE.md "
            "section 4), or set UWCT_YOLO_DIR."
        )
    return path


def list_split_images(
    split: str = "test",
    dataset_dir: str | os.PathLike[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    """List the images of a built split, sorted by name. Read-only.

    These files may be hardlinks or symlinks to `data/raw/**` (CLAUDE.md rule 2), so they are
    opened for reading only and never written back.
    """
    if split not in SPLIT_STATUS:
        known = ", ".join(sorted(SPLIT_STATUS))
        raise DetectError(f"Unknown split '{split}'. Known splits: {known}.")
    directory = resolve_dataset_dir(dataset_dir) / "images" / split
    if not directory.is_dir():
        raise DetectError(f"Split '{split}' has no image folder at {directory}")
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise DetectError(f"No images found in {directory}")
    return files[:limit] if limit else files


def label_path_for(image_path: str | os.PathLike[str]) -> Path:
    """Map `.../images/<split>/<stem>.<ext>` to `.../labels/<split>/<stem>.txt`."""
    image = Path(image_path)
    parts = list(image.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    else:
        raise DetectError(f"Not a path inside a built YOLO split (no 'images' folder): {image}")
    return Path(*parts).with_suffix(".txt")


def parse_yolo_labels(
    text: str, image_width: int, image_height: int, class_names: dict[int, str] | None = None
) -> list[TruthBox]:
    """Parse YOLO label lines into pixel-space boxes. Pure function, no I/O.

    Accepts the project's own label format: `class x_center y_center width height`, normalised.
    A trailing confidence column, as written by some tools, is ignored. Blank lines are skipped,
    which is how a confirmed-negative image is stored: an empty file, not a missing one.
    """
    if image_width <= 0 or image_height <= 0:
        raise DetectError(f"image size must be positive, got {image_width}x{image_height}")

    names = class_names or {0: "lobster"}
    boxes: list[TruthBox] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 5:
            raise DetectError(
                f"Malformed label on line {number}: expected 5 fields "
                f"(class x_center y_center width height), got {len(fields)}: {stripped!r}"
            )
        try:
            class_id = int(float(fields[0]))
            xc, yc, w, h = (float(v) for v in fields[1:5])
        except ValueError as exc:
            raise DetectError(f"Malformed label on line {number}: {stripped!r} ({exc})") from exc

        boxes.append(
            TruthBox(
                x1=(xc - w / 2) * image_width,
                y1=(yc - h / 2) * image_height,
                x2=(xc + w / 2) * image_width,
                y2=(yc + h / 2) * image_height,
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
            )
        )
    return boxes


def load_ground_truth(
    image_path: str | os.PathLike[str], image_width: int, image_height: int
) -> list[TruthBox] | None:
    """Read the ground truth for a built-split image.

    Returns the boxes, an empty list for a confirmed-negative image (empty label file), or
    `None` when no label file exists. `None` is never silently read as "no lobster": only a
    source whose missing-label policy is `negative` may support that inference, and the built
    splits contain a label file for every image (CLAUDE.md rule 6).
    """
    path = label_path_for(image_path)
    if not path.is_file():
        return None
    return parse_yolo_labels(path.read_text(encoding="utf-8"), image_width, image_height)


def _iou(a: Detection | TruthBox, b: Detection | TruthBox) -> float:
    """Intersection over union of two pixel boxes. Pure function."""
    left, right = max(a.x1, b.x1), min(a.x2, b.x2)
    top, bottom = max(a.y1, b.y1), min(a.y2, b.y2)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def match_detections(
    detections: Sequence[Detection], truth: Sequence[TruthBox], iou_threshold: float = 0.5
) -> Match:
    """Greedily pair predictions with ground truth by IoU, highest confidence first.

    Describes one image so a human can see which boxes agree. It is not an evaluation: no
    precision, recall or mAP is produced here, and these counts must not be aggregated over the
    test split (CLAUDE.md rule 7).
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise DetectError(f"iou_threshold must be in (0, 1], got {iou_threshold}")

    order = sorted(range(len(detections)), key=lambda i: -detections[i].confidence)
    taken: set[int] = set()
    matched: list[tuple[int, int, float]] = []

    for d_index in order:
        best_t, best_iou = -1, 0.0
        for t_index, t in enumerate(truth):
            if t_index in taken or t.class_id != detections[d_index].class_id:
                continue
            score = _iou(detections[d_index], t)
            if score > best_iou:
                best_t, best_iou = t_index, score
        if best_t >= 0 and best_iou >= iou_threshold:
            taken.add(best_t)
            matched.append((d_index, best_t, best_iou))

    matched.sort()
    paired = {d for d, _t, _s in matched}
    return Match(
        matched=tuple(matched),
        unmatched_detections=tuple(i for i in range(len(detections)) if i not in paired),
        unmatched_truth=tuple(i for i in range(len(truth)) if i not in taken),
        iou_threshold=iou_threshold,
    )


def draw_ground_truth(image: Any, truth: Sequence[TruthBox]) -> Any:
    """Return a copy of `image` (PIL) with ground-truth boxes drawn. Never modifies the original.

    Drawn underneath the predictions: compose as
    `draw_detections(draw_ground_truth(img, truth), detections)`.
    """
    from PIL import ImageDraw

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    thickness = max(2, round(min(canvas.size) / 300))
    for t in truth:
        # Amber, with a dark dashed overlay on the horizontal edges, so a ground-truth box stays
        # distinguishable from a prediction sitting exactly on top of it.
        draw.rectangle((t.x1, t.y1, t.x2, t.y2), outline=(255, 176, 0), width=thickness)
        step = max(2, thickness * 6)
        for x in range(int(t.x1), int(t.x2), step):
            end = min(x + step // 2, t.x2)
            draw.line((x, t.y1, end, t.y1), fill=(24, 16, 4), width=max(1, thickness // 2))
            draw.line((x, t.y2, end, t.y2), fill=(24, 16, 4), width=max(1, thickness // 2))
    return canvas
