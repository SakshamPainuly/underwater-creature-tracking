"""Streamlit GUI: run the trained detector on single images and inspect the result.

    .venv/Scripts/streamlit run scripts/gui_app.py        (Windows, local CPU environment)
    streamlit run scripts/gui_app.py                       (any machine with requirements-gui.txt)

Thin layer only (CLAUDE.md section 7, PROMPT.md GUI checklist): every decision lives in
src/detect/predict.py. This file renders widgets and calls it.

Sources are read-only. Nothing is written to the raw dataset, nothing is written to the built
splits (whose image files may be hardlinks into the raw root, CLAUDE.md rule 2). Exports are
produced in memory and downloaded by the browser, so the app never writes to disk at all.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from detect import (  # noqa: E402  (path setup must precede the import)
    ENHANCED_SOURCE,
    HELD_OUT_SPLITS,
    IOU_SOURCE,
    MODELS,
    checkpoint_image_source,
    dataset_for_split,
    index_all_enhanced,
    LIMITATIONS,
    MATCH_IOU_SOURCE,
    SPLIT_STATUS,
    DetectError,
    detect,
    detections_csv_rows,
    draw_detections,
    draw_ground_truth,
    list_source_images,
    list_split_images,
    load_ground_truth,
    load_model,
    match_detections,
    recorded_scores,
    resolve_weights,
    sweet_spot_for,
    to_yolo_lines,
)

st.set_page_config(page_title="Lobster detector", page_icon="🦞", layout="wide")

# Streamlit drops a widget's value on any run that stops before drawing it (e.g. while a missing
# checkpoint stops the page). Re-assigning keeps the chosen source and images across such runs.
for _key in [k for k in st.session_state
             if k in ("source", "model_choice") or str(k).startswith(("pick_", "compare_"))]:
    st.session_state[_key] = st.session_state[_key]

# label -> (kind, name). "raw" reads a configured raw source; "split" reads a built YOLO split.
SOURCES = {
    "zExtra — never seen (unlabelled)": ("raw", "zExtra"),
    "test split — held out (labelled)": ("split", "test"),
    "Zenodo sk — new site, held out (labelled)": ("split", "test_zenodo_sk"),
    "val split — used for tuning (labelled)": ("split", "val"),
    "train split — fitted on (labelled)": ("split", "train"),
}
MODEL_LABELS = [m.label for m in MODELS]
NO_COMPARE = "none"


@st.cache_resource(show_spinner="Loading the detector...")
def _model(weights: str):
    """Load once per weights path and keep it in memory across reruns."""
    return load_model(weights)


@st.cache_data(show_spinner=False)
def _images(kind: str, name: str) -> list[str]:
    if kind == "raw":
        return [str(p) for p in list_source_images(name)]
    return [str(p) for p in list_split_images(name, dataset_for_split(name))]


@st.cache_data(show_spinner=False)
def _enhanced_index() -> dict[str, Path]:
    """Filename -> enhanced image, built once per session (read-only)."""
    return index_all_enhanced()


@st.cache_data(show_spinner="Looking up the checkpoint's validated settings...")
def _sweet_spot(weights: str, size: int, mtime: float):
    """Hash the checkpoint once per file version (size + mtime) and look up its validated settings."""
    return sweet_spot_for(weights)


IMGSZ_CHOICES = [320, 480, 640, 960, 1280]


def _apply_sweet_spot(spot) -> None:
    """Put every slider at its validated value. Runs before the widgets are drawn."""
    st.session_state["conf"] = spot.conf
    st.session_state["iou"] = spot.iou
    st.session_state["imgsz"] = spot.imgsz
    st.session_state["match_iou"] = spot.match_iou


def _checkpoint(path_text: str):
    """(path, stat) of a checkpoint, or raise DetectError with where to find it."""
    path = resolve_weights(path_text)
    return path, path.stat()


def _counts(detections, truth, match_iou: float) -> None:
    """Per-image agreement with the labels. Counts only, never a metric (CLAUDE.md rule 7)."""
    st.metric("Detections", len(detections))
    if truth is None:
        return
    match = match_detections(detections, truth, iou_threshold=match_iou)
    c1, c2, c3 = st.columns(3)
    c1.metric("Agree", len(match.matched), help="Prediction over a labelled box.")
    c2.metric("Extra", len(match.unmatched_detections), help="Predicted, nothing labelled there.")
    c3.metric("Missed", len(match.unmatched_truth), help="Labelled, nothing predicted over it.")


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        return b""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


st.title("Lobster detector")
st.caption(
    "Qualitative inspection of a trained YOLO detector. What you see here is an observation, "
    "not a measurement: no accuracy score is computed in this app."
)

with st.sidebar:
    # The model decides the image type: E027 and E001 see raw images, E004 sees Harsh's enhanced
    # copies of the same images (D-017). Each is shown only the image type it was evaluated on.
    st.header("Model")
    chosen_model = MODELS[MODEL_LABELS.index(st.radio(
        "Detector", MODEL_LABELS, key="model_choice",
        help="E027 is the final model. E001 is the first baseline, the GUI's earlier default. "
             "E004 is shown Harsh's enhanced images.",
    ))]
    st.caption(chosen_model.description)
    st.caption(recorded_scores(chosen_model.experiment_id).caption())
    source = chosen_model.image_source
    enhanced = source == ENHANCED_SOURCE
    paired_weights, paired_exp = chosen_model.weights, chosen_model.experiment_id

    weights_input = st.text_input(
        "Checkpoint",
        value=str(paired_weights).replace("\\", "/"),
        key=f"weights_{paired_exp}",  # each model remembers its own checkpoint
        help=f"{paired_exp}'s best.pt. Relative to the repository root, or an absolute path. "
             "Weights are never committed.",
    )
    try:
        weights_path, stat = _checkpoint(weights_input)
        st.caption(f"{weights_path.name} — {stat.st_size / 1e6:.1f} MB")
    except DetectError as exc:
        st.error(str(exc))
        st.info(f"{paired_exp}'s best.pt location and sha256 are in "
                f"`experiments/records/{paired_exp}/run_record.json`. Put the file at "
                f"`{str(paired_weights).replace(chr(92), '/')}`, or pick another model.")
        st.stop()

    # Optional second model on the same image, at that model's own validated settings.
    compare_choices = [NO_COMPARE] + [m.label for m in MODELS
                                      if m.image_source == source and m.experiment_id != paired_exp]
    compare_model, compare_path, compare_sweet = None, None, None
    if len(compare_choices) > 1:
        compare_label = st.selectbox(
            "Compare side by side with", compare_choices, key=f"compare_{paired_exp}",
            help="Runs a second model on the same image, at that model's own validated settings "
                 "(its val-chosen confidence and training size). The sliders below move only the "
                 "first model.",
        )
        if compare_label != NO_COMPARE:
            compare_model = MODELS[MODEL_LABELS.index(compare_label)]
            try:
                compare_path, compare_stat = _checkpoint(str(compare_model.weights))
                compare_sweet = _sweet_spot(str(compare_path), compare_stat.st_size, compare_stat.st_mtime)
            except DetectError as exc:
                st.warning(f"No side-by-side: {exc}")
                compare_model = None

    # Validated slider values for THIS checkpoint (sha256 -> experiments/records/<id>/).
    sweet = _sweet_spot(str(weights_path), stat.st_size, stat.st_mtime)
    if st.session_state.get("sweet_spot_for") != (str(weights_path), stat.st_size, stat.st_mtime):
        _apply_sweet_spot(sweet)  # new checkpoint (or first run): start at its sweet spot
        st.session_state["sweet_spot_for"] = (str(weights_path), stat.st_size, stat.st_mtime)
    if sweet.known:
        st.caption(f"Recognised as **{sweet.experiment_id}** from its sha256.")
        trained_on = checkpoint_image_source(sweet.experiment_id)
        if trained_on and trained_on != source:
            st.warning(f"{sweet.experiment_id} was trained on {trained_on} images, but "
                       f"{'enhanced' if enhanced else 'raw'} images are shown. This pairing was never "
                       f"evaluated; {paired_exp} is the matching model.")
    else:
        st.warning("This checkpoint is not in experiments/records, so its sliders start at E001's "
                   "settings, which were not validated for it.")

    st.header("Detection settings")
    st.button("Reset to sweet spot", on_click=_apply_sweet_spot, args=(sweet,), width="stretch",
              help="Put every slider back to its validated value for this checkpoint.")
    if sweet.imgsz not in IMGSZ_CHOICES:
        IMGSZ_CHOICES = sorted(IMGSZ_CHOICES + [sweet.imgsz])
    conf = st.slider(
        "Confidence threshold", 0.01, 0.99, step=0.01, key="conf",
        help=f"Sweet spot {sweet.conf}: {sweet.conf_source} Lower finds more lobsters and invents more.",
    )
    iou = st.slider("NMS IoU", 0.1, 0.95, step=0.05, key="iou",
                    help=f"Sweet spot {IOU_SOURCE} How much two boxes may overlap before one is "
                         "suppressed.")
    imgsz = st.select_slider("Inference size (px)", IMGSZ_CHOICES, key="imgsz",
                             help=f"Sweet spot {sweet.imgsz} px: {sweet.imgsz_source} Other sizes are "
                                  "untested.")
    show_conf = st.checkbox("Draw confidence labels", value=True)

    st.header("Ground truth")
    show_truth = st.checkbox("Overlay labelled boxes", value=True,
                             help="Available for the built splits only. zExtra has no labels.")
    match_iou = st.slider("Match IoU", 0.1, 0.95, step=0.05, key="match_iou",
                          help=f"Sweet spot {MATCH_IOU_SOURCE} How much a prediction must overlap a "
                               "labelled box to count as the same object, for the per-image "
                               "comparison below.")

    off = [name for name, value, target in (
        ("confidence", conf, sweet.conf), ("NMS IoU", iou, sweet.iou),
        ("inference size", imgsz, sweet.imgsz), ("match IoU", match_iou, sweet.match_iou),
    ) if abs(value - target) > 1e-9]
    if off:
        st.caption(f"⚠ Not the validated setting: {', '.join(off)}. Results at these values are "
                   "observations, not the evaluated configuration.")
    else:
        st.caption("✓ All sliders at the validated sweet spot.")

    st.header("What this cannot tell you")
    for item in LIMITATIONS:
        st.caption(f"• {item}")

tab_browse, tab_upload = st.tabs(["Browse a source", "Upload an image"])

image = None
image_name = ""
image_path: Path | None = None

with tab_browse:
    label = st.radio("Source", list(SOURCES), horizontal=False, key="source")
    kind, name = SOURCES[label]

    if kind == "raw":
        st.markdown(
            "**zExtra** is the one set the model has never seen in any form — excluded from every "
            "dataset version, never labelled, never trained on, never evaluated (D-003). It is the "
            "honest answer to *how does this behave on data it has never met*. There is no ground "
            "truth here, so a miss and a correct rejection look identical."
        )
    else:
        (st.warning if name in HELD_OUT_SPLITS else st.info)(f"**{name} split** — {SPLIT_STATUS[name]}")
        if name in HELD_OUT_SPLITS and abs(conf - sweet.conf) > 1e-9:
            who = sweet.experiment_id or "the model"
            st.error(
                f"The confidence slider is at {conf:.2f}, not {who}'s frozen operating point "
                f"{sweet.conf}. Viewing test images at another threshold is fine; **carrying "
                "that value forward as the model's setting is threshold selection on test** and "
                "is forbidden (CLAUDE.md rule 7). Choose thresholds on val."
            )

    try:
        paths = _images(kind, name)
    except DetectError as exc:
        paths = []
        st.warning(str(exc))

    if paths:
        names = [Path(p).name for p in paths]
        chosen_name = st.selectbox(
            f"Image ({len(paths)} available)", names, key=f"pick_{kind}_{name}",
            help="Type to filter. Filenames start with the session ID.",
        )
        chosen = Path(paths[names.index(chosen_name)])
        from PIL import Image  # local import: heavy dependency, GUI-only

        shown = chosen
        if enhanced:
            try:
                shown = _enhanced_index().get(chosen.name)
            except DetectError as exc:
                st.error(str(exc))
                st.stop()
            if shown is None:
                st.info(f"No enhanced copy of **{chosen.name}** exists. Harsh's enhanced images cover "
                        "the 2,105 v1 train/val/test images and, when its delivery is on this machine, "
                        "the 436 zExtra images (D-024); Zenodo sk has none. Pick a raw-image model "
                        "(E027 or E001) to view it.")
                st.stop()
        # image_path stays the split image, so ground truth is read from the same label file;
        # enhanced copies have the original size and pixel grid (D-017, D-018).
        image, image_name, image_path = Image.open(shown), chosen.name, chosen

with tab_upload:
    uploaded = st.file_uploader(
        "Any image — including one the model has not been trained on",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )
    if uploaded is not None and enhanced:
        st.warning("An uploaded image has no enhanced copy, and Harsh's enhancement is not part of "
                   "this app. Pick a raw-image model (E027 or E001) to run it.")
    elif uploaded is not None:
        from PIL import Image  # local import: heavy dependency, GUI-only

        image, image_name, image_path = Image.open(uploaded), uploaded.name, None

if image is None:
    st.info("Pick an image above to run the detector.")
    st.stop()

truth = None
if image_path is not None and kind == "split":  # raw sources (zExtra) have no label files
    try:
        truth = load_ground_truth(image_path, image.width, image.height)
    except DetectError as exc:  # a malformed label file must be visible, not silently ignored
        st.warning(f"Ground truth unavailable: {exc}")

try:
    model = _model(str(weights_path))
    with st.spinner(f"Detecting with {paired_exp}..."):
        detections = detect(model, image, conf=conf, iou=iou, imgsz=imgsz)
    compare_detections = None
    if compare_model is not None:
        with st.spinner(f"Detecting with {compare_model.experiment_id}..."):
            compare_detections = detect(_model(str(compare_path)), image, conf=compare_sweet.conf,
                                        iou=compare_sweet.iou, imgsz=compare_sweet.imgsz)
except DetectError as exc:
    st.error(str(exc))
    st.stop()


def _canvas(dets):
    out = draw_ground_truth(image, truth) if truth and show_truth else image
    return draw_detections(out, dets, show_confidence=show_conf)


canvas = _canvas(detections)
shown_as = "enhanced (Harsh, fusion_mild)" if enhanced else "raw"
legend = "🟩 prediction   🟧 ground truth (dashed)"

if compare_detections is not None:
    # Same image, two models, each at its own validated settings.
    st.caption(f"{image_name} · {shown_as} · {image.width} × {image.height} px"
               + (f" · per-image counts at match IoU {match_iou:.2f}" if truth is not None else ""))
    panels = (
        (paired_exp, detections, conf, imgsz, canvas),
        (compare_model.experiment_id, compare_detections, compare_sweet.conf, compare_sweet.imgsz,
         _canvas(compare_detections)),
    )
    for column, (exp, dets, c, size, img) in zip(st.columns(2, gap="medium"), panels):
        with column:
            st.image(img, caption=f"{exp} · conf ≥ {c:.2f} · {size} px", width="stretch")
            _counts(dets, truth, match_iou)
    if truth and show_truth:
        st.caption(legend)
    if truth is not None and not truth:
        st.caption("This image is a **confirmed negative** (empty label file): every box drawn on it "
                   "is a false positive.")
    st.caption("Counts describe this one image. They are not precision or recall, and aggregating "
               "them over a held-out split would be a test evaluation (CLAUDE.md rule 7).")

else:
    left, right = st.columns([3, 1], gap="large")

    with left:
        st.image(canvas, caption=f"{image_name} · {shown_as} · {sweet.experiment_id or weights_path.name}",
                 width="stretch")
        if truth and show_truth:
            st.caption(legend)

    with right:
        st.caption(f"{image.width} × {image.height} px, inferred at {imgsz} px, conf ≥ {conf:.2f}")
        if truth is not None:
            st.divider()
            st.caption(f"**This image only**, at match IoU {match_iou:.2f}")
        _counts(detections, truth, match_iou)
        if truth is None:
            st.caption(
                "No label file for this image, so nothing here can be called right or wrong. A "
                "missing label is not a negative (CLAUDE.md rule 6)."
            )
        else:
            if not truth:
                st.caption(
                    "This image is a **confirmed negative**: an empty label file, not a missing one. "
                    "Every box drawn on it is a false positive."
                )
            st.caption(
                "Counts describe this one image. They are not precision or recall, and aggregating "
                "them over a held-out split would be a test evaluation — allowed only for a declared "
                "final candidate, and logged in docs/EXPERIMENTS.md (CLAUDE.md rule 7)."
            )

        st.divider()
        if detections:
            st.write("Confidence per detection")
            st.bar_chart({"confidence": [d.confidence for d in detections]}, height=160)
        else:
            st.info(
                "Nothing above the threshold. Lower it to see what the model considered and rejected — "
                "absence here is not evidence that no lobster is present."
            )

rows = detections_csv_rows(detections, image_name, conf, imgsz)
if rows:
    st.subheader(f"Detections ({paired_exp})")
    st.dataframe(rows, width="stretch", hide_index=True)

    stem = Path(image_name).stem
    col_csv, col_yolo = st.columns(2)
    col_csv.download_button(
        "Download detections (CSV)", _csv_bytes(rows), f"{stem}_detections.csv", "text/csv",
        width="stretch",
    )
    col_yolo.download_button(
        "Download labels (YOLO .txt)",
        ("\n".join(to_yolo_lines(detections, image.width, image.height)) + "\n").encode("utf-8"),
        f"{stem}.txt",
        "text/plain",
        width="stretch",
        help="Same format as the project's own labels, so it opens in LabelImg for correction.",
    )
