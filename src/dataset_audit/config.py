"""Dataset-specific paths and naming conventions.

This is the one module that knows about "Set 100" / "Set 2000" / "zExtra Photos"
and the GoPro-derived filename convention. If a future dataset uses different
folder names or a different filename scheme, only this file (and the regexes
below) should need to change.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# On Colab / the lab GPU the raw data may live outside the repo (e.g. copied from Drive
# to local disk); point UWCT_RAW_ROOT at the folder that contains "Set 100", "Set 2000", etc.
BIOLOGY_ROOT = Path(os.environ.get("UWCT_RAW_ROOT", PROJECT_ROOT / "data" / "raw" / "biology"))

# Each entry: (source_name, image_dir, label_dir_or_None)
# label_dir is None when the images have no annotation folder at all (zExtra Photos).
DATASET_SOURCES = [
    (
        "Set100",
        BIOLOGY_ROOT / "Set 100",
        BIOLOGY_ROOT / "Set 100",  # images and labels are co-located here
    ),
    (
        "Set2000",
        BIOLOGY_ROOT / "Set 2000" / "batch_2 (2k - imgOnly)" / "batch_2 (2k - imgOnly)",
        BIOLOGY_ROOT / "Set 2000" / "batch_2_2k_labels" / "batch_2_2k_labels",
    ),
    (
        "zExtra",
        BIOLOGY_ROOT / "zExtra Photos",
        None,
    ),
]

# What a missing label file means, per source. Explicit allowlist: a source without an entry is
# an error, never an implicit "negative".
#   "negative": documented convention that no label file = no lobster (Set 2000, per the docx)
#   "pending":  annotation status unresolved -> excluded, never treated as negative (zExtra)
#   "error":    every image is documented as labeled, so a missing label is a data error (Set 100)
MISSING_LABEL_POLICY = {"Set100": "error", "Set2000": "negative", "zExtra": "pending"}

SPREADSHEET_PATH = BIOLOGY_ROOT / "What's been done.xlsx"

# Filename pattern observed across all three sources:
#   {trip}_{SITE}_{session}_{lightmode}_{date}_{GoProSerial}[_|scene]{frameNumber}.png
# e.g. "06_HERR_02_CW_20200901_GP299683_00001.png"
#      "16_HERR_01_CW_20220509_GP141792scene00001.png"  (Set100 uses "scene" instead of "_")
#
# "video_id"   = deployment code + GoPro file ID (one recording file).
# "session_id" = video_id with the GoPro file ID stripped -- the split unit. Some sessions
#                contain several recording files ("multi-file sessions"); all of a session's
#                files always stay in one split, since they share the dive, place and animals.
#                VERIFIED: the files share one session code. INFERRED: the IDs (same 4-digit
#                file number, consecutive 2-digit prefix) look like sequential GoPro chapters.
#                UNKNOWN -- requires confirmation: whether any were simultaneous cameras.
VIDEO_ID_PATTERN = r"^(?P<video_id>.+)(?:_|scene)(?P<frame>\d+)$"
GOPRO_SUFFIX_PATTERN = r"_GP\d+$"
