"""Checks around E010 detection on a Tritonia sequence (D-044 phase 2), standard library only.

The Ultralytics call itself lives in scripts/predict_tritonia_sequence.py (GPU side, CLAUDE.md §7). Here: verify the
sequence folder against its manifest and the weights against their registered sha256 before any prediction, and
summarise the per-frame prediction files (Ultralytics `cls x y w h conf`, normalised, as D-033's point eval).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

FINAL_DECISION = "D-044"


class DetectionError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_sequence(seq_dir: Path, final: str | None = None, verify: bool = True) -> tuple[dict, list[dict]]:
    """Read sequence.json + manifest.csv; refuse a test sequence without the final decision; check every frame."""
    seq_dir = Path(seq_dir)
    try:
        meta = json.loads((seq_dir / "sequence.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise DetectionError(f"{seq_dir}: no readable sequence.json ({e})") from e
    if meta.get("split") == "test" and final != FINAL_DECISION:
        raise DetectionError(f"REFUSED: test sequence needs --final {FINAL_DECISION} (rule 7)")
    if meta.get("split") not in ("val", "test"):
        raise DetectionError(f"{seq_dir}: unexpected split {meta.get('split')!r}")
    man = seq_dir / "manifest.csv"
    if sha256_file(man) != meta.get("manifest_sha256"):
        raise DetectionError(f"{man}: sha256 differs from sequence.json")
    with open(man, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != meta["totals"]["frames"]:
        raise DetectionError(f"{man}: {len(rows)} rows, sequence.json says {meta['totals']['frames']}")
    if verify:
        bad = [r["image"] for r in rows if not (seq_dir / r["image"]).is_file()
               or sha256_file(seq_dir / r["image"]) != r["png_sha256"]]
        if bad:
            raise DetectionError(f"{len(bad)} frame(s) missing or changed, first: {bad[0]}")
    return meta, rows


def check_weights(path: Path, expected_sha256: str) -> str:
    got = sha256_file(path)
    if got != expected_sha256:
        raise DetectionError(f"{path}: sha256 {got} is not the registered {expected_sha256}")
    return got


def summarize_predictions(pred_dir: Path, stems: list[str], conf: float) -> dict:
    """Count frames and detections (at or above conf) over the expected per-frame files; fail on a missing file."""
    frames_with, n = 0, 0
    for s in stems:
        p = Path(pred_dir) / f"{s}.txt"
        if not p.is_file():
            raise DetectionError(f"missing prediction file {p}")
        k = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 6:
                raise DetectionError(f"{p.name}: expected 'cls x y w h conf', got {line!r}")
            k += float(parts[5]) >= conf
        frames_with += k > 0
        n += k
    return {"frames": len(stems), "frames_with_detections": frames_with, "detections": n, "at_conf": conf}
