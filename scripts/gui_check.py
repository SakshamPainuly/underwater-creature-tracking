#!/usr/bin/env python
"""Check that the GUI can run on this machine, before starting it.

    python scripts/gui_check.py            (add --no-detect to skip the test detection)

Reports: the Python and package versions, every species and model in configs/gui_species.json
with whether its checkpoint is present and recognised, the number of images in every folder
source, and one test detection with the first available model. Read-only: nothing is written.
Exit code: 0 = the GUI can start (missing optional checkpoints are listed, not fatal), 1 = it cannot.
"""

import argparse
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from detect import (  # noqa: E402
    DetectError,
    detect,
    file_sha256,
    find_experiment_for_checkpoint,
    list_folder_images,
    load_model,
    load_species,
    sweet_spot_for,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-detect", action="store_true", help="skip the test detection")
    args = parser.parse_args()
    problems: list[str] = []

    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
    for package in ("streamlit", "ultralytics", "torch", "PIL"):
        try:
            module = __import__(package)
            print(f"  {package:12s} {getattr(module, '__version__', '?')}")
        except ImportError:
            problems.append(f"{package} is not installed (see README.md, step 2)")
            print(f"  {package:12s} MISSING")

    try:
        species = load_species()
    except DetectError as exc:
        print(f"GUI CHECK FAILED: {exc}")
        sys.exit(1)

    first = None  # (model choice, weights path, an image) for the test detection
    present = 0  # checkpoints found
    for s in species:
        print(f"\n{s.label}")
        images = []
        for src in s.sources:
            if src.kind != "folder":
                print(f"  source  {src.label}: {src.kind} source (needs the repository's data)")
                continue
            try:
                found = list_folder_images(src.name)
            except DetectError:
                found = []
            images += found
            print(f"  source  {src.label}: {len(found)} images")
        for m in s.models:
            path = PROJECT_ROOT / m.weights
            if not path.is_file():
                print(f"  model   {m.experiment_id}: checkpoint NOT PRESENT -> put it at {m.weights.as_posix()}")
                continue
            present += 1
            exp = find_experiment_for_checkpoint(file_sha256(path))
            spot = sweet_spot_for(path, declared_conf=m.declared_conf, declared_imgsz=m.declared_imgsz,
                                  declared_source=m.declared_source)
            origin = "from its record" if exp and not spot.conf_declared else (
                "declared in gui_species.json" if spot.conf_declared else "NOT VALIDATED (default)")
            print(f"  model   {m.experiment_id}: {path.stat().st_size / 1e6:.1f} MB, "
                  f"{'recognised as ' + exp if exp else 'not in experiments/records'}, "
                  f"conf {spot.conf} at {spot.imgsz} px ({origin})")
            if first is None and images:
                first = (m, path, images[0], spot)

    if present == 0:
        problems.append("no checkpoint is present for any model, so the GUI has nothing to run")
    elif first is None:
        print("\ntest detection skipped: no model has a folder source with images on this machine")
    elif not args.no_detect and not problems:
        m, path, image, spot = first
        try:
            from PIL import Image

            model = load_model(path)
            start = time.time()
            found = detect(model, Image.open(image), conf=spot.conf, iou=spot.iou, imgsz=spot.imgsz)
            print(f"\ntest detection: {m.experiment_id} on {image.name}: {len(found)} boxes "
                  f"in {time.time() - start:.1f} s (first run includes loading)")
        except Exception as exc:  # any failure here means the GUI would fail the same way
            problems.append(f"test detection failed: {exc}")

    if problems:
        print("\nGUI CHECK FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("\nGUI CHECK PASSED. Start it with:  streamlit run scripts/gui_app.py")


if __name__ == "__main__":
    main()
