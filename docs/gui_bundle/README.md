# DeepSight GUI: portable folder

A self-contained copy of the DeepSight detection GUI. It runs on macOS, Windows or Linux without
the project repository or the raw dataset. It carries the lobster detectors only (teammates'
other species are sent separately):

| Species | Model in this folder | Status |
|---|---|---|
| Lobster | E027 (final model) and E001 (first baseline, for side-by-side comparison) | works out of the box |

## Start (macOS)

1. Open Terminal in this folder. Python 3.10, 3.11 or 3.12 is needed (`python3 --version`).
2. Create the environment and install (about 5 minutes, about 1 GB):

   ```
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.14.0 torchvision==0.29.0
   python -m pip install -r requirements-gui.txt
   ```

   If pip cannot find those exact torch versions, install the newest: `python -m pip install torch torchvision`.
3. Check the setup (about 30 seconds). It must end with `GUI CHECK PASSED`:

   ```
   python scripts/gui_check.py
   ```

4. Start the app. It opens in the browser at http://localhost:8501:

   ```
   streamlit run scripts/gui_app.py
   ```

Next time, only `source .venv/bin/activate` and step 4 are needed.

## Where things are

`HOW_THINGS_ARE_STORED.txt` lists every folder and file. `CLAUDE.md` is the full context for an
AI assistant (or a person) continuing the work: what the project did, how the GUI is built and
the rules it keeps.

This is a **prototype** demo GUI, not a deployed software package. `PACKAGING.md` describes how
to turn it into one, if that is ever wanted.
