# From prototype to software package

This folder is a **prototype**: a local demo GUI for the lobster detector, good enough for the team
demo and the report. Nothing here has to change to run it. This file is for later, if Harsh (or
anyone) wants to turn it into a real software package. Read `CLAUDE.md` first; the rules in its
section 6 still apply to a package.

Status marks: OBSERVED = checked in this folder's code; ADVICE = a recommended approach, not yet
tried on this code; UNKNOWN = must be confirmed with the person named.

## 1. What makes it a prototype today (OBSERVED)

| Area | Prototype now | What a package needs |
|---|---|---|
| Finding files | Every module finds the folder from its own location (`Path(__file__).parents[...]` in `scripts/gui_app.py`, `scripts/gui_check.py`, `src/detect/predict.py`, `species.py`, `models.py`, `sweet_spot.py`). Works only while the folder layout is kept. | One configurable data folder (section 3a). |
| Weights | Two `.pt` files (73 MB) inside the folder. | Downloaded once, checked by sha256 (section 3b). |
| Starting it | `streamlit run scripts/gui_app.py` from the folder. | One command, e.g. `deepsight-gui` (section 3c). |
| Install | `requirements-gui.txt` + manual torch line. | `pyproject.toml` with dependencies (section 3c). |
| Users | One person, local browser, no login, nothing saved to disk. | Same is fine for a desktop tool; a shared server needs more (section 4). |
| Tests | None in this folder (the CPU tests live in the project repository: `tests/test_species.py` and others). | Tests shipped and run before every release. |
| Platforms | Checked on Windows only; macOS not yet run. | Checked on every platform it is released for. |
| Repository leftovers | `predict.py` still has the `raw` and `split` image sources and the raw-folder guard (`UWCT_RAW_ROOT`); `enhanced.py` is used only by E004, which is not in this folder. | Remove or keep, but they are dead code here. |

## 2. What must stay true in any package

1. **Validated settings come from records.** The app recognises a checkpoint by its sha256 in
   `experiments/records/*/run_record.json` and takes its confidence (E027: 0.269, E001: 0.298) and
   inference size from that model's records. Ship those record files with the package, unchanged.
   Do not hard-code the numbers elsewhere.
2. **Inference on CPU by default.** The validated settings were checked on CPU. Apple's GPU (`mps`)
   may give slightly different boxes: offer it only as an option, and say it is not the validated
   setting.
3. **No accuracy metrics in the app, pixels only, test images not used for tuning**: CLAUDE.md
   section 6, rules 1–5.
4. **Keep the provenance idea of `MANIFEST.json`**: every release states the repository commit it
   came from and the sha256 of each model file.

## 3. Steps (ADVICE: a plan, not yet tried on this code)

### a. One data folder instead of positions in the folder
Add one setting, for example an environment variable `DEEPSIGHT_HOME`, read in one place (a small
`src/detect/paths.py`), defaulting to today's behaviour (the folder the code sits in). Replace each
`PROJECT_ROOT = Path(__file__)...` with that one value. The data folder holds:

```
DEEPSIGHT_HOME/
  configs/gui_species.json
  experiments/records/E027/ ...  experiments/records/E001/ ...
  weights/E027_best.pt  weights/E001_best.pt
  samples/ ...                   (optional demo images)
```

Change one module at a time and run `python scripts/gui_check.py` after each change: it must still
print `GUI CHECK PASSED`.

### b. Weights: download once, check the sha256
Do not put the `.pt` files inside the Python package (too large for a wheel). On first start,
download them from where the team keeps them (UNKNOWN: ask Saksham where they may be hosted) into
`DEEPSIGHT_HOME/weights/`, then compare the sha256 with `outputs.best.sha256` in that model's
`run_record.json` and refuse a mismatch. The app already shows "Recognised as E027" only on a match.

### c. Packaging and a start command
A starting point for `pyproject.toml` (UNTESTED: adjust and test before use):

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "deepsight-gui"
version = "0.1.0"
requires-python = ">=3.10,<3.13"
dependencies = ["ultralytics==8.4.155", "streamlit==1.64.0", "torch", "torchvision"]

[project.scripts]
deepsight-gui = "deepsight_gui.launch:main"

[tool.setuptools.packages.find]
where = ["src"]
```

- Move `scripts/gui_app.py` into the package (e.g. `src/deepsight_gui/app.py`) and add a small
  `launch.py` whose `main()` starts Streamlit on that file (`streamlit.web.cli` with
  `["streamlit", "run", <path to app.py>]`).
- Keep `ultralytics` pinned: the checkpoints were trained with 8.4.155. torch can float, but record
  the version each release was checked with.
- Install for development with `python -m pip install -e .`, then `deepsight-gui` starts the app.

### d. Tests
Ask Saksham for the repository's CPU tests (`tests/test_species.py` and the GUI-related tests) and
adapt them to the package layout. At minimum: the species file parses, each model's checkpoint
matches its record, the declared/recorded settings are what the records say, and one detection on
a sample image returns boxes (that is what `gui_check.py` does today).

### e. Clean-up
Remove the repository-only parts listed in section 1 if the package will never read the full
dataset. Keep the split of labour: logic in the package modules, widgets in the app file,
configuration in `gui_species.json`.

### f. Release checklist
- [ ] Tests pass on every platform released for (at least macOS and Windows).
- [ ] `gui_check.py` (or its package equivalent) passes on a fresh install.
- [ ] Version number and a short changelog.
- [ ] Release notes name the repository commit and the sha256 of each model.
- [ ] Redistribution rights confirmed (section 4).

## 4. Before sharing it outside the team (UNKNOWN: must be confirmed)

- **Our lobster frames and the models trained on them** come from the StFX lab dataset. Whether they
  may be published outside the course is UNKNOWN: do not upload the sample images or the weights to
  a public site, app store or public repository without permission. A package can ship without
  sample images (users open their own).
- **Zenodo sample frames** (`samples/lobster/images/test_zenodo_sk/`): Zenodo record 15755673,
  CC BY 4.0 (Fiorina, Obst, Sundelof, Ljungberg, Sande 2025). They may be shared with attribution.
- **Libraries:** Ultralytics is AGPL-3.0 (or a commercial licence). A package that uses it and is
  distributed or offered as a network service must follow the AGPL or obtain a licence. Check this
  before any public release.
- **A shared web server** (not just a local tool) also needs: upload size limits, no storing of
  uploaded images unless the users agree, and enough CPU (OBSERVED on Windows: E027's first
  detection took about 6 s including model loading; speed per image under load is UNKNOWN).

## 5. Keep the repository in step

The project repository is the source of truth. Send Saksham every change (or open a pull request
on the repository), so the next exported folder includes it. Changes made only in this folder are
lost at the next export.
