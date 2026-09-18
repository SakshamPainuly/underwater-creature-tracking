# Colab GPU runbook (P4)

How to train a YOLO experiment on Google Colab reproducibly. The notebook
`notebooks/colab_train_baseline.ipynb` runs exactly these steps; every check lives in the scripts.
Rules: CLAUDE.md (hard rules 1–4 and 7, section 6). Decision: D-012.

## One-time setup

1. **GitHub token (private repo).** Create a *fine-grained* personal access token:
   - Repository access: only `SakshamPainuly/underwater-creature-tracking`.
   - Permissions: **Contents: Read-only**; nothing else.
   - Do not reuse the GitHub CLI credential or a classic token.

   In Colab, open **Secrets** (key icon), add `UWCT_GITHUB_TOKEN`, and allow notebook access.
   See "GitHub authentication" below for how the notebook uses it.
2. **Raw training subset on Drive.** Upload these three folders to `MyDrive/uwct/raw/biology/`,
   keeping the same relative paths. Do not upload `zExtra Photos/`.

   ```
   MyDrive/uwct/raw/biology/Set 100/                                              (all files)
   MyDrive/uwct/raw/biology/Set 2000/batch_2 (2k - imgOnly)/batch_2 (2k - imgOnly)/
   MyDrive/uwct/raw/biology/Set 2000/batch_2_2k_labels/batch_2_2k_labels/
   ```

   Treat this copy as read-only. The notebook copies it to local disk and checks it against the raw
   fingerprint.
3. **Runs folder:** `MyDrive/uwct/runs/` (created by the notebook). It holds all run folders,
   checkpoints and run records, outside Git.
4. **Code:** commit and push the code to run on `p4/colab-baseline`, then note the 40-character sha
   from `git rev-parse HEAD`. The notebook runs only that exact commit.

## Every session (notebook cells 1–10)

| Cell | Step | Must show |
|---|---|---|
| 1 | Set `COMMIT` (other settings are fixed) | — |
| 2 | `nvidia-smi` | a GPU |
| 3 | Mount Drive | — |
| 4 | Checkout `COMMIT` with the read-only token | `working tree: clean` |
| 5 | `pip install -r requirements-gpu.txt` | `ultralytics 8.4.155` |
| 6 | `scripts/env_report.py` | GPU, CUDA, torch |
| 7 | Copy Set 100 + Set 2000 to `/content/raw/biology`, set `UWCT_RAW_ROOT` | no zExtra |
| 8 | `check_raw_unchanged.py --root /content/raw/biology --subset training` | `RAW DATA MATCHES FINGERPRINT` |
| 9 | `prepare_yolo_dataset.py --version v1` | frozen v1 verified, build summary |
| 10 | `verify_yolo_dataset.py` | `VERIFIED` |

## GitHub authentication (cell 4)

Clone and reconnect/fetch both go through `git_remote()`, which uses Git's standard
`GIT_ASKPASS` mechanism.

1. The cell writes `/content/uwct_git_askpass.sh`, outside the repo. It contains **no secret**: it
   answers Git's username prompt with `x-access-token` and its password prompt with
   `printf "%s\n" "$UWCT_GIT_TOKEN"`.
2. For each clone or fetch, `git_remote()` calls `userdata.get("UWCT_GITHUB_TOKEN")` at that
   moment. It puts the token into a copy of the environment passed only to that one `git`
   process (`subprocess.run(..., env=env)`), with `GIT_ASKPASS` pointing at the helper and
   `GIT_TERMINAL_PROMPT=0`. The copy is cleared right after the call.
3. The command line is `git -c credential.helper= clone|fetch ...`, with a plain
   `https://github.com/...` URL. The empty `credential.helper` disables every credential helper,
   so Git stores nothing.

Why the token is not exposed in process arguments:
- A command line (`/proc/<pid>/cmdline`, `ps`) is readable by every process on the machine. Here
  no command line contains the token: not git's arguments, not the remote URL, and not the helper
  invocation. Git runs the helper with the prompt text as its only argument.
- The token travels in the child process *environment* (`/proc/<pid>/environ`, readable only by
  the same user or root). The helper returns it to Git over a pipe on stdout. `printf` and
  `echo` are shell builtins, so the token is never the argument of another process.
- Nothing is written to disk: not `.git/config` (checked after each call), not the remote URL,
  not a credential store, not the notebook. On errors, stderr has the token replaced by `***`.

Limitation: the Colab VM runs everything as root, so code running in the same runtime could read
a git process's environment during a clone or fetch. The same code could read Colab Secrets
directly. The token is read-only and limited to this repository; revoke it on GitHub if in doubt.

## Training (cells 11–14)

```
python scripts/train_yolo.py --config configs/experiments/E001_yolo_baseline.yaml \
    --runs-root /content/drive/MyDrive/uwct/runs --weights-dir /content/weights --preflight-only
... --smoke        # 1 epoch, 10% of train, val on val        -> runs/E001_smoke_<utc>/
...                # full run (E001: 100 epochs max, patience 30) -> runs/E001/
... --resume       # after a disconnect: rerun cells 1-10 with the same COMMIT first
... --finalize-only  # recompute val metrics + run_record.json for a finished run
```

Before any training, `train_yolo.py` stops (exit 1, nothing trained) unless all of these hold:
- A Linux machine with a CUDA GPU; `ultralytics` equals the version pinned in the config (8.4.155).
- The config passes the stdlib validator, and PyYAML reads it identically.
- The git checkout is clean. `--allow-dirty` is accepted for preflight/smoke only and is recorded.
- The dataset view matches frozen v1:
  - `FROZEN.json` hashes and the manifest hashes it was built from;
  - the recomputed fingerprint equals the config's `dataset_fingerprint`;
  - no zExtra images, and exactly the built files on disk.
- Ultralytics' own label scan of train/val gives the built counts.
- The exact checkpoint in the config (`yolo26s.pt`) loads. If it does not, the run **stops**;
  another model is never substituted.
- For a full run of a child experiment (e.g. E002), the parent is already in
  `experiments/registry.csv`.

Training receives `data_trainval.yaml`, written into the run folder. It has only `path`,
`train`, `val` and `names`: **the test split does not exist for the training process**. Early
stopping and `best.pt` selection use val. After training, `best.pt` is validated on `split="val"`.

## Run folder (`MyDrive/uwct/runs/<run>/`)

| File | Content |
|---|---|
| `config.yaml` | copy of the experiment config |
| `data_trainval.yaml` | train + val data file (no test) |
| `preflight.json` | everything checked before the first launch |
| `launches.jsonl` | start/end of each launch (full, resume) |
| `train/` | Ultralytics output: `args.yaml`, `results.csv`, `weights/best.pt`, `weights/last.pt` (saved every epoch) |
| `val_best/` | validation of `best.pt` on val |
| `run_record.json` | the CLAUDE.md §5 record (below) |

`run_record.json` records:
- the git commit, branch and dirty flag;
- config path, sha256 and contents;
- dataset version, fingerprint, manifest sha256s and the Ultralytics train/val counts;
- the pretrained checkpoint's name, path and sha256, and the Ultralytics version;
- the full environment report (Python, torch, CUDA, cuDNN, GPU, VRAM, driver), at preflight and at
  finalize;
- the exact training kwargs, the resolved Ultralytics `args.yaml`, and the seed;
- launches and duration, and epochs completed;
- `best.pt`, `last.pt` and `results.csv` with path, size and sha256;
- val P, R, mAP50 and mAP50-95; `test_metrics` is always `null`;
- the v1 limitations.

## After a run (local PC)

1. Download `run_record.json` from the run folder. **Do not download or commit weights.** They stay
   on Drive and are referenced by path and sha256.
2. `python scripts/register_run.py <path>/run_record.json --dry-run`, then run it again without
   `--dry-run`. This appends a row to `experiments/registry.csv`. It refuses smoke runs, incomplete
   records and duplicates.
3. Update the run log in `docs/EXPERIMENTS.md`, then propose a commit (CLAUDE.md §8).

## FINAL TEST EVALUATION (cells 16–17)

The **only** use of the test split (CLAUDE.md rule 7). It is done once per declared final
candidate, after its training run is finished, registered and reviewed. The results never feed
back into training, early stopping, thresholds, hyperparameters or model choice.

1. **Declare the candidate first (local PC):**
   - add a row to `experiments/final_candidates.csv`
     (`experiment_id,declared_utc,decision,reason`, with a docs/DECISIONS.md id);
   - add a line under the test policy in `docs/EXPERIMENTS.md`;
   - commit and push (after approval).
2. **Colab:** set `COMMIT` to that commit, run cells 1–10, then set `RUN_FINAL_TEST = True` in
   cell 16 and run it:
   ```
   python scripts/final_test_eval.py --experiment E001 \
       --runs-root /content/drive/MyDrive/uwct/runs --confirm-final-test
   ```
   `src/training/final_test.py` refuses, with the test split untouched, unless:
   - the machine is a GPU machine with a clean checkout;
   - the experiment is declared at this commit;
   - `run_record.json` is a valid finished **full** run and `best.pt` still matches its sha256;
   - the dataset view verifies and its fingerprint equals the training run's fingerprint;
   - Ultralytics reads the test split with the built counts;
   - no completed final test exists for the run.
3. **What it does, in order:**
   - **a.** The operating confidence is chosen on **val**: `best.pt`, the run's
     `data_trainval.yaml`, and the confidence that maximizes F1. This is fixed before step b.
   - **b.** `YOLO(best.pt).val(data=<data/yolo/lobster_v1/data.yaml, which includes test>,
     split="test", plots=True)`.
   - **c.** Prediction examples: one test image per test session, drawn at the val-chosen
     confidence.
4. **Outputs** (Drive, outside Git) in `runs/<id>/final_test/`:

   | Path | Content |
   |---|---|
   | `attempts.jsonl` | every attempt, including interrupted ones |
   | `val_threshold/` | the val run that fixes the confidence |
   | `test/` | Ultralytics plots: confusion matrix (raw and normalized), PR, F1, P and R curves, `val_batch*_labels.jpg` / `val_batch*_pred.jpg` |
   | `predictions/` | prediction examples with boxes and confidence scores |
   | `final_test_record.json` | see below |

   `final_test_record.json` holds:
   - metrics: mAP50 and mAP50-95; P, R and F1 at the val-chosen confidence; Ultralytics' own P
     and R, which use the max-F1 confidence **on test** and are labelled as such; the confusion
     matrix; speed;
   - the declaration row, the training and evaluation commits, the dataset fingerprint and test
     counts, `best.pt` sha256, and the sha256 of every artifact;
   - the environment and the v1 limitations.
5. Cell 17 displays everything under a **FINAL TEST EVALUATION** heading.
6. **Afterwards (local PC):** add the evaluation to the Test-evaluation log in
   `docs/EXPERIMENTS.md` (date, experiment, reason, result).

Plot file names depend on the installed Ultralytics version. Cell 17 says so when a plot is
missing, rather than failing.

## Resuming after a disconnect

Colab's local disk (`/content`) is wiped, but the run folder on Drive keeps `last.pt`, which is
saved every epoch.
1. Rerun cells 1–10 with the **same** `COMMIT`. This recreates `/content/uwct/data/yolo/lobster_v1`
   at the same path, and `data_trainval.yaml` points there.
2. Run cell 14 (`--resume`). It refuses to resume if the commit, config sha256, dataset fingerprint
   or Ultralytics version differs from `preflight.json`.

## Known limitations

- torch/CUDA come from the Colab image and are not pinned; they are recorded per run.
  `deterministic=True` and a fixed seed reduce, but do not remove, GPU non-determinism.
- Within a run, `best.pt` is chosen by Ultralytics' built-in fitness on val, not by val mAP50 alone.
  Experiments are compared on val mAP50 (docs/EXPERIMENTS.md). The exact fitness weights of 8.4.155
  are UNKNOWN — requires confirmation from the installed source.
- The raw-copy check is structural (paths and sizes; PNG pixels are not hashed).
- Val metrics use Ultralytics defaults (conf 0.001, IoU 0.7). The operating confidence is chosen
  on val only inside the final-test stage. Per-session and per-condition results, and false
  positives on negative images, are not computed yet.
- The final test depends on Ultralytics exposing `box.px`, `p_curve`, `r_curve` and `f1_curve`
  and `confusion_matrix`, and on its plot file names. These are UNKNOWN for 8.4.155 until the
  first Colab run; if the curves are missing, the stage stops before touching test.
