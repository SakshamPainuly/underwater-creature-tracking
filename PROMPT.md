# PROMPT.md — Reusable task template

Copy, fill in the <placeholders>, and delete the checklist sections that don't apply.
Project rules are in CLAUDE.md and are not repeated here.

###ROLE###
Senior computer-vision / ML research engineer on the Underwater Creature Video Tracking project.
Follow CLAUDE.md. <Optional nuance, e.g. "review only, no edits".>

###CONTEXT###
Phase: <P# name>. Why now: <motivation / weekly milestone>.
Relevant decisions: <docs/DECISIONS.md entries>.

###CURRENT STATE###
- Dataset version: <v1 + fingerprint from build_summary.json>
- Latest relevant experiment(s): <E### + key val metrics, or "none">
- Known open issues: <from docs/DATASET.md / docs/EXPERIMENTS.md>
- Machine: <local Windows | Colab | Kaggle>

###TASK###
<One clear objective.>
Steps (if known): 1. <...>
Out of scope: <what NOT to do>

###CONSTRAINTS###
- All CLAUDE.md hard rules apply.
- Task-specific: <e.g. "val split only", "no new dependencies", "do not change model size">
- Compute budget: <e.g. "smoke test only" / "≤ N GPU hours">

###FILES TO INSPECT###
- CLAUDE.md, README.md, <docs/...>
- <src/... and scripts/... relevant to the task>
- <manifests / configs / run records relevant to the task>

###SUCCESS CRITERIA###
- <Observable outcome 1>
- <Observable outcome 2>
- No CLAUDE.md rule violated; raw check passes.

###VALIDATION###
- `python -m unittest discover tests`
- `python scripts/check_raw_unchanged.py` (structural; add `--full` if content proof is needed)
- <task-specific checks>

###OUTPUT FORMAT###
1. What was inspected
2. What changed (files created / modified)
3. Results (OBSERVED / INFERRED / LIMITATION labels; no invented numbers)
4. Validation performed and outcomes
5. Risks / open questions
6. Proposed commit plan (if files changed) — not executed
7. Proposed next step

###STOP CONDITION###
Stop after <deliverable>. Do not commit. Do not start <next phase>. Wait for approval.

---
## Task-type checklists (keep the one that applies)
- Dataset: which version? Frozen versions are read-only; new work = new draft version with parent +
  decision entry. Label policy for new sources. Raw check.
- Training (GPU): env_report → verify → smoke → E### config (parent + changed variable) →
  run_record.json → registry row. Persistent checkpoints.
- Evaluation: val only unless it's a declared final test evaluation; threshold chosen on val;
  per-condition breakdown; false positives on negatives; failure-case images.
- Video / tracking: real video or sparse frames? (Sparse frames = "timelapse analysis", not tracking.)
  Which session/split does it come from? fps/stride recorded; detections/tracks schema; qualitative
  only without ID ground truth.
- Analytics: px / frames (seconds only with known fps); state calibration status.
- GUI: thin layer over existing functions; no logic in the UI.
- Debugging: reproduce → root cause → test → minimal fix.
- Refactoring: concrete defect/need; behavior-preserving; byte-identical outputs where applicable.
