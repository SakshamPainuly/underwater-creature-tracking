"""Deployment/session-level, leakage-aware train/val/test split.

Unit of assignment is a *session* (one dive/deployment, possibly spanning several
recording files -- see config.py), never an individual image. This
is a deliberate response to the leakage risk documented in the Phase 0 audit:
frames from the same session share the same lobsters, background, and lighting.

The split is deterministic given SPLIT_SEED -- re-running produces the same
assignment, which is what "reproducible" means here.
"""

import random
from dataclasses import dataclass

SPLIT_SEED = 20260917  # fixed for reproducibility; change deliberately, not silently
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
LOW_LABEL_RATE_THRESHOLD = 0.30
MIN_STRATUM_FOR_FULL_SPLIT = 3


@dataclass
class SplitResult:
    session_id: str
    source_set: str
    split: str
    flags: list


def _apportion(n: int) -> dict:
    """Largest-remainder apportionment of n items into train/val/test by TARGET_RATIOS."""
    raw = {k: n * r for k, r in TARGET_RATIOS.items()}
    counts = {k: int(v) for k, v in raw.items()}
    remainder = n - sum(counts.values())
    # give leftover items to the splits with the largest fractional remainder
    order = sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)
    for k in order[:remainder]:
        counts[k] += 1
    return counts


def build_split(rows: list) -> list:
    """rows: list of dicts from run_audit (deployment_metadata rows).

    Returns a list of SplitResult, one per input row, covering every session
    (including excluded ones, so the manifest is a complete audit trail).
    """
    results = []
    candidates = []

    for row in rows:
        flags = []
        if row["n_videos"] > 1:
            # Historical flag name (means "multi-file session"); kept so v1 regenerates byte-identically.
            flags.append("multi_camera_session")
        if not row["substrate"] or not row["visibility"]:
            flags.append("missing_substrate_or_visibility_metadata")
        if row["n_images"] > 0 and row["n_labeled"] / row["n_images"] < LOW_LABEL_RATE_THRESHOLD:
            flags.append(
                f"low_label_rate_{row['n_labeled']}_of_{row['n_images']}_verify_not_a_qc_gap"
            )

        if row["source_set"] == "Set100":
            flags.append("single_video_low_diversity_forced_train_only_never_val_or_test")
            results.append(SplitResult(row["session_id"], row["source_set"], "train", flags))
            continue

        if row["n_labeled"] == 0:
            flags.append("excluded_unlabeled_pending_annotation_not_confirmed_negative")
            results.append(SplitResult(row["session_id"], row["source_set"], "excluded", flags))
            continue

        candidates.append((row, flags))

    # Stratify by lighting_flag only -- it's the one condition variable populated
    # for all Set2000 sessions. Substrate/visibility are missing for most Set2000
    # rows (flagged above per-session) and are NOT used as a stratification key
    # because doing so on ~5/44 known values would be misleading, not rigorous.
    strata = {}
    for row, flags in candidates:
        strata.setdefault(row["lighting_flag"], []).append((row, flags))

    for stratum_key, items in strata.items():
        items_sorted = sorted(items, key=lambda ri: ri[0]["session_id"])
        rng = random.Random(SPLIT_SEED + int(stratum_key))
        rng.shuffle(items_sorted)

        n = len(items_sorted)
        counts = _apportion(n)
        stratum_flags = []
        if n < MIN_STRATUM_FOR_FULL_SPLIT:
            stratum_flags.append(
                f"stratum_lighting_flag={stratum_key}_too_small_n={n}_split_may_be_imbalanced"
            )

        idx = 0
        for split_name in ("val", "test", "train"):  # fill val/test first so rounding favors train
            for _ in range(counts[split_name]):
                row, flags = items_sorted[idx]
                idx += 1
                results.append(
                    SplitResult(
                        row["session_id"],
                        row["source_set"],
                        split_name,
                        flags + stratum_flags,
                    )
                )

    return results
