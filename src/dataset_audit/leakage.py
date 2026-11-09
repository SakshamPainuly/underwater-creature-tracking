"""Leakage check for a dataset version that mixes originals, derived images and external data (D-031).

The check reads a mixed manifest (CSV, one row per image) and a group table, and never writes.

Mixed manifest columns:
  image_id   unique id of the image (e.g. "<origin>/<kind>/<filename>")
  origin     "ours" (Set 100 / Set 2000) or "zenodo"
  kind       "original", "colour_change", "copy_paste" or "diffusion"
  group_id   split unit of the image: session for ours, site-camera-date group for zenodo
             (for a derived image: the group of its background)
  split      "train", "val", "test" or "excluded"
  ancestors  ";"-joined image_ids of the ORIGINAL images it was built from: the background
             first, then the source image of every pasted lobster. Empty for originals.
  path       optional: image path relative to --image-root, for the duplicate checks
  declared_heldout  optional: the delivery's own claim that the image uses a held-out
             (val/test) group ("yes"/"no"); it must agree with the computed answer

Group table: (origin, group_id) -> split. For ours it comes from split_manifest.csv; other
origins get their own table (columns origin, group_id, split).

Rules (an error for every violation):
  1. Values are from the vocabularies above and image_ids are unique.
  2. An original sits in a known group and has the group's split, or "excluded".
  3. A derived image names at least one ancestor, and every ancestor is an original row.
  4. A derived image is never in val or test: evaluation uses real, unmodified images only.
  5. A derived image in train has every ancestor in a train group. An excluded original from a
     held-out group still counts as held out.
  6. No file content is shared between train, val and test (sha256), and no image pair across
     those splits is a near-duplicate (dHash Hamming distance <= max_distance).
"""

import csv
import hashlib
from collections import Counter
from pathlib import Path

ORIGINS = ("ours", "zenodo")
KINDS = ("original", "colour_change", "copy_paste", "diffusion")
SPLITS = ("train", "val", "test", "excluded")
EVAL_SPLITS = ("val", "test")
_YES, _NO = {"yes", "true", "1"}, {"no", "false", "0"}


class LeakageError(RuntimeError):
    pass


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def groups_from_split_manifest(path: Path, origin: str = "ours") -> dict:
    """(origin, session_id) -> split, from a split_manifest.csv."""
    return {(origin, r["session_id"]): r["split"] for r in read_csv(path)}


def groups_from_table(path: Path) -> dict:
    """(origin, group_id) -> split, from a CSV with columns origin, group_id, split."""
    return {(r["origin"], r["group_id"]): r["split"] for r in read_csv(path)}


def ancestors_of(row: dict) -> list:
    return [a.strip() for a in (row.get("ancestors") or "").split(";") if a.strip()]


def heldout_groups_used(row: dict, by_id: dict, groups: dict) -> list:
    """Sorted (origin, group_id) of the row's ancestors whose group is in val or test."""
    out = set()
    for a in ancestors_of(row):
        anc = by_id.get(a)
        if anc is not None:
            key = (anc["origin"], anc["group_id"])
            if groups.get(key) in EVAL_SPLITS:
                out.add(key)
    return sorted(out)


def check_manifest(rows: list, groups: dict) -> list:
    """Return a sorted list of rule violations (rules 1-5); empty means the manifest passes."""
    errors = []
    for key, split in groups.items():
        if split not in SPLITS:
            errors.append(f"group {key[0]}:{key[1]}: unknown split {split!r}")

    ids = Counter(r["image_id"] for r in rows)
    errors += [f"{i}: image_id appears {n} times" for i, n in ids.items() if n > 1]
    by_id = {r["image_id"]: r for r in rows}

    for r in rows:
        i, kind, split = r["image_id"], r["kind"], r["split"]
        bad = [f"{col} {r[col]!r}" for col, vocab in (("origin", ORIGINS), ("kind", KINDS), ("split", SPLITS))
               if r[col] not in vocab]
        if bad:
            errors.append(f"{i}: unknown " + ", ".join(bad))
            continue
        anc = ancestors_of(r)

        if kind == "original":
            gsplit = groups.get((r["origin"], r["group_id"]))
            if gsplit is None:
                errors.append(f"{i}: group {r['origin']}:{r['group_id']} is not in the group table")
            elif split not in (gsplit, "excluded"):
                errors.append(f"{i}: split {split} but its group {r['group_id']} is {gsplit}")
            if anc:
                errors.append(f"{i}: an original must not list ancestors")
            continue

        if not anc:
            errors.append(f"{i}: derived image ({kind}) lists no ancestors")
        for a in anc:
            if a not in by_id:
                errors.append(f"{i}: ancestor {a} is not in the manifest")
            elif by_id[a]["kind"] != "original":
                errors.append(f"{i}: ancestor {a} is {by_id[a]['kind']}, not an original")
        if split in EVAL_SPLITS:
            errors.append(f"{i}: derived image ({kind}) in {split}; val and test hold originals only")
        used = heldout_groups_used(r, by_id, groups)
        if split == "train" and used:
            errors.append(f"{i}: in train but built from held-out group(s) "
                          + ", ".join(f"{o}:{g} ({groups[(o, g)]})" for o, g in used))
        else:
            # Every ancestor must have a known group, or rule 5 cannot be checked.
            for a in anc:
                if a in by_id and (by_id[a]["origin"], by_id[a]["group_id"]) not in groups:
                    errors.append(f"{i}: ancestor {a} has a group that is not in the group table")

        declared = (r.get("declared_heldout") or "").strip().lower()
        if declared:
            if declared not in _YES | _NO:
                errors.append(f"{i}: declared_heldout {declared!r} is not yes/no")
            elif (declared in _YES) != bool(used):
                errors.append(f"{i}: delivery says held-out={declared}, computed {'yes' if used else 'no'}")
    return sorted(set(errors))


def summary(rows: list) -> dict:
    """split -> kind -> image count, for the report."""
    out = {}
    for r in rows:
        out.setdefault(r["split"], Counter())[f"{r['origin']}/{r['kind']}"] += 1
    return {s: dict(sorted(c.items())) for s, c in sorted(out.items())}


def _in_scope(rows: list) -> list:
    return [r for r in rows if r["split"] in ("train",) + EVAL_SPLITS]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_duplicates(rows: list, digests: dict) -> list:
    """Errors for file content shared by images in different splits. digests: image_id -> sha256."""
    splits_of = {}
    for r in _in_scope(rows):
        if r["image_id"] in digests:
            splits_of.setdefault(digests[r["image_id"]], []).append((r["split"], r["image_id"]))
    errors = []
    for members in splits_of.values():
        if len({s for s, _ in members}) > 1:
            errors.append("identical files across splits: " + ", ".join(f"{i} ({s})" for s, i in sorted(members)))
    return sorted(errors)


def dhash(path: Path, size: int = 8) -> int:
    """64-bit difference hash of an image (grayscale, (size+1) x size). Needs Pillow."""
    from PIL import Image  # heavy import kept local: only the duplicate check needs it

    with Image.open(path) as im:
        px = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS).tobytes()  # 1 byte/pixel
    bits = 0
    for y in range(size):
        for x in range(size):
            bits = (bits << 1) | (px[y * (size + 1) + x] > px[y * (size + 1) + x + 1])
    return bits


def near_duplicates(rows: list, hashes: dict, max_distance: int = 4) -> list:
    """Errors for image pairs in different splits whose dHash differs in <= max_distance bits."""
    by_split = {}
    for r in _in_scope(rows):
        if r["image_id"] in hashes:
            by_split.setdefault(r["split"], []).append((r["image_id"], hashes[r["image_id"]]))
    errors = []
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        for ia, ha in by_split.get(a, []):
            for ib, hb in by_split.get(b, []):
                d = (ha ^ hb).bit_count()
                if d <= max_distance:
                    errors.append(f"near-duplicate (dHash distance {d}): {ia} ({a}) ~ {ib} ({b})")
    return sorted(errors)


def image_paths(rows: list, image_root: Path) -> dict:
    """image_id -> absolute path for in-scope rows with a path; missing files raise."""
    out, missing = {}, []
    for r in _in_scope(rows):
        rel = (r.get("path") or "").strip()
        if not rel:
            missing.append(f"{r['image_id']}: no path")
            continue
        p = Path(image_root) / rel
        if not p.is_file():
            missing.append(f"{r['image_id']}: {p} not found")
        out[r["image_id"]] = p
    if missing:
        raise LeakageError(f"{len(missing)} image(s) cannot be read, e.g. " + "; ".join(missing[:5]))
    return out
