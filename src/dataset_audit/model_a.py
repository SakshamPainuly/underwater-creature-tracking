"""Mixed manifest of dataset version v4 (Model A, D-031) from Harsh's deliveries and UWCT v1.

Reads only: the parent version's image_manifest.csv (our originals and their v1 splits), the raw
label files, and Harsh's index CSVs, image and label folders. Harsh's own split and data.yaml are
never used (CLAUDE.md §10); every split comes from v1 sessions or the Zenodo group table below.

One row per image, in the leakage.py schema plus:
  path          image, relative to the folder that holds the raw roots (e.g. "biology/Set 100/x.png")
  label_path    label file, same base; empty for our negatives (Set 2000: no label file = negative)
  n_boxes       boxes in the label file
  label_status  "positive" or "negative" (our originals keep v1's "negative_confirmed")
  review        Harsh's label-review outcome and confidence, when the frame was reviewed
  exclude_reason  why a row is "excluded" (empty otherwise)
  delivery      raw root and folder the image comes from

Split rules (D-031 points 4-6):
  * our originals keep the v1 split; in train only, frames Harsh's review excluded and "kept as
    empty frame" verdicts of low confidence are excluded (val and test keep every v1 frame);
  * Zenodo: group = site_camera_date; site "mpa" is train, site "sk" is a held-out test, and the
    frames without a site ("REC_") are excluded;
  * a derived image is train only if every ancestor (background and every pasted lobster's
    source) is a train original that is not excluded; otherwise it is excluded.
Harsh's zip-1 colour-change folder is not used: harsh_v1_colour_2026-11 replaces it (D-031).
"""

import csv
import re
from collections import Counter
from pathlib import Path

ZENODO_ROOT, ZENODO_FOLDER = "harsh_model_a_2026-11", "4 Zenodo originals"
REVIEW_CSV = ("harsh_model_a_2026-11", "label_review_verdicts.csv")
ZENODO_SPLIT_BY_SITE = {"mpa": "train", "sk": "test"}
NO_SITE_GROUP = "REC"
_ZENODO_GROUP_RE = re.compile(r"^kso__((mpa|sk)_cam\d+_\d{6})_")
LABEL_TOL = 1e-6

# (raw root, folder as written in the root's derived_images_index.csv, origin, kind)
DERIVED_FOLDERS = [
    ("harsh_v1_colour_2026-11", "ours", "ours", "colour_change"),
    ("harsh_v1_colour_2026-11", "zenodo", "zenodo", "colour_change"),
    ("harsh_model_a_2026-11", "3 Synthetic - copy-paste/ours", "ours", "copy_paste"),
    ("harsh_model_a_2026-11", "3 Synthetic - copy-paste/zenodo", "zenodo", "copy_paste"),
    ("harsh_model_a_2026-11", "5 Diffusion - DM_underwater (train only)", "ours", "diffusion"),
    ("harsh_v1_extra_2026-11", "2 Augmented - colour change", "ours", "colour_change"),
    ("harsh_v1_extra_2026-11", "3 Synthetic - copy-paste", "ours", "copy_paste"),
    ("harsh_v1_extra_2026-11", "5 Diffusion - DM_underwater (10 sessions)", "ours", "diffusion"),
]
# Delivered but not used for v4, with the reason (reported, never read as data).
UNUSED_FOLDERS = {
    ("harsh_model_a_2026-11", "1 Original - our lobsters"): "originals come from the raw biology files and v1",
    ("harsh_model_a_2026-11", "2 Augmented - colour change"): "replaced by harsh_v1_colour_2026-11 (water colours from v1 train only)",
}

FIELDNAMES = ["image_id", "origin", "kind", "group_id", "split", "ancestors", "path", "label_path",
              "n_boxes", "label_status", "review", "exclude_reason", "declared_heldout", "delivery"]


class ModelAError(RuntimeError):
    pass


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_label(path: Path) -> list:
    """Boxes of a YOLO label file as (class, x, y, w, h); raises ModelAError on a malformed line."""
    boxes = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            cls, vals = int(parts[0]), [float(v) for v in parts[1:]]
        except ValueError:
            raise ModelAError(f"{path}:{n}: not a number: {line!r}") from None
        if len(vals) != 4 or cls != 0 or not all(0.0 <= v <= 1.0 for v in vals) or min(vals[2:]) <= 0:
            raise ModelAError(f"{path}:{n}: expected 'class 0' and 4 coordinates in [0, 1]: {line!r}")
        boxes.append((cls, *vals))
    return boxes


def same_boxes(a: list, b: list, tol: float = LABEL_TOL) -> bool:
    """True when a and b hold the same boxes (any order), coordinates within tol."""
    if len(a) != len(b):
        return False
    rest = list(b)
    for box in a:
        hit = next((i for i, o in enumerate(rest)
                    if o[0] == box[0] and all(abs(p - q) <= tol for p, q in zip(o[1:], box[1:]))), None)
        if hit is None:
            return False
        rest.pop(hit)
    return True


def contains_boxes(outer: list, inner: list, tol: float = LABEL_TOL) -> bool:
    """True when every box of inner is in outer (each matched once)."""
    rest = list(outer)
    for box in inner:
        hit = next((i for i, o in enumerate(rest)
                    if o[0] == box[0] and all(abs(p - q) <= tol for p, q in zip(o[1:], box[1:]))), None)
        if hit is None:
            return False
        rest.pop(hit)
    return True


def zenodo_group(filename: str) -> tuple:
    """(site, group_id) of a Zenodo image name; frames without a site get NO_SITE_GROUP."""
    m = _ZENODO_GROUP_RE.match(filename)
    return (m.group(2), m.group(1)) if m else ("", NO_SITE_GROUP)


def folder_files(root: Path, folder: str) -> dict:
    """stem -> (image file, label file) for <folder>/images and <folder>/labels; unpaired files raise."""
    images = {p.stem: p for p in (root / folder / "images").iterdir() if p.is_file()}
    labels = {p.stem: p for p in (root / folder / "labels").iterdir() if p.suffix == ".txt"}
    unpaired = sorted(set(images) ^ set(labels))
    if unpaired:
        raise ModelAError(f"{root.name}/{folder}: {len(unpaired)} image(s) without label or label without "
                          f"image, e.g. {unpaired[:3]}")
    return {s: (images[s], labels[s]) for s in images}


def _rel(base: str, root: Path, path: Path) -> str:
    return f"{base}/{path.relative_to(root).as_posix()}"


def ours_originals(v1_rows: list, source_dirs: dict, biology_root: Path, review: dict) -> list:
    """Rows for our labelled originals (zExtra is never part of v4, as in v1)."""
    rows = []
    for r in v1_rows:
        if r["split"] not in ("train", "val", "test") or r["label_status"] not in ("positive", "negative_confirmed"):
            continue
        image_dir, label_dir = source_dirs[r["source_set"]]
        stem = Path(r["filename"]).stem
        verdict = review.get(stem)
        reason = ""
        if r["split"] == "train" and verdict:
            if verdict["outcome_in_canonical_v2"].startswith("excluded"):
                reason = "label_review_excluded"
            elif verdict["outcome_in_canonical_v2"] == "kept as empty frame" and verdict["confidence"] == "low":
                reason = "label_review_low_confidence_empty"
        positive = r["label_status"] == "positive"
        rows.append({
            "image_id": f"ours/original/{r['filename']}", "origin": "ours", "kind": "original",
            "group_id": r["session_id"], "split": "excluded" if reason else r["split"], "ancestors": "",
            "path": _rel("biology", biology_root, image_dir / r["filename"]),
            "label_path": _rel("biology", biology_root, label_dir / f"{stem}.txt") if positive else "",
            "n_boxes": int(r["n_instances"]), "label_status": r["label_status"],
            "review": f"{verdict['outcome_in_canonical_v2']} ({verdict['confidence']})" if verdict else "",
            "exclude_reason": reason, "declared_heldout": "", "delivery": f"biology/{r['source_set']}",
        })
    return rows


def zenodo_originals(root: Path) -> tuple:
    """(rows, groups table rows) for the Zenodo originals."""
    rows, groups = [], {}
    for stem, (img, lbl) in sorted(folder_files(root, ZENODO_FOLDER).items()):
        site, group = zenodo_group(img.name)
        split = ZENODO_SPLIT_BY_SITE.get(site, "excluded")
        groups[group] = split
        n = len(read_label(lbl))
        rows.append({
            "image_id": f"zenodo/original/{img.name}", "origin": "zenodo", "kind": "original",
            "group_id": group, "split": split, "ancestors": "",
            "path": _rel(root.name, root, img), "label_path": _rel(root.name, root, lbl), "n_boxes": n,
            "label_status": "positive" if n else "negative", "review": "",
            "exclude_reason": "" if site else "zenodo_no_site", "declared_heldout": "",
            "delivery": f"{root.name}/{ZENODO_FOLDER}",
        })
    return rows, [{"origin": "zenodo", "group_id": g, "split": s} for g, s in sorted(groups.items())]


def _declared_heldout(d: dict, pastes: list, origin: str) -> str:
    """Harsh's own claim that a derived image of ours uses a v1 val/test session ("" for Zenodo)."""
    if origin != "ours":
        return ""
    if d.get("uses_uwct_v1_val_or_test_session"):
        return d["uses_uwct_v1_val_or_test_session"]
    splits = [d.get("base_split_uwct_v1") or d.get("base_session_split_uwct_v1", "")]
    splits += [p.get("source_split_uwct_v1") or p.get("source_session_split_uwct_v1", "") for p in pastes]
    return "yes" if any(s != "train" for s in splits) else "no"


def derived_rows(roots: dict, originals: dict, labels_of: dict, problems: list) -> list:
    """Rows for every image of DERIVED_FOLDERS. originals: stem -> original row; labels_of: stem ->
    boxes of the original. Label checks go to problems (one string each)."""
    rows = []
    indexes, pastes = {}, {}
    for base in {b for b, _f, _o, _k in DERIVED_FOLDERS}:
        indexes[base] = read_csv(roots[base] / "derived_images_index.csv")
        pf = roots[base] / "pasted_lobsters_index.csv"
        for p in read_csv(pf) if pf.is_file() else []:
            pastes.setdefault((base, p["derived_image"]), []).append(p)
    for base, folder, origin, kind in DERIVED_FOLDERS:
        root = roots[base]
        files = folder_files(root, folder)
        index = [d for d in indexes[base] if d["folder"] == folder]
        listed = {d["derived_image"] for d in index}
        if listed != set(files) or len(listed) != len(index):
            raise ModelAError(f"{base}/{folder}: index lists {len(index)} images, folder has {len(files)}; "
                              f"unlisted {sorted(set(files) - listed)[:3]}, missing {sorted(listed - set(files))[:3]}")
        for d in index:
            img, lbl = files[d["derived_image"]]
            paste = sorted(pastes.get((base, d["derived_image"]), []), key=lambda p: int(p["paste_no"]))
            stems = [d["base_image"]] + [p["source_image"] for p in paste]
            missing = [s for s in stems if s not in originals]
            if missing:
                raise ModelAError(f"{base}/{folder}/{img.name}: ancestor(s) not among the originals: {missing}")
            anc = [originals[s] for s in stems]
            boxes = read_label(lbl)
            where = f"{base}/{folder}/{lbl.name}"
            if kind == "copy_paste":
                if len(paste) != int(d.get("pasted_lobsters") or len(paste)) or not paste:
                    problems.append(f"{where}: index says {d.get('pasted_lobsters')} pasted lobsters, "
                                    f"pasted_lobsters_index lists {len(paste)}")
                if len(boxes) != len(labels_of[stems[0]]) + len(paste):
                    problems.append(f"{where}: {len(boxes)} boxes, expected background "
                                    f"{len(labels_of[stems[0]])} + pasted {len(paste)}")
                elif not contains_boxes(boxes, labels_of[stems[0]]):
                    problems.append(f"{where}: background boxes not all kept")
            elif not same_boxes(boxes, labels_of[stems[0]]):
                problems.append(f"{where}: boxes differ from the original's ({len(boxes)} vs "
                                f"{len(labels_of[stems[0]])})")
            if any(a["split"] in ("val", "test") or a["exclude_reason"] == "zenodo_no_site" for a in anc):
                reason = "ancestor_not_train"
            elif any(a["split"] != "train" for a in anc):
                reason = "ancestor_label_excluded"
            else:
                reason = ""
            rows.append({
                "image_id": f"{origin}/{kind}/{img.name}", "origin": origin, "kind": kind,
                "group_id": anc[0]["group_id"], "split": "excluded" if reason else "train",
                "ancestors": ";".join(a["image_id"] for a in anc),
                "path": _rel(base, root, img), "label_path": _rel(base, root, lbl), "n_boxes": len(boxes),
                "label_status": "positive" if boxes else "negative", "review": "",
                "exclude_reason": reason, "declared_heldout": _declared_heldout(d, paste, origin),
                "delivery": f"{base}/{folder}",
            })
    return rows


def build(v1_rows: list, source_dirs: dict, biology_root: Path, roots: dict) -> tuple:
    """(manifest rows, zenodo group rows, report). Raises ModelAError when a label check fails."""
    review = {r["image"]: r for r in read_csv(roots[REVIEW_CSV[0]] / REVIEW_CSV[1])}
    ours = ours_originals(v1_rows, source_dirs, biology_root, review)
    known = {Path(r["filename"]).stem for r in v1_rows}
    unknown = sorted(set(review) - known)
    if unknown:
        raise ModelAError(f"{len(unknown)} reviewed frame(s) not in the parent manifest, e.g. {unknown[:3]}")
    zen, groups = zenodo_originals(roots[ZENODO_ROOT])

    originals, labels_of = {}, {}
    for r in ours + zen:
        stem = Path(r["image_id"].split("/", 2)[2]).stem
        if stem in originals:
            raise ModelAError(f"two originals share the stem {stem}")
        originals[stem] = r
        if r["origin"] == "ours":
            labels_of[stem] = read_label(biology_root / r["label_path"].split("/", 1)[1]) if r["label_path"] else []
        else:
            labels_of[stem] = read_label(roots[ZENODO_ROOT] / r["label_path"].split("/", 1)[1])

    problems = []
    derived = derived_rows(roots, originals, labels_of, problems)
    if problems:
        raise ModelAError(f"{len(problems)} label problem(s), e.g. " + "; ".join(problems[:5]))
    rows = ours + zen + derived
    report = {
        "rows": len(rows),
        "by_delivery": {k: dict(sorted(v.items())) for k, v in sorted(_count(rows, "delivery").items())},
        "excluded_by_reason": dict(sorted(Counter(r["exclude_reason"] for r in rows if r["exclude_reason"]).items())),
        "zenodo_boxes_by_site": {s or NO_SITE_GROUP: sum(r["n_boxes"] for r in zen
                                                          if zenodo_group(r["image_id"].split("/", 2)[2])[0] == s)
                                 for s in ("mpa", "sk", "")},
        "unused_folders": {f"{b}/{f}": why for (b, f), why in UNUSED_FOLDERS.items()},
        "label_checks": "colour change and diffusion boxes equal the original's; copy-paste boxes = background "
                        f"boxes + one per pasted lobster (tolerance {LABEL_TOL})",
    }
    boxes = Counter()
    for r in rows:
        boxes[f"{r['split']}/{r['origin']}/{r['kind']}"] += r["n_boxes"]
    report["boxes_by_split"] = dict(sorted(boxes.items()))
    return rows, groups, report


def _count(rows: list, key: str) -> dict:
    out = {}
    for r in rows:
        out.setdefault(r[key], Counter())[r["split"] if not r["exclude_reason"] else f"excluded:{r['exclude_reason']}"] += 1
    return out


def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
