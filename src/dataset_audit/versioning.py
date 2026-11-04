"""Immutable dataset versions.

A dataset version = the manifest files that define the dataset (deployment metadata, split,
image manifest, consistency report). Frozen versions are registered with their sha256 hashes in
data/processed/FROZEN.json and can never be written to -- there is no override. Changes go into a
new *draft* version created from a frozen parent with a docs/DECISIONS.md entry; a draft becomes
frozen only through scripts/freeze_dataset_version.py (requires user approval, see CLAUDE.md).

v1 lives directly in data/processed/ (its location predates versioning); later versions live in
data/processed/<name>/ with a VERSION.json.
"""

import csv
import hashlib
import json
import re
import shutil
from datetime import date
from pathlib import Path

from .config import DEFAULT_IMAGE_SOURCE, IMAGE_SOURCES, PROJECT_ROOT

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DECISIONS_PATH = PROJECT_ROOT / "docs" / "DECISIONS.md"
REGISTRY_NAME = "FROZEN.json"
VERSION_FILE = "VERSION.json"
_NAME_RE = re.compile(r"^v\d+(\.\d+)?$")


class VersionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry(root: Path = PROCESSED_ROOT) -> dict:
    path = root / REGISTRY_NAME
    if not path.is_file():
        return {"versions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def is_frozen(name: str, root: Path = PROCESSED_ROOT) -> bool:
    return name in load_registry(root)["versions"]


def version_dir(name: str, root: Path = PROCESSED_ROOT) -> Path:
    if not _NAME_RE.match(name):
        raise VersionError(f"invalid version name {name!r} (expected e.g. v1, v1.1, v2)")
    entry = load_registry(root)["versions"].get(name)
    if entry is not None:
        return (root / entry["dir"]).resolve()
    return (root / name).resolve()


def _protected(root: Path) -> tuple:
    registry = load_registry(root)
    files = {(root / REGISTRY_NAME).resolve()}
    dirs = []
    for entry in registry["versions"].values():
        d = (root / entry["dir"]).resolve()
        files.update((d / f).resolve() for f in entry["files"])
        if d != root.resolve():  # v1 shares data/processed/ with other versions: protect its files only
            dirs.append(d)
    return files, dirs


def assert_writable(path: Path, root: Path = PROCESSED_ROOT) -> None:
    p = Path(path).resolve()
    files, dirs = _protected(root)
    if p in files or any(p == d or d in p.parents for d in dirs):
        raise VersionError(
            f"refusing to write {p}: it belongs to a frozen dataset version (see {REGISTRY_NAME}). "
            "Frozen versions are immutable; create a new draft version instead."
        )


def verify_frozen(name: str, root: Path = PROCESSED_ROOT) -> None:
    entry = load_registry(root)["versions"].get(name)
    if entry is None:
        raise VersionError(f"version {name} is not frozen")
    d = root / entry["dir"]
    problems = []
    for fname, expected in sorted(entry["files"].items()):
        p = d / fname
        if not p.is_file():
            problems.append(f"missing: {fname}")
        elif sha256_file(p) != expected:
            problems.append(f"hash mismatch: {fname}")
    if problems:
        raise VersionError(f"frozen version {name} failed verification: " + "; ".join(problems))


def decision_exists(decision_id: str, decisions_path: Path = DECISIONS_PATH) -> bool:
    if not decision_id or not decisions_path.is_file():
        return False
    return re.search(rf"\b{re.escape(decision_id)}\b", decisions_path.read_text(encoding="utf-8")) is not None


def read_version_meta(name: str, root: Path = PROCESSED_ROOT) -> dict | None:
    path = version_dir(name, root) / VERSION_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def image_source_of(name: str, root: Path = PROCESSED_ROOT) -> str:
    """The image source a version's images come from (D-017); versions without one use raw images."""
    meta = read_version_meta(name, root) or {}
    return meta.get("image_source", DEFAULT_IMAGE_SOURCE)


def open_draft(name: str, parent: str | None, decision: str | None,
               root: Path = PROCESSED_ROOT, decisions_path: Path = DECISIONS_PATH,
               image_source: str | None = None) -> Path:
    """Return the directory of draft version `name`, creating it from `parent` if needed.

    `image_source` (D-017) names where the new version's images come from; by default it is the
    parent's. Labels, splits and manifests are always the parent's until changed deliberately.
    """
    if is_frozen(name, root):
        raise VersionError(
            f"dataset version {name} is frozen and immutable ({REGISTRY_NAME}); there is no override. "
            f"To correct it or add data, add a decision to docs/DECISIONS.md and create a draft, e.g. "
            f"--version {name}.1 --from-version {name} --decision D-XXX"
        )
    d = version_dir(name, root)
    meta = read_version_meta(name, root)
    if meta is not None:
        existing_source = meta.get("image_source", DEFAULT_IMAGE_SOURCE)
        if ((parent and parent != meta["parent"]) or (decision and decision != meta["decision"])
                or (image_source and image_source != existing_source)):
            raise VersionError(f"draft {name} already exists with parent={meta['parent']}, "
                               f"decision={meta['decision']}, image_source={existing_source}")
        return d

    if not parent or not decision:
        raise VersionError(f"version {name} does not exist; create it with --from-version <frozen "
                           "parent> --decision <DECISIONS.md id>")
    if not is_frozen(parent, root):
        raise VersionError(f"parent version {parent} must be frozen")
    if not decision_exists(decision, decisions_path):
        raise VersionError(f"decision {decision} not found in {decisions_path}; record it first")
    if d.exists():
        raise VersionError(f"{d} exists but has no {VERSION_FILE}; refusing to reuse it")
    image_source = image_source or image_source_of(parent, root)
    if image_source not in IMAGE_SOURCES:
        raise VersionError(f"unknown image source {image_source!r}; known: {sorted(IMAGE_SOURCES)}")

    d.mkdir(parents=True)
    parent_entry = load_registry(root)["versions"][parent]
    for fname in parent_entry["files"]:  # a draft starts as a copy of its parent
        shutil.copyfile(root / parent_entry["dir"] / fname, d / fname)
    meta = {
        "name": name, "parent": parent, "decision": decision, "status": "draft",
        "created": date.today().isoformat(),
        "approved_split_changes": [],   # session_ids whose split may differ from the parent
        "approved_label_changes": [],   # filenames whose label_status may differ from the parent
    }
    if image_source != DEFAULT_IMAGE_SOURCE:  # drafts on raw images keep the VERSION.json shape
        meta["image_source"] = image_source
    (d / VERSION_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return d


def open_new_lineage(name: str, decision: str, dataset_kind: str, extra: dict | None = None,
                     root: Path = PROCESSED_ROOT, decisions_path: Path = DECISIONS_PATH) -> Path:
    """Create an empty draft version with no parent: a different dataset, not a lobster extension.

    Used for the Tritonia sea slug data (D-029/D-030). VERSION.json records `parent: null` and its
    `dataset_kind`; freeze() then skips the parent-extension check, which has no parent to compare
    with. The caller writes the manifests into the returned directory before freezing.
    """
    if is_frozen(name, root):
        raise VersionError(f"dataset version {name} is frozen and immutable ({REGISTRY_NAME}); no override")
    if not dataset_kind or dataset_kind == "lobster":
        raise VersionError("a new lineage needs a dataset_kind other than the lobster data")
    if not decision_exists(decision, decisions_path):
        raise VersionError(f"decision {decision} not found in {decisions_path}; record it first")
    d = version_dir(name, root)
    if d.exists():
        raise VersionError(f"{d} already exists; a new lineage starts in a new version folder")
    d.mkdir(parents=True)
    meta = {"name": name, "parent": None, "decision": decision, "status": "draft",
            "created": date.today().isoformat(), "dataset_kind": dataset_kind, **(extra or {})}
    (d / VERSION_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return d


def _csv_map(path: Path, key: str, value: str) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        return {r[key]: r[value] for r in csv.DictReader(f)}


def check_extends_parent(name: str, root: Path = PROCESSED_ROOT) -> list:
    """Problems that stop draft `name` from being a valid extension of its parent."""
    meta = read_version_meta(name, root)
    if meta is None:
        return [f"{name} has no {VERSION_FILE}"]
    if meta.get("parent") is None:  # a new lineage (open_new_lineage) has nothing to extend
        return [] if meta.get("dataset_kind") not in (None, "lobster") else [f"{name} has no parent"]
    d, pd = version_dir(name, root), version_dir(meta["parent"], root)
    problems = []

    ok_sessions = set(meta.get("approved_split_changes", []))
    parent_split = _csv_map(pd / "split_manifest.csv", "session_id", "split")
    draft_split = _csv_map(d / "split_manifest.csv", "session_id", "split")
    for sid, split in parent_split.items():
        if sid in ok_sessions:
            continue
        if sid not in draft_split:
            problems.append(f"parent session {sid} missing from draft split")
        elif draft_split[sid] != split:
            problems.append(f"session {sid} moved {split} -> {draft_split[sid]} without approval")

    ok_images = set(meta.get("approved_label_changes", []))
    parent_labels = _csv_map(pd / "image_manifest.csv", "filename", "label_status")
    draft_labels = _csv_map(d / "image_manifest.csv", "filename", "label_status")
    for fname, status in parent_labels.items():
        if fname in ok_images:
            continue
        if fname not in draft_labels:
            problems.append(f"parent image {fname} missing from draft manifest")
        elif draft_labels[fname] != status:
            problems.append(f"image {fname} label_status {status} -> {draft_labels[fname]} without approval")
    return problems


def freeze(name: str, root: Path = PROCESSED_ROOT, decisions_path: Path = DECISIONS_PATH) -> dict:
    if is_frozen(name, root):
        raise VersionError(f"{name} is already frozen")
    meta = read_version_meta(name, root)
    if meta is None or meta.get("status") != "draft":
        raise VersionError(f"{name} is not a draft version")
    if not decision_exists(meta["decision"], decisions_path):
        raise VersionError(f"decision {meta['decision']} not found in {decisions_path}")
    problems = check_extends_parent(name, root)
    if problems:
        raise VersionError(f"{name} does not extend {meta['parent']}: " + "; ".join(problems[:10]))

    d = version_dir(name, root)
    meta["status"] = "frozen"
    (d / VERSION_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    entry = {
        "dir": d.relative_to(root.resolve()).as_posix(),
        "parent": meta["parent"],
        "decision": meta["decision"],
        "frozen_on": date.today().isoformat(),
        "files": {p.name: sha256_file(p) for p in sorted(d.iterdir()) if p.is_file()},
    }
    registry = load_registry(root)
    registry["versions"][name] = entry
    (root / REGISTRY_NAME).write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return entry


def add_version_args(parser) -> None:
    parser.add_argument("--version", required=True,
                        help="dataset version to write (frozen versions are refused; no override)")
    parser.add_argument("--from-version", help="frozen parent when creating a new draft")
    parser.add_argument("--decision", help="docs/DECISIONS.md id justifying a new draft")


def output_dir_for(args, root: Path = PROCESSED_ROOT) -> Path:
    return open_draft(args.version, args.from_version, args.decision, root)
