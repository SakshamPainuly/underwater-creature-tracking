#!/usr/bin/env python
"""Raw-data fingerprint: record it once, then check raw data against it.

Default fingerprint = STRUCTURAL / COMPLETENESS check:
    every file's relative path + size, and sha256 of .txt / .xlsx / .docx contents.
    PNG (and other binary) files are covered by path + size only, so the default check
    does NOT cryptographically prove PNG pixel content is unchanged.
--full additionally sha256-hashes every file (cryptographic content check; slow, ~9 GB).

    python scripts/check_raw_unchanged.py --write [--full]     # create the fingerprint (once)
    python scripts/check_raw_unchanged.py [--full]             # check local raw data
    python scripts/check_raw_unchanged.py --root /content/raw/biology --subset training
                                          # check a partial copy (e.g. Colab upload)
Exit code: 0 = match, 1 = differences found, 2 = usage/fingerprint problem.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.config import BIOLOGY_ROOT, DATASET_SOURCES, PROJECT_ROOT

FINGERPRINT_PATH = PROJECT_ROOT / "data" / "processed" / "raw_fingerprint.json"
CONTENT_HASHED_SUFFIXES = {".txt", ".xlsx", ".docx"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute(root: Path, full: bool) -> dict:
    files = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        hashed = full or p.suffix.lower() in CONTENT_HASHED_SUFFIXES
        files[rel] = {"size": p.stat().st_size, "sha256": _sha256(p) if hashed else None}
    return files


def training_prefixes() -> list:
    """Folders of sources that have annotation folders (the ones uploaded for training)."""
    prefixes = []
    for _, image_dir, label_dir in DATASET_SOURCES:
        if label_dir is None:
            continue
        for d in {Path(image_dir), Path(label_dir)}:
            prefixes.append(d.relative_to(BIOLOGY_ROOT).as_posix() + "/")
    return sorted(prefixes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=BIOLOGY_ROOT)
    ap.add_argument("--write", action="store_true", help="create the fingerprint (refuses if it exists)")
    ap.add_argument("--full", action="store_true", help="hash every file's contents, incl. PNGs")
    ap.add_argument("--subset", choices=["all", "training"], default="all")
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f"raw root not found: {args.root}")

    if args.write:
        if FINGERPRINT_PATH.exists():
            print(f"{FINGERPRINT_PATH} already exists; the fingerprint is written once and never "
                  "overwritten. Delete it deliberately (with a docs/DECISIONS.md entry) to re-record.")
            sys.exit(2)
        files = compute(args.root, args.full)
        doc = {"kind": "full" if args.full else "structural", "n_files": len(files), "files": files}
        FINGERPRINT_PATH.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {doc['kind']} fingerprint of {len(files)} files -> {FINGERPRINT_PATH}")
        return

    if not FINGERPRINT_PATH.is_file():
        sys.exit(f"no fingerprint at {FINGERPRINT_PATH}; create it with --write")
    ref = json.loads(FINGERPRINT_PATH.read_text(encoding="utf-8"))
    if args.full and ref["kind"] != "full":
        print("The recorded fingerprint is structural (no PNG content hashes), so --full cannot be "
              "compared against it.")
        sys.exit(2)

    expected = ref["files"]
    current = compute(args.root, args.full)
    if args.subset == "training":
        prefixes = training_prefixes()
        keep = lambda rel: any(rel.startswith(p) for p in prefixes)
        expected = {k: v for k, v in expected.items() if keep(k)}
        current = {k: v for k, v in current.items() if keep(k)}

    missing = sorted(set(expected) - set(current))
    added = sorted(set(current) - set(expected))
    size_changed, hash_changed = [], []
    for rel in sorted(set(expected) & set(current)):
        e, c = expected[rel], current[rel]
        if e["size"] != c["size"]:
            size_changed.append(rel)
        elif e["sha256"] and c["sha256"] and e["sha256"] != c["sha256"]:
            hash_changed.append(rel)

    check = "full content (cryptographic)" if args.full else "structural (paths+sizes; txt/xlsx/docx hashed)"
    print(f"check: {check}; subset: {args.subset}; root: {args.root}")
    print(f"files expected: {len(expected)}  present: {len(current)}")
    for name, items in (("missing", missing), ("added", added),
                        ("size changed", size_changed), ("content changed", hash_changed)):
        print(f"  {name}: {len(items)}")
        for rel in items[:20]:
            print(f"    {rel}")
    ok = not (missing or added or size_changed or hash_changed)
    print("RAW DATA MATCHES FINGERPRINT" if ok else "RAW DATA DIFFERS FROM FINGERPRINT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
