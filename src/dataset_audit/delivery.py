"""Unpack a delivered image ZIP into a new read-only image source root (D-017).

Only `images/<folder>/<name>.png` members are kept, re-rooted at `images/<folder>/`, whatever
top-level folder the ZIP adds. macOS metadata (`__MACOSX/`, `.DS_Store`, `._*`) is skipped. The
result is the same tree as the local ingest, so it can be checked against the source's recorded
content fingerprint (scripts/check_raw_unchanged.py --full).
"""

import re
import shutil
import zipfile
from pathlib import Path

_IMAGE_MEMBER_RE = re.compile(r"(?:^|/)(images/[^/]+/[^/]+\.png)$")


class DeliveryError(RuntimeError):
    pass


def image_members(names: list) -> dict:
    """ZIP member name -> relative output path, for the delivered PNGs only."""
    out = {}
    for name in names:
        parts = name.split("/")
        if "__MACOSX" in parts or parts[-1].startswith("._") or parts[-1] == ".DS_Store":
            continue
        m = _IMAGE_MEMBER_RE.search(name)
        if m:
            out[name] = m.group(1)
    targets = list(out.values())
    if len(set(targets)) != len(targets):
        raise DeliveryError("the ZIP holds the same images/<folder>/<name>.png more than once")
    return out


def unpack_delivery(zip_path: Path, out_dir: Path) -> int:
    """Extract the delivered PNGs into a new, empty `out_dir`; return how many were written."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise DeliveryError(f"{out_dir} is not empty; an image source root is written once")
    with zipfile.ZipFile(zip_path) as z:
        members = image_members(z.namelist())
        if not members:
            raise DeliveryError(f"{zip_path}: no images/<folder>/<name>.png members found")
        for name, rel in sorted(members.items()):
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f, 1 << 20)
    return len(members)
