"""Read-only parsing of 'What's been done.xlsx' using only the standard library.

Only the 'Counts' sheet is used here (deployment-level substrate/visibility/
lighting/done/sent columns). The workbook is never opened for writing.
"""

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import SPREADSHEET_PATH

_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# Shape of a deployment/session code, e.g. "16_HERR_01_CW_20220509".
# Used to filter out section headers / blank rows / unrelated columns in the sheet.
_CODE_SHAPE_RE = re.compile(r"^\d{2}_[A-Za-z0-9]+_\d{2}_[A-Za-z]+_\d{8}$")

_LIGHTING_KEYWORDS = ("redlight", "red light", "night", "lowlight", "low light")


def _col_row(cell_ref: str):
    m = re.match(r"([A-Z]+)(\d+)", cell_ref)
    letters, row = m.groups()
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1, int(row) - 1


def _load_shared_strings(z: zipfile.ZipFile) -> list:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    shared = []
    for si in root.findall("main:si", _NS):
        texts = si.findall(".//main:t", _NS)
        shared.append("".join(t.text or "" for t in texts))
    return shared


def _load_sheet_grid(z: zipfile.ZipFile, sheet_path: str, shared: list) -> dict:
    root = ET.fromstring(z.read(sheet_path))
    grid = {}
    for row in root.findall(".//main:sheetData/main:row", _NS):
        for c in row.findall("main:c", _NS):
            ref = c.get("r")
            if not ref:
                continue
            col, r = _col_row(ref)
            t = c.get("t")
            v_el = c.find("main:v", _NS)
            is_el = c.find("main:is", _NS)
            val = None
            if is_el is not None:
                texts = is_el.findall(".//main:t", _NS)
                val = "".join(tt.text or "" for tt in texts)
            elif v_el is not None:
                raw = v_el.text
                val = shared[int(raw)] if t == "s" else raw
            grid[(r, col)] = val
    return grid


def _clean(v):
    if v is None:
        return ""
    return str(v).strip()


def load_session_metadata() -> dict:
    """Returns {session_id: {substrate, visibility, lighting_flag, lighting_notes,
    free_notes, done, sent, files_used, spreadsheet_n_images}}.
    """
    path = Path(SPREADSHEET_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"Spreadsheet not found: {path}")

    with zipfile.ZipFile(path) as z:
        shared = _load_shared_strings(z)
        # 'Counts' is the first sheet (xl/worksheets/sheet1.xml) -- verified during
        # the Phase 0 audit against xl/workbook.xml's <sheets> ordering.
        grid = _load_sheet_grid(z, "xl/worksheets/sheet1.xml", shared)

    max_row = max(r for r, _ in grid) if grid else -1
    metadata = {}
    for r in range(max_row + 1):
        code = _clean(grid.get((r, 0)))
        if not _CODE_SHAPE_RE.match(code):
            continue  # section header, blank row, or unrelated summary cell

        files_used = _clean(grid.get((r, 1)))
        n_images_txt = _clean(grid.get((r, 2)))
        notes = _clean(grid.get((r, 3)))
        substrate = _clean(grid.get((r, 4)))
        visibility = _clean(grid.get((r, 5)))
        done = _clean(grid.get((r, 6))).lower()
        sent = _clean(grid.get((r, 7))).lower()

        combined_text = f"{notes} {visibility}".lower()
        lighting_flag = any(kw in combined_text for kw in _LIGHTING_KEYWORDS)

        try:
            spreadsheet_n_images = int(n_images_txt) if n_images_txt else None
        except ValueError:
            spreadsheet_n_images = None

        if code in metadata:
            # A handful of sessions were filmed by two GoPros and appear as two
            # spreadsheet rows with the same code (e.g. one row per recording file).
            # Their substrate/visibility/lighting values were identical in the
            # source data on inspection; we just union the files_used field and
            # keep the first row's other fields rather than silently overwriting.
            existing = metadata[code]
            existing["files_used"] = "; ".join(
                filter(None, [existing["files_used"], files_used])
            )
            if spreadsheet_n_images and existing["spreadsheet_n_images"]:
                existing["spreadsheet_n_images"] += spreadsheet_n_images
            continue

        metadata[code] = {
            "substrate": substrate,
            "visibility": visibility,
            "lighting_flag": lighting_flag,
            "free_notes": notes,
            "done": done,
            "sent": sent,
            "files_used": files_used,
            "spreadsheet_n_images": spreadsheet_n_images,
        }

    return metadata
