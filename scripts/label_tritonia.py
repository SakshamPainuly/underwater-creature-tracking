#!/usr/bin/env python
"""Draw sea slug boxes on the Tritonia labelling batch (D-029) -- a small window, standard library only.

    python scripts/label_tritonia.py --batch D:\\uwct_work\\tritonia_label_batch1

Mouse:  drag = draw a box (always sea_slug) | drag a box edge or corner = resize it
        right-click a box = delete it
Keys:   Right / n = next frame     Left / p = previous frame     h = show/hide the + hints
        z = zoom 1x/2x             u = undo last change             e = exclude this frame (asks why)
        c = contrast on/off (display only; each camera view stretched on its own)
        o = add a note             x = export ZIP for the ingest  q = save and quit
Every frame is saved when you leave it (labels/<frame>.txt next to the batch, see --labels).
A box across a camera border is refused as you draw it. Rules: docs/TRITONIA_LABELLING.md.

Suggested boxes (D-031 proposal; off with --no-suggest): a frame with no label file yet opens with
boxes suggested from the image next to each cross (orange), and a red circle on each cross that got
no suggestion. Check every one: delete wrong ones, resize loose ones, draw missing ones. Leaving the
frame saves what is on screen as your reviewed labels; labels/_provenance/<frame>.json records
which boxes were suggested, and x also writes a provenance CSV next to the ZIP.
s = replace this frame's boxes with fresh suggestions (u undoes it), e.g. to redo loose hand boxes.

--split val,test shows only the frames of those D-030 splits (day split, seed 0, 3 val + 3 test days);
x still exports every labelled frame of the batch.
"""

import argparse
import csv
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_audit.config import RAW_ROOTS
from dataset_audit.tritonia_dataset import TILE_COLS, TILE_ROWS
from dataset_audit.tritonia_point_eval import frames_in_splits
from dataset_audit import tritonia_labeller as L
from dataset_audit import tritonia_prelabel as P

GRAB_SCREEN_PX = 6  # how close (screen px) the mouse must be to a box edge to resize it
COLOURS = {"hand": "#00e676", "suggested": "#ff9100", "suggested_resized": "#00e676"}


class Labeller:
    def __init__(self, root: tk.Tk, batch: Path, labels: Path, suggest: bool = True,
                 splits: set[str] | None = None):
        self.root, self.batch, self.labels, self.suggest = root, batch, labels, suggest
        self.origins, self.sugg = [], None  # origin per box; suggestion record of this frame or None
        with open(batch / "manifest.csv", newline="", encoding="utf-8") as f:
            self.all_stems = [Path(r["image"]).stem for r in csv.DictReader(f)]
        self.stems = frames_in_splits(self.all_stems, splits) if splits else list(self.all_stems)
        if not self.stems:
            raise SystemExit(f"no frames in splits {sorted(splits)}")
        self.notes = batch / "notes.txt"
        self.excluded = L.excluded_stems(self.notes, set(self.all_stems))
        first = tk.PhotoImage(file=str(batch / "images" / f"{self.stems[0]}.png"))
        self.w, self.h = first.width(), first.height()
        self.zoom, self.show_hints, self.boxes, self.drag = 1, False, [], None
        self.resizing, self.history, self.box_items = None, [], []
        self.contrast = True  # faint slugs are much easier to see; never changes saved files
        # start at the first frame still to do: unlabelled; with --split also labelled by hand but
        # not yet checked against suggestions (no provenance record)
        todo = (lambda s: not L.provenance_path(labels, s).is_file()) if splits else                (lambda s: not (labels / f"{s}.txt").is_file())
        self.index = next((i for i, s in enumerate(self.stems) if s not in self.excluded and todo(s)), 0)

        self.status = tk.Label(root, anchor="w", font=("Segoe UI", 10, "bold"))
        self.status.pack(fill="x")
        self.message = tk.Label(root, anchor="w", fg="#b00020", font=("Segoe UI", 10))
        self.message.pack(fill="x")
        frame = tk.Frame(root)
        frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(frame, bg="black", highlightthickness=0, cursor="crosshair")
        sx = tk.Scrollbar(frame, orient="horizontal", command=self.canvas.xview)
        sy = tk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=sx.set, yscrollcommand=sy.set)
        sy.pack(side="right", fill="y")
        sx.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        tk.Label(root, anchor="w", fg="#444", justify="left", text=(
            "drag = box (sea_slug)   drag edge/corner = resize   right-click = delete box   u = undo   s = re-suggest   h = hints   c = contrast   z = zoom   "
            "Right/n = next   Left/p = previous   e = exclude frame   o = note   x = export ZIP   q = quit"
        )).pack(fill="x")

        c = self.canvas
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Button-3>", self.on_delete)
        c.bind("<Motion>", self.on_motion)
        for keys, fn in ((("<Right>", "n"), lambda: self.move(1)), (("<Left>", "p"), lambda: self.move(-1)),
                         (("h",), self.toggle_hints), (("z",), self.toggle_zoom), (("u",), self.undo),
                         (("c",), self.toggle_contrast), (("s",), self.resuggest),
                         (("e",), self.exclude), (("o",), self.note), (("x",), self.export), (("q",), self.quit)):
            for k in keys:
                root.bind(k, lambda _e, fn=fn: fn())
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.load()

    # ------------------------------------------------------------ frames
    @property
    def stem(self) -> str:
        return self.stems[self.index]

    def load(self):
        path = self.labels / f"{self.stem}.txt"
        self.history = []
        if path.is_file():
            self.boxes = L.load_label(path, self.w, self.h)
            rec = L.load_provenance(self.labels, self.stem, len(self.boxes))
            self.origins = list(rec["origins"]) if rec else ["hand"] * len(self.boxes)
            self.sugg = ({k: rec[k] for k in ("suggested_total", "suggested_deleted", "flags")}
                         if rec else None)
        elif self.suggest and self.stem not in self.excluded:
            self.apply_suggestions()
        else:
            self.boxes, self.origins, self.sugg = [], [], None
        self.draw()

    def apply_suggestions(self):
        _, _, px = L.read_gray_png(self.batch / "images" / f"{self.stem}.png")
        found = P.suggest_frame(px, self.w, self.h, P.read_points(self.batch / "points" / f"{self.stem}.csv"))
        self.boxes = [r["box"] for r in found if r["box"] is not None]
        self.origins = ["suggested"] * len(self.boxes)
        flags = [{"x": r["cross"][0], "y": r["cross"][1], "flag": r["flag"]} for r in found if r["box"] is None]
        self.sugg = {"suggested_total": len(self.boxes), "suggested_deleted": 0, "flags": flags}
        self.message.config(text=(
            f"{len(self.boxes)} suggested boxes (orange): check each one."
            + (f"  {len(flags)} cross(es) with no suggestion (red circle): look there yourself." if flags else "")))

    def resuggest(self):
        """s: replace this frame's boxes with fresh suggestions (u brings the old boxes back)."""
        if self.stem in self.excluded:
            return
        self.history.append((list(self.boxes), list(self.origins), self.sugg and dict(self.sugg)))
        self.apply_suggestions()
        self.draw()

    def save(self) -> bool:
        """Save the current frame; False if the user cancels leaving an unboxed, unlabelled frame."""
        if self.stem in self.excluded:
            return True
        path = self.labels / f"{self.stem}.txt"
        if not self.boxes and not path.is_file():
            answer = messagebox.askyesnocancel(
                "No boxes", f"{self.stem} has no boxes.\n\nYes = save it as 'no slug in this frame'\n"
                            "No = leave it unlabelled for now\nCancel = stay here")
            if answer is None:
                return False
            if answer is False:
                return True
        L.save_label(path, self.boxes, self.w, self.h)
        if self.sugg is not None:
            L.save_provenance(self.labels, self.stem, self.origins, self.sugg["suggested_total"],
                              self.sugg["suggested_deleted"], self.sugg["flags"], P.PARAMS)
        return True

    def move(self, step: int):
        if not self.save():
            return
        self.index = (self.index + step) % len(self.stems)
        self.message.config(text="")
        self.load()

    # ------------------------------------------------------------ drawing
    def draw(self):
        c, z = self.canvas, self.zoom
        c.delete("all")
        folder = "hints" if self.show_hints else "images"
        path = self.batch / folder / f"{self.stem}.png"
        if self.contrast:
            w, h, px = L.read_gray_png(path)
            img = tk.PhotoImage(data=L.pgm(w, h, L.stretch_tiles(px, w, h, TILE_COLS, TILE_ROWS)), format="PPM")
        else:
            img = tk.PhotoImage(file=str(path))
        self.photo = img.zoom(z) if z > 1 else img
        c.create_image(0, 0, image=self.photo, anchor="nw")
        c.configure(scrollregion=(0, 0, self.w * z, self.h * z))
        for i in range(1, TILE_COLS):
            c.create_line(self.w * i / TILE_COLS * z, 0, self.w * i / TILE_COLS * z, self.h * z, fill="#e0c000", dash=(4, 4))
        for i in range(1, TILE_ROWS):
            c.create_line(0, self.h * i / TILE_ROWS * z, self.w * z, self.h * i / TILE_ROWS * z, fill="#e0c000", dash=(4, 4))
        self.box_items = [c.create_rectangle(xa * z, ya * z, xb * z, yb * z, outline=COLOURS[o], width=2)
                          for (xa, ya, xb, yb), o in zip(self.boxes, self.origins)]
        for f in (self.sugg or {}).get("flags", []):
            c.create_oval((f["x"] - 9) * z, (f["y"] - 9) * z, (f["x"] + 9) * z, (f["y"] + 9) * z,
                          outline="#ff1744", width=2)
        if self.stem in self.excluded:
            c.create_text(self.w * z / 2, self.h * z / 2, text="EXCLUDED (see notes.txt)", fill="#ff5252",
                          font=("Segoe UI", 24, "bold"))
        p = L.progress(self.labels, self.stems, self.excluded)
        n_sugg = self.origins.count("suggested")
        unchecked = f" ({n_sugg} orange = suggested, not changed by you)" if n_sugg else ""
        self.status.config(text=(
            f"Frame {self.index + 1}/{len(self.stems)}  {self.stem}   boxes: {len(self.boxes)}"
            f"{unchecked}   "
            f"[{'HINTS' if self.show_hints else 'clean'} | contrast {'on' if self.contrast else 'off'} | zoom {z}x]     "
            f"done {p['labelled']}  excluded {p['excluded']}  left {p['left']}"))

    def to_image(self, e) -> tuple[float, float]:
        return self.canvas.canvasx(e.x) / self.zoom, self.canvas.canvasy(e.y) / self.zoom

    def remember(self):
        """Keep the boxes as they are now, so u can go back to them."""
        self.history.append((list(self.boxes), list(self.origins), self.sugg and dict(self.sugg)))

    def handle_at(self, e):
        x, y = self.to_image(e)
        return L.find_handle(self.boxes, x, y, GRAB_SCREEN_PX / self.zoom)

    def on_motion(self, e):
        if self.drag or self.resizing:
            return
        h = None if self.stem in self.excluded else self.handle_at(e)
        edges = h[1] if h else ""
        cursor = ("sizing" if len(edges) == 2 else "sb_h_double_arrow" if edges in ("l", "r")
                  else "sb_v_double_arrow" if edges else "crosshair")
        if self.canvas.cget("cursor") != cursor:
            self.canvas.config(cursor=cursor)

    def on_press(self, e):
        if self.stem in self.excluded:
            self.message.config(text="This frame is excluded; boxes are not used.")
            return
        h = self.handle_at(e)
        if h:
            self.resizing = (h[0], h[1], self.boxes[h[0]])
            self.canvas.itemconfig(self.box_items[h[0]], outline="#40c4ff", dash=(3, 2))
            return
        x, y = self.to_image(e)
        self.drag = (x, y, self.canvas.create_rectangle(x * self.zoom, y * self.zoom, x * self.zoom, y * self.zoom,
                                                       outline="#40c4ff", width=2, dash=(3, 2)))

    def on_drag(self, e):
        if self.resizing:
            i, edges, box = self.resizing
            xa, ya, xb, yb = L.resize_box(box, edges, *self.to_image(e), self.w, self.h)
            z = self.zoom
            self.canvas.coords(self.box_items[i], xa * z, ya * z, xb * z, yb * z)
        elif self.drag:
            x0, y0, item = self.drag
            x, y = self.to_image(e)
            self.canvas.coords(item, x0 * self.zoom, y0 * self.zoom, x * self.zoom, y * self.zoom)

    def on_release(self, e):
        if self.resizing:
            i, edges, box = self.resizing
            self.resizing = None
            new = L.resize_box(box, edges, *self.to_image(e), self.w, self.h)
            problems = L.check_new_box(new, self.w, self.h)
            if problems:
                self.message.config(text="Resize not kept: " + "; ".join(problems))
            elif new != box:
                self.remember()
                self.boxes[i] = new
                if self.origins[i] == "suggested":
                    self.origins[i] = "suggested_resized"
                self.message.config(text="")
            self.draw()
            return
        if not self.drag:
            return
        x0, y0, _ = self.drag
        self.drag = None
        box = L.normalise_px(x0, y0, *self.to_image(e), self.w, self.h)
        problems = L.check_new_box(box, self.w, self.h)
        if problems:
            self.message.config(text="Box not kept: " + "; ".join(problems) +
                                     ("  (one box per camera view)" if "border" in problems[0] else ""))
        else:
            self.remember()
            self.boxes.append(box)
            self.origins.append("hand")
            self.message.config(text="")
        self.draw()

    def on_delete(self, e):
        x, y = self.to_image(e)
        inside = [i for i, b in enumerate(self.boxes) if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
        if inside:
            self.remember()
            i = min(inside, key=lambda k: (self.boxes[k][2] - self.boxes[k][0]) * (self.boxes[k][3] - self.boxes[k][1]))
            del self.boxes[i]
            if self.origins.pop(i) != "hand" and self.sugg is not None:
                self.sugg["suggested_deleted"] += 1
            self.draw()

    def undo(self):
        """Go back one change on this frame (a drawn, resized or deleted box)."""
        if self.history:
            self.boxes, self.origins, self.sugg = self.history.pop()
            self.draw()

    # ------------------------------------------------------------ notes, export
    def toggle_hints(self):
        self.show_hints = not self.show_hints
        self.draw()

    def toggle_contrast(self):
        self.contrast = not self.contrast
        self.draw()

    def toggle_zoom(self):
        self.zoom = 2 if self.zoom == 1 else 1
        self.draw()

    def exclude(self):
        if self.stem in self.excluded:
            return
        reason = simpledialog.askstring("Exclude frame", f"Why can {self.stem} not be labelled completely?",
                                        parent=self.root)
        if reason and reason.strip():
            L.append_note(self.notes, self.stem, "exclude", reason)
            self.excluded.add(self.stem)
            self.draw()

    def note(self):
        reason = simpledialog.askstring("Note", f"Note for {self.stem} (e.g. 'cross of track 12 has no visible slug'):",
                                        parent=self.root)
        if reason and reason.strip():
            L.append_note(self.notes, self.stem, "note", reason)
            self.message.config(text="Note saved to notes.txt")

    def export(self):
        if not self.save():
            return
        p = L.progress(self.labels, self.all_stems, self.excluded)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        out = self.batch.parent / f"tritonia_batch1_yolo_{stamp}.zip"
        n = L.export_zip(self.labels, self.all_stems, out)
        prov = self.batch.parent / f"tritonia_batch1_provenance_{stamp}.csv"
        L.export_provenance_csv(self.labels, self.all_stems, self.w, self.h, prov)
        messagebox.showinfo("Exported", f"{n} label files written to\n{out}\n"
                                        f"box origins (hand / suggested) in\n{prov}\n\n"
                                        f"done {p['labelled']}, excluded {p['excluded']}, left {p['left']}")

    def quit(self):
        if self.save():
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=Path, required=True, help="labelling batch folder (manifest.csv, images/, hints/)")
    parser.add_argument("--labels", type=Path, help="where label files go (default: <batch>/../tritonia_batch1_labels)")
    parser.add_argument("--no-suggest", action="store_true", help="open unlabelled frames empty (no suggested boxes)")
    parser.add_argument("--split", help="only frames of these D-030 splits, comma-separated (train, val, test)")
    args = parser.parse_args()
    batch = args.batch.resolve()
    labels = (args.labels or batch.parent / "tritonia_batch1_labels").resolve()
    for forbidden in (*RAW_ROOTS, batch / "images", batch / "hints"):
        f = Path(forbidden).resolve()
        if labels == f or f in labels.parents:
            sys.exit(f"REFUSED: labels folder {labels} is inside {f}")
    if not (batch / "manifest.csv").is_file():
        sys.exit(f"{batch} has no manifest.csv; pass the labelling batch folder")
    labels.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.title("Tritonia sea slug labelling (D-029)")
    root.geometry("1100x620")
    splits = {x.strip() for x in args.split.split(",")} if args.split else None
    if splits and not splits <= {"train", "val", "test"}:
        sys.exit(f"--split takes train, val, test; got {args.split}")
    Labeller(root, batch, labels, suggest=not args.no_suggest, splits=splits)
    root.mainloop()


if __name__ == "__main__":
    main()
