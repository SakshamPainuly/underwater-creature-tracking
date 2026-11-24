"""Background-contrast preprocessing (D-045) on made-up frames in a temporary folder (no project data is read)."""

import csv
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dataset_audit import tritonia_bgc as B
from dataset_audit import tritonia_dataset as T
from dataset_audit import versioning as V
from training import experiment as X

DAYS = ["20130519", "20130520", "20130521", "20130522", "20130523"]
W, H = 1056, 480
SLUG = "1 0.100000 0.200000 0.050000 0.100000\n"
LABELLED_T = (100, 300)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frame_bytes(day: str, t: int) -> bytes:
    return f"png {day} {t}".encode()


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def make_batch(d: Path) -> list[str]:
    """2 frames per day with points/, labels.txt, manifest.csv and batch.json (a labelling batch)."""
    (d / "images").mkdir(parents=True)
    (d / "points").mkdir()
    (d / "labels.txt").write_text("lobster\nsea_slug\n", encoding="utf-8")
    rows, stems = [], []
    for day in DAYS:
        for t in LABELLED_T:
            stem = f"{day}all_15spf_t{t:04d}"
            (d / "images" / f"{stem}.png").write_bytes(frame_bytes(day, t))
            (d / "points" / f"{stem}.csv").write_text(f"track,point,x,y,t\n1,1,105.6,96.0,{t}\n", encoding="utf-8")
            rows.append({"image": f"images/{stem}.png", "source_tif": f"{day}all_15spf.tif", "t": t,
                         "page_index": t - 1, "n_points": 1, "tracks": "1", "png_sha256": sha(frame_bytes(day, t))})
            stems.append(stem)
    write_csv(d / "manifest.csv", rows)
    stacks = {f"{day}all_15spf.tif": {"n_frames": 600, "width": W, "height": H, "tif_bytes": 1} for day in DAYS}
    (d / "batch.json").write_text(json.dumps({"seed": 0, "n": 10, "min_gap": 40, "classes": T.CLASS_NAMES,
                                              "totals": {}, "source_stacks": stacks}), encoding="utf-8")
    return stems


def make_sequence(d: Path, step: int = 10, days=DAYS) -> None:
    """Every `step`-th frame of each day plus the labelled frames (same bytes as in the batch)."""
    (d / "images").mkdir(parents=True)
    rows = []
    for day in days:
        for t in sorted(set(range(1, 601, step)) | set(LABELLED_T)):
            name = f"{day}all_15spf_t{t:04d}.png"
            (d / "images" / name).write_bytes(frame_bytes(day, t))
            rows.append({"image": f"images/{name}", "day": day, "source_tif": f"{day}all_15spf.tif", "t": t,
                         "page_index": t - 1, "n_gt_points": 0, "png_sha256": sha(frame_bytes(day, t))})
    write_csv(d / "manifest.csv", rows)


def fake_process(frame, neighbours, params):
    return b"bgc " + Path(frame).name.encode() + b" " + str(len(neighbours)).encode(), 1.5


def fake_write(path, img):
    Path(path).write_bytes(img)


class OffsetsTest(unittest.TestCase):
    def test_offsets_are_symmetric_and_nearest_first(self):
        offs = B.target_offsets()
        self.assertEqual(offs[:4], [-20, 20, -44, 44])
        self.assertEqual(offs[-2:], [-240, 240])
        self.assertEqual(len(offs), 2 * B.PARAMS["n_per_side"])

    def test_dense_pool_hits_every_target_exactly(self):
        picks = B.pick_neighbours(1000, list(range(1, 2001)))
        self.assertEqual([a - 1000 for _, a in picks], B.target_offsets())

    def test_sparse_pool_takes_nearest_within_tolerance_on_the_right_side(self):
        pool = list(range(3, 2000, 8))  # about one frame in 8, like batch 2
        for off, a in B.pick_neighbours(1003, pool):
            self.assertLessEqual(abs(a - (1003 + off)), B.PARAMS["target_tol_frames"])
            self.assertGreaterEqual((a - 1003) * (1 if off > 0 else -1), B.PARAMS["min_offset_frames"])
        self.assertEqual(B.pick_neighbours(1003, pool), B.pick_neighbours(1003, list(reversed(pool))))

    def test_never_uses_close_frames_or_itself_and_fails_when_too_few(self):
        with self.assertRaisesRegex(T.TritoniaError, "only 0 background frames"):
            B.pick_neighbours(100, [95, 99, 100, 101, 110, 119])
        start = B.pick_neighbours(1, list(range(1, 400)))  # day start: one side only
        self.assertEqual(len(start), B.PARAMS["n_per_side"])
        self.assertTrue(all(a > 1 for _, a in start))


class BgcChainTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.t = Path(self._tmp.name)
        self.batch, self.seq = self.t / "batch", self.t / "seq"
        self.stems = make_batch(self.batch)
        make_sequence(self.seq)
        notes = self.t / "notes.txt"
        notes.write_text("", encoding="utf-8")
        z = self.t / "export.zip"
        with zipfile.ZipFile(z, "w") as f:
            for s in self.stems:
                f.writestr(f"{s}.txt", SLUG)
        T.ingest_boxes(z, notes, self.batch, self.t / "boxes")
        self.processed = self.t / "processed"
        self.processed.mkdir()
        self.decisions = self.t / "DECISIONS.md"
        self.decisions.write_text("## D-037 — parent\n## D-045 — test\n", encoding="utf-8")
        T.make_version("v7", "D-037", self.t / "boxes", self.batch, seed=0, n_val=1, n_test=1,
                       root=self.processed, decisions_path=self.decisions)
        V.freeze("v7", root=self.processed, decisions_path=self.decisions)
        self.rows = T._read_csv(self.processed / "v7" / "image_manifest.csv")

    def tearDown(self):
        self._tmp.cleanup()

    def preprocess(self, out="bgc"):
        planned = B.plan(self.rows, B.day_pools(self.batch, [self.seq]))
        return B.write_processed_batch(self.batch, self.rows, planned, self.t / out, "v7",
                                       process=fake_process, write_png=fake_write, inputs=[self.seq])

    def test_full_chain_preprocess_version_fingerprint_freeze_build_verify(self):
        report = self.preprocess()
        self.assertEqual(report["frames"], 10)
        self.assertEqual(set(report["per_split"]), {"train", "val", "test"})
        self.assertLessEqual(report["per_split"]["val"]["mean_abs_offset_error"], 5.0)  # a 10-frame grid: within 5
        out = self.t / "bgc"
        for s in self.stems:
            self.assertTrue((out / "points" / f"{s}.csv").is_file())
        meta = json.loads((out / "batch.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["preprocessing"]["method"], "bgc1")
        self.assertEqual(meta["source_stacks"], json.loads((self.batch / "batch.json").read_text())["source_stacks"])

        d = B.make_bgc_version("v9", "D-045", "v7", out, root=self.processed, decisions_path=self.decisions)
        self.assertEqual((d / "split_manifest.csv").read_bytes(),
                         (self.processed / "v7" / "split_manifest.csv").read_bytes())
        v9 = T._read_csv(d / "image_manifest.csv")
        self.assertEqual([{k: v for k, v in r.items() if k != "image_sha256"} for r in v9],
                         [{k: v for k, v in r.items() if k != "image_sha256"} for r in self.rows])
        self.assertTrue(all(a["image_sha256"] != b["image_sha256"] for a, b in zip(v9, self.rows)))
        vmeta = V.read_version_meta("v9", self.processed)
        self.assertEqual((vmeta["derived_from"], vmeta["parent"], vmeta["frame_batch_name"]), ("v7", None, "bgc"))
        self.assertEqual(vmeta["label_source"], "boxes")

        expected = B.draft_fingerprint("v9", self.t / "boxes", self.t / "fp", root=self.processed)
        V.freeze("v9", root=self.processed, decisions_path=self.decisions)
        summary = T.build_view("v9", out / "images", self.t / "boxes", self.t / "view", "8.4.155",
                               root=self.processed)
        self.assertEqual(summary["dataset_fingerprint"], expected)
        self.assertEqual(summary["image_source"], "bgc")
        cfg = {"dataset_version": "v9", "dataset_fingerprint": expected, "ultralytics_version": "8.4.155"}
        X.verify_built_dataset(self.t / "view", cfg, processed_root=self.processed)
        with self.assertRaisesRegex(T.TritoniaError, "missing or not the frame"):  # raw frames are not v9's
            T.build_view("v9", self.batch / "images", self.t / "boxes", self.t / "view2", "8.4.155",
                         root=self.processed)

    def test_version_refuses_a_batch_made_from_other_frames(self):
        self.preprocess()
        pm = self.t / "bgc" / "preprocess_manifest.csv"
        rows = T._read_csv(pm)
        rows[0]["source_png_sha256"] = "0" * 64
        T._write_csv(pm, rows, list(rows[0]))
        with self.assertRaisesRegex(T.TritoniaError, "preprocess_report"):  # report hash no longer matches
            B.make_bgc_version("v9", "D-045", "v7", self.t / "bgc", root=self.processed, decisions_path=self.decisions)
        report = json.loads((self.t / "bgc" / "preprocess_report.json").read_text(encoding="utf-8"))
        report["preprocess_manifest_sha256"] = V.sha256_file(pm)
        (self.t / "bgc" / "preprocess_report.json").write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(T.TritoniaError, "processed from another frame"):
            B.make_bgc_version("v9", "D-045", "v7", self.t / "bgc", root=self.processed, decisions_path=self.decisions)
        self.assertFalse((self.processed / "v9").exists())

    def test_plan_fails_without_neighbours_or_on_disagreeing_sources(self):
        with self.assertRaisesRegex(T.TritoniaError, "background frames found"):
            B.plan(self.rows, B.day_pools(self.batch, []))  # the batch alone has 2 frames per day
        bad = self.t / "seq_bad"
        make_sequence(bad, days=DAYS[:1])
        (bad / "images" / f"{DAYS[0]}all_15spf_t0100.png").write_bytes(b"other")
        rows = T._read_csv(bad / "manifest.csv")
        for r in rows:
            if r["image"].endswith("t0100.png"):
                r["png_sha256"] = sha(b"other")
        write_csv(bad / "manifest.csv", rows)
        with self.assertRaisesRegex(T.TritoniaError, "disagree"):
            B.day_pools(self.batch, [bad])

    def test_refuses_output_inside_an_input_or_existing(self):
        planned = B.plan(self.rows, B.day_pools(self.batch, [self.seq]))
        for out in (self.batch / "bgc", self.seq / "bgc"):
            with self.assertRaisesRegex(T.TritoniaError, "overlaps"):
                B.write_processed_batch(self.batch, self.rows, planned, out, "v7", process=fake_process,
                                        write_png=fake_write, inputs=[self.seq])
        (self.t / "exists").mkdir()
        with self.assertRaisesRegex(T.TritoniaError, "already exists"):
            B.write_processed_batch(self.batch, self.rows, planned, self.t / "exists", "v7",
                                    process=fake_process, write_png=fake_write)


try:
    import cv2  # noqa: F401
    import numpy as np
except ImportError:  # local Windows PC: no numpy/OpenCV (CLAUDE.md rule 3); runs on Kaggle
    np = None


@unittest.skipIf(np is None, "numpy + OpenCV not installed (runs on Kaggle)")
class ProcessFrameTest(unittest.TestCase):
    def test_channels(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rng = np.random.default_rng(0)
            base = rng.integers(40, 60, size=(H, W), dtype=np.uint8)
            frame = base.copy()
            frame[100:110, 200:215] += 15  # a "slug" 15 grey levels brighter than usual
            paths = []
            for i in range(9):
                p = tmp / f"n{i}.png"
                cv2.imwrite(str(p), base)
                paths.append(p)
            cv2.imwrite(str(tmp / "f.png"), frame)
            bgr, mad = B.process_frame(tmp / "f.png", paths)
            self.assertEqual(bgr.shape, (H, W, 3))
            self.assertTrue((bgr[:, :, 2] == frame).all())               # R = original
            self.assertTrue((bgr[100:110, 200:215, 1] == 60).all())      # G = 15 * gain
            self.assertEqual(int(bgr[:, :, 1].sum()), 60 * 150)          # zero elsewhere
            self.assertAlmostEqual(mad, 15 * 150 / (H * W))


if __name__ == "__main__":
    unittest.main()
