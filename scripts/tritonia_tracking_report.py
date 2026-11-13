#!/usr/bin/env python
r"""Charts and tables of the D-043 Tritonia tracking results, val vs the final test (reporting only).

    python scripts/tritonia_tracking_report.py --test-run D:\uwct_tritonia\runs\T002_test \
        --test-refs D:\uwct_tritonia\runs\T002_test_ref_no-link D:\uwct_tritonia\runs\T002_test_ref_gt \
        --test-trajectories D:\uwct_tritonia\runs\T002_test_trajectories --out D:\uwct_tritonia\runs\T002_test_report

Reads only logged results: val from experiments/records/T001/ and experiments/records/tracking_d043/phase5/, test
from the given folders (scores.json / summary.json of the single final test, D-043 phase 6). Writes out/charts/*.svg
(3 bar charts, dataviz reference palette slots 1-2: val blue, test orange) and out/tables.md + out/tables.csv.
Nothing here changes a setting; the test result is reported, never used to choose (rule 7). Px only, no units.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VAL_SCORES = REPO / "experiments/records/T001/selected_scores.json"
VAL_RECORD = REPO / "experiments/records/T001/run_record.json"
VAL_TRAJ = REPO / "experiments/records/tracking_d043/phase5/summary.json"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated, adjacent pair)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
METRICS = ("HOTA", "DetA", "AssA", "IDF1")


def load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _bar(x, y_top, w, y0, colour, r=4):
    h = y0 - y_top
    if h <= 0:
        return ""
    r = min(r, h, w / 2)
    return (f'<path d="M{x:.1f},{y0:.1f} V{y_top + r:.1f} Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} '
            f'H{x + w - r:.1f} Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f} V{y0:.1f} Z" fill="{colour}"/>')


def bar_chart(title: str, subtitle: str, cats: list[str], bars: list[list[tuple]], legend: list[tuple],
              label: set, note: str, ymax: float = 1.0) -> str:
    """bars[i] = [(value, colour), ...] for category i (1 or 2 bars). label = {(i, j)} bars that get a value label."""
    W, H, L, R, T, B = 760, 440, 56, 20, 96, 64
    pw, ph = W - L - R, H - T - B
    y0 = T + ph
    y = lambda v: T + ph * (1 - max(0.0, min(v, ymax)) / ymax)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="Segoe UI, Arial, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
           f'<text x="16" y="28" font-size="17" font-weight="600" fill="{INK}">{title}</text>',
           f'<text x="16" y="48" font-size="12" fill="{INK2}">{subtitle}</text>']
    lx = 16
    for name, c in legend:
        out.append(f'<rect x="{lx}" y="62" width="12" height="12" rx="3" fill="{c}"/>')
        out.append(f'<text x="{lx + 18}" y="72" font-size="12" fill="{INK}">{name}</text>')
        lx += 18 + 7.2 * len(name) + 22
    step = 0.25 if ymax >= 1 else 0.1
    for k in range(int(round(ymax / step)) + 1):
        v = step * k
        out.append(f'<line x1="{L}" y1="{y(v):.1f}" x2="{W - R}" y2="{y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{L - 8}" y="{y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="{INK2}">{v:.2f}</text>')
    slot = pw / len(cats)
    bw = min(44.0, slot * 0.3)
    for i, (cat, group) in enumerate(zip(cats, bars)):
        cx = L + slot * (i + 0.5)
        n = len(group)
        x_start = cx - (n * bw + (n - 1) * 2) / 2
        for j, (v, c) in enumerate(group):
            x = x_start + j * (bw + 2)
            out.append(_bar(x, y(v), bw, y0, c))
            if (i, j) in label:
                out.append(f'<text x="{x + bw / 2:.1f}" y="{y(v) - 6:.1f}" font-size="12" font-weight="600" '
                           f'text-anchor="middle" fill="{INK}">{v:.3f}</text>')
        out.append(f'<text x="{cx:.1f}" y="{y0 + 20}" font-size="12" text-anchor="middle" fill="{INK}">{cat}</text>')
    out.append(f'<line x1="{L}" y1="{y0}" x2="{W - R}" y2="{y0}" stroke="{INK2}" stroke-width="1"/>')
    out.append(f'<text x="16" y="{H - 14}" font-size="11" fill="{INK2}">{note}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-run", type=Path, required=True)
    ap.add_argument("--test-refs", type=Path, nargs=2, required=True, help="no-link folder, then clicks-as-detections folder")
    ap.add_argument("--test-trajectories", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"REFUSED: {args.out} already exists")
    vs, vr, vt = load(VAL_SCORES), load(VAL_RECORD), load(VAL_TRAJ)
    ts = load(args.test_run / "scores.json")
    tf, tc = load(args.test_refs[0] / "scores.json"), load(args.test_refs[1] / "scores.json")
    tt = load(args.test_trajectories / "summary.json")
    for s, want in ((ts, "tracker"), (tf, "no-link"), (tc, "gt")):
        if s["split"] != "test" or s["final"] != "D-043" or s["source"]["mode"] != want:
            sys.exit(f"FAILED: unexpected test result file (mode {s['source']['mode']}, split {s['split']})")
    if {k: ts["params"][k] for k in ("conf", "gate_px", "max_gap", "min_len")} != vs["params"]:
        sys.exit("FAILED: the test run did not use the T002 settings")
    vm, tm = vs["mean_over_days"], ts["mean_over_days"]
    va, ta = vs["all_days"], ts["all_days"]
    refs = {"val": {"no linking": vr["references_at_selected"]["no_link"]["HOTA"], "T002": vm["HOTA"],
                    "clicks as detections": vr["references_at_selected"]["clicks_as_detections"]["HOTA"]},
            "test": {"no linking": tf["mean_over_days"]["HOTA"], "T002": tm["HOTA"],
                     "clicks as detections": tc["mean_over_days"]["HOTA"]}}
    (args.out / "charts").mkdir(parents=True)
    legend = [("val (3 days, used to choose settings)", BLUE), ("final test (3 days, run once)", ORANGE)]
    note = "Tritonia sparse-frame/timelapse tracking (15 s per frame), scored on clicked frames, 10 px; D-043."
    charts = {
        "1_val_vs_test_metrics.svg": bar_chart(
            "T002 tracker: val vs final test", "Mean over 3 days; higher is better (0 to 1)", list(METRICS),
            [[(vm[m], BLUE), (tm[m], ORANGE)] for m in METRICS], legend, {(0, 0), (0, 1)}, note),
        "2_references_hota.svg": bar_chart(
            "Detection limits tracking: HOTA with E010 detections vs with the clicks",
            "Same tracker settings (T002); mean over 3 days", ["no linking (floor)", "T002 on E010 detections",
                                                               "clicks as detections (ceiling)"],
            [[(refs["val"][k], BLUE), (refs["test"][k], ORANGE)] for k in refs["val"]], legend,
            {(1, 0), (1, 1), (2, 0), (2, 1)}, note),
        "3_hota_per_day.svg": bar_chart(
            "T002 HOTA per day", "Val days (blue) were used to choose the settings; test days (orange) only scored once",
            sorted(vs["per_day"]) + sorted(ts["per_day"]),
            [[(vs["per_day"][d]["all_cameras"]["HOTA"], BLUE)] for d in sorted(vs["per_day"])]
            + [[(ts["per_day"][d]["all_cameras"]["HOTA"], ORANGE)] for d in sorted(ts["per_day"])],
            legend, set(), note + " Values in tables.md.", ymax=0.5),
    }
    for name, svg in charts.items():
        (args.out / "charts" / name).write_text(svg, encoding="utf-8")

    rows = []
    def add(table, split, key, **vals):
        rows.append({"table": table, "split": split, "key": key, **vals})  # rounded only when written
    for split, m, a, s in (("val", vm, va, vs), ("test", tm, ta, ts)):
        add("overall", split, "mean over 3 days", HOTA=m["HOTA"], DetA=m["DetA"], AssA=m["AssA"], LocA=m["LocA"],
            IDF1=m["IDF1"], MOTA=m["MOTA"], IDSW=a["IDSW"], TP=a["TP"], FN=a["FN"], FP=a["FP"],
            clicks_matched=a["TP"] / (a["TP"] + a["FN"]), MT=a["MT"], PT=a["PT"], ML=a["ML"])
        for d, v in sorted(s["per_day"].items()):
            x = v["all_cameras"]
            add("per_day", split, d, HOTA=x["HOTA"], DetA=x["DetA"], AssA=x["AssA"], IDF1=x["IDF1"], MOTA=x["MOTA"],
                IDSW=x["IDSW"], eval_frames=v["eval_frames"])
        for k, v in refs[split].items():
            add("references_HOTA", split, k, HOTA=v)
    for split, t in (("val", vt), ("test", tt)):
        for d, v in sorted(t["per_day"].items()):
            add("trajectories", split, d, clicked_tracks=v["clicks"]["tracks"], clicks_median_path_px=v["clicks"]["median_path_px"],
                tracker_tracks=v["tracker"]["tracks"], tracker_median_path_px=v["tracker"]["median_path_px"],
                matched_half=v["matched"]["with_a_tracker_track_covering_half"])
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(args.out / "tables.csv", "x", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        wr.writeheader()
        wr.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    md = ["# Tritonia tracking (D-043): T002, val vs final test", "",
          "Final test run once (2026-11-13), settings frozen before it (T002 = T001's val choice: conf 0.16, gate 20 px, "
          "max gap 60, min length 10) on E010 detections. Scored on clicked frames, 10 px, cameras summed per day. "
          "Sparse-frame/timelapse analysis (15 s per frame); px only. Test sessions are new days of the same setup "
          "(not new sites); UNKNOWN whether every slug was clicked, so unmatched detections are an upper bound on false positives.", ""]
    for table, cols in (("overall", ["HOTA", "DetA", "AssA", "LocA", "IDF1", "MOTA", "IDSW", "TP", "FN", "FP", "clicks_matched", "MT", "PT", "ML"]),
                        ("per_day", ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "eval_frames"]),
                        ("references_HOTA", ["HOTA"]),
                        ("trajectories", ["clicked_tracks", "clicks_median_path_px", "tracker_tracks", "tracker_median_path_px", "matched_half"])):
        md += [f"## {table}", "", "| split | " + ("day" if table in ("per_day", "trajectories") else "run") + " | " + " | ".join(cols) + " |",
               "|---|---|" + "---:|" * len(cols)]
        for r in rows:
            if r["table"] == table:
                md.append(f"| {r['split']} | {r['key']} | " + " | ".join(
                    (f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |")
        md.append("")
    (args.out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"test mean HOTA {tm['HOTA']:.3f} (val {vm['HOTA']:.3f}) | wrote {args.out}")


if __name__ == "__main__":
    main()
