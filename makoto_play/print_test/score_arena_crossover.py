"""Crossover detection rates of two printed sheets in fixed arena cameras (see README.md).

The two sheets sit on one holder; it is still, turned 180 deg by hand, then still again, so each
of the two cameras sees each sheet once. Per camera and still phase:
  - the phase is the longest run of fitted frames in its search window whose main-sheet pose
    (grid centre and angle) stays within CENTRE_TOL_PX / ANGLE_TOL_DEG of the run's first frame,
    with no gap over MAX_GAP_S between fitted frames;
  - every frame of that run counts: tags are placed with the phase's median homography (refined
    on the frame's own reads when at least 8 match), so a frame whose own fit failed still counts
    its misses;
  - a tag-frame whose mean intensity is OCCLUDED_DIFF off that tag's phase median (a hand above
    it) is dropped;
  - a wrong ID is a read on no sheet's grid that sits on a counted tag of this sheet with another ID.

Usage:
  python makoto_play/print_test/score_arena_crossover.py OUT_DIR configs/arena_20261020_1829.json
The config names the videos, the phase search windows (s) and the sheet filling each camera in
each phase, read from the printed footers (cut_footers.py). Check the main-sheet centres this
prints against cut_footers.py: the largest fit is taken as the main sheet.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

import sheet_grid as sg

ANGLE_TOL_DEG, CENTRE_TOL_PX = 1.0, 3.0
MAX_GAP_S = 1.0                            # a longer run of unfitted frames ends a still phase
OCCLUDED_DIFF = 25.0                       # grey levels off the tag's phase median
OTHER_SHEET_SIDES = 10                     # a second fit this many marker sides away is the other sheet
DEFAULT_NPZ = Path(__file__).resolve().parents[2] / "aruco_detection/custom_dicts/custom_4x4_A100_d4_20260410_103938.npz"


def check_config(config):
    sheets = set(config["main"].values())
    if len(sheets) != 2:
        raise SystemExit(f"config must name exactly two sheets, got {sorted(sheets)}")
    for cam in config["cams"]:
        for ph in config["phases"]:
            if f"{cam}:{ph}" not in config["main"]:
                raise SystemExit(f"config 'main' has no sheet for {cam}:{ph}")
    for cam in config["cams"]:
        if {config["main"][f"{cam}:{ph}"] for ph in config["phases"]} != sheets:
            raise SystemExit(f"{cam} must see each sheet once (that is the crossover)")
    return sorted(sheets, key=lambda s: float(s.removesuffix("mm")))


def still_run(fits, window_s, n_frames, fps):
    """Longest run of fitted frames in the window whose main-sheet pose stays at the run's first pose."""
    t0, t1 = window_s
    cand = [i for i in range(int(t0 * fps), min(int(t1 * fps), n_frames)) if fits.get(i)]
    poses = {i: sg.pose(fits[i][0][0]) for i in cand}            # largest fit = main sheet
    best, start = [], 0
    while start < len(cand):
        c0, a0 = poses[cand[start]]
        end = start
        while end + 1 < len(cand):
            c, a = poses[cand[end + 1]]
            if (np.linalg.norm(c - c0) > CENTRE_TOL_PX or abs((a - a0 + 180) % 360 - 180) > ANGLE_TOL_DEG
                    or cand[end + 1] - cand[end] > MAX_GAP_S * fps):
                break
            end += 1
        if end - start + 1 > len(best):
            best = cand[start:end + 1]
        start = end + 1
    if not best:
        raise SystemExit(f"no fitted frames in the {t0}-{t1} s window")
    return best, poses[best[0]]


def phase_views(cam, fits, results, phases, main, fps):
    """(phase, sheet, 'main'|'partial', H, frames) for every sheet in view in each still phase."""
    sheets = sorted(set(main.values()))
    views = []
    for ph, window in phases.items():
        fitted, (ctr, ang) = still_run(fits, window, len(results), fps)
        frames = list(range(fitted[0], fitted[-1] + 1))      # also the frames whose own fit failed
        H_main = sg.norm(np.median([sg.norm(fits[i][0][0]) for i in fitted], axis=0))
        side_px = np.median([sg.match_threshold_px(results[i]["prod_ecr1"][1]) for i in fitted]) / sg.MATCH_MM * sg.MARKER_MM
        others = [sg.norm(H) for i in frames for H, _ in fits.get(i, [])[1:]
                  if np.linalg.norm(sg.pose(H)[0] - ctr) > OTHER_SHEET_SIDES * side_px]
        sheet = main[f"{cam}:{ph}"]
        other = next(s for s in sheets if s != sheet)
        print(f"{cam} phase {ph}: {len(frames)} still frames ({frames[0] / fps:.1f}-{frames[-1] / fps:.1f} s), "
              f"main = {sheet} at ({ctr[0]:.0f}, {ctr[1]:.0f}) px, grid angle {ang:.1f} deg"
              + (f"; partial {other} also in view" if len(others) >= 10 else ""))
        views.append((ph, sheet, "main", H_main, frames))
        if len(others) >= 10:
            views.append((ph, other, "partial", sg.norm(np.median(others, axis=0)), frames))
    return views


def tag_means(video, views, shape):
    """view index -> (frames, 100) mean grey level of each in-frame tag's bounding box (NaN outside)."""
    boxes = []
    for _, _, _, H, _ in views:
        q = sg.marker_quads(H)
        boxes.append((sg.in_frame(q, shape), np.floor(q.min(axis=1)).astype(int), np.ceil(q.max(axis=1)).astype(int)))
    wanted = {i for v in views for i in v[4]}
    means = defaultdict(dict)
    for i, gray in sg.frames(video):
        if i not in wanted:
            continue
        for k, (_, _, _, _, frames) in enumerate(views):
            if i in frames:
                inside, lo, hi = boxes[k]
                m = np.full(100, np.nan)
                for t in np.flatnonzero(inside):
                    m[t] = gray[lo[t, 1]:hi[t, 1], lo[t, 0]:hi[t, 0]].mean()
                means[k][i] = m
    return {k: np.array([means[k][i] for i in v[4]]) for k, v in enumerate(views)}, [b[0] for b in boxes]


def score_camera(cam, video, npz, workers, out, phases, main):
    results = sg.detect_video(video, npz, workers, cache_dir=out)
    shape, fps = sg.frame_shape(video), sg.video_fps(video)
    fits = {i: sg.fit_sheets(*results[i]["prod_ecr1"]) for i in results}
    views = phase_views(cam, fits, results, phases, main, fps)
    means, insides = tag_means(video, views, shape)
    rows = []
    for k, (ph, sheet, kind, H, frames) in enumerate(views):
        with warnings.catch_warnings():                  # tags outside the frame are all-NaN columns
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(means[k], axis=0)
        clear = np.abs(means[k] - median) < OCCLUDED_DIFF   # NaN -> False
        vis = clear & insides[k][None]
        others = [v[3] for j, v in enumerate(views) if v[0] == ph and j != k]
        for cfg in sg.CONFIGS:
            hits = np.zeros((len(frames), 100), bool)
            wrong = off_position = 0
            for r, i in enumerate(frames):
                ids, corners = results[i][cfg]
                _, mine = sg.match_sheet(H, ids, corners, np.ones(len(ids), bool))
                hits[r, ids[mine]] = True
                taken = mine.copy()
                for H_other in others:
                    taken |= sg.match_sheet(H_other, ids, corners, ~taken)[1]
                w, o = sg.wrong_reads(H, ids, corners, ~taken, vis[r])
                wrong, off_position = wrong + w, off_position + o
            per_tag = np.where(vis.sum(0) > 0, (hits & vis).sum(0) / np.maximum(vis.sum(0), 1), np.nan)
            rows.append(dict(cam=cam, phase=ph, sheet=sheet, kind=kind, cfg=cfg, frames=len(frames),
                             visible=int(vis.sum()), read=int((hits & vis).sum()), wrong=wrong,
                             off_position=off_position, hidden=int((~clear & insides[k][None]).sum()),
                             per_tag=per_tag, tags=int(insides[k].sum())))
    return rows


def report(rows, sheets, cams):
    for cfg in sg.CONFIGS:
        print(f"\n=== {cfg} ({sg.CONFIG_NAMES[cfg]}) ===")
        print(f"  {'camera':>6} {'phase':>5} {'sheet':>7} {'view':>7} {'frames':>6} {'tags':>4} {'hidden':>6} "
              f"{'rate':>6} {'>=99%':>6} {'>=90%':>6} {'wrong':>5} {'wrong %':>8} {'off-pos':>7}")
        for r in rows:
            if r["cfg"] != cfg:
                continue
            pt = r["per_tag"][~np.isnan(r["per_tag"])]
            print(f"  {r['cam']:>6} {r['phase']:>5} {r['sheet']:>7} {r['kind']:>7} {r['frames']:>6} {r['tags']:>4} "
                  f"{r['hidden']:>6} {r['read'] / max(r['visible'], 1):>6.3f} {int(np.sum(pt >= 0.99)):>6} "
                  f"{int(np.sum(pt >= 0.90)):>6} {r['wrong']:>5} "
                  f"{100 * r['wrong'] / max(r['read'] + r['wrong'], 1):>7.2f}% {r['off_position']:>7}")
        main = {(r["cam"], r["sheet"]): r for r in rows if r["cfg"] == cfg and r["kind"] == "main"}
        print("  crossover (main views):")
        for sheet in sheets:
            views = [main[c, sheet] for c in cams]
            rates = [v["read"] / max(v["visible"], 1) for v in views]
            wrong, read = sum(v["wrong"] for v in views), sum(v["read"] for v in views)
            print(f"    {sheet:>7}: " + "  ".join(f"{c} {x:.3f}" for c, x in zip(cams, rates))
                  + f"  mean {np.mean(rates):.3f}; wrong IDs {wrong} = {100 * wrong / max(read + wrong, 1):.2f} % of reads")
        s1, s2 = sheets
        for cam in cams:
            a, b = main[cam, s1]["per_tag"], main[cam, s2]["per_tag"]
            ok = ~np.isnan(a) & ~np.isnan(b)
            print(f"    {cam}, same tag on both sheets ({int(ok.sum())} tags): {s1} better {int(np.sum(a[ok] > b[ok]))}, "
                  f"{s2} better {int(np.sum(a[ok] < b[ok]))}, equal {int(np.sum(a[ok] == b[ok]))}; "
                  f"mean per-tag rate {np.mean(a[ok]):.3f} vs {np.mean(b[ok]):.3f}")
    print("\n  wrong = another ID read on a counted tag; off-pos = the tag's own ID read 0.8-1.2 mm off "
          "its placed centre (counted as a miss)")


def run(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(f"config {args.config}: {json.dumps(config)}")
    sheets = check_config(config)
    rows = []
    for cam, video in config["cams"].items():
        rows += score_camera(cam, video, args.npz, args.workers, args.out_dir, config["phases"], config["main"])
    report(rows, sheets, list(config["cams"]))
    with open(args.out_dir / "per_tag_rates.csv", "w", encoding="utf-8") as f:
        f.write("cam,phase,sheet,view,config,tag,rate\n")
        for r in rows:
            for t, v in enumerate(r["per_tag"]):
                if not np.isnan(v):
                    f.write(f"{r['cam']},{r['phase']},{r['sheet']},{r['kind']},{r['cfg']},{t},{v:.4f}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", type=Path)
    p.add_argument("config", help="JSON: cams {cam: video}, phases {name: [t0, t1] s}, main {'cam:phase': sheet}")
    p.add_argument("--npz", default=DEFAULT_NPZ, help="dictionary of the printed sheets (default: Custom_A)")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sg.run_logged(args.out_dir, "score_arena_crossover", run, args)


if __name__ == "__main__":
    main()
