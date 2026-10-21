"""Detection rates of two printed sheets in a handheld video (phone), left vs right (see README.md).

Each frame: fit up to two sheets (sheet_grid.fit_sheets), label them left/right by their relative
position in the sheet's own grid frame, and track single-sheet frames. A sheet too poorly read to
fit is placed from the other one via their fixed offset on the table, so its misses still count.
Only tags whose whole marker is inside the frame count. Compares the same tag in the same frame,
also by marker size and by which sheet is in better focus (sharpness of the printed ID labels,
which are identical on every sheet and not shrunk).

Limits: a single-sheet frame is labelled by the nearer last-seen sheet position, which can go stale
while the phone pans; check the labels on a few frames. Label boxes near the frame edge enter the
sharpness score even when partly outside the frame.

Usage:
  python makoto_play/print_test/score_phone_video.py VIDEO OUT_DIR --left-label 0.03mm --right-label 0mm
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import sheet_grid as sg
from score_arena_crossover import DEFAULT_NPZ

SHARP_PX_PER_MM = 20
ROW_BINS = tuple(range(11))                            # print rows 0-9
PX_BINS = (0, 40, 50, 60, 70, 1000)                   # marker side in px
FOCUS_BINS = (-9, -0.3, -0.1, 0.1, 0.3, 9)             # log(left sharpness / right sharpness)


def grid_centre(H):
    return sg.project(H, sg.GRID_CENTRE[None])[0]


def assign_sides(fits):
    """frame -> ['left'|'right', ...] in fit order: relative position when two fit, tracking otherwise."""
    labels, last, pending = {}, {}, []
    order = sorted(fits)
    for i in order:
        if len(fits[i]) == 2:
            (Ha, _), (Hb, _) = fits[i]
            gb = sg.project(np.linalg.inv(Ha), grid_centre(Hb)[None])[0]   # sheet b in sheet a's grid
            labels[i] = ["left", "right"] if gb[0] > sg.GRID_CENTRE[0] else ["right", "left"]
            last = {labels[i][0]: grid_centre(Ha), labels[i][1]: grid_centre(Hb)}
        elif len(fits[i]) == 1:
            if last:
                c = grid_centre(fits[i][0][0])
                labels[i] = [min(last, key=lambda s: np.linalg.norm(last[s] - c))]
                last[labels[i][0]] = c
            else:
                pending.append(i)
    for i in reversed(pending):                          # before the first two-sheet frame
        nxt = next((j for j in order if j > i and len(labels.get(j, [])) == 2), None)
        if nxt is not None:
            ref = {s: grid_centre(fits[nxt][k][0]) for k, s in enumerate(labels[nxt])}
            c = grid_centre(fits[i][0][0])
            labels[i] = [min(ref, key=lambda s: np.linalg.norm(ref[s] - c))]
    return labels


def complete_fits(fits, sides, results):
    """Place the sheet missing from single-sheet frames from the other one (they do not move)."""
    offsets = []
    for i, lab in sides.items():
        if len(lab) == 2:
            H = dict(zip(lab, (f[0] for f in fits[i])))
            offsets.append(sg.norm(np.linalg.inv(H["right"]) @ H["left"]))   # left grid -> right grid
    if not offsets:
        return 0
    T = np.median(offsets, axis=0)
    c = np.append(sg.GRID_CENTRE, 1.0)
    ref = (T @ c)[:2] / (T @ c)[2]
    dev = [np.linalg.norm((t @ c)[:2] / (t @ c)[2] - ref) for t in offsets]
    print(f"  sheet offset from {len(offsets)} two-sheet frames: left grid centre at ({ref[0]:.1f}, {ref[1]:.1f}) mm "
          f"in the right grid; scatter median {np.median(dev):.2f}, 95% {np.percentile(dev, 95):.2f} mm")
    placed = 0
    for i, lab in list(sides.items()):
        if len(lab) == 1:
            Hk, mk = fits[i][0]
            Hm = Hk @ np.linalg.inv(T) if lab[0] == "left" else Hk @ T
            fits[i] = fits[i] + [sg.match_sheet(Hm, *results[i], ~mk)]
            sides[i] = lab + ["right" if lab[0] == "left" else "left"]
            placed += 1
    return placed


def label_sharpness(video, placed):
    """(frame, side) -> var(Laplacian) / var(intensity) over the in-view ID labels, sheet warped to 20 px/mm."""
    A = np.array([[SHARP_PX_PER_MM, 0, 2 * SHARP_PX_PER_MM], [0, SHARP_PX_PER_MM, 2 * SHARP_PX_PER_MM], [0, 0, 1.0]])
    size = (int((9 * sg.PITCH_X + 4) * SHARP_PX_PER_MM), int((9 * sg.PITCH_Y + 5) * SHARP_PX_PER_MM))
    boxes = []
    for gx, gy in sg.GRID:   # label text sits 1.25-2.3 mm below the tag centre
        x0, x1 = (np.array([gx - 0.9, gx + 0.9]) + 2) * SHARP_PX_PER_MM
        y0, y1 = (np.array([gy + 1.25, gy + 2.3]) + 2) * SHARP_PX_PER_MM
        boxes.append((slice(int(y0), int(y1)), slice(int(x0), int(x1))))
    out = {}
    for i, gray in sg.frames(video):
        for side, (H, vis) in placed.get(i, {}).items():
            if vis.sum() >= 10:
                canon = cv2.warpPerspective(gray, A @ np.linalg.inv(H), size, flags=cv2.INTER_AREA)
                lap = cv2.Laplacian(canon.astype(np.float32), cv2.CV_32F)
                px = np.concatenate([canon[b].ravel() for t, b in enumerate(boxes) if vis[t]]).astype(np.float32)
                lp = np.concatenate([lap[b].ravel() for t, b in enumerate(boxes) if vis[t]])
                out[i, side] = float(lp.var() / max(px.var(), 1e-6))
    return out


def score(results, shape):
    """Rows (one per sheet-frame and config), per-tag reads, and the ecr=1 placements for sharpness."""
    rows, per_tag, placed = [], [], {}
    for cfg in sg.CONFIGS:
        res = {i: results[i][cfg] for i in results}
        fits = {i: sg.fit_sheets(*res[i]) for i in res}
        sides = assign_sides(fits)
        print(f"[{cfg}]")
        n_placed = complete_fits(fits, sides, res)
        print(f"  sheets placed from the other sheet: {n_placed} frames; "
              f"frames with neither sheet fitted: {sum(1 for i in res if i not in sides)}")
        for i in sorted(sides):
            ids, corners = res[i]
            taken = np.any([m for _, m in fits[i]], axis=0)
            for (H, m), side in zip(fits[i], sides[i]):
                quads = sg.marker_quads(H)
                vis = sg.in_frame(quads, shape)
                hit = np.zeros(100, bool)
                hit[ids[m]] = True
                hit &= vis
                wrong = sg.wrong_reads(H, ids, corners, ~taken, vis)[0]
                px = float(np.median(np.linalg.norm(quads[vis, 1] - quads[vis, 0], axis=1))) if vis.any() else np.nan
                rows.append(dict(frame=i, config=cfg, side=side, visible=int(vis.sum()), correct=int(hit.sum()),
                                 wrong=wrong, marker_px=round(px, 1)))
                per_tag += [(i, cfg, side, t, bool(hit[t])) for t in np.flatnonzero(vis)]
                if cfg == "prod_ecr1":
                    placed.setdefault(i, {})[side] = (H, vis)
    return rows, per_tag, placed


def binned(pairs, edges, names):
    """Read rate of each sheet over (left read, right read, key) pairs, per [lo, hi) bin of the key."""
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [(a, b) for a, b, k in pairs if lo <= k < hi]
        if sel:
            a, b = np.array(sel, float).T
            print(f"    [{lo:>5}, {hi:>5}): {names['left']} {a.mean():.3f}  {names['right']} {b.mean():.3f}  "
                  f"diff {a.mean() - b.mean():+.3f}  (pairs {len(sel)})")


def summarize(rows, per_tag, sharp, names):
    px = defaultdict(list)
    for r in rows:
        px[r["frame"], r["config"]].append(r["marker_px"])
    for cfg in sg.CONFIGS:
        print(f"\n=== {cfg} ({sg.CONFIG_NAMES[cfg]}) ===")
        for side in ("left", "right"):
            r = [x for x in rows if x["config"] == cfg and x["side"] == side and x["visible"]]
            vis, cor = sum(x["visible"] for x in r), sum(x["correct"] for x in r)
            print(f"  {names[side]:>7}: {len(r)} frames, {cor}/{vis} tags read = {cor / max(vis, 1):.3f}, "
                  f"wrong IDs {sum(x['wrong'] for x in r)}")
        seen = defaultdict(dict)
        for i, c, s, t, h in per_tag:
            if c == cfg:
                seen[i, t][s] = h
        pairs = [(v["left"], v["right"], i, t) for (i, t), v in seen.items() if len(v) == 2]
        if not pairs:
            continue
        a = np.array([[l, r] for l, r, _, _ in pairs], float)
        print(f"  same tag, same frame, both sheets ({len(pairs)} pairs): {names['left']} {a[:, 0].mean():.3f}, "
              f"{names['right']} {a[:, 1].mean():.3f}; only {names['left']} read {int(np.sum(a[:, 0] > a[:, 1]))}, "
              f"only {names['right']} read {int(np.sum(a[:, 0] < a[:, 1]))}")
        print("  by print row (row 0 = IDs 0-9 at the top):")
        binned([(l, r, t // 10) for l, r, _, t in pairs], ROW_BINS, names)
        print("  by marker size in px (mean of the two sheets in that frame):")
        binned([(l, r, np.nanmean(px[i, cfg])) for l, r, i, _ in pairs], PX_BINS, names)
        print("  by focus balance, log(label sharpness left / right):")
        binned([(l, r, np.log(sharp[i, "left"] / sharp[i, "right"])) for l, r, i, _ in pairs
                if (i, "left") in sharp and (i, "right") in sharp], FOCUS_BINS, names)


def run(args):
    names = {"left": args.left_label, "right": args.right_label}
    print(f"video {args.video}\nleft sheet = {names['left']}, right sheet = {names['right']}")
    results = sg.detect_video(args.video, args.npz, args.workers)
    shape = sg.frame_shape(args.video)
    print(f"{len(results)} frames, {shape[1]}x{shape[0]}")
    rows, per_tag, placed = score(results, shape)
    sharp = label_sharpness(args.video, placed)
    with open(args.out_dir / "per_frame.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["frame", "config", "sheet", "visible", "correct", "wrong", "marker_px",
                                           "label_sharpness"])
        wr.writeheader()
        for r in rows:
            wr.writerow({**{k: v for k, v in r.items() if k != "side"}, "sheet": names[r["side"]],
                         "label_sharpness": sharp.get((r["frame"], r["side"]), "")})
    with open(args.out_dir / "per_tag.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "config", "sheet", "tag", "read"])
        wr.writerows((i, c, names[s], t, int(h)) for i, c, s, t, h in per_tag)
    summarize(rows, per_tag, sharp, names)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("video")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--left-label", default="left")
    p.add_argument("--right-label", default="right")
    p.add_argument("--npz", default=DEFAULT_NPZ)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sg.run_logged(args.out_dir, "score_phone_video", run, args)


if __name__ == "__main__":
    main()
