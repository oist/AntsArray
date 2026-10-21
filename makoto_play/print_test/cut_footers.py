"""Which sheet is which: cut every fitted sheet's footer ("Black shrink: ...") upright out of chosen frames.

Also prints each camera's sheet poses once a second, which shows the still phases for the config.

Usage:
  python makoto_play/print_test/cut_footers.py OUT_DIR 2,20 cam19=VIDEO cam20=VIDEO
Writes OUT_DIR/footers.jpg and caches the reads as OUT_DIR/<video stem>_detections.pkl (shared
with score_arena_crossover.py when it writes to the same OUT_DIR).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import sheet_grid as sg
from score_arena_crossover import DEFAULT_NPZ

PX_PER_MM = 30
X0, X1, Y0, Y1 = -2.0, 36.0, 38.0, 41.0   # footer band in grid mm (tag 0 centre = origin)


def run(args):
    A = np.array([[PX_PER_MM, 0, -X0 * PX_PER_MM], [0, PX_PER_MM, -Y0 * PX_PER_MM], [0, 0, 1.0]])
    size = (int((X1 - X0) * PX_PER_MM), int((Y1 - Y0) * PX_PER_MM))
    strips = []
    for cam, video in (a.split("=", 1) for a in args.cams):
        results = sg.detect_video(video, args.npz, args.workers, cache_dir=args.out_dir)
        fps = sg.video_fps(video)
        print(f"{cam}: {len(results)} frames at {fps:g} fps")
        for t in range(int(len(results) / fps) + 1):
            i = min(round(t * fps), len(results) - 1)
            poses = [f"n={m.sum():3d} c=({c[0]:5.0f},{c[1]:5.0f}) a={a:7.1f}"
                     for H, m in sg.fit_sheets(*results[i]["prod_ecr1"]) for c, a in [sg.pose(H)]]
            print(f"  t={t:2d}s reads={len(results[i]['prod_ecr1'][0]):3d} | " + " | ".join(poses))
        wanted = {round(t * fps): t for t in args.times}
        for i, gray in sg.frames(video):
            if i > max(wanted):
                break
            if i not in wanted:
                continue
            for k, (H, m) in enumerate(sg.fit_sheets(*results[i]["prod_ecr1"])):
                strip = cv2.cvtColor(cv2.warpPerspective(gray, A @ np.linalg.inv(H), size, flags=cv2.INTER_AREA),
                                     cv2.COLOR_GRAY2BGR)
                c = sg.pose(H)[0]
                cv2.putText(strip, f"{cam} t={wanted[i]:g}s fit{k} n={m.sum()} centre=({c[0]:.0f},{c[1]:.0f})",
                            (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                strips.append(strip)
    if not strips:
        raise SystemExit(f"no sheet fitted at {args.times} s in any camera")
    path = args.out_dir / "footers.jpg"
    cv2.imwrite(str(path), np.vstack(strips))
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", type=Path)
    p.add_argument("times", type=lambda s: [float(t) for t in s.split(",")], help="seconds, e.g. 2,20")
    p.add_argument("cams", nargs="+", help="CAM=VIDEO")
    p.add_argument("--npz", default=DEFAULT_NPZ)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sg.run_logged(args.out_dir, "cut_footers", run, args)


if __name__ == "__main__":
    main()
