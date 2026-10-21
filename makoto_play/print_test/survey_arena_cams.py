"""Which arena cameras see the printed sheets: reads at a few times per camera, plus a contact sheet.

Usage:
  python makoto_play/print_test/survey_arena_cams.py OUT_DIR "Y:/cam*_2026-10-20-19-*.mkv"
Per camera: frame count, and at each sample time the reads, distinct IDs, the most copies of one ID
(= sheets in view) and the median marker side in px. Reads use the production settings without bit
correction (ecr=0), so they are a little lower than production. OUT_DIR/contact.jpg tiles every
camera at 20 s with the reads outlined.
"""

from __future__ import annotations

import argparse
import glob
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import sheet_grid as sg
from score_arena_crossover import DEFAULT_NPZ

SAMPLE_S = (5, 20, 35, 50)
THUMB = (503, 380)


def survey(job):
    path, npz = job
    cv2.setNumThreads(1)
    det = sg.detector(sg.load_custom_dictionary(npz)[0], 0.0)
    cap = cv2.VideoCapture(str(path))
    fps = sg.video_fps(path)
    want = {round(s * fps): s for s in SAMPLE_S}
    rows, thumb, i = [], None, 0
    while cap.grab():
        if i in want:
            _, f = cap.retrieve()
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            c = Counter() if ids is None else Counter(int(x) for x in ids.ravel())
            side = float(np.median([np.linalg.norm(k[0][1] - k[0][0]) for k in corners])) if corners else 0.0
            rows.append((want[i], sum(c.values()), len(c), max(c.values(), default=0), side))
            if want[i] == SAMPLE_S[1]:
                for k in corners:
                    cv2.polylines(f, [k[0].astype(np.int32)], True, (0, 0, 255), 6)
                thumb = cv2.resize(f, THUMB, interpolation=cv2.INTER_AREA)
                cv2.putText(thumb, Path(path).name[:5], (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        i += 1
    return Path(path).name, i, rows, thumb


def run(args):
    videos = sorted(glob.glob(args.pattern))
    if not videos:
        raise SystemExit(f"no videos match {args.pattern}")
    with Pool(min(13, len(videos))) as pool:
        results = pool.map(survey, [(v, str(args.npz)) for v in videos])
    for name, n, rows, _ in results:
        print(f"{name[:5]}  {n:>5} frames  " + "  ".join(
            f"t{t:>2}s: {reads:>3} reads {ids:>3} IDs x{mult} ({side:>4.0f}px)" for t, reads, ids, mult, side in rows))
    thumbs = [t if t is not None else np.zeros(THUMB[::-1] + (3,), np.uint8) for *_, t in results]
    thumbs += [np.zeros_like(thumbs[0])] * (-len(thumbs) % 5)
    cv2.imwrite(str(args.out_dir / "contact.jpg"),
                np.vstack([np.hstack(thumbs[r:r + 5]) for r in range(0, len(thumbs), 5)]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir", type=Path)
    p.add_argument("pattern", help='glob of one session\'s videos, e.g. "Y:/cam*_2026-10-20-19-*.mkv"')
    p.add_argument("--npz", default=DEFAULT_NPZ)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sg.run_logged(args.out_dir, "survey_arena_cams", run, args)


if __name__ == "__main__":
    main()
