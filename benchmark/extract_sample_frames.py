#!/usr/bin/env python3
"""Extract evenly-spaced frames from a video, saved as grayscale PNGs."""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-frames", type=int, default=100)
    p.add_argument("--max-read", type=int, default=7200,
                   help="Max raw frames to read sequentially (default: 7200)")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    read_limit = min(args.max_read, total) if args.max_read > 0 else total

    # Read sequentially, pick evenly spaced frames
    target_indices = set(np.linspace(0, read_limit - 1, args.n_frames, dtype=int))

    saved = 0
    for idx in tqdm(range(read_limit), desc="Reading"):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in target_indices:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(out / f"frame_{idx:06d}.png"), gray)
            saved += 1

    cap.release()
    print(f"Saved {saved} frames to {out}")


if __name__ == "__main__":
    main()
