#!/usr/bin/env python3
"""Cache all-bodypoint motion evidence for sleep tuning in one stitched track."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from analysis.sleep_motion_utils import compute_sleep_motion_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=None, help="Input per-track parquet. Defaults to $TRACK_PATH.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to $TASK_OUTPUT_DIR.")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--mm_per_px", type=float, default=0.016)
    parser.add_argument(
        "--cache_max_gap_frames",
        type=int,
        default=120,
        help=(
            "Largest bodypoint sample gap retained in the cache. Downstream analysis can choose any "
            "smaller gap without rereading pose data. Default: 120."
        ),
    )
    args = parser.parse_args()

    track = args.track or Path(os.environ["TRACK_PATH"])
    out_dir = args.out or Path(os.environ["TASK_OUTPUT_DIR"])
    metadata = compute_sleep_motion_cache(
        track,
        out_dir,
        fps=float(args.fps),
        mm_per_px=float(args.mm_per_px),
        cache_max_gap_frames=int(args.cache_max_gap_frames),
    )
    print(
        f"Wrote sleep-motion cache for {metadata['track_name']}: "
        f"{metadata['n_cached_frames']:,} observed frames x {metadata['n_bodypoints']} bodypoints "
        f"across a {metadata['n_frames']:,}-frame span, "
        f"{metadata['n_valid_speed_values']:,} valid speeds"
    )


if __name__ == "__main__":
    main()
