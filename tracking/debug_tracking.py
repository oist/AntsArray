# run_tracking.py
# Small CLI to run get_complete_tracks with visualization for debugging.
# Consolidated: only --video input is required.
# ArUco/SLEAP file paths are inferred from the video filename by replacing the extension:
#   <video_without_ext>_aruco_detections.csv
#   <video_without_ext>_slaep_data.csv   # note: "slaep" as requested

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tracking.tracking_utils import get_complete_tracks


def load_df(p: Path) -> pd.DataFrame:
    ext = p.suffix.lower()
    if ext in {".csv"}:
        return pd.read_csv(p)
    if ext in {".pkl", ".pickle"}:
        return pd.read_pickle(p)
    if ext in {".parquet"}:
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported table format: {ext}. Use csv, pkl, or parquet.")


def infer_detection_paths(video_path: Path) -> tuple[Path, Path]:
    """
    Given a video path like /path/to/file.avi (or .mp4, .mov, etc.),
    return:
      /path/to/file_aruco_detections.csv
      /path/to/file_slaep_data.csv
    """
    # Remove just the final suffix; robust to multi-dot filenames.
    base_no_ext = Path(str(video_path.with_suffix("")))
    aruco = Path(f"{base_no_ext}_aruco_detections.csv")
    # The user explicitly asked for `_sleap_data.csv` (note the 'a'/'e' order).
    sleap = Path(f"{base_no_ext}_sleap_data.csv")
    return aruco, sleap


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run tracking with on-screen visualization for debugging."
    )
    # Single required input
    ap.add_argument("--video", required=True, type=Path, help="Video file path")

    # Optional outputs / parameters
    ap.add_argument("--output", type=Path, default=None, help="Optional tracks .pkl output")
    ap.add_argument("--video-out", type=Path, default=None, help="Optional annotated MP4 output")
    ap.add_argument("--anchor-bodypoint", type=int, default=0)
    ap.add_argument("--max-distance", type=float, default=100.0)
    ap.add_argument("--aruco-sleap-max-distance", type=float, default=None)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--harvest-crops", action="store_true")
    ap.add_argument("--crops-dir", type=Path, default=None)
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--harvest-interval", type=int, default=5)
    args = ap.parse_args()

    # Check video exists
    if not args.video.exists():
        sys.exit(f"Missing video file: {args.video}")

    # Infer detection file paths from the video filename
    aruco_path, sleap_path = infer_detection_paths(args.video)
    #import pdb; pdb.set_trace()

    if not aruco_path.exists():
        sys.exit(f"Missing inferred ArUco file: {aruco_path}")
    if not sleap_path.exists():
        sys.exit(f"Missing inferred SLEAP file: {sleap_path}")
    if args.harvest_crops and args.crops_dir is None:
        sys.exit("Provide --crops-dir when --harvest-crops is set.")

    aruco_df = load_df(aruco_path)
    sleap_df = load_df(sleap_path)

    # Required columns check
    need_aruco_cols = {"Frame", "Instance", "X", "Y"}
    need_sleap_cols = {"Frame", "Instance", "Bodypoint", "X", "Y"}
    if not need_aruco_cols.issubset(aruco_df.columns):
        missing = need_aruco_cols - set(aruco_df.columns)
        sys.exit(f"ArUco file missing columns: {sorted(missing)}")
    if not need_sleap_cols.issubset(sleap_df.columns):
        missing = need_sleap_cols - set(sleap_df.columns)
        sys.exit(f"SLEAP file missing columns: {sorted(missing)}")

    # Visualize is hardwired to True for debugging
    _ = get_complete_tracks(
        output_path=args.output,
        aruco_detection=aruco_df,
        sleap_detection=sleap_df,
        video_file=args.video,
        anchor_bodypoint=args.anchor_bodypoint,
        visualize=True,
        video_out_path=args.video_out,
        harvest_crops=args.harvest_crops,
        crops_output_dir=args.crops_dir,
        crop_size=args.crop_size,
        max_distance=args.max_distance,
        aruco_sleap_max_distance=args.aruco_sleap_max_distance,
        start_frame=args.start_frame,
        harvest_interval=args.harvest_interval,
    )

    print("Tracking finished. Press 'q' in the window to stop early.")


if __name__ == "__main__":
    main()
