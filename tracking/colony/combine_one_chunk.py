#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from tracking.colony.panorama_io import (  # noqa: E402
    extract_key,
    infer_side,
    load_aruco_pkl,
    load_sleap_pkl,
    pick_one,
)


def process_one(
    input_file: Path,
    output_root: Path,
    *,
    max_distance: float = 100.0,
    lost_track_max_frames: int = 120,
    lost_track_max_distance: float | None = None,
    lost_track_aruco_max_distance: float | None = None,
    skip_existing: bool = False,
) -> None:
    """
    Flat-folder behavior:
      - derive key (<dataset>_chunkNNN) from input_file basename
      - derive side from filename
      - locate matching sleap+aruco PKLs for that key+side in input_file.parent
      - write output to output_root/<key>_<side>.parquet
    """
    if not input_file.is_file():
        raise FileNotFoundError(f"input_file does not exist: {input_file}")

    output_root.mkdir(parents=True, exist_ok=True)

    bn = input_file.name
    key = extract_key(bn)
    if key is None:
        raise ValueError(f"Could not extract <dataset>_chunkNNN key from filename: {bn}")

    side = infer_side(bn)
    if side is None:
        raise ValueError(f"Could not infer side (left/right) from filename: {bn}")

    out_parquet = output_root / f"{key}_{side}.parquet"
    if skip_existing and out_parquet.exists():
        logging.info("Skipping existing %s", out_parquet)
        return

    in_dir = input_file.parent

    # Match only within the same key and side.
    # We look for detector-specific patterns so we don't confuse aruco vs sleap.
    sleap_matches = list(in_dir.glob(f"{key}*sleap_panorama_x*{side}*.pkl"))
    aruco_matches = list(in_dir.glob(f"{key}*aruco_panorama_x*{side}*.pkl"))

    sleap_file = pick_one(sleap_matches, label=f"{key} side={side} sleap")
    aruco_file = pick_one(aruco_matches, label=f"{key} side={side} aruco")

    if sleap_file is None or aruco_file is None:
        logging.error(
            "Missing required files for key=%s side=%s in %s (sleap=%d, aruco=%d)",
            key,
            side,
            in_dir,
            len(sleap_matches),
            len(aruco_matches),
        )
        # Fail the job so SLURM shows it clearly (instead of silently skipping).
        raise SystemExit(2)

    logging.info("Key: %s", key)
    logging.info("Side: %s", side)
    logging.info("SLEAP: %s", sleap_file.name)
    logging.info("ARUCO: %s", aruco_file.name)

    from tracking.core.tracking_utils import get_complete_tracks

    sleap_det = load_sleap_pkl(sleap_file).dropna(
        subset=["Frame", "Instance", "Bodypoint", "X", "Y"]
    )
    aruco_det, num_frames = load_aruco_pkl(aruco_file)

    get_complete_tracks(
        output_path=str(out_parquet),
        aruco_detection=aruco_det,
        sleap_detection=sleap_det,
        num_frames=num_frames,
        max_distance=max_distance,
        lost_track_max_frames=lost_track_max_frames,
        lost_track_max_distance=lost_track_max_distance,
        lost_track_aruco_max_distance=lost_track_aruco_max_distance,
        stream_output=True,
    )
    logging.info("Wrote %s", out_parquet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        required=True,
        type=Path,
        help="Representative PKL file (used to derive key and side).",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=Path,
        help="Flat output directory.",
    )
    parser.add_argument("--max_distance", type=float, default=100.0)
    parser.add_argument("--lost_track_max_frames", type=int, default=120)
    parser.add_argument("--lost_track_max_distance", type=float, default=None)
    parser.add_argument("--lost_track_aruco_max_distance", type=float, default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Do not overwrite existing output parquet.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    process_one(
        args.input_file,
        args.output_path,
        max_distance=args.max_distance,
        lost_track_max_frames=args.lost_track_max_frames,
        lost_track_max_distance=args.lost_track_max_distance,
        lost_track_aruco_max_distance=args.lost_track_aruco_max_distance,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
