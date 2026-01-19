#!/usr/bin/env python3
import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tracking.tracking_utils import get_complete_tracks  # noqa: E402

# Example filename:
#   20251118_121514_chunk000_aruco_panorama_x_left1740.pkl
# Key should be:
#   20251118_121514_chunk000
KEY_RE = re.compile(r"^(.+_chunk[0-9]{3})", re.IGNORECASE)
SIDES = ("left", "right")


from pathlib import Path
import pandas as pd

def load_sleap_pkl(path: Path) -> pd.DataFrame:
    # SLEAP outputs are DataFrame pickles
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"SLEAP PKL did not contain a DataFrame: {path} (type={type(df)})")
    return df

def load_aruco_pkl(path: Path) -> tuple[pd.DataFrame, int]:
    # ArUco outputs are dict payloads: {"detections": DataFrame, "num_frames": int}
    payload = pd.read_pickle(path)
    if not isinstance(payload, dict) or "detections" not in payload:
        raise TypeError(f"ARUCO PKL did not contain expected dict payload: {path} (type={type(payload)})")

    det = payload["detections"]
    if not isinstance(det, pd.DataFrame):
        raise TypeError(f"ARUCO payload['detections'] is not a DataFrame: {path} (type={type(det)})")

    num_frames = int(payload.get("num_frames", -1))
    return det, num_frames


def extract_key(filename: str) -> str | None:
    m = KEY_RE.match(filename)
    return m.group(1) if m else None


def infer_side(filename: str) -> str | None:
    fn = filename.lower()
    for s in SIDES:
        if f"_x_{s}" in fn or f"_{s}" in fn:
            return s
    return None


def pick_one(matches: list[Path], label: str) -> Path | None:
    """Pick lexicographically first to match submit script behavior."""
    if not matches:
        return None
    matches_sorted = sorted(matches, key=lambda p: p.name)
    if len(matches_sorted) > 1:
        logging.warning(
            "%s has %d matching files; using: %s",
            label,
            len(matches_sorted),
            matches_sorted[0].name,
        )
    return matches_sorted[0]


def process_one(input_file: Path, output_root: Path) -> None:
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

    sleap_det = load_sleap_pkl(sleap_file).dropna()
    aruco_det, num_frames = load_aruco_pkl(aruco_file)


    out_parquet = output_root / f"{key}_{side}.parquet"
    get_complete_tracks(
        output_path=str(out_parquet),
        aruco_detection=aruco_det,
        sleap_detection=sleap_det,
        num_frames=num_frames
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    process_one(args.input_file, args.output_path)


if __name__ == "__main__":
    main()
