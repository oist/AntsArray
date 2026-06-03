#!/usr/bin/env python3
"""Stage 3: turn selected_frames.csv into SLEAP labeling chunks.

Thin orchestration over scripts/sleap_salvage_project.py:

  1. Reads selected_frames.csv (Stage 2 output) and matches each (camera,
     frame_idx) to the source .slp + .avi in --data-dir.
  2. Writes salvage_input.csv with the 3-column schema that
     sleap_salvage_project expects: sleap, video, frame.
  3. Runs ``sleap_salvage_project build-master --source-mode slp
     --promote-predictions`` to produce the master rescue .slp.
  4. Runs ``sleap_salvage_project split --package`` to fan out the master
     into per-labeler .pkg.slp chunks ready for the GUI.

Output layout under --out-dir:
  salvage_input.csv
  master/master_<prefix>.slp                + manifest_resolved.csv etc.
  chunks/<prefix>_chunk01.pkg.slp ... chunkNN  + chunk_manifest.csv
  build_training_chunks_<ts>.log
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import pandas as pd

CAM_RE = re.compile(r"^(cam\d{2})_")
SALVAGE_SCRIPT = Path(__file__).with_name("sleap_salvage_project.py")


def slp_frame_set(slp_path: Path) -> set[int]:
    """Read frame_idx values from a .slp HDF5 directly (no sleap_io load)."""
    with h5py.File(slp_path, "r") as f:
        return set(int(x) for x in f["frames"]["frame_idx"][:])


def discover_camera_files(data_dir: Path) -> dict[str, dict[str, Path]]:
    """Map cam_id -> {slp, video}. Both must exist; otherwise the cam is skipped."""
    out: dict[str, dict[str, Path]] = {}
    for slp in sorted(data_dir.glob("cam*.slp")):
        m = CAM_RE.match(slp.name)
        if not m:
            continue
        cam = m.group(1)
        video = slp.with_suffix(".avi")
        if not video.exists():
            logging.warning("[%s] no .avi sibling for %s", cam, slp.name)
            continue
        out[cam] = {"slp": slp, "video": video}
    return out


def write_salvage_input(
    selected: pd.DataFrame,
    cam_files: dict[str, dict[str, Path]],
    out_csv: Path,
    dropped_csv: Path,
) -> tuple[int, int]:
    """Write the 3-column CSV that sleap_salvage_project expects.
    Filters out frames not present in the source .slp (inventory may include
    frame indices that the original prediction didn't write to .slp).
    Returns (n_kept, n_dropped); raises if any camera in selected is unmapped."""
    missing_cams = sorted(set(selected["camera"]) - set(cam_files))
    if missing_cams:
        raise RuntimeError(
            f"selected_frames.csv references cameras with no .slp/.avi pair: {missing_cams}"
        )

    rows_kept: list[dict] = []
    rows_dropped: list[dict] = []
    drops_per_cam: dict[str, int] = {}

    for cam in sorted(selected["camera"].unique()):
        slp_path = cam_files[cam]["slp"]
        video_path = cam_files[cam]["video"]
        present = slp_frame_set(slp_path)
        cam_rows = selected[selected["camera"] == cam]
        for _, r in cam_rows.iterrows():
            f = int(r["frame_idx"])
            if f in present:
                rows_kept.append({
                    "sleap": str(slp_path).replace("\\", "/"),
                    "video": str(video_path).replace("\\", "/"),
                    "frame": f,
                })
            else:
                rows_dropped.append({
                    "camera": cam,
                    "frame_idx": f,
                    "stratum": r.get("stratum", ""),
                    "reason": "frame_not_in_source_slp",
                })
                drops_per_cam[cam] = drops_per_cam.get(cam, 0) + 1

    df = pd.DataFrame(rows_kept, columns=["sleap", "video", "frame"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if rows_dropped:
        pd.DataFrame(rows_dropped).to_csv(dropped_csv, index=False)
        for cam, n in sorted(drops_per_cam.items()):
            logging.warning("[%s] dropped %d frame(s) not present in source .slp", cam, n)
        logging.warning("dropped frames detailed in: %s", dropped_csv)

    return len(rows_kept), len(rows_dropped)


def run_salvage(args: list[str]) -> None:
    cmd = [sys.executable, str(SALVAGE_SCRIPT), *args]
    logging.info("running: %s", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"salvage script exited with {res.returncode}: {cmd}")


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_path, encoding="utf-8")],
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selected-csv", type=Path, required=True,
                   help="Output of select_training_frames.py")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory with cam*.slp and cam*.avi")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to write salvage_input.csv, master/, chunks/")
    p.add_argument("--chunks", type=int, default=6,
                   help="Number of per-labeler .pkg.slp chunks (default 6)")
    p.add_argument("--prefix", type=str, default="block01_20260515_round1")
    p.add_argument("--context-frames", type=int, default=0,
                   help="If >0, embed N neighbor frames each side for visual context")
    p.add_argument("--skip-master", action="store_true",
                   help="Re-use existing master .slp from a prior run")
    p.add_argument("--skip-split", action="store_true",
                   help="Build master only, no chunking yet")
    args = p.parse_args()

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(out_dir / f"build_training_chunks_{ts}.log")
    logging.info("selected_csv=%s", args.selected_csv)
    logging.info("data_dir=%s", args.data_dir)
    logging.info("out_dir=%s", out_dir)

    selected = pd.read_csv(args.selected_csv)
    logging.info("selected_frames: %d rows across %d cams",
                 len(selected), selected["camera"].nunique())

    cam_files = discover_camera_files(args.data_dir.resolve())
    if not cam_files:
        logging.error("no cam*.slp/.avi pairs in %s", args.data_dir)
        return 1
    logging.info("discovered %d cam .slp/.avi pairs", len(cam_files))

    salvage_csv = out_dir / "salvage_input.csv"
    dropped_csv = out_dir / "dropped_frames.csv"
    n_kept, n_dropped = write_salvage_input(selected, cam_files, salvage_csv, dropped_csv)
    logging.info("wrote %s (%d kept, %d dropped)", salvage_csv, n_kept, n_dropped)
    if n_kept == 0:
        logging.error("no frames survived .slp-presence filter")
        return 1

    master_dir = out_dir / "master"
    master_dir.mkdir(exist_ok=True)
    master_slp = master_dir / f"master_{args.prefix}.slp"

    if not args.skip_master:
        run_salvage([
            "build-master", str(salvage_csv),
            "--out", str(master_slp),
            "--source-mode", "slp",
            "--promote-predictions",
        ])
        logging.info("master built: %s", master_slp)
    else:
        if not master_slp.exists():
            logging.error("--skip-master requested but %s does not exist", master_slp)
            return 1
        logging.info("re-using existing master: %s", master_slp)

    if args.skip_split:
        logging.info("--skip-split set, done")
        return 0

    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    split_args = [
        "split", str(master_slp),
        "--chunks", str(args.chunks),
        "--out-dir", str(chunks_dir),
        "--prefix", args.prefix,
        "--package",
    ]
    if args.context_frames > 0:
        split_args += ["--context-frames", str(args.context_frames)]
    run_salvage(split_args)
    logging.info("chunks written to %s", chunks_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
