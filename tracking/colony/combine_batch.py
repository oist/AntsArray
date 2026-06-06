#!/usr/bin/env python3
"""
Batch driver for colony panorama PKLs.

This is the batch entry point for colony panorama PKLs. It handles discovery,
grouping, and local/SLURM execution while keeping combine_one_chunk.py as
the single-chunk worker.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tracking.colony.panorama_io import SIDES, extract_key  # noqa: E402


@dataclass(frozen=True)
class ChunkJob:
    key: str
    side: str
    representative_file: Path


def _detector_rank(path: Path) -> int:
    name = path.name
    if "aruco_panorama_x" in name:
        return 0
    if "sleap_panorama_x" in name:
        return 1
    return 2


def discover_jobs(input_folder: Path, sides: Iterable[str]) -> list[ChunkJob]:
    jobs: list[ChunkJob] = []

    for side in sides:
        candidates = sorted(input_folder.glob(f"*_x_{side}*.pkl"))
        by_key: dict[str, list[Path]] = {}

        for fp in candidates:
            key = extract_key(fp.name)
            if key is None:
                logging.warning("Skipping file without <dataset>_chunkNNN key: %s", fp.name)
                continue
            by_key.setdefault(key, []).append(fp)

        for key, fps in sorted(by_key.items()):
            aruco_count = sum(1 for fp in fps if "aruco_panorama_x" in fp.name)
            sleap_count = sum(1 for fp in fps if "sleap_panorama_x" in fp.name)
            if aruco_count == 0 or sleap_count == 0:
                logging.warning(
                    "Skipping key=%s side=%s because files are incomplete (aruco=%d, sleap=%d)",
                    key,
                    side,
                    aruco_count,
                    sleap_count,
                )
                continue

            if aruco_count > 1 or sleap_count > 1:
                logging.warning(
                    "key=%s side=%s has duplicate detector files (aruco=%d, sleap=%d); "
                    "worker will choose lexicographic first per detector",
                    key,
                    side,
                    aruco_count,
                    sleap_count,
                )

            representative = sorted(fps, key=lambda fp: (_detector_rank(fp), fp.name))[0]
            jobs.append(ChunkJob(key=key, side=side, representative_file=representative))

    return jobs


def run_local(
    jobs: list[ChunkJob],
    output_path: Path,
    *,
    max_distance: float = 90.0,
    lost_track_max_frames: int = 120,
    lost_track_max_distance: float | None = None,
    lost_track_aruco_max_distance: float | None = None,
    skip_existing: bool = False,
) -> None:
    from tracking.colony.combine_one_chunk import process_one

    for job in jobs:
        logging.info("Processing key=%s side=%s", job.key, job.side)
        process_one(
            job.representative_file,
            output_path,
            max_distance=max_distance,
            lost_track_max_frames=lost_track_max_frames,
            lost_track_max_distance=lost_track_max_distance,
            lost_track_aruco_max_distance=lost_track_aruco_max_distance,
            skip_existing=skip_existing,
        )


def submit_slurm(
    jobs: list[ChunkJob],
    output_path: Path,
    *,
    logs_dir: Path,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    job_name: str,
    conda_env: str,
    conda_bin: str,
    python_bin: str | None = None,
    max_distance: float = 90.0,
    lost_track_max_frames: int = 120,
    lost_track_max_distance: float | None = None,
    lost_track_aruco_max_distance: float | None = None,
    skip_existing: bool = False,
) -> list[str]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).with_name("combine_one_chunk.py").resolve()
    python_cmd = python_bin or f'{conda_bin} run -n "{conda_env}" python'
    job_ids: list[str] = []

    for job in jobs:
        sbatch_script = logs_dir / f"sbatch_{job_name}_{job.key}_{job.side}.sh"
        sbatch_script.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    f"#SBATCH -J {job_name}_{job.side}_{job.key}",
                    f"#SBATCH -p {partition}",
                    f"#SBATCH -c {cpus}",
                    f"#SBATCH --mem={mem}",
                    f"#SBATCH -t {time_limit}",
                    f"#SBATCH -o {logs_dir}/{job_name}_{job.side}_{job.key}_%j.out",
                    f"#SBATCH -e {logs_dir}/{job_name}_{job.side}_{job.key}_%j.err",
                    "",
                    "set -euo pipefail",
                    "export PYTHONNOUSERSITE=1",
                    "",
                    'echo "Running on host: $(hostname)"',
                    f'echo "Input file: {job.representative_file}"',
                    f'echo "Output root: {output_path}"',
                    f'echo "Python command: {python_cmd}"',
                    "",
                    " ".join(
                        [
                            f'{python_cmd} "{script_path}"',
                            f'--input_file "{job.representative_file}"',
                            f'--output_path "{output_path}"',
                            f"--max_distance {float(max_distance)}",
                            f"--lost_track_max_frames {int(lost_track_max_frames)}",
                            (
                                ""
                                if lost_track_max_distance is None
                                else f"--lost_track_max_distance {float(lost_track_max_distance)}"
                            ),
                            (
                                ""
                                if lost_track_aruco_max_distance is None
                                else f"--lost_track_aruco_max_distance {float(lost_track_aruco_max_distance)}"
                            ),
                            "" if not skip_existing else "--skip_existing",
                        ]
                    ).strip(),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        sbatch_script.chmod(0o755)
        result = subprocess.run(
            ["sbatch", "--parsable", str(sbatch_script)],
            check=True,
            text=True,
            capture_output=True,
        )
        job_id = result.stdout.strip()
        job_ids.append(job_id)
        logging.info(
            "Submitted key=%s side=%s -> job %s",
            job.key,
            job.side,
            job_id,
        )
    return job_ids


def parse_sides(value: str) -> tuple[str, ...]:
    if value == "both":
        return SIDES
    if value not in SIDES:
        raise ValueError("--side must be left, right, or both")
    return (value,)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch combine colony panorama PKLs into track parquets.")
    parser.add_argument("--input_folder", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--runner", choices=("local", "slurm"), default="local")
    parser.add_argument("--logs_dir", type=Path, default=Path("logs"))
    parser.add_argument("--partition", default="compute")
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--time", default="0-24:00:00")
    parser.add_argument("--job_name", default="combine_tracks")
    parser.add_argument("--conda_env", default="aruco_env")
    parser.add_argument("--conda_bin", default="conda")
    parser.add_argument("--python_bin", default=None, help="Python executable to use inside submitted chunk jobs.")
    parser.add_argument("--job_ids_file", type=Path, default=None, help="Write submitted SLURM job IDs here.")
    parser.add_argument("--max_distance", type=float, default=90.0)
    parser.add_argument("--lost_track_max_frames", type=int, default=120)
    parser.add_argument("--lost_track_max_distance", type=float, default=None)
    parser.add_argument("--lost_track_aruco_max_distance", type=float, default=None)
    parser.add_argument("--skip_existing", action="store_true", help="Do not overwrite existing chunk parquets.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.input_folder.is_dir():
        raise NotADirectoryError(args.input_folder)

    args.output_path.mkdir(parents=True, exist_ok=True)
    jobs = discover_jobs(args.input_folder, parse_sides(args.side))
    if not jobs:
        raise RuntimeError(f"No complete ArUco/SLEAP chunk jobs found in {args.input_folder}")

    logging.info("Prepared %d chunk-side jobs", len(jobs))
    if args.runner == "local":
        run_local(
            jobs,
            args.output_path,
            max_distance=args.max_distance,
            lost_track_max_frames=args.lost_track_max_frames,
            lost_track_max_distance=args.lost_track_max_distance,
            lost_track_aruco_max_distance=args.lost_track_aruco_max_distance,
            skip_existing=args.skip_existing,
        )
    else:
        job_ids = submit_slurm(
            jobs,
            args.output_path,
            logs_dir=args.logs_dir,
            partition=args.partition,
            cpus=args.cpus,
            mem=args.mem,
            time_limit=args.time,
            job_name=args.job_name,
            conda_env=args.conda_env,
            conda_bin=args.conda_bin,
            python_bin=args.python_bin,
            max_distance=args.max_distance,
            lost_track_max_frames=args.lost_track_max_frames,
            lost_track_max_distance=args.lost_track_max_distance,
            lost_track_aruco_max_distance=args.lost_track_aruco_max_distance,
            skip_existing=args.skip_existing,
        )
        if args.job_ids_file is not None:
            args.job_ids_file.parent.mkdir(parents=True, exist_ok=True)
            args.job_ids_file.write_text("\n".join(job_ids) + ("\n" if job_ids else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
