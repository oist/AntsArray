#!/usr/bin/env python3
"""Batch submit directed ant-ant interaction jobs for chunk track parquets."""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tracking.colony.interaction_one_chunk import process_chunk  # noqa: E402


SIDES = ("left", "right")


@dataclass(frozen=True)
class InteractionJob:
    chunk_file: Path
    chunk_stem: str
    side: str
    output_file: Path


def infer_side(path: Path) -> str | None:
    match = re.search(r"_(left|right)(?:\.parquet)?$", path.name)
    return match.group(1) if match else None


def parse_sides(value: str) -> tuple[str, ...]:
    if value == "both":
        return SIDES
    if value not in SIDES:
        raise ValueError("--side must be left, right, or both")
    return (value,)


def safe_label(value: str, max_len: int = 80) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return (label or "job")[:max_len]


def submit_sbatch(args: list[str]) -> str:
    result = subprocess.run(args, check=True, text=True, capture_output=True)
    return result.stdout.strip().splitlines()[-1]


def discover_jobs(
    input_folder: Path,
    output_path: Path,
    *,
    sides: Iterable[str],
    skip_existing: bool,
    chunks: set[str] | None,
) -> list[InteractionJob]:
    jobs: list[InteractionJob] = []
    sides_set = set(sides)
    chunk_set = None if chunks is None else {str(chunk).zfill(3) for chunk in chunks}

    for chunk_file in sorted(input_folder.glob("*.parquet")):
        side = infer_side(chunk_file)
        if side not in sides_set:
            continue
        chunk_match = re.search(r"_chunk(\d{3})_", chunk_file.name)
        if chunk_set is not None and (chunk_match is None or chunk_match.group(1) not in chunk_set):
            continue

        output_file = output_path / chunk_file.name
        if skip_existing and output_file.exists():
            continue
        jobs.append(
            InteractionJob(
                chunk_file=chunk_file,
                chunk_stem=chunk_file.stem,
                side=side,
                output_file=output_file,
            )
        )

    return jobs


def run_local(
    jobs: list[InteractionJob],
    *,
    mm_per_px: float,
    interaction_radius_mm: float,
    micro_interaction_distance_mm: float,
    antenna_bodypoints: tuple[int, ...],
    frame_start: int,
    max_frames: int | None,
    frame_step: int,
    frame_batch_size: int,
    progress_every_frames: int,
    skip_existing: bool,
) -> None:
    for job in jobs:
        logging.info("Processing chunk=%s", job.chunk_file.name)
        process_chunk(
            chunk_file=job.chunk_file,
            output_path=job.output_file,
            mm_per_px=mm_per_px,
            interaction_radius_mm=interaction_radius_mm,
            micro_interaction_distance_mm=micro_interaction_distance_mm,
            antenna_bodypoints=antenna_bodypoints,
            frame_start=frame_start,
            max_frames=max_frames,
            frame_step=frame_step,
            frame_batch_size=frame_batch_size,
            progress_every_frames=progress_every_frames,
            skip_existing=skip_existing,
        )


def write_worker_script(
    *,
    script_path: Path,
    job: InteractionJob,
    logs_dir: Path,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    job_name: str,
    python_cmd: str,
    worker_script: Path,
    mm_per_px: float,
    interaction_radius_mm: float,
    micro_interaction_distance_mm: float,
    antenna_bodypoints: tuple[int, ...],
    frame_start: int,
    max_frames: int | None,
    frame_step: int,
    frame_batch_size: int,
    progress_every_frames: int,
    skip_existing: bool,
) -> None:
    antenna_args = " ".join(f"--antenna_bodypoint {int(bp)}" for bp in antenna_bodypoints)
    max_frames_arg = "none" if max_frames is None else str(int(max_frames))
    skip_existing_arg = "--skip_existing" if skip_existing else ""
    safe_chunk = safe_label(job.chunk_stem, max_len=70)
    job_label = safe_label(f"{job_name}_{job.side}_{job.chunk_stem}", max_len=40)
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"#SBATCH -J {job_label}",
                f"#SBATCH -p {partition}",
                f"#SBATCH -c {cpus}",
                f"#SBATCH --mem={mem}",
                f"#SBATCH -t {time_limit}",
                f"#SBATCH -o {logs_dir}/{safe_chunk}_%j.out",
                f"#SBATCH -e {logs_dir}/{safe_chunk}_%j.err",
                "",
                "set -euo pipefail",
                "export PYTHONNOUSERSITE=1",
                "",
                'echo "Running on host: $(hostname)"',
                f'echo "Chunk file: {job.chunk_file}"',
                f'echo "Output file: {job.output_file}"',
                f'echo "Python command: {python_cmd}"',
                "python --version || true",
                "",
                " ".join(
                    [
                        f'{python_cmd} "{worker_script}"',
                        f'--chunk_file "{job.chunk_file}"',
                        f'--output_path "{job.output_file}"',
                        f"--mm_per_px {float(mm_per_px)}",
                        f"--interaction_radius_mm {float(interaction_radius_mm)}",
                        f"--micro_interaction_distance_mm {float(micro_interaction_distance_mm)}",
                        antenna_args,
                        f"--frame_start {int(frame_start)}",
                        f"--max_frames {max_frames_arg}",
                        f"--frame_step {int(frame_step)}",
                        f"--frame_batch_size {int(frame_batch_size)}",
                        f"--progress_every_frames {int(progress_every_frames)}",
                        skip_existing_arg,
                    ]
                ).strip(),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def submit_completion_job(
    *,
    name: str,
    done_file: Path,
    dependency_ids: list[str],
    logs_dir: Path,
    partition: str,
    sbatch_bin: str,
) -> str:
    script_path = logs_dir / f"sbatch_{safe_label(name)}_complete.sh"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"#SBATCH -J {safe_label(name, max_len=40)}_done",
                f"#SBATCH -p {partition}",
                "#SBATCH -c 1",
                "#SBATCH --mem=1G",
                "#SBATCH -t 0-00:10:00",
                f"#SBATCH -o {logs_dir}/{safe_label(name)}_complete_%j.out",
                f"#SBATCH -e {logs_dir}/{safe_label(name)}_complete_%j.err",
                "",
                "set -euo pipefail",
                f'mkdir -p "{done_file.parent}"',
                f'printf "completed %s\\n" "$(date "+%Y-%m-%d %H:%M:%S")" > "{done_file}"',
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    dependency = ":".join(dependency_ids)
    return submit_sbatch([sbatch_bin, "--parsable", f"--dependency=afterok:{dependency}", str(script_path)])


def submit_transfer_job(
    *,
    output_path: Path,
    bucket_output_path: Path,
    dependency_id: str | None,
    logs_dir: Path,
    partition: str,
    sbatch_bin: str,
    job_name: str,
) -> str:
    transfer_script = logs_dir / f"sbatch_{safe_label(job_name)}_transfer.sh"
    marker_name = f"{safe_label(job_name)}_transfer_complete.ok"
    transfer_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"#SBATCH -J {safe_label(job_name, max_len=35)}_xfer",
                f"#SBATCH -p {partition}",
                "#SBATCH -c 1",
                "#SBATCH --mem=4G",
                "#SBATCH -t 0-04:00:00",
                f"#SBATCH -o {logs_dir}/{safe_label(job_name)}_transfer_%j.out",
                f"#SBATCH -e {logs_dir}/{safe_label(job_name)}_transfer_%j.err",
                "",
                "set -euo pipefail",
                f'mkdir -p "{bucket_output_path}"',
                f'echo "rsync {output_path}/ -> {bucket_output_path}/"',
                f'rsync -a --partial --protect-args "{output_path}/" "{bucket_output_path}/"',
                (
                    f'printf "completed %s\\nsource={output_path}\\ndestination={bucket_output_path}\\n" '
                    f'"$(date "+%Y-%m-%d %H:%M:%S")" > "{bucket_output_path / marker_name}"'
                ),
                f'echo "TRANSFER_TO_BUCKET_COMPLETE destination={bucket_output_path} marker={bucket_output_path / marker_name}"',
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    transfer_script.chmod(0o755)
    command = [sbatch_bin, "--parsable"]
    if dependency_id is not None:
        command.append(f"--dependency=afterok:{dependency_id}")
    command.append(str(transfer_script))
    return submit_sbatch(command)


def submit_slurm(
    jobs: list[InteractionJob],
    *,
    output_path: Path,
    logs_dir: Path,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    job_name: str,
    conda_env: str,
    conda_bin: str,
    python_bin: str | None,
    sbatch_bin: str,
    mm_per_px: float,
    interaction_radius_mm: float,
    micro_interaction_distance_mm: float,
    antenna_bodypoints: tuple[int, ...],
    frame_start: int,
    max_frames: int | None,
    frame_step: int,
    frame_batch_size: int,
    progress_every_frames: int,
    skip_existing: bool,
    job_ids_file: Path | None,
    complete_job_id_file: Path | None,
    complete_marker_path: Path,
    transfer_job_id_file: Path | None,
    bucket_output_path: Path | None,
) -> tuple[list[str], str, str | None]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    worker_script = Path(__file__).with_name("interaction_one_chunk.py").resolve()
    python_cmd = python_bin or f'{conda_bin} run -n "{conda_env}" python'
    worker_job_ids: list[str] = []

    for job_index, job in enumerate(jobs):
        safe_chunk = safe_label(job.chunk_stem)
        script_path = logs_dir / f"sbatch_{safe_label(job_name)}_{safe_chunk}.sh"
        write_worker_script(
            script_path=script_path,
            job=job,
            logs_dir=logs_dir,
            partition=partition,
            cpus=cpus,
            mem=mem,
            time_limit=time_limit,
            job_name=job_name,
            python_cmd=python_cmd,
            worker_script=worker_script,
            mm_per_px=mm_per_px,
            interaction_radius_mm=interaction_radius_mm,
            micro_interaction_distance_mm=micro_interaction_distance_mm,
            antenna_bodypoints=antenna_bodypoints,
            frame_start=frame_start,
            max_frames=max_frames,
            frame_step=frame_step,
            frame_batch_size=frame_batch_size,
            progress_every_frames=progress_every_frames,
            skip_existing=skip_existing,
        )
        job_id = submit_sbatch([sbatch_bin, "--parsable", str(script_path)])
        worker_job_ids.append(job_id)
        logging.info(
            "Submitted interaction job %d/%d chunk=%s -> %s",
            job_index + 1,
            len(jobs),
            job.chunk_file.name,
            job_id,
        )

    if job_ids_file is not None:
        job_ids_file.parent.mkdir(parents=True, exist_ok=True)
        job_ids_file.write_text("\n".join(worker_job_ids) + ("\n" if worker_job_ids else ""), encoding="utf-8")

    complete_job_id = submit_completion_job(
        name=f"{job_name}_all",
        done_file=complete_marker_path,
        dependency_ids=worker_job_ids,
        logs_dir=logs_dir,
        partition=partition,
        sbatch_bin=sbatch_bin,
    )
    logging.info("Submitted interaction completion marker -> %s", complete_job_id)

    if complete_job_id_file is not None:
        complete_job_id_file.parent.mkdir(parents=True, exist_ok=True)
        complete_job_id_file.write_text(f"{complete_job_id}\n", encoding="utf-8")

    transfer_job_id = None
    if bucket_output_path is not None:
        transfer_job_id = submit_transfer_job(
            output_path=output_path,
            bucket_output_path=bucket_output_path,
            dependency_id=complete_job_id,
            logs_dir=logs_dir,
            partition=partition,
            sbatch_bin=sbatch_bin,
            job_name=job_name,
        )
        logging.info("Submitted interaction bucket transfer -> %s", transfer_job_id)
        if transfer_job_id_file is not None:
            transfer_job_id_file.parent.mkdir(parents=True, exist_ok=True)
            transfer_job_id_file.write_text(f"{transfer_job_id}\n", encoding="utf-8")

    return worker_job_ids, complete_job_id, transfer_job_id


def parse_optional_int(value: str | None) -> int | None:
    if value is None or str(value).lower() in {"none", "off", "all"}:
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_folder", type=Path, required=True, help="Folder with chunk track parquet files.")
    parser.add_argument("--output_path", type=Path, required=True, help="Flash output root for interaction results.")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--runner", choices=("local", "slurm"), default="local")
    parser.add_argument("--logs_dir", type=Path, default=Path("logs"))
    parser.add_argument("--partition", default="compute")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--mem", default="16G")
    parser.add_argument("--time", default="0-12:00:00")
    parser.add_argument("--job_name", default="interactions")
    parser.add_argument("--conda_env", default="aruco_env")
    parser.add_argument("--conda_bin", default="conda")
    parser.add_argument("--python_bin", default=None, help="Python executable to use inside submitted jobs.")
    parser.add_argument("--sbatch_bin", default="sbatch")
    parser.add_argument("--job_ids_file", type=Path, default=None)
    parser.add_argument("--complete_job_id_file", type=Path, default=None)
    parser.add_argument("--complete_marker_path", type=Path, default=None)
    parser.add_argument("--transfer_job_id_file", type=Path, default=None)
    parser.add_argument("--bucket_output_path", type=Path, default=None)
    parser.add_argument("--mm_per_px", type=float, default=0.016)
    parser.add_argument("--interaction_radius_mm", type=float, default=8.0)
    parser.add_argument("--micro_interaction_distance_mm", type=float, default=1.0)
    parser.add_argument("--antenna_bodypoint", action="append", type=int, default=None)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--max_frames", default=None, help="None/all means process each full chunk.")
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--frame_batch_size", type=int, default=3000)
    parser.add_argument("--progress_every_frames", type=int, default=500)
    parser.add_argument("--chunk", action="append", default=None, help="Only process this chunk number. May be repeated.")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.input_folder.is_dir():
        raise NotADirectoryError(args.input_folder)

    antenna_bodypoints = (
        tuple(int(bp) for bp in args.antenna_bodypoint)
        if args.antenna_bodypoint
        else (4, 5, 6, 7, 8, 9)
    )
    chunks = None if args.chunk is None else {str(chunk).zfill(3) for chunk in args.chunk}
    jobs = discover_jobs(
        args.input_folder,
        args.output_path,
        sides=parse_sides(args.side),
        skip_existing=bool(args.skip_existing),
        chunks=chunks,
    )
    complete_marker_path = args.complete_marker_path or (args.output_path / "interactions_complete.ok")
    if not jobs:
        args.output_path.mkdir(parents=True, exist_ok=True)
        done_file = complete_marker_path
        done_file.parent.mkdir(parents=True, exist_ok=True)
        done_file.write_text("completed no work\n", encoding="utf-8")
        logging.info("No interaction jobs to submit. Wrote %s", done_file)
        if args.runner == "slurm" and args.bucket_output_path is not None:
            transfer_job_id = submit_transfer_job(
                output_path=args.output_path,
                bucket_output_path=args.bucket_output_path,
                dependency_id=None,
                logs_dir=args.logs_dir,
                partition=args.partition,
                sbatch_bin=args.sbatch_bin,
                job_name=args.job_name,
            )
            logging.info("Submitted interaction bucket transfer for existing outputs -> %s", transfer_job_id)
            if args.transfer_job_id_file is not None:
                args.transfer_job_id_file.parent.mkdir(parents=True, exist_ok=True)
                args.transfer_job_id_file.write_text(f"{transfer_job_id}\n", encoding="utf-8")
        return

    logging.info("Prepared %d chunk interaction jobs", len(jobs))
    max_frames = parse_optional_int(args.max_frames)
    if args.runner == "local":
        run_local(
            jobs,
            mm_per_px=float(args.mm_per_px),
            interaction_radius_mm=float(args.interaction_radius_mm),
            micro_interaction_distance_mm=float(args.micro_interaction_distance_mm),
            antenna_bodypoints=antenna_bodypoints,
            frame_start=int(args.frame_start),
            max_frames=max_frames,
            frame_step=int(args.frame_step),
            frame_batch_size=int(args.frame_batch_size),
            progress_every_frames=int(args.progress_every_frames),
            skip_existing=bool(args.skip_existing),
        )
        complete_marker_path.parent.mkdir(parents=True, exist_ok=True)
        complete_marker_path.write_text("completed local\n", encoding="utf-8")
        return

    submit_slurm(
        jobs,
        output_path=args.output_path,
        logs_dir=args.logs_dir,
        partition=args.partition,
        cpus=int(args.cpus),
        mem=args.mem,
        time_limit=args.time,
        job_name=args.job_name,
        conda_env=args.conda_env,
        conda_bin=args.conda_bin,
        python_bin=args.python_bin,
        sbatch_bin=args.sbatch_bin,
        mm_per_px=float(args.mm_per_px),
        interaction_radius_mm=float(args.interaction_radius_mm),
        micro_interaction_distance_mm=float(args.micro_interaction_distance_mm),
        antenna_bodypoints=antenna_bodypoints,
        frame_start=int(args.frame_start),
        max_frames=max_frames,
        frame_step=int(args.frame_step),
        frame_batch_size=int(args.frame_batch_size),
        progress_every_frames=int(args.progress_every_frames),
        skip_existing=bool(args.skip_existing),
        job_ids_file=args.job_ids_file,
        complete_job_id_file=args.complete_job_id_file,
        complete_marker_path=complete_marker_path,
        transfer_job_id_file=args.transfer_job_id_file,
        bucket_output_path=args.bucket_output_path,
    )


if __name__ == "__main__":
    main()
