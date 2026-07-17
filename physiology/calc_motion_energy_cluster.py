#!/usr/bin/env python3
"""Chunked Deigo motion-energy pipeline for long ant physiology videos.

The normal single-process script trusts the AVI frame count and processes one
video serially. This version submits a Deigo pipeline:

1. split each AVI into approximately 5-minute chunks on /flash,
2. process chunks in a Slurm array on compute nodes,
3. assemble chunk motion-energy files into one final .me file,
4. publish the final .me file to the requested output directory.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shlex
import subprocess
import time
from pathlib import Path


DEFAULT_ARUCO_PYTHON = "/bucket/ReiterU/sam/miniforge3/envs/aruco_env/bin/python"


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def safe_name(value: str, max_len: int = 80) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    out = out.strip("._-") or "video"
    return out[:max_len]


def default_work_root() -> Path:
    user = os.environ.get("USER", "samuel-reiter")
    return Path("/flash/ReiterU/ant_tmp") / user / "ant_physiology_motion_energy"


def normalize_deigo_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path_text = str(path_value)
    replacements = {
        "/home/sam-reiter/bucket/": "/bucket/",
        "/home/sam-reiter/saionHome/AntsArray/": "/home/s/samuel-reiter/AntsArray/",
    }
    for old_prefix, new_prefix in replacements.items():
        if path_text.startswith(old_prefix):
            path_text = new_prefix + path_text[len(old_prefix):]
            break
    return Path(path_text).expanduser()


def write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o775)


def run_sbatch(script: Path) -> str:
    out = subprocess.check_output(["sbatch", "--parsable", str(script)], text=True)
    return out.strip()


def conda_env_activate_snippet(env_name: str) -> str:
    return f"""source ~/.bashrc >/dev/null 2>&1 || true
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "/bucket/ReiterU/sam/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "/bucket/ReiterU/sam/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate {shlex.quote(env_name)}
"""


def env_block(env_activate: str | None) -> str:
    if env_activate:
        return env_activate.rstrip() + "\n"
    return conda_env_activate_snippet("aruco_env")


def module_block(module_name: str | None) -> str:
    if not module_name:
        return ""
    return (
        "if command -v module >/dev/null 2>&1; then\n"
        f"    module load {shlex.quote(module_name)} || true\n"
        "fi\n"
    )


def resolve_submit_mask_path(experiment_path: Path, mask_path: str | Path | None) -> Path | None:
    if mask_path:
        resolved = normalize_deigo_path(mask_path)
        if resolved is None:
            return None
        return resolved.resolve()

    matches = sorted(experiment_path.glob("*mask.png"))
    if not matches:
        print(f"[INFO] no default mask matched {experiment_path / '*mask.png'}; processing without a mask")
        return None
    if len(matches) > 1:
        examples = ", ".join(str(path) for path in matches[:5])
        raise ValueError(
            f"Multiple default masks matched {experiment_path / '*mask.png'}: {examples}. "
            "Pass --mask-path explicitly."
        )
    print(f"[INFO] using default mask: {matches[0]}")
    return matches[0].resolve()


def submit(args: argparse.Namespace) -> None:
    experiment_path = normalize_deigo_path(args.experiment_path).resolve()
    output_path = (
        normalize_deigo_path(args.output_path).resolve()
        if args.output_path
        else experiment_path
    )
    work_root = normalize_deigo_path(args.work_root).resolve()
    script_path = Path(__file__).resolve()
    mask_path = resolve_submit_mask_path(experiment_path, args.mask_path)
    videos = sorted(glob.glob(str(experiment_path / args.video_glob)))
    if not videos:
        raise FileNotFoundError(f"No videos matched {experiment_path / args.video_glob}")

    for video_str in videos:
        video = Path(video_str).resolve()
        video_base = video.name
        stem = safe_name(video.stem)
        final_output = output_path / f"{video_base}.me"
        if final_output.exists() and not args.overwrite:
            print(f"[SKIP] {final_output} exists; use --overwrite to recompute")
            continue

        video_work = work_root / stem
        chunk_dir = video_work / "chunks"
        chunk_me_dir = video_work / "chunk_motion_energy"
        assembled_dir = video_work / "assembled"
        logs_dir = video_work / "logs"
        jobs_dir = video_work / "jobs"
        manifest = video_work / "chunk_manifest.txt"
        assembled_output = assembled_dir / f"{video_base}.me"
        jobs_file = video_work / "pipeline.jobs"
        for path in (chunk_dir, chunk_me_dir, assembled_dir, logs_dir, jobs_dir):
            path.mkdir(parents=True, exist_ok=True)

        split_script = jobs_dir / f"split_{stem}.sbatch"
        process_script = jobs_dir / f"process_{stem}.sbatch"
        assemble_script = jobs_dir / f"assemble_{stem}.sbatch"
        publish_script = jobs_dir / f"publish_{stem}.sbatch"

        overwrite_int = 1 if args.overwrite else 0
        mask_arg = str(mask_path) if mask_path is not None else ""
        process_mask_lines = (
            'if [[ -n "$MASK_PATH" ]]; then cmd+=(--mask-path "$MASK_PATH"); fi\n'
        )
        assemble_mask_lines = (
            'if [[ -n "$MASK_PATH" ]]; then cmd+=(--mask-path "$MASK_PATH"); fi\n'
        )

        split_text = f"""#!/bin/bash -l
#SBATCH -t {args.split_time}
#SBATCH -c {args.split_cpus}
#SBATCH --partition={args.split_partition}
#SBATCH --mem={args.split_mem}
#SBATCH -J mesplit-{stem[:40]}
#SBATCH -o {q(logs_dir / "split_%j.out")}
#SBATCH -e {q(logs_dir / "split_%j.err")}
set -euo pipefail
shopt -s nullglob

{env_block(args.env_activate)}{module_block(args.ffmpeg_module)}

VIDEO={q(video)}
CHUNK_DIR={q(chunk_dir)}
CHUNK_ME_DIR={q(chunk_me_dir)}
ASSEMBLED_OUTPUT={q(assembled_output)}
FINAL_OUTPUT={q(final_output)}
MANIFEST={q(manifest)}
PROCESS_SCRIPT={q(process_script)}
ASSEMBLE_SCRIPT={q(assemble_script)}
PUBLISH_SCRIPT={q(publish_script)}
JOBS_FILE={q(jobs_file)}
CHUNK_SECONDS={int(args.chunk_seconds)}
CONCURRENCY={int(args.process_concurrency)}
OVERWRITE={overwrite_int}
STEM={q(stem)}

mkdir -p "$CHUNK_DIR" "$CHUNK_ME_DIR" "$(dirname "$ASSEMBLED_OUTPUT")" "$(dirname "$JOBS_FILE")"
chmod 2775 "$CHUNK_DIR" "$CHUNK_ME_DIR" "$(dirname "$ASSEMBLED_OUTPUT")" || true

if (( OVERWRITE )); then
    rm -f "$CHUNK_DIR"/"$STEM"_chunk_*.avi "$CHUNK_ME_DIR"/*.me "$ASSEMBLED_OUTPUT" "$ASSEMBLED_OUTPUT.tmp" "$MANIFEST"
fi

if ! compgen -G "$CHUNK_DIR/${{STEM}}_chunk_*.avi" >/dev/null; then
    ffmpeg -hide_banner -nostdin -y -fflags +genpts -i "$VIDEO" \\
        -map 0:v:0 -an -c copy -copyinkf -f segment -segment_time "$CHUNK_SECONDS" \\
        -reset_timestamps 1 "$CHUNK_DIR/${{STEM}}_chunk_%05d.avi"
fi

find "$CHUNK_DIR" -maxdepth 1 -type f -name "${{STEM}}_chunk_*.avi" | sort > "$MANIFEST"
chunk_count=$(wc -l < "$MANIFEST")
if (( chunk_count == 0 )); then
    echo "[ERR] no chunks produced for $VIDEO" >&2
    exit 1
fi

array_max=$(( chunk_count - 1 ))
: > "$JOBS_FILE"
printf 'video\\t%s\\nchunks\\t%s\\nmanifest\\t%s\\n' "$VIDEO" "$chunk_count" "$MANIFEST" >> "$JOBS_FILE"
proc_jid=$(sbatch --parsable --array=0-${{array_max}}%${{CONCURRENCY}} "$PROCESS_SCRIPT")
printf 'process\\t%s\\n' "$proc_jid" >> "$JOBS_FILE"
asm_jid=$(sbatch --parsable --dependency=afterok:$proc_jid "$ASSEMBLE_SCRIPT")
printf 'assemble\\t%s\\n' "$asm_jid" >> "$JOBS_FILE"
pub_jid=$(sbatch --parsable --dependency=afterok:$asm_jid "$PUBLISH_SCRIPT")
printf 'publish\\t%s\\n' "$pub_jid" >> "$JOBS_FILE"
cat "$JOBS_FILE"
"""

        process_text = f"""#!/bin/bash -l
#SBATCH -t {args.process_time}
#SBATCH --cpus-per-task={args.process_cpus}
#SBATCH --partition={args.process_partition}
#SBATCH --mem={args.process_mem}
#SBATCH -J meproc-{stem[:40]}
#SBATCH -o {q(logs_dir / "process_%A_%a.out")}
#SBATCH -e {q(logs_dir / "process_%A_%a.err")}
set -euo pipefail

{env_block(args.env_activate)}

MANIFEST={q(manifest)}
CHUNK_ME_DIR={q(chunk_me_dir)}
PIPELINE={q(script_path)}
PYTHON_EXE={q(args.python)}
SOURCE_VIDEO={q(video)}
MASK_PATH={q(mask_arg)}
OVERWRITE={overwrite_int}

task_id="${{SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}}"
line_no=$(( task_id + 1 ))
chunk_path=$(sed -n "${{line_no}}p" "$MANIFEST")
if [[ -z "$chunk_path" || ! -s "$chunk_path" ]]; then
    echo "[ERR] missing chunk for task $task_id: $chunk_path" >&2
    exit 1
fi
chunk_name=$(basename "$chunk_path")
chunk_output="$CHUNK_ME_DIR/${{chunk_name}}.me"

cmd=("$PYTHON_EXE" -u "$PIPELINE" process-chunk \\
    --chunk-path "$chunk_path" \\
    --chunk-output "$chunk_output" \\
    --source-video "$SOURCE_VIDEO" \\
    --chunk-index "$task_id")
{process_mask_lines}if (( OVERWRITE )); then cmd+=(--overwrite); fi

echo "[INFO] processing task=$task_id chunk=$chunk_path"
"${{cmd[@]}}"
"""

        assemble_text = f"""#!/bin/bash -l
#SBATCH -t {args.assemble_time}
#SBATCH --cpus-per-task={args.assemble_cpus}
#SBATCH --partition={args.assemble_partition}
#SBATCH --mem={args.assemble_mem}
#SBATCH -J measm-{stem[:40]}
#SBATCH -o {q(logs_dir / "assemble_%j.out")}
#SBATCH -e {q(logs_dir / "assemble_%j.err")}
set -euo pipefail

{env_block(args.env_activate)}

PIPELINE={q(script_path)}
PYTHON_EXE={q(args.python)}
MANIFEST={q(manifest)}
CHUNK_ME_DIR={q(chunk_me_dir)}
ASSEMBLED_OUTPUT={q(assembled_output)}
SOURCE_VIDEO={q(video)}
MASK_PATH={q(mask_arg)}
OVERWRITE={overwrite_int}

cmd=("$PYTHON_EXE" -u "$PIPELINE" assemble \\
    --manifest "$MANIFEST" \\
    --chunk-output-dir "$CHUNK_ME_DIR" \\
    --output-file "$ASSEMBLED_OUTPUT" \\
    --source-video "$SOURCE_VIDEO")
{assemble_mask_lines}if (( OVERWRITE )); then cmd+=(--overwrite); fi

"${{cmd[@]}}"
"""

        publish_text = f"""#!/bin/bash -l
#SBATCH -t {args.publish_time}
#SBATCH --cpus-per-task=1
#SBATCH --partition={args.publish_partition}
#SBATCH --mem={args.publish_mem}
#SBATCH -J mepub-{stem[:40]}
#SBATCH -o {q(logs_dir / "publish_%j.out")}
#SBATCH -e {q(logs_dir / "publish_%j.err")}
set -euo pipefail
umask 0002

ASSEMBLED_OUTPUT={q(assembled_output)}
FINAL_OUTPUT={q(final_output)}
OVERWRITE={overwrite_int}

if [[ ! -s "$ASSEMBLED_OUTPUT" ]]; then
    echo "[ERR] assembled output missing: $ASSEMBLED_OUTPUT" >&2
    exit 1
fi
if [[ -e "$FINAL_OUTPUT" && "$OVERWRITE" != "1" ]]; then
    echo "[SKIP] final output exists: $FINAL_OUTPUT"
    exit 0
fi

mkdir -p "$(dirname "$FINAL_OUTPUT")"
tmp="${{FINAL_OUTPUT}}.tmp.${{SLURM_JOB_ID:-$$}}"
rsync -a "$ASSEMBLED_OUTPUT" "$tmp"
mv -f "$tmp" "$FINAL_OUTPUT"
chmod 664 "$FINAL_OUTPUT" || true
echo "[DONE] published $FINAL_OUTPUT"
"""

        write_executable(split_script, split_text)
        write_executable(process_script, process_text)
        write_executable(assemble_script, assemble_text)
        write_executable(publish_script, publish_text)

        if args.dry_run:
            print(f"[DRY-RUN] wrote scripts for {video}")
            print(f"          split script: {split_script}")
            continue

        jid = run_sbatch(split_script)
        print(f"[SUBMITTED] {video} split job {jid}; work dir {video_work}")


def load_mask(mask_path: str | None, expected_shape: tuple[int, int] | None = None):
    if not mask_path:
        return None
    import cv2

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError(f"Could not read mask image from {mask_path}")
    mask = mask_img > 0
    if expected_shape is not None and mask.shape != expected_shape:
        raise ValueError(f"Mask dimensions {mask.shape} do not match frame size {expected_shape}")
    return mask


def process_chunk(args: argparse.Namespace) -> None:
    import cv2
    import h5py
    import numpy as np

    chunk_path = Path(args.chunk_path)
    output = Path(args.chunk_output)
    tmp_output = output.with_name(output.name + ".tmp")
    if output.exists() and not args.overwrite:
        print(f"[SKIP] chunk output exists: {output}")
        return
    if tmp_output.exists():
        tmp_output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    cap = None
    writer = None
    try:
        cap = cv2.VideoCapture(str(chunk_path))
        reported_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        succ, img = cap.read()
        if not succ:
            raise ValueError(f"Unable to read first frame from {chunk_path}")

        img_g_old = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        first_gray = img_g_old.copy()
        last_gray = img_g_old.copy()
        mask = load_mask(args.mask_path, img_g_old.shape)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        resize_chunk = max(1, int(args.resize_chunk))
        writer = h5py.File(tmp_output, "w")
        dset = writer.create_dataset(
            "motion_energy",
            shape=(0,),
            maxshape=(None,),
            chunks=(resize_chunk,),
            dtype="float32",
        )
        writer.attrs.create("source_video", str(args.source_video), dtype=h5py.special_dtype(vlen=str))
        writer.attrs.create("chunk_path", str(chunk_path), dtype=h5py.special_dtype(vlen=str))
        writer.attrs.create("chunk_index", int(args.chunk_index), dtype="int64")
        writer.attrs.create("fps", fps, dtype="float32")
        writer.attrs.create("reported_frame_count", reported_length, dtype="int64")
        if args.mask_path:
            writer.attrs.create("mask", str(args.mask_path), dtype=h5py.special_dtype(vlen=str))

        tally = 0
        started_at = time.monotonic()
        last_report_at = started_at
        print(
            f"[START] chunk={chunk_path.name} index={args.chunk_index} "
            f"reported_frames={reported_length} fps={fps:g} output={output}",
            flush=True,
        )
        while cap.isOpened():
            succ, img = cap.read()
            if not succ:
                break
            if tally >= dset.shape[0]:
                dset.resize((dset.shape[0] + resize_chunk,))

            img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frame_diff = np.abs(np.int32(img_g) - np.int32(img_g_old))
            if mask is not None:
                frame_diff[~mask] = 0
            dset[tally] = np.sum(frame_diff)
            img_g_old = img_g
            last_gray = img_g
            tally += 1
            now = time.monotonic()
            if tally % int(args.progress_frames) == 0 or now - last_report_at >= float(args.progress_seconds):
                elapsed = max(1e-9, now - started_at)
                frame_rate = tally / elapsed
                pct = (
                    100.0 * tally / reported_length
                    if reported_length > 0
                    else float("nan")
                )
                pct_text = f"{pct:5.1f}%" if reported_length > 0 else "  n/a"
                print(
                    f"[PROGRESS] chunk={chunk_path.name} index={args.chunk_index} "
                    f"frames={tally:,}/{reported_length if reported_length > 0 else '?'} "
                    f"({pct_text}) elapsed={elapsed/60:.1f}min rate={frame_rate:.1f}fps",
                    flush=True,
                )
                writer.flush()
                last_report_at = now

        dset.resize((tally,))
        writer.create_dataset("first_gray", data=first_gray, compression="gzip", compression_opts=1)
        writer.create_dataset("last_gray", data=last_gray, compression="gzip", compression_opts=1)
        writer.attrs.create("n_frames_processed", tally, dtype="int64")
        writer.attrs.create("complete", True, dtype="bool")
        cap.release()
        cap = None
        writer.close()
        writer = None
        os.replace(tmp_output, output)
        elapsed = max(1e-9, time.monotonic() - started_at)
        print(
            f"[DONE] {chunk_path}: {tally:,} frames in {elapsed/60:.1f}min "
            f"({tally/elapsed:.1f}fps)",
            flush=True,
        )
    except Exception:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.close()
        if tmp_output.exists():
            tmp_output.unlink()
        raise


def assemble(args: argparse.Namespace) -> None:
    import h5py
    import numpy as np

    manifest = Path(args.manifest)
    chunk_output_dir = Path(args.chunk_output_dir)
    output = Path(args.output_file)
    tmp_output = output.with_name(output.name + ".tmp")
    if output.exists() and not args.overwrite:
        print(f"[SKIP] assembled output exists: {output}")
        return
    if tmp_output.exists():
        tmp_output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    chunk_paths = [Path(line.strip()) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chunk_paths:
        raise ValueError(f"Manifest has no chunks: {manifest}")

    chunk_me_paths = [chunk_output_dir / f"{path.name}.me" for path in chunk_paths]
    missing = [path for path in chunk_me_paths if not path.exists()]
    if missing:
        examples = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} chunk motion-energy files, examples: {examples}")

    mask = None
    first_shape = None
    lengths: list[int] = []
    fps_values: list[float] = []
    for path in chunk_me_paths:
        with h5py.File(path, "r") as h5:
            if "motion_energy" not in h5:
                raise KeyError(f"{path} has no motion_energy dataset")
            n = int(h5["motion_energy"].shape[0])
            lengths.append(n)
            fps_values.append(float(h5.attrs.get("fps", np.nan)))
            if first_shape is None and "first_gray" in h5:
                first_shape = tuple(int(v) for v in h5["first_gray"].shape)

    if args.mask_path and first_shape is not None:
        mask = load_mask(args.mask_path, first_shape)

    total = int(sum(lengths))
    writer = h5py.File(tmp_output, "w")
    try:
        out = writer.create_dataset("motion_energy", shape=(total,), dtype="float32")
        chunk_start = writer.create_dataset("chunk_start_frame", shape=(len(chunk_me_paths),), dtype="int64")
        chunk_n = writer.create_dataset("chunk_n_frames", shape=(len(chunk_me_paths),), dtype="int64")
        string_dtype = h5py.special_dtype(vlen=str)
        chunk_file_ds = writer.create_dataset("chunk_file", shape=(len(chunk_me_paths),), dtype=string_dtype)

        offset = 0
        prev_last_gray = None
        fps = next((value for value in fps_values if np.isfinite(value) and value > 0), np.nan)
        for idx, (chunk_path, me_path, n_frames) in enumerate(zip(chunk_paths, chunk_me_paths, lengths)):
            with h5py.File(me_path, "r") as h5:
                values = h5["motion_energy"][:]
                if n_frames and prev_last_gray is not None and "first_gray" in h5:
                    first_gray = h5["first_gray"][:]
                    boundary_diff = np.abs(np.int32(first_gray) - np.int32(prev_last_gray))
                    if mask is not None:
                        boundary_diff[~mask] = 0
                    values[0] = np.sum(boundary_diff)
                if "last_gray" in h5:
                    prev_last_gray = h5["last_gray"][:]
                out[offset : offset + n_frames] = values
                chunk_start[idx] = offset
                chunk_n[idx] = n_frames
                chunk_file_ds[idx] = str(chunk_path)
                offset += n_frames

        writer.attrs.create("source_video", str(args.source_video), dtype=string_dtype)
        writer.attrs.create("fps", float(fps), dtype="float32")
        writer.attrs.create("n_frames_processed", total, dtype="int64")
        writer.attrs.create("chunk_count", len(chunk_me_paths), dtype="int64")
        writer.attrs.create("complete", True, dtype="bool")
        if args.mask_path:
            writer.attrs.create("mask", str(args.mask_path), dtype=string_dtype)
    except Exception:
        writer.close()
        if tmp_output.exists():
            tmp_output.unlink()
        raise
    else:
        writer.close()
        os.replace(tmp_output, output)
        print(f"[DONE] assembled {len(chunk_me_paths)} chunks, {total} frames -> {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "calc_motion_energy_cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    submit_p = sub.add_parser("submit", help="Submit the Deigo chunked motion-energy pipeline.")
    submit_p.add_argument("--experiment-path", required=True, help="Path containing videos.")
    submit_p.add_argument(
        "--output-path",
        default=None,
        help="Where final .me files should be published. Defaults to --experiment-path.",
    )
    submit_p.add_argument(
        "--mask-path",
        default=None,
        help="Optional binary mask image. Defaults to the single *mask.png in --experiment-path, if present.",
    )
    submit_p.add_argument("--video-glob", default="*.avi", help="Video glob inside experiment path.")
    submit_p.add_argument("--work-root", default=str(default_work_root()), help="Flash work root for chunks and logs.")
    submit_p.add_argument("--chunk-seconds", type=int, default=5 * 60, help="Chunk duration for ffmpeg segmenting.")
    submit_p.add_argument("--process-concurrency", type=int, default=24, help="Max concurrent chunk array tasks.")
    submit_p.add_argument("--python", default=DEFAULT_ARUCO_PYTHON, help="Python executable to run in Slurm jobs.")
    submit_p.add_argument(
        "--env-activate",
        default=None,
        help="Shell snippet run before Python in Slurm jobs. Default activates conda aruco_env.",
    )
    submit_p.add_argument("--ffmpeg-module", default="", help="Optional module to load before ffmpeg; empty disables.")
    submit_p.add_argument("--overwrite", action="store_true", help="Overwrite existing final and intermediate outputs.")
    submit_p.add_argument("--dry-run", action="store_true", help="Write scripts but do not submit.")
    submit_p.add_argument("--split-partition", default="short")
    submit_p.add_argument("--split-time", default="2:00:00")
    submit_p.add_argument("--split-cpus", type=int, default=4)
    submit_p.add_argument("--split-mem", default="16G")
    submit_p.add_argument("--process-partition", default="compute")
    submit_p.add_argument("--process-time", default="0-4")
    submit_p.add_argument("--process-cpus", type=int, default=2)
    submit_p.add_argument("--process-mem", default="8G")
    submit_p.add_argument("--assemble-partition", default="short")
    submit_p.add_argument("--assemble-time", default="0-2")
    submit_p.add_argument("--assemble-cpus", type=int, default=2)
    submit_p.add_argument("--assemble-mem", default="8G")
    submit_p.add_argument("--publish-partition", default="datacp")
    submit_p.add_argument("--publish-time", default="0-1")
    submit_p.add_argument("--publish-mem", default="2G")
    submit_p.set_defaults(func=submit)

    chunk_p = sub.add_parser("process-chunk", help="Process one chunk; intended for Slurm array tasks.")
    chunk_p.add_argument("--chunk-path", required=True)
    chunk_p.add_argument("--chunk-output", required=True)
    chunk_p.add_argument("--source-video", required=True)
    chunk_p.add_argument("--chunk-index", type=int, required=True)
    chunk_p.add_argument("--mask-path", default=None)
    chunk_p.add_argument("--resize-chunk", type=int, default=100000)
    chunk_p.add_argument("--progress-frames", type=int, default=5000)
    chunk_p.add_argument("--progress-seconds", type=float, default=30.0)
    chunk_p.add_argument("--overwrite", action="store_true")
    chunk_p.set_defaults(func=process_chunk)

    assemble_p = sub.add_parser("assemble", help="Assemble chunk .me files into one .me file.")
    assemble_p.add_argument("--manifest", required=True)
    assemble_p.add_argument("--chunk-output-dir", required=True)
    assemble_p.add_argument("--output-file", required=True)
    assemble_p.add_argument("--source-video", required=True)
    assemble_p.add_argument("--mask-path", default=None)
    assemble_p.add_argument("--overwrite", action="store_true")
    assemble_p.set_defaults(func=assemble)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
