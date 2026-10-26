#!/usr/bin/env python3
"""Durable Deigo-login coordinator for SLEAP -> per-recording tracks -> stitching.

Run under nohup from a frozen checkout. GPU arrays run on Saion, CPU arrays on
Deigo, and only this login-side process publishes CPU results to bucket. Failed
tasks are retried from completion receipts, never from stale file presence.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

SSH = ["ssh", "-x", "-oBatchMode=yes", "-oConnectTimeout=30",
       "-oServerAliveInterval=15", "-oServerAliveCountMax=4"]


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def command(args: list[str], host: str | None = None) -> str:
    scheduler_query = args[0] in {"squeue", "sacct"}
    if host:
        args = SSH + [host, "bash -lc " + shlex.quote(shlex.join(args))]
    attempt = 0
    while True:
        try:
            p = subprocess.run(args, capture_output=True, text=True,
                               timeout=90 if scheduler_query else None)
        except subprocess.TimeoutExpired:
            if not scheduler_query:
                raise
            print("[WAIT] Scheduler query timed out; retrying in 60 seconds", flush=True)
            time.sleep(60)
            continue
        if p.returncode == 0:
            return p.stdout.strip()
        transient = ("unable to contact slurm controller", "socket timed out",
                     "connection refused", "connection reset", "connection timed out",
                     "connection closed", "temporary failure in name resolution",
                     "slurm_persist_conn_open", "communication connection failure")
        if scheduler_query and any(marker in p.stderr.lower() for marker in transient):
            print(f"[WAIT] Scheduler unavailable; retrying in 60 seconds: {p.stderr.strip()}", flush=True)
            time.sleep(60)
            continue
        if attempt == 4:
            raise RuntimeError(f"{shlex.join(args)}: {p.stderr}")
        print(f"[RETRY] {p.stderr.strip()}", flush=True)
        time.sleep(10 * (attempt + 1))
        attempt += 1


def is_live(job: str, host: str | None) -> bool:
    # A failed query is never interpreted as an empty queue.
    # Query by user: querying a finished/purged job ID can itself exit nonzero.
    active = command(["squeue", "-h", "-u", os.environ["USER"], "-o", "%F"], host)
    return bool(set(job.split(",")) & {line.strip() for line in active.splitlines()})


def submit(script: Path, count: int, concurrency: int, name: str, host: str | None) -> str:
    # Adopt a live same-name submission before retrying a controller timeout.
    for attempt in range(5):
        existing = command(["squeue", "-u", os.environ["USER"], "-h", "-n", name, "-o", "%A"], host)
        if existing:
            return existing.splitlines()[0]
        args = ["sbatch", "--parsable", "--job-name", name, f"--array=0-{count - 1}%{concurrency}", str(script)]
        if host:
            args = SSH + [host, "bash -lc " + shlex.quote(shlex.join(args))]
        p = subprocess.run(args, capture_output=True, text=True)
        if p.returncode == 0:
            job = p.stdout.strip().split(";")[0]
            if not job.isdigit():
                raise ValueError(f"Unexpected sbatch response: {p.stdout}")
            return job
        if "Invalid account" in p.stderr or "Invalid partition" in p.stderr:
            raise RuntimeError(p.stderr.strip())
        print(f"[SUBMIT RETRY] {p.stderr}", flush=True)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Could not submit {script}")


def gpu_done(manifest: dict, pair: list[int]) -> bool:
    r = manifest["recordings"][pair[0]]
    c = r["chunks"][pair[1]]
    directory = Path(manifest["output"]) / "detections" / r["session"]
    try:
        state = json.loads((directory / f"{c['stem']}.done.json").read_text())
        return (state["contract_sha256"] == manifest["contract_sha256"] and state["frames"] == c["frames"]
                and (directory / f"{c['stem']}.slp").stat().st_size == state["slp_bytes"]
                and (directory / f"{c['stem']}_sleap_data.h5").stat().st_size == state["h5_bytes"])
    except (OSError, ValueError, KeyError):
        return False


def cpu_done(manifest: dict, output: Path, key: str, stage: str) -> bool:
    suffix = "tracked" if stage == "track" else "stitched"
    try:
        state = json.loads((output / "state" / f"{key}.{suffix}.json").read_text())
        return state["contract_sha256"] == manifest["contract_sha256"] and Path(state["output"]).is_file()
    except (OSError, ValueError, KeyError):
        return False


def gpu_script(args, manifest: dict, worklist: Path, label: str, smoke: int = 0) -> Path:
    script = args.jobs / f"{label}.sbatch"
    gres = "gpu:" + (args.gpu_type + ":" if args.gpu_type else "") + "1"
    lines = ["#!/bin/bash -l", "#SBATCH --partition=" + args.partition, "#SBATCH --gres=" + gres,
             "#SBATCH -c 8", "#SBATCH --mem=64G", "#SBATCH -t " + ("00:15:00" if smoke else "02:00:00"),
             f"#SBATCH -o {args.jobs}/{label}_%A_%a.out", f"#SBATCH -e {args.jobs}/{label}_%A_%a.err",
             "set -eo pipefail", "export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1", "module use /apps/unit/ReiterU/.modulefiles",
             "module load " + shlex.quote(manifest["model"]["module"]),
             'SLEAP_PY="$(dirname "$(readlink -f "$(command -v sleap-nn)")")/python"']
    call = [str(Path(__file__).with_name("infer.py")), "--manifest", str(args.manifest),
            "--worklist", str(worklist), "--scratch", str(args.gpu_scratch),
            "--converter", str(Path(__file__).resolve().parents[2] / "detection_pipeline/scripts/sleap2h5.py")]
    if smoke:
        call.extend(["--smoke-frames", str(smoke)])
    lines.append('"$SLEAP_PY" ' + shlex.join(call))
    script.write_text("\n".join(lines) + "\n")
    return script


def cpu_script(args, stage: str, worklist: Path, label: str) -> Path:
    script = args.jobs / f"{label}.sbatch"
    call = [args.python, str(Path(__file__)), "worker", "--manifest", str(args.manifest),
            "--cpu-scratch", str(args.cpu_scratch), "--stage", stage, "--worklist", str(worklist)]
    script.write_text("\n".join(["#!/bin/bash -l", "#SBATCH --partition=compute", "#SBATCH -c 4",
                                 "#SBATCH --mem=24G", "#SBATCH -t 12:00:00",
                                 f"#SBATCH -o {args.jobs}/{label}_%A_%a.out",
                                 f"#SBATCH -e {args.jobs}/{label}_%A_%a.err",
                                 "set -eo pipefail", "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
                                 shlex.join(call)]) + "\n")
    return script


def wait_for_job(job: str, host: str | None):
    while is_live(job, host):
        time.sleep(60)


def wait_for_prerequisites(job: str, host: str) -> None:
    """Fail promptly if an upstream job fails while its dependents remain queued."""
    required = set(job.split(","))
    failures = {"BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL",
                "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT"}
    missing_polls = 0
    while True:
        report = command(["sacct", "-X", "-n", "-P", "-j", job,
                          "--format=JobID,State"], host)
        states = {}
        for line in report.splitlines():
            fields = line.split("|")
            if len(fields) >= 2 and fields[0].strip():
                state = fields[1].strip().split()
                if state:
                    states[fields[0].strip()] = state[0].rstrip("+")
        failed = {key: value for key, value in states.items() if value in failures}
        if failed:
            raise RuntimeError(f"Export/smoke prerequisite failed: {failed}")
        # Slurm may report only array elements (123_0), with no parent record.
        represented = {key.split("_", 1)[0] for key in states}
        if required <= represented and all(value == "COMPLETED" for value in states.values()):
            return
        # Accounting can lag behind the queue. Allow it to catch up, but never
        # mistake absent accounting records for successful validation.
        missing_polls = 0 if is_live(job, host) else missing_polls + 1
        if missing_polls >= 10:
            raise RuntimeError(f"Export/smoke completion could not be verified: {states}")
        time.sleep(60)


def coordinate(args):
    args.jobs.mkdir(parents=True, exist_ok=True)
    lock = (args.jobs / "coordinator.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(args.manifest.read_text())
    run_state = args.jobs / "coordinator.json"
    state = json.loads(run_state.read_text()) if run_state.exists() else dict(jobs=[], phase="starting")
    save(args.jobs / "launch.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    # Resume the previously launched array after a login process restart.
    if state.get("active_job"):
        wait_for_job(state["active_job"], state.get("active_host"))
    if args.prerequisite_job and not (state.get("prerequisite_complete") and state.get("prerequisite_job") == args.prerequisite_job):
        state["phase"] = "waiting_for_export_and_smoke"
        state["prerequisite_job"] = args.prerequisite_job
        save(run_state, state)
        print(f"[WAIT] export/smoke prerequisite {args.prerequisite_job}", flush=True)
        wait_for_prerequisites(args.prerequisite_job, "saion")
        state["prerequisite_complete"] = True
        save(run_state, state)
    pairs = [[ri, ci] for ri, r in enumerate(manifest["recordings"]) for ci in range(len(r["chunks"]))]
    pairs.sort(key=lambda pair: (pair[1], pair[0]))
    for stage in ("infer", "track", "stitch"):
        state["phase"] = stage
        items = pairs if stage == "infer" else ([r["key"] for r in manifest["recordings"]] if stage == "track" else manifest["cameras"])
        done = (lambda x: gpu_done(manifest, x)) if stage == "infer" else (lambda x: cpu_done(manifest, args.cpu_scratch, x, stage))
        attempts = {}
        while True:
            pending = [x for x in items if not done(x)]
            state.update(completed=len(items) - len(pending), total=len(items), active_job=None, active_host=None)
            save(run_state, state)
            print(f"[{stage}] {state['completed']}/{len(items)} complete", flush=True)
            if not pending:
                break
            wave = pending[:args.wave_size] if stage == "infer" else pending
            for item in wave:
                key = str(item)
                attempts[key] = attempts.get(key, 0) + 1
                if attempts[key] > args.retries:
                    raise RuntimeError(f"{stage} failed after {args.retries} attempts: {item}; inspect {args.jobs}")
            label = f"{stage}_{len(state['jobs']):03d}"
            worklist = args.jobs / f"{label}.json"
            save(worklist, wave)
            host = "saion" if stage == "infer" else None
            script = gpu_script(args, manifest, worklist, label) if host else cpu_script(args, stage, worklist, label)
            job = submit(script, len(wave), args.gpus if host else args.cpu_concurrency,
                         f"singleant_{manifest['contract_sha256'][:8]}_{label}", host)
            state.update(active_job=job, active_host=host)
            state["jobs"].append(dict(stage=stage, job=job, host=host, script=str(script), worklist=str(worklist)))
            save(run_state, state)
            print(f"[SUBMITTED] {stage} {job} ({len(wave)} tasks)", flush=True)
            wait_for_job(job, host)
    state["phase"] = "publishing"
    save(run_state, state)
    destination = Path(manifest["output"])
    for folder in ("tracks", "coverage", "stitched", "state"):
        command(["rsync", "-a", "--chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw",
                 str(args.cpu_scratch / folder), str(destination) + "/"])
    pipeline = Path(__file__).with_name("pipeline.py")
    print(command([args.python, str(pipeline), "finalize", "--manifest", str(args.manifest)]), flush=True)
    state.update(phase="complete", active_job=None, active_host=None)
    save(run_state, state)
    save(destination / "jobs.json", state)
    print("[COMPLETE] " + str(destination), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--jobs", type=Path, required=True)
        p.add_argument("--gpu-scratch", type=Path, required=True)
        p.add_argument("--cpu-scratch", type=Path, required=True)
        p.add_argument("--python", default="/apps/unit/ReiterU/ant_tracking/venv/bin/python")
        p.add_argument("--partition", default="short-a100")
        p.add_argument("--gpu-type", default="", help="Optional Slurm GPU type, e.g. v100 on the gpu partition")
        p.add_argument("--gpus", type=int, default=32)
        p.add_argument("--cpu-concurrency", type=int, default=12)
        p.add_argument("--wave-size", type=int, default=256)
        p.add_argument("--retries", type=int, default=3)
        p.add_argument("--prerequisite-job", help="Wait for successful Saion export/smoke job IDs (comma-separated) before production")
        if name == "smoke":
            p.add_argument("--frames", type=int, default=240)
            p.add_argument("--recording-index", type=int, default=0)
            p.add_argument("--chunk-index", type=int, default=0)
    p = sub.add_parser("worker")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cpu-scratch", type=Path, required=True)
    p.add_argument("--worklist", type=Path, required=True)
    p.add_argument("--stage", choices=("track", "stitch"), required=True)
    args = parser.parse_args()
    if args.command in ("run", "smoke"):
        if min(args.gpus, args.cpu_concurrency, args.wave_size, args.retries) < 1:
            parser.error("Concurrency, wave size, and retry count must be positive")
    if args.command == "worker":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from tracking.single_ant.workflow import stitch_camera, track_recording
        manifest = json.loads(args.manifest.read_text())
        manifest["detections_root"] = manifest["output"]
        manifest["output"] = str(args.cpu_scratch)
        key = json.loads(args.worklist.read_text())[int(os.environ["SLURM_ARRAY_TASK_ID"])]
        (track_recording if args.stage == "track" else stitch_camera)(manifest, key)
    elif args.command == "smoke":
        args.jobs.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(args.manifest.read_text())
        label = f"smoke_{args.recording_index}_{args.chunk_index}"
        worklist = args.jobs / f"{label}.json"
        save(worklist, [[args.recording_index, args.chunk_index]])
        script = gpu_script(args, manifest, worklist, label, args.frames)
        job = submit(script, 1, 1, f"singleant_{manifest['contract_sha256'][:8]}_{label}", "saion")
        save(args.jobs / f"{label}.job.json", dict(job=job, script=str(script)))
        print(job)
    else:
        try:
            (args.jobs / "FAILED.json").unlink(missing_ok=True)
            coordinate(args)
        except Exception as error:
            save(args.jobs / "FAILED.json", dict(error=repr(error), time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
            raise


if __name__ == "__main__":
    main()
