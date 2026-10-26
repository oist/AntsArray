#!/usr/bin/env python3
"""One Slurm GPU task: stage one video, use colony SLEAP inference, publish.

Only a successfully exited predictor + converter may create a completion
receipt. Bucket upload of the receipt is last, after both result files.
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

import h5py
import numpy as np

SSH = ["ssh", "-x", "-oBatchMode=yes", "-oConnectTimeout=30",
       "-oServerAliveInterval=15", "-oServerAliveCountMax=4"]


def run(command: list[str], retries: int = 1) -> None:
    for attempt in range(retries):
        print("[CMD] " + shlex.join(command), flush=True)
        result = subprocess.run(command)
        if result.returncode == 0:
            return
        if attempt + 1 < retries:
            time.sleep(10 * (attempt + 1))
    raise subprocess.CalledProcessError(result.returncode, command)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--worklist", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--converter", type=Path, required=True)
    p.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--smoke-frames", type=int)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    exported = json.loads((Path(manifest["model"]["export_dir"]) / "export_metadata.json").read_text())
    expected_sources = ("centroid:" + str(Path(manifest["model"]["centroid"]) / "best.ckpt")
                        + ";centered_instance:" + str(Path(manifest["model"]["instance"]) / "best.ckpt"))
    if exported.get("checkpoint_path") != expected_sources:
        raise ValueError("Exported model sources do not match the manifest's checkpoint pair")
    for key in ("centroid_peak_threshold", "instance_peak_threshold", "onnx_sha256"):
        actual = exported.get(key, exported.get("peak_threshold") if key.endswith("threshold") else None)
        if key in manifest["model"] and actual != manifest["model"][key]:
            raise ValueError(f"Exported {key} does not match the inference contract")
    record_index, chunk_index = json.loads(args.worklist.read_text())[args.task]
    recording = manifest["recordings"][record_index]
    chunk = recording["chunks"][chunk_index]
    if manifest["model"].get("build_trt_if_missing") and manifest["model"]["runtime"] == "tensorrt":
        export_dir = Path(manifest["model"]["export_dir"])
        with (export_dir / ".build.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (export_dir / "model.trt").exists():
                from tune_export import build_trt
                build_trt(export_dir, recording["height"], recording["width"], manifest["model"]["batch_size"])
        exported = json.loads((export_dir / "export_metadata.json").read_text())
    work = args.scratch / ("smoke" if args.smoke_frames else "chunks") / chunk["stem"]
    work.mkdir(parents=True, exist_ok=True)
    source = Path(manifest["root"]) / chunk["video"]
    video = work / source.name
    destination = Path(manifest["output"]) / ("smoke" if args.smoke_frames else "detections") / recording["session"]
    frames = min(args.smoke_frames or chunk["frames"], chunk["frames"])
    # Fetch through the login node: /bucket is not writable/mounted on all GPUs.
    run(["rsync", "-a", "--protect-args", "-e", shlex.join(SSH), f"saion:{source}", str(video)], retries=5)
    if video.stat().st_size != chunk["bytes"]:
        raise ValueError(f"Incomplete video transfer: {video}")
    slp = work / f"{chunk['stem']}.slp"
    h5 = work / f"{chunk['stem']}_sleap_data.h5"
    receipt = work / f"{chunk['stem']}.done.json"
    # Resume only a locally completed inference from this exact contract.
    resume = False
    if receipt.exists() and slp.exists() and h5.exists():
        state = json.loads(receipt.read_text())
        resume = (state.get("contract_sha256") == manifest["contract_sha256"] and state.get("frames") == frames
                  and state.get("slp_bytes") == slp.stat().st_size and state.get("h5_bytes") == h5.stat().st_size)
    start = time.monotonic()
    if not resume:
        receipt.unlink(missing_ok=True)
        slp.unlink(missing_ok=True)
        h5.unlink(missing_ok=True)
        run(["sleap-nn", "predict", "-m", manifest["model"]["export_dir"], "-i", str(video),
             "-o", str(slp), "--runtime", manifest["model"]["runtime"], "--batch_size",
             str(manifest["model"]["batch_size"]), "--frames", f"0-{frames - 1}",
             "--device", manifest["model"].get("device", "cuda")])
        with h5py.File(slp) as h:
            actual = h["frames"]["frame_idx"]
            if len(actual) and (np.min(actual) < 0 or np.max(actual) >= frames):
                raise ValueError(f"Out of range SLEAP frames: {slp}")
        run([sys.executable, str(args.converter), str(slp), str(work), "--expected-frames", str(frames)])
        with h5py.File(h5, "r+") as h:
            if "sleap_data" not in h or int(h.attrs["expected_frames"]) != frames:
                raise ValueError(f"Invalid converted SLEAP output: {h5}")
            h.attrs["single_ant_contract"] = manifest["contract_sha256"]
            h.attrs["model_centroid_sha256"] = manifest["model"]["centroid_sha256"]
            h.attrs["model_instance_sha256"] = manifest["model"]["instance_sha256"]
            h.attrs["node_names"] = json.dumps(exported.get("node_names", []))
            h.attrs["centroid_peak_threshold"] = exported.get("centroid_peak_threshold", exported["peak_threshold"])
            h.attrs["instance_peak_threshold"] = exported.get("instance_peak_threshold", exported["peak_threshold"])
            h.attrs["export_onnx_sha256"] = exported.get("onnx_sha256", "")
        state = dict(contract_sha256=manifest["contract_sha256"], frames=frames, video=chunk["video"],
                     slp_bytes=slp.stat().st_size, h5_bytes=h5.stat().st_size,
                     elapsed_seconds=time.monotonic() - start, job_id=os.environ.get("SLURM_JOB_ID"))
        receipt.write_text(json.dumps(state, indent=2) + "\n")
    run(SSH + ["saion", shlex.join(["mkdir", "-p", str(destination)])], retries=5)
    for files in ([slp, h5], [receipt]):
        run(["rsync", "-a", "--protect-args", "--chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw", "-e", shlex.join(SSH),
             *map(str, files), f"saion:{destination}/"], retries=5)
    # Only remove the staged input after the durable completion receipt lands.
    video.unlink()
    print(f"[COMPLETE] {chunk['stem']} frames={frames} seconds={time.monotonic() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
