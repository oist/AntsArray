#!/usr/bin/env python3
"""Login-side transfer of a completed interaction-only fanout to bucket.

Compute nodes write flash; this watcher runs on a login node with bucket write
access. A new destination is published only after every worker succeeded.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time


def publish(run_dir: Path, *, poll_seconds=120, timeout_hours=72):
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    source = Path(manifest["output_dir"])
    destination = Path(manifest["bucket_output_dir"])
    deadline = time.monotonic() + timeout_hours*3600
    lock = (run_dir / "publish.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    marker = source / "interactions_complete.ok"
    print(f"Waiting for {marker}", flush=True)
    while not marker.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Fanout did not finish; inspect jobs in {run_dir}")
        time.sleep(poll_seconds)
    import pyarrow.parquet as pq

    for name in manifest["expected_files"]:
        path = source / name
        metadata = json.loads(path.with_suffix(".metadata.json").read_text())
        if metadata["parameters"] != manifest["parameters"]:
            raise ValueError(f"Parameter mismatch: {path}")
        if path.stat().st_size != metadata["output_size"]:
            raise ValueError(f"Output size mismatch: {path}")
        if pq.ParquetFile(path).metadata.num_rows != metadata.get("n_pair_detections", metadata.get("n_directed_detections")):
            raise ValueError(f"Output row-count mismatch: {path}")
    if destination.exists():
        existing = destination / "run_manifest.json"
        if not existing.is_file() or json.loads(existing.read_text())["run_id"] != manifest["run_id"]:
            raise FileExistsError(f"Refusing to mix separate runs in {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Transferring {len(manifest['expected_files'])} completed chunks to {destination}", flush=True)
    subprocess.run(["rsync", "-a", "--partial", "--protect-args", f"{source}/", f"{destination}/"], check=True)
    for name in manifest["expected_files"]:
        if (destination/name).stat().st_size != (source/name).stat().st_size:
            raise ValueError(f"Transferred size mismatch: {name}")
    ready = destination / "transfer_complete.ok"
    temp = ready.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(run_id=manifest["run_id"], completed_at=datetime.now(timezone.utc).isoformat(),
                                   n_chunks=len(manifest["expected_files"])), indent=2)+"\n")
    temp.replace(ready)
    print(f"PUBLISHED: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=120)
    parser.add_argument("--timeout-hours", type=float, default=72)
    args = parser.parse_args()
    publish(args.run_dir, poll_seconds=args.poll_seconds, timeout_hours=args.timeout_hours)


if __name__ == "__main__":
    main()
