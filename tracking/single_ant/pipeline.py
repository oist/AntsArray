#!/usr/bin/env python3
"""Prepare, run, and verify the SLEAP-only single-ant pipeline.

Run ``python -m tracking.single_ant.pipeline --help`` from the repository root.
Cluster inference/CPU fanout is provided by ``submit.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracking.single_ant.workflow import (
    Settings, finalize, load_manifest, prepare, stitch_camera, track_recording,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Freeze and validate videos, masks, times, and model provenance")
    p.add_argument("--input-folder", "--input_folder", required=True, type=Path)
    p.add_argument("--out-dir", "--out_dir", required=True, type=Path)
    p.add_argument("--model-centroid", required=True, type=Path)
    p.add_argument("--model-instance", required=True, type=Path)
    p.add_argument("--export-dir", required=True, type=Path)
    p.add_argument("--export-metadata", type=Path, help="Freeze effective thresholds and graph hash from a validated export")
    p.add_argument("--build-trt-if-missing", action="store_true", help="Compile a validated ONNX graph in the first GPU worker")
    p.add_argument("--timezone", default="Asia/Tokyo")
    p.add_argument("--sleap-module", default="sleap-nn/0.3.3")
    p.add_argument("--anchor-bodypoint", type=int, default=1)
    p.add_argument("--min-score", type=float, default=0.2)
    p.add_argument("--min-nodes", type=int, default=2)
    p.add_argument("--max-distance", type=float, default=100.0)
    p.add_argument("--reacquire-seconds", type=float, default=5.0)
    for name in ("track", "stitch", "finalize", "run"):
        p = commands.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--work-output", type=Path, help="Write on compute scratch; keep detections at the manifest's output")
        if name == "track":
            p.add_argument("--recording", required=True)
        elif name == "stitch":
            p.add_argument("--camera", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.input_folder, args.out_dir, model_centroid=args.model_centroid,
                         model_instance=args.model_instance, export_dir=args.export_dir,
                         timezone=args.timezone, module=args.sleap_module,
                         export_metadata=args.export_metadata, build_trt_if_missing=args.build_trt_if_missing,
                         settings=Settings(args.anchor_bodypoint, args.min_score, args.min_nodes,
                                           args.max_distance, args.reacquire_seconds))
        print(f"Prepared {len(result['recordings'])} recordings / {len(result['cameras'])} cameras")
        return
    manifest = load_manifest(args.manifest)
    if args.work_output:
        manifest["detections_root"] = manifest["output"]
        manifest["output"] = str(args.work_output)
    if args.command == "track":
        result = track_recording(manifest, args.recording)
    elif args.command == "stitch":
        result = stitch_camera(manifest, args.camera)
    elif args.command == "finalize":
        result = finalize(manifest)
    else:
        for recording in manifest["recordings"]:
            track_recording(manifest, recording["key"])
        for camera in manifest["cameras"]:
            stitch_camera(manifest, camera)
        result = finalize(manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
