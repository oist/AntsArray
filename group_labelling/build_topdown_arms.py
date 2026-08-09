#!/usr/bin/env python3
"""Cut the flattened chunks into camera-disjoint training arms and a frozen test set.

Each of the six labelling chunks covers its own cameras -- the 25 cameras are
partitioned across chunks, none shared. That matters more than it looks: an arm
trained on chunk06 alone sees only cam06/cam10/cam24, so comparing it against an
arm trained on chunks 01-05 confounds *label quality* with *camera coverage*.
Two things follow, and this script implements both.

First, the held-out cameras are removed from **every** arm, so no arm is scored on
a camera it trained on. Frames are split by camera rather than at random because
neighbouring frames of one camera are near-identical -- a random split leaks, and
cross-camera generalisation is the known weak point.

Second, the arm list includes a camera-matched control. ``A3`` (all six chunks,
collaborator labels) and ``A5`` (the same six with chunk06 replaced by the reviewed
version) differ *only* by the review, so A3 vs A5 measures what reviewing is worth.
A1 vs A2 cannot, since those two never share a camera.

Cameras are matched to video indices by exact frame-index set equality against the
manifest, not by position, because video order inside a package is not guaranteed
to survive a GUI re-save.

``A4`` additionally drops user instances with at least ``--max-hidden-antennae``
invisible antenna nodes. A hidden keypoint is not "no label" to sleap-nn: it is
rendered as an all-zero confidence map, which actively teaches the model to predict
nothing there. Dropping the worst offenders tests whether that supervision hurts.

``--extra-arm`` names an arm out of packages that already exist. That is how the
incremental loop feeds a freshly extracted review batch back in without rebuilding
the chunk split::

    --extra-arm rev_batch2=/work/.../reviewed_batch1.pkg.slp,/work/.../reviewed_batch2.pkg.slp

Example::

    python build_topdown_arms.py --flat-dir flat/ --manifest chunk_manifest.csv \
        --out-dir arms/ --report arms/arms.json
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import sleap_io as sio

from topdown_common import (
    antenna_indices,
    chunk_number,
    map_videos_to_cameras,
    read_manifest,
    save_package,
    stem_of,
    videos_of,
    write_report,
)

DEFAULT_HOLDOUT = ("cam25", "cam22", "cam15", "cam23", "cam21", "cam10")


def hidden_antennae(instance, antenna_idx: list[int]) -> int:
    """Count invisible antenna nodes; Instance.numpy() gives NaN for invisible."""
    points = np.asarray(instance.numpy())
    return int(sum(bool(np.isnan(points[i]).any()) for i in antenna_idx))


def package_counts(path: Path) -> tuple[int, int]:
    """(frames, user instances) read straight from HDF5, without decoding images."""
    with h5py.File(path, "r") as h5:
        n_frames = int(len(h5["frames"])) if "frames" in h5 else 0
        n_user = int((h5["instances"][:]["instance_type"] == 0).sum()) if "instances" in h5 else 0
    return n_frames, n_user


def write(labels: sio.Labels, frames: list, path: Path, quiet: bool) -> dict:
    if not frames:
        return {"path": None, "n_frames": 0, "n_user_instances": 0}
    started = time.time()
    save_package(frames, videos_of(frames, labels), labels.skeletons, path)
    stats = {
        "path": str(path),
        "n_frames": len(frames),
        "n_user_instances": sum(len(f.user_instances) for f in frames),
    }
    if not quiet:
        print(
            f"     -> {path.name}: {stats['n_frames']} frames, "
            f"{stats['n_user_instances']} user inst ({time.time() - started:.1f}s)"
        )
    return stats


def process_file(
    src: Path, cameras: dict[str, set[int]], holdout: set[str], args: argparse.Namespace
) -> dict:
    labels = sio.load_slp(str(src))
    mapping = map_videos_to_cameras(labels, cameras, src.name)
    antenna_idx = antenna_indices(labels)

    train_frames, test_frames, filtered_frames = [], [], []
    per_camera: dict[str, int] = defaultdict(int)
    dropped = 0

    for frame in labels:
        camera = mapping[labels.videos.index(frame.video)]
        per_camera[camera] += len(frame.user_instances)
        if camera in holdout:
            test_frames.append(frame)
            continue
        train_frames.append(frame)

        keep = [
            inst
            for inst in frame.user_instances
            if hidden_antennae(inst, antenna_idx) < args.max_hidden_antennae
        ]
        dropped += len(frame.user_instances) - len(keep)
        if keep:
            filtered_frames.append(
                sio.LabeledFrame(video=frame.video, frame_idx=frame.frame_idx, instances=keep)
            )

    if not args.quiet:
        print(f"[..] {src.name}: " + ", ".join(f"v{v}={c}" for v, c in sorted(mapping.items())))

    stem, out = stem_of(src), args.out_dir
    result = {
        "src": str(src),
        "cameras": {cam: per_camera[cam] for cam in sorted(per_camera)},
        "holdout_cameras_present": sorted(set(mapping.values()) & holdout),
        "train": write(labels, train_frames, out / f"{stem}_train.pkg.slp", args.quiet),
        "test": write(labels, test_frames, out / f"{stem}_test.pkg.slp", args.quiet),
    }
    if not args.skip_filtered:
        result["train_filtered"] = write(
            labels, filtered_frames, out / f"{stem}_trainfilt.pkg.slp", args.quiet
        )
        result["instances_dropped_by_filter"] = dropped
    return result


def collect(results: dict[str, dict], keys: list[str], field: str) -> dict:
    paths, frames, instances = [], 0, 0
    for key in keys:
        part = results[key].get(field) or {}
        if part.get("path"):
            paths.append(part["path"])
            frames += part["n_frames"]
            instances += part["n_user_instances"]
    return {"train": paths, "n_frames": frames, "n_user_instances": instances}


def extra_arms(specs: list[str]) -> dict[str, dict]:
    """``NAME=path[,path...]`` -> an arm entry shaped exactly like ``collect``'s."""
    arms: dict[str, dict] = {}
    for spec in specs:
        name, _, joined = spec.partition("=")
        paths = [Path(p) for p in joined.split(",") if p.strip()]
        if not name or not paths:
            raise SystemExit(f"[ERR] --extra-arm wants NAME=path[,path...], got {spec!r}")
        frames = instances = 0
        for path in paths:
            if not path.is_file():
                raise SystemExit(f"[ERR] --extra-arm {name}: no such package {path}")
            n_frames, n_user = package_counts(path)
            frames += n_frames
            instances += n_user
        arms[name] = {
            "train": [str(p) for p in paths],
            "n_frames": frames,
            "n_user_instances": instances,
        }
    return arms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--flat-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--holdout-cameras",
        default=",".join(DEFAULT_HOLDOUT),
        help="Excluded from every arm. One per chunk; cam10 is the reviewed gold camera.",
    )
    parser.add_argument(
        "--max-hidden-antennae",
        type=int,
        default=3,
        help="A4 drops user instances with at least this many invisible antenna nodes.",
    )
    parser.add_argument("--skip-filtered", action="store_true", help="Do not build the A4 inputs.")
    parser.add_argument(
        "--extra-arm",
        action="append",
        default=[],
        metavar="NAME=PKG[,PKG]",
        help="Name an arm from packages that already exist, e.g. an extracted review "
        "batch. Repeatable.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    holdout = {c.strip() for c in args.holdout_cameras.split(",") if c.strip()}
    manifest = read_manifest(args.manifest)

    files = sorted(p for p in args.flat_dir.glob("*.slp") if not p.name.startswith("."))
    if not files:
        raise SystemExit(f"[ERR] no label files in {args.flat_dir}")

    results: dict[str, dict] = {}
    for src in files:
        key = f"ch{chunk_number(src):02d}" + ("_REVIEWED" if "REVIEWED" in src.name else "")
        results[key] = process_file(src, manifest[chunk_number(src)], holdout, args)

    unreviewed = [k for k in results if not k.endswith("_REVIEWED")]
    reviewed6 = [k for k in results if k.endswith("_REVIEWED")]
    ch01to05 = [k for k in unreviewed if k != "ch06"]

    if not reviewed6:
        raise SystemExit("[ERR] no *_REVIEWED file found; A1 and A5 cannot be built")

    arms = {
        # Phase 1: centroid needs no review -- the anchor node is ~99% visible everywhere.
        "centroid_all6": collect(results, unreviewed, "train"),
        # Phase 2 arms, all warm-started, all scored on the same frozen cameras.
        "ci_A1_ch06reviewed": collect(results, reviewed6, "train"),
        "ci_A2_ch01to05": collect(results, ch01to05, "train"),
        "ci_A3_all6_unreviewed": collect(results, unreviewed, "train"),
        "ci_A5_all6_ch06reviewed": collect(results, ch01to05 + reviewed6, "train"),
    }
    if not args.skip_filtered:
        arms["ci_A4_all6_filtered"] = collect(results, unreviewed, "train_filtered")
    arms.update(extra_arms(args.extra_arm))

    report = {
        "holdout_cameras": sorted(holdout),
        "max_hidden_antennae": args.max_hidden_antennae,
        "per_file": results,
        "arms": arms,
        "test_gold": [results[k]["test"]["path"] for k in reviewed6 if results[k]["test"]["path"]],
        "test_broad": [
            results[k]["test"]["path"] for k in unreviewed if results[k]["test"]["path"]
        ],
    }

    print("\n=== arms ===")
    for name, arm in arms.items():
        print(
            f"  {name:<26} {len(arm['train'])} file(s)  "
            f"{arm['n_frames']:>5} frames  {arm['n_user_instances']:>6} user inst"
        )
    print(f"  {'test_gold (reviewed)':<26} {len(report['test_gold'])} file(s)")
    print(f"  {'test_broad':<26} {len(report['test_broad'])} file(s)")

    write_report(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
