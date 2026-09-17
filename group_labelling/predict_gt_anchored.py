#!/usr/bin/env python3
"""Run a centered-instance model over existing labels, cropping on ground truth.

Passing *only* a centered-instance checkpoint to ``sleap-nn predict`` -- no centroid
model -- puts ``CentroidCrop`` into ``use_gt_centroids=True`` mode: every crop is
taken around a labelled instance's own anchor node, and one prediction comes back
per labelled instance. That is what makes a clean model-vs-label comparison
possible. Running the full top-down pipeline instead would fold centroid error and
detection misses into the score and break the 1:1 correspondence, so the ranking
would end up measuring the centroid model rather than the instance a reviewer has
to fix.

Two properties of a trained config must be undone before it can predict anywhere
but the node it trained on:

  data_pipeline_fw   ``torch_dataset_cache_img_disk`` replays a cache directory that
                     lived on the training node's /scratch and was deleted when the
                     job ended. Rewritten to ``torch_dataset``.
  cache_img_path     that now-absent path itself. Rewritten to null.

``--peak_threshold 0.0`` is equally mandatory for these models: at any higher
threshold no peak clears it and every frame comes back empty.

The patched config and a symlink to the checkpoint go in a staging directory, so
the published model is never modified and /bucket stays read-only.

Example::

    python predict_gt_anchored.py \
        --model /work/.../models_reviewed260/reviewed260_centered_instance/reviewed260_centered_instance \
        --out-dir /work/.../review_selection/pred \
        /bucket/.../flat/block01_*_chunk0[1-5]*.pkg.slp
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import yaml

from topdown_common import stem_of, write_report

PLAIN_FW = "torch_dataset"


def patch_config(model_dir: Path, staged: Path) -> dict:
    """Stage the model with a config that can run off the node that trained it."""
    config_path = model_dir / "training_config.yaml"
    checkpoint = model_dir / "best.ckpt"
    if not config_path.is_file():
        raise SystemExit(f"[ERR] no training_config.yaml in {model_dir}")
    if not checkpoint.is_file():
        raise SystemExit(f"[ERR] no best.ckpt in {model_dir}")

    config = yaml.safe_load(config_path.read_text())

    head = config.get("model_config", {}).get("head_configs", {})
    if head.get("centered_instance") is None:
        raise SystemExit(
            f"[ERR] {model_dir} has no centered_instance head; ground-truth anchoring "
            f"needs one. Heads present: {sorted(k for k, v in head.items() if v)}"
        )

    data = config.get("data_config", {})
    before = {
        "data_pipeline_fw": data.get("data_pipeline_fw"),
        "cache_img_path": data.get("cache_img_path"),
    }
    data["data_pipeline_fw"] = PLAIN_FW
    data["cache_img_path"] = None
    data["use_existing_imgs"] = False
    config["data_config"] = data

    staged.mkdir(parents=True, exist_ok=True)
    (staged / "training_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    link = staged / "best.ckpt"
    if link.is_symlink() or link.exists():
        link.unlink()
    # Symlinked, not copied: the checkpoint is large and is read-only where it lives.
    link.symlink_to(checkpoint.resolve())

    return {
        "staged": str(staged),
        "patched_from": before,
        "anchor_part": head["centered_instance"]["confmaps"].get("anchor_part"),
        "crop_size": config.get("data_config", {}).get("preprocessing", {}).get("crop_size"),
    }


def predict(staged: Path, src: Path, out: Path, args: argparse.Namespace) -> dict:
    command = [
        str(args.sleap_nn_bin / "sleap-nn"),
        "predict",
        "--data_path", str(src),
        "--model_paths", str(staged),
        "--output_path", str(out),
        "--peak_threshold", str(args.peak_threshold),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--only_labeled_frames",
        "--no-restore_source_videos",
    ]
    if args.dry_run:
        print("[dry-run] " + " ".join(command))
        return {"command": command, "skipped": True}

    print(f"[..] {src.name} -> {out.name}", flush=True)
    started = time.time()
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"[ERR] sleap-nn predict failed ({result.returncode}) on {src.name}")
    if not out.is_file():
        raise SystemExit(f"[ERR] sleap-nn predict wrote nothing to {out}")

    elapsed = time.time() - started
    print(f"[OK] {out.name} in {elapsed / 60:.1f} min", flush=True)
    return {"command": command, "output": str(out), "seconds": round(elapsed, 1)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Flattened label packages.")
    parser.add_argument("--model", type=Path, required=True, help="Centered-instance model dir.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Where the patched config and checkpoint symlink go. Default: <out-dir>/model.",
    )
    parser.add_argument(
        "--peak-threshold",
        dest="peak_threshold",
        type=float,
        default=0.0,
        help="Must stay 0.0 for these models; anything higher returns empty frames.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Frames per batch, NOT crops. Every instance in a batched frame becomes "
            "its own 640px crop, so the centered-instance model sees batch_size x "
            "instances_per_frame samples at once. At 30 ants/frame, 16 frames is 450 "
            "crops and the decoder's bilinear upsample overflows INT_MAX. Keep "
            "batch_size x max_instances_per_frame under ~400."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sleap-nn-bin",
        type=Path,
        default=Path("/apps/unit/ReiterU/sleap-nn/0.3.1/tools/sleap-nn/bin"),
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    staged = args.staging_dir or (args.out_dir / "model")

    info = patch_config(args.model, staged)
    print(f"[OK] staged model at {staged}")
    print(f"     anchor_part={info['anchor_part']!r}  crop_size={info['crop_size']}")
    print(f"     patched {info['patched_from']} -> data_pipeline_fw={PLAIN_FW}, cache_img_path=None")

    results = []
    for src in args.inputs:
        if not src.is_file():
            raise SystemExit(f"[ERR] not a file: {src}")
        out = args.out_dir / f"{stem_of(src)}.pred.slp"
        if out.is_file() and not args.overwrite:
            print(f"[skip] {out.name} exists")
            results.append({"input": str(src), "output": str(out), "skipped": True})
            continue
        results.append({"input": str(src), **predict(staged, src, out, args)})

    write_report(args.report, {"model": str(args.model), "staged": info, "runs": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
