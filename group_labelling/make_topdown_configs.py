#!/usr/bin/env python3
"""Write one sleap-nn training config per arm, warm-started off the current model.

The architecture is copied verbatim from the deployed model's own
``training_config.yaml`` rather than regenerated. That is not stylistic: warm start
loads a state dict into the new model, so the backbone geometry (``filters``,
``filters_rate``, ``max_stride``, ``output_stride``) and the head's node list have
to match the checkpoint exactly or the weights will not fit. ``sleap-nn config
--auto`` picks different defaults (filters 24 / rate 1.5 / max_stride 64 for the
centroid, against the checkpoint's 16 / 2.0 / 32), so it cannot serve as the base.

The node names carry deliberate irregularities -- ``'aruco '`` has a trailing space,
``'antenna L_2'`` uses a space where its siblings use an underscore, and
``'antenna_L3'`` drops one. They are reproduced exactly; "tidying" them renames the
head channels and silently breaks the warm start.

Two settings deserve comment:

* ``early_stopping.patience`` is the one raised to 40 -- it governs when to stop.
  ``reduce_lr_on_plateau.patience`` is deliberately left far lower: across 200
  epochs a patience of 40 would let the learning rate anneal only three or four
  times, and a fine-tune wants it to settle.
* ``validation_fraction`` drops from 0.35 to 0.15. The 0.35 split existed because
  there was no held-out set; now that whole cameras are held out, that extra 20%
  is better spent on training.

The learning rate is lowered for the warm start, since the original 1e-4 is a
from-scratch rate and would wash the pretrained weights out in the first epochs.

Batch size is fixed, not searched. A VRAM sweep finds the largest batch that fits,
and on this data that is not the batch to train with: throughput peaks well below the
memory limit and then falls off, and a larger batch also means fewer gradient updates
per epoch on a dataset this small. Measurement settled it at 6 for the centered
instance and 8 for the centroid, so those are the defaults and the sweep is gone.

Example::

    python make_topdown_configs.py --arms arms/arms.json --out-dir cfg/ \
        --centroid-model /bucket/.../250408_141245.centroid \
        --instance-model /bucket/.../250408_141245.centered_instance \
        --ckpt-root /work/ReiterU/Ants/topdown_20260803/models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from topdown_common import write_report

CENTROID_ARM = "centroid_all6"
DEFAULT_BATCH_CENTROID = 8
DEFAULT_BATCH_INSTANCE = 6


def configure(
    base: DictConfig,
    train_paths: list[str],
    test_path: str | None,
    ckpt: Path,
    batch: int,
    run_name: str,
    ckpt_dir: Path,
    args: argparse.Namespace,
) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))

    cfg.data_config.train_labels_path = list(train_paths)
    cfg.data_config.val_labels_path = None
    cfg.data_config.validation_fraction = args.validation_fraction
    cfg.data_config.test_file_path = test_path

    # `cache_img_path` is only read when the framework is the disk-caching one --
    # under the deployed config's plain `torch_dataset` it is silently ignored and
    # crops accumulate in host RAM instead (~21 GB/epoch at 100 steps x batch 128
    # of 640x640 float32, which OOM-killed every arm at epoch 8 regardless of
    # dataset size). sleap-nn also documents that `torch_dataset` requires
    # num_workers=0, since video backends cannot be pickled across workers.
    # Cache to node-local disk: /bucket is read-only on compute nodes and Lustre
    # would be slower than the node's own NVMe.
    cfg.data_config.data_pipeline_fw = args.data_pipeline_fw
    cfg.data_config.cache_img_path = str(Path(args.cache_root) / run_name)
    cfg.data_config.use_existing_imgs = False
    cfg.data_config.delete_cache_imgs_after_training = True

    # Warm start. `pretrained_*_weights` load weights into a fresh optimizer;
    # `resume_ckpt_path` would instead restore optimizer/epoch/scheduler state,
    # which is wrong when moving to a different dataset.
    cfg.model_config.pretrained_backbone_weights = str(ckpt)
    cfg.model_config.pretrained_head_weights = str(ckpt)
    cfg.trainer_config.resume_ckpt_path = None

    cfg.trainer_config.train_data_loader.batch_size = batch
    cfg.trainer_config.val_data_loader.batch_size = batch
    cfg.trainer_config.train_data_loader.num_workers = args.num_workers
    cfg.trainer_config.val_data_loader.num_workers = args.num_workers

    # The deployed configs pin train_steps_per_epoch (483 for the centered
    # instance, 200 for the centroid) to values computed for the original dataset
    # at its original batch size. Carried over unchanged they no longer describe an
    # epoch: at batch 128 483 steps would draw 61,824 instances from a ~13,500
    # instance training set -- roughly 4.5 passes per "epoch", pushing 200 epochs
    # past 24h and making early-stopping patience mean something entirely
    # different. Clearing it lets the trainer derive the length from the data.
    cfg.trainer_config.train_steps_per_epoch = None
    cfg.trainer_config.min_train_steps_per_epoch = args.min_train_steps

    # The deployed config's intensity augmentation is a legacy-TF artifact: written
    # when images were 0-255 and left with every probability at 1.0. A local run
    # that trained successfully had them all at 0.0, so that is reproduced here
    # rather than the converted file's values.
    if not args.intensity_aug:
        intensity = cfg.data_config.augmentation_config.intensity
        for key in ("gaussian_noise_p", "contrast_p", "brightness_p", "uniform_noise_p"):
            if key in intensity:
                intensity[key] = 0.0

    # Ants appear at every orientation and the centered-instance crop is not
    # orientation-normalised, so the original +/-15 degrees is needlessly narrow.
    geometric = cfg.data_config.augmentation_config.geometric
    geometric.rotation_min = -float(args.rotation)
    geometric.rotation_max = float(args.rotation)

    cfg.trainer_config.max_epochs = args.max_epochs
    cfg.trainer_config.early_stopping.patience = args.early_stopping_patience
    cfg.trainer_config.lr_scheduler.reduce_lr_on_plateau.patience = args.plateau_patience
    cfg.trainer_config.optimizer.lr = args.lr
    cfg.trainer_config.ckpt_dir = str(ckpt_dir)
    cfg.trainer_config.run_name = run_name
    cfg.trainer_config.save_ckpt = True
    cfg.trainer_config.seed = args.seed
    cfg.trainer_config.use_wandb = False
    # Rendering prediction overlays every epoch costs wall clock and writes into
    # the checkpoint dir; the frozen test set is the real read on quality.
    cfg.trainer_config.visualize_preds_during_training = args.visualize

    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arms", type=Path, required=True, help="arms.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--centroid-model", type=Path, required=True, help="dir with best.ckpt")
    parser.add_argument("--instance-model", type=Path, required=True, help="dir with best.ckpt")
    parser.add_argument(
        "--batch-centroid",
        type=int,
        default=DEFAULT_BATCH_CENTROID,
        help="Centroid batch size; the measured throughput optimum.",
    )
    parser.add_argument(
        "--batch-instance",
        type=int,
        default=DEFAULT_BATCH_INSTANCE,
        help="Centered-instance batch size; settled by measurement, do not raise blind.",
    )
    parser.add_argument("--ckpt-root", type=Path, required=True)
    parser.add_argument("--cache-root", default="/scratch/sleapnn_cache")
    parser.add_argument(
        "--data-pipeline-fw",
        default="torch_dataset_cache_img_disk",
        choices=["torch_dataset", "torch_dataset_cache_img_memory", "torch_dataset_cache_img_disk"],
        help="Disk caching by default; the other two hold crops in host RAM.",
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument(
        "--min-train-steps",
        type=int,
        default=100,
        help="Floor on steps per epoch. Kept below the natural epoch length so it "
        "does not silently stretch epochs on these small datasets.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--plateau-patience", type=int, default=8)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Scale with batch size: a recipe that trains at batch 6 / lr 1e-4 is "
        "~21x under-powered per sample if the rate is not raised for batch 128.",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=180.0,
        help="Rotation augmentation range in degrees (+/-).",
    )
    parser.add_argument(
        "--intensity-aug",
        action="store_true",
        help="Keep the deployed config's intensity augmentation. Off by default.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arms_doc = json.loads(args.arms.read_text())
    arms = arms_doc["arms"]

    gold = arms_doc.get("test_gold") or []
    test_path = gold[0] if gold else None
    if test_path is None:
        print("[!!] arms.json has no test_gold; configs will have test_file_path unset")

    bases = {
        "centroid": OmegaConf.load(args.centroid_model / "training_config.yaml"),
        "centered_instance": OmegaConf.load(args.instance_model / "training_config.yaml"),
    }
    ckpts = {
        "centroid": args.centroid_model / "best.ckpt",
        "centered_instance": args.instance_model / "best.ckpt",
    }
    batches = {
        "centroid": args.batch_centroid,
        "centered_instance": args.batch_instance,
    }
    for kind, ckpt in ckpts.items():
        if not ckpt.is_file():
            raise SystemExit(f"[ERR] warm-start checkpoint missing: {ckpt}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}

    for name, arm in arms.items():
        if not arm["train"]:
            print(f"[!!] {name}: no training files, skipped")
            continue
        kind = "centroid" if name == CENTROID_ARM else "centered_instance"
        cfg = configure(
            bases[kind],
            arm["train"],
            test_path,
            ckpts[kind],
            batches[kind],
            name,
            args.ckpt_root / name,
            args,
        )
        out = args.out_dir / f"{name}.yaml"
        OmegaConf.save(cfg, out)
        index[name] = {
            "config": str(out),
            "model_type": kind,
            "batch_size": batches[kind],
            "warm_start": str(ckpts[kind]),
            "n_train_files": len(arm["train"]),
            "n_frames": arm["n_frames"],
            "n_user_instances": arm["n_user_instances"],
            "ckpt_dir": str(args.ckpt_root / name),
        }
        print(
            f"[OK] {name:<26} {kind:<17} batch={batches[kind]:<4} "
            f"{arm['n_frames']:>5} frames -> {out.name}"
        )

    write_report(args.report, index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
