#!/usr/bin/env python3
"""Convert a legacy SLEAP (TF/Keras, sleap<=1.4) UNet model directory into a
sleap-nn Lightning checkpoint so `sleap-nn export` (ONNX/TensorRT) can consume it.

Background: sleap-nn 0.2.0 inference (`track`/`predict`) auto-loads legacy
`best_model.h5` models, but the `export` CLI only reads `best.ckpt` and has no
legacy fallback (sleap_nn/export/cli.py). `export/utils.load_training_config`
already understands the legacy `training_config.json`, so the only missing
artifact is the checkpoint. This script materializes it.

The produced checkpoint matches exactly what `_load_lightning_model` rebuilds:
the LightningModule is constructed from the same cfg-derived kwargs the export
CLI uses, the legacy Keras->PyTorch weights (via legacy_models.load_legacy_model)
are loaded into its inner `self.model`, and the full module state_dict is saved.
On reload, `LightningModule.load_from_checkpoint(**cfg_kwargs)` does a strict
state_dict load, which lines up because both sides build
`Model(backbone_type, backbone_config[bt], head_configs[mt], model_type)`.

Usage:
    legacy_to_ckpt.py --model-dir <legacy dir> --out-dir <dir> [--verify]

<out-dir> receives best.ckpt + a copy of training_config.json, which is exactly
what `sleap-nn export <out-dir>` needs.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from sleap_nn.export.utils import (
    load_training_config,
    resolve_backbone_type,
    resolve_model_type,
)
from sleap_nn.legacy_models import load_legacy_model
from sleap_nn.training.lightning_modules import (
    BottomUpLightningModule,
    BottomUpMultiClassLightningModule,
    CentroidLightningModule,
    SingleInstanceLightningModule,
    TopDownCenteredInstanceLightningModule,
    TopDownCenteredInstanceMultiClassLightningModule,
)

LIGHTNING_CLS = {
    "centroid": CentroidLightningModule,
    "centered_instance": TopDownCenteredInstanceLightningModule,
    "single_instance": SingleInstanceLightningModule,
    "bottomup": BottomUpLightningModule,
    "multi_class_topdown": TopDownCenteredInstanceMultiClassLightningModule,
    "multi_class_bottomup": BottomUpMultiClassLightningModule,
}


def module_kwargs(cfg, model_type, backbone_type):
    """Mirror exactly the kwargs sleap_nn/export/cli.py::_load_lightning_model passes."""
    return dict(
        model_type=model_type,
        backbone_type=backbone_type,
        backbone_config=cfg.model_config.backbone_config,
        head_configs=cfg.model_config.head_configs,
        pretrained_backbone_weights=cfg.model_config.pretrained_backbone_weights,
        pretrained_head_weights=cfg.model_config.pretrained_head_weights,
        init_weights=cfg.model_config.init_weights,
        lr_scheduler=cfg.trainer_config.lr_scheduler,
        online_mining=cfg.trainer_config.online_hard_keypoint_mining.online_mining,
        hard_to_easy_ratio=cfg.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
        min_hard_keypoints=cfg.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
        max_hard_keypoints=cfg.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
        loss_scale=cfg.trainer_config.online_hard_keypoint_mining.loss_scale,
        optimizer=cfg.trainer_config.optimizer_name,
        learning_rate=cfg.trainer_config.optimizer.lr,
        amsgrad=cfg.trainer_config.optimizer.amsgrad,
    )


def legacy_input_scaling(model_dir: Path):
    """Read data.preprocessing.input_scaling from a legacy SLEAP training_config.json.

    load_sleap_config drops this field (it defaults cfg scale to 1.0), but the
    legacy centroid is typically trained at input_scaling<1. Carrying it into the
    sleap-nn cfg is required for both a buildable TRT profile and correct inference.
    """
    p = model_dir / "training_config.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return float(d["data"]["preprocessing"]["input_scaling"])
    except Exception:
        return None


def lightning_version():
    try:
        import lightning

        return getattr(lightning, "__version__", "unknown")
    except Exception:
        try:
            import pytorch_lightning

            return getattr(pytorch_lightning, "__version__", "unknown")
        except Exception:
            return "unknown"


def convert(model_dir: Path, out_dir: Path, verify: bool, scale_override=None) -> None:
    cfg = load_training_config(model_dir)
    model_type = resolve_model_type(cfg)
    backbone_type = resolve_backbone_type(cfg)
    print(f"[INFO] {model_dir}")
    print(f"[INFO] model_type={model_type} backbone_type={backbone_type}")

    # Carry legacy input_scaling into cfg.data_config.preprocessing.scale.
    # load_sleap_config drops it (defaults to 1.0); without this the centroid
    # exports/infers at full resolution instead of its trained scale.
    scale = scale_override if scale_override is not None else legacy_input_scaling(model_dir)
    cfg_scale = cfg.data_config.preprocessing.scale
    if scale is not None:
        OmegaConf.set_struct(cfg, False)
        cfg.data_config.preprocessing.scale = scale
        print(f"[INFO] preprocessing.scale: {cfg_scale} -> {scale} (from legacy input_scaling)")
    else:
        print(f"[INFO] preprocessing.scale unchanged: {cfg_scale}")

    if backbone_type != "unet":
        sys.exit(f"[ERR] only legacy UNet models are supported; got backbone_type={backbone_type}")

    kwargs = module_kwargs(cfg, model_type, backbone_type)
    cls = LIGHTNING_CLS.get(model_type)
    if cls is None:
        sys.exit(f"[ERR] unsupported model_type: {model_type}")

    module = cls(**kwargs)
    legacy_model = load_legacy_model(str(model_dir), load_weights=True)

    missing, unexpected = module.model.load_state_dict(legacy_model.state_dict(), strict=False)
    if missing or unexpected:
        print("[ERR] state_dict mismatch between legacy model and lightning inner model")
        print(f"      missing keys ({len(missing)}): {missing[:20]}")
        print(f"      unexpected keys ({len(unexpected)}): {unexpected[:20]}")
        sys.exit(2)

    module.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write the resolved sleap-nn cfg (with corrected scale) as training_config.yaml.
    # load_training_config prefers .yaml, so this is what export/inference will read.
    OmegaConf.save(cfg, out_dir / "training_config.yaml")
    print(f"[OK] wrote {out_dir / 'training_config.yaml'}")

    ckpt = {
        "state_dict": module.state_dict(),
        "pytorch-lightning_version": lightning_version(),
        "epoch": 0,
        "global_step": 0,
        "hyper_parameters": {},
    }
    ckpt_path = out_dir / "best.ckpt"
    torch.save(ckpt, ckpt_path)
    n_params = sum(v.numel() for v in module.state_dict().values())
    print(f"[OK] wrote {ckpt_path} (state_dict tensors={len(ckpt['state_dict'])}, elems={n_params})")

    if verify:
        verify_roundtrip(cls, out_dir, cfg, kwargs)


def verify_roundtrip(cls, out_dir: Path, cfg, kwargs) -> None:
    """Reload exactly as sleap_nn/export/cli.py does, then run a dummy forward."""
    print("[INFO] verifying load_from_checkpoint round-trip ...")
    reloaded = cls.load_from_checkpoint(
        checkpoint_path=str(out_dir / "best.ckpt"),
        map_location="cpu",
        weights_only=False,
        **kwargs,
    )
    reloaded.eval()
    in_ch = int(cfg.model_config.backbone_config[kwargs["backbone_type"]]["in_channels"])
    x = torch.rand(1, in_ch, 256, 256)
    with torch.no_grad():
        out = reloaded.model(x)
    keys = list(out.keys()) if isinstance(out, dict) else type(out).__name__
    print(f"[OK] round-trip load + forward succeeded; output heads={keys}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="legacy model dir (best_model.h5 + training_config.json)")
    ap.add_argument("--out-dir", required=True, help="output dir for best.ckpt + training_config")
    ap.add_argument("--verify", action="store_true", help="reload checkpoint and run a dummy forward")
    ap.add_argument("--scale", type=float, default=None,
                    help="override preprocessing.scale (default: legacy input_scaling from json)")
    args = ap.parse_args()
    convert(Path(args.model_dir), Path(args.out_dir), args.verify, args.scale)
