#!/usr/bin/env python3
"""Rebuild the final_v2 training set and write its config with a different crop size.

final_v2 (published as 260803_topdown.centered_instance) trained on two packages that
lived only in /work and are gone:

  ch06_full_v2.pkg.slp          all 260 reviewed ch06 frames after the cam24 re-review
  ch01to05_reviewed31.pkg.slp   the 31 ch01-05 frames reviewed before training (2026-08-09)

Both are rebuilt here from what the bucket still holds, and checked against the
291 frames / 6,283 user instances recorded in work_records_20260819/final_v2/arms.json:

  ch06   arms_cam24/*_REVIEWED_train.pkg.slp (200 frames) + *_REVIEWED_test.pkg.slp
         (the 60 re-reviewed cam24 frames), copied as they are; sleap-nn takes both.
  ch01-05 review_ch01to05/ch01to05_dense_n3000.slp, the reviewer's save, holds the
         same 142 frames as the 2026-08-06 package it was opened from; the frames whose
         labels differ between the two are the ones that were reviewed (31 of them).
         They are written out with the package's embedded images.

The config is final_v2's own training_config.yaml with only these changes: crop size,
local label/cache/checkpoint paths, the warm-start checkpoint path (same A0 weights),
and the local-machine settings (single device, 8 dataloader workers). Everything else --
seed, augmentation, optimiser, steps per epoch, early stopping -- is kept, so a
difference in the result is down to the crop.

    & "C:\\Program Files\\uv-tools\\tools\\sleap-nn\\Scripts\\python.exe" `
        makoto_play\\prepare_final_v2_crop.py --out-dir D:\\crop768_train --crop 768
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import numpy as np
import sleap_io as sio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "group_labelling"))
from topdown_common import save_package  # noqa: E402

BUCKET = Path("Z:/ReiterU/Ants/SLEAP_files")
TOPDOWN = BUCKET / "Group_labelling/20260803_topdown"
CH06 = [
    TOPDOWN / "arms_cam24/block01_20260515_round1_chunk06_Sorida_Juliet_corrected_"
              "Sorida-Juliet-recorrected_REVIEWED_train.pkg.slp",
    TOPDOWN / "arms_cam24/block01_20260515_round1_chunk06_Sorida_Juliet_corrected_"
              "Sorida-Juliet-recorrected_REVIEWED_test.pkg.slp",
]
REVIEW_PKG = TOPDOWN / "review_ch01to05/ch01to05_dense_n3000.pkg.slp"
REVIEW_SAVE = TOPDOWN / "review_ch01to05/ch01to05_dense_n3000.slp"
BASE_CONFIG = BUCKET / "Simple_skeleton/20260803_models/260803_topdown.centered_instance/training_config.yaml"
WARM_START = BUCKET / "Simple_skeleton_nn/250408_141245.centered_instance/best.ckpt"
EXPECTED = {"frames": 291, "user_instances": 6283}   # work_records_20260819/final_v2/arms.json

_LOG = None


def log(msg: str = "") -> None:
    print(msg, flush=True)
    if _LOG:
        _LOG.write(msg + "\n")
        _LOG.flush()


def copy_if_needed(src: Path, dst: Path) -> Path:
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        log(f"[skip] {dst.name} already copied")
        return dst
    log(f"[..] copy {src.name}")
    shutil.copy2(src, dst)
    return dst


def user_points(frame) -> list[np.ndarray]:
    return [inst.numpy() for inst in frame.user_instances]


def same_labels(a: list[np.ndarray], b: list[np.ndarray]) -> bool:
    return len(a) == len(b) and all(np.allclose(x, y, equal_nan=True) for x, y in zip(a, b))


def build_reviewed(dst: Path) -> tuple[int, int]:
    """Write the reviewed ch01-05 frames, with the package's images, to ``dst``."""
    package = sio.load_slp(str(REVIEW_PKG))
    saved = sio.load_slp(str(REVIEW_SAVE), open_videos=False)
    if len(package.labeled_frames) != len(saved.labeled_frames):
        raise SystemExit(f"[ERR] {REVIEW_SAVE.name} has {len(saved.labeled_frames)} frames, "
                         f"the package {len(package.labeled_frames)}; they are not one review")
    skeleton = package.skeletons[0]
    frames = []
    for before, after in zip(package.labeled_frames, saved.labeled_frames):
        if int(before.frame_idx) != int(after.frame_idx):
            raise SystemExit(f"[ERR] frame order differs at frame_idx {before.frame_idx} / {after.frame_idx}")
        reviewed = user_points(after)
        if same_labels(user_points(before), reviewed):
            continue
        instances = [sio.Instance.from_numpy(points, skeleton=skeleton) for points in reviewed]
        frames.append(sio.LabeledFrame(video=before.video, frame_idx=before.frame_idx, instances=instances))
    videos = [v for v in package.videos if any(f.video is v for f in frames)]
    save_package(frames, videos, [skeleton], dst)
    n_inst = sum(len(f.instances) for f in frames)
    log(f"[OK] {dst.name}: {len(frames)} reviewed frames, {n_inst} user instances")
    return len(frames), n_inst


def count(path: Path) -> tuple[int, int]:
    labels = sio.load_slp(str(path), open_videos=False)
    with_user = [f for f in labels.labeled_frames if f.user_instances]
    return len(with_user), sum(len(f.user_instances) for f in with_user)


def write_config(out_dir: Path, labels: list[Path], crop: int, warm_start: Path, batch: int | None) -> Path:
    config = yaml.safe_load(BASE_CONFIG.read_text())
    run_name = f"final_v2_c{crop}"
    data = config["data_config"]
    data["train_labels_path"] = [str(p) for p in labels]
    data["cache_img_path"] = str(out_dir / "cache" / run_name)
    data["preprocessing"]["crop_size"] = crop

    model = config["model_config"]
    model["pretrained_backbone_weights"] = str(warm_start)
    model["pretrained_head_weights"] = str(warm_start)

    trainer = config["trainer_config"]
    for loader in ("train_data_loader", "val_data_loader"):
        # 8 spawn workers run fine on Windows; with 0 the GPU idles (0.66 vs 6.05 it/s at crop 768)
        trainer[loader]["num_workers"] = 8
        if batch:
            trainer[loader]["batch_size"] = batch
    trainer["trainer_devices"] = 1
    trainer["ckpt_dir"] = str(out_dir / "models")
    trainer["run_name"] = run_name
    trainer["resume_ckpt_path"] = None

    path = out_dir / f"{run_name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def main(argv: list[str] | None = None) -> int:
    global _LOG
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--crop", type=int, required=True)
    parser.add_argument("--warm-start", type=Path, default=WARM_START,
                        help="A0 centered-instance checkpoint final_v2 was warm-started from.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override final_v2's batch size (6) only if the GPU cannot hold it.")
    args = parser.parse_args(argv)

    label_dir = args.out_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    _LOG = (args.out_dir / f"prepare_log_{dt.datetime.now():%Y%m%d_%H%M%S}.txt").open("w", encoding="utf-8")
    if args.crop % 64:
        raise SystemExit(f"[ERR] crop {args.crop} is not a multiple of the backbone's max_stride 64")

    labels = [copy_if_needed(src, label_dir / src.name) for src in CH06]
    reviewed = label_dir / "ch01to05_reviewed31.pkg.slp"
    if reviewed.is_file():
        log(f"[skip] {reviewed.name} already built")
    else:
        build_reviewed(reviewed)
    labels.append(reviewed)

    totals = np.array([count(p) for p in labels]).sum(axis=0)
    log(f"training set: {totals[0]} frames, {totals[1]} user instances "
        f"(final_v2 recorded {EXPECTED['frames']} / {EXPECTED['user_instances']})")
    if totals[0] != EXPECTED["frames"] or abs(int(totals[1]) - EXPECTED["user_instances"]) > 2:
        raise SystemExit("[ERR] the rebuilt set does not match final_v2's; not writing a config")

    config = write_config(args.out_dir, labels, args.crop, args.warm_start, args.batch_size)
    log(f"[OK] config -> {config}")
    # PYTHONUTF8: rich's progress spinner cannot be written to a cp1252-redirected log on Windows
    log(f'train with: PYTHONUTF8=1 sleap-nn train "{config}" > train_log.txt 2>&1')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
