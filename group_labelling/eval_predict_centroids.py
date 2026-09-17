#!/usr/bin/env python3
"""Stage B: run BOTH centroid models over the SAME anchor clip, in one task.

Running both models inside a single task on a single clip is what makes the
pairing exact rather than nominal -- there is no opportunity for a different
decode, frame ordering or preprocessing path to creep between them.

Two config properties must be undone before a published model can predict off
the node that trained it (same treatment as predict_gt_anchored.patch_config,
minus its centered_instance requirement, which hard-exits on a centroid-only
model):
  data_pipeline_fw  -> torch_dataset  (the trained value assumes a cached store)
  cache_img_path    -> null           (points at /scratch on the training node)
  use_existing_imgs -> false
The published model is never modified; /bucket is read-only anyway.

--peak_threshold 0.0 is mandatory for this family: at any higher threshold no
peak clears it. That returns every confmap local maximum, so the operating point
is chosen afterwards in analysis by thinning on the within-model score quantile
(EVAL_HARNESS_DESIGN.md section 6). Do not "fix" this by raising the threshold --
it would hard-code one operating point and destroy the frontier.

Contract: group_labelling/EVAL_HARNESS_DESIGN.md section 6.

    python group_labelling/eval_predict_centroids.py \
        --out-dir /work/ReiterU/centroid_eval --key block01_cam10
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import MODELS, save_table, setup_logging, write_json  # noqa: E402

SLEAP_NN_BIN = Path("/apps/unit/ReiterU/sleap-nn/0.3.1/tools/sleap-nn/bin")

# 3x the observed max ant count. peak_threshold 0.0 returns every confmap local
# maximum (~55,850/frame measured), so an ingest cap is mandatory -- without it a
# single camera-model pair writes ~5.5 GB of noise peaks.
K_CAP = 150


def stage_model(model_dir: Path, staged: Path) -> dict:
    cfg_path = model_dir / "training_config.yaml"
    ckpt = model_dir / "best.ckpt"
    if not cfg_path.is_file():
        raise SystemExit(f"no training_config.yaml in {model_dir}")
    if not ckpt.is_file():
        raise SystemExit(f"no best.ckpt in {model_dir}")

    cfg = yaml.safe_load(cfg_path.read_text())
    heads = cfg.get("model_config", {}).get("head_configs", {})
    centroid = heads.get("centroid")
    if centroid is None:
        raise SystemExit(f"{model_dir} has no centroid head; heads: "
                         f"{sorted(k for k, v in heads.items() if v)}")

    data = cfg.setdefault("data_config", {})
    before = {"data_pipeline_fw": data.get("data_pipeline_fw"),
              "cache_img_path": data.get("cache_img_path")}
    data["data_pipeline_fw"] = "torch_dataset"
    data["cache_img_path"] = None
    data["use_existing_imgs"] = False

    staged.mkdir(parents=True, exist_ok=True)
    (staged / "training_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    link = staged / "best.ckpt"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(ckpt.resolve())  # large, and read-only where it lives

    prep = data.get("preprocessing", {})
    conf = centroid.get("confmaps", {})
    return {
        "staged": str(staged),
        "patched_from": before,
        "anchor_part": conf.get("anchor_part"),
        "sigma": conf.get("sigma"),
        "output_stride": conf.get("output_stride"),
        "scale": prep.get("scale"),
    }


def run_predict(staged: Path, clip: Path, out_slp: Path, batch_size: int,
                device: str, dry_run: bool, k_cap: int) -> dict:
    cmd = [
        str(SLEAP_NN_BIN / "sleap-nn"), "predict",
        "--data_path", str(clip),
        "--model_paths", str(staged),
        "--output_path", str(out_slp),
        "--peak_threshold", "0.0",
        "--centroid_only",
        "--centroid-output", "centroid",
        "--max_instances", str(k_cap),
        "--batch_size", str(batch_size),
        "--device", device,
    ]
    if dry_run:
        logging.info("[dry-run] %s", " ".join(cmd))
        return {"command": cmd, "skipped": True}
    t0 = time.time()
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise SystemExit(f"sleap-nn predict failed ({rc}) on {clip.name}")
    if not out_slp.is_file():
        raise SystemExit(f"sleap-nn predict wrote nothing to {out_slp}")
    return {"command": cmd, "seconds": round(time.time() - t0, 1)}


def slp_to_points(slp_path: Path, coord_scale: float, width: int, height: int,
                  k_cap: int) -> pd.DataFrame:
    """Flatten centroid predictions to (clip_frame, x, y, score).

    With --centroid-output centroid the peaks live in the file's top-level
    ``centroids`` group, NOT in LabeledFrame.instances (which is empty). Read
    them straight out of HDF5: it is both correct and far cheaper than
    materialising the object graph for tens of millions of peaks.

    Two defensive filters, both measured as necessary on cam10:
      * peak_threshold 0.0 returns EVERY confmap local maximum -- ~55,850 per
        frame, not ~23 -- and integral refinement diverges on the flat noise
        peaks, emitting coordinates up to 3.1e6 and -inf. Non-finite and
        out-of-frame peaks are dropped.
      * only the top `k_cap` peaks per frame by score are kept. k_cap must sit
        far above any operating point (pi_native is ~23/frame); at 150 the top
        peaks are all finite and in-frame, and the score gap is stark -- the
        top 23 score 0.88-1.01 against a 0.0002 median.

    coord_scale converts peaks to full-resolution pixels. Gate 0e measured 1.0
    on this build (median nearest-tag distance 2.1-2.7 px at 1x versus 600-1300
    px at 2x), i.e. sleap-nn already returns full-resolution coordinates despite
    preprocessing.scale 0.5. Every constant in the design is full-resolution.
    """
    with h5py.File(slp_path, "r") as h:
        if "centroids" not in h:
            raise SystemExit(
                f"{slp_path} has no 'centroids' group -- was --centroid-output "
                f"centroid passed? keys: {sorted(h.keys())}")
        c = h["centroids"]
        frame = np.asarray(c["frame_idx"][:], np.int64)
        x = np.asarray(c["x"][:], np.float64) * coord_scale
        y = np.asarray(c["y"][:], np.float64) * coord_scale
        score = np.asarray(c["score"][:], np.float64)

    keep = (np.isfinite(x) & np.isfinite(y) & np.isfinite(score)
            & (x >= -1) & (x <= width + 1) & (y >= -1) & (y <= height + 1))
    n_dropped = int((~keep).sum())
    df = pd.DataFrame({"clip_frame": frame[keep], "x": x[keep], "y": y[keep],
                       "score": score[keep]})
    logging.info("  %d raw peaks, %d dropped as non-finite/out-of-frame",
                 len(frame), n_dropped)
    if k_cap > 0:
        df = (df.sort_values("score", ascending=False)
                .groupby("clip_frame", sort=False)
                .head(k_cap)
                .reset_index(drop=True))
    return df.sort_values(["clip_frame", "score"], ascending=[True, False]) \
             .reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--key", type=str, default=None)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--coord-scale", type=float, default=None,
                   help="override; default reads stage0_report.json, else 1.0")
    p.add_argument("--k-cap", type=int, default=K_CAP,
                   help="max peaks kept per frame by score (0 = no cap)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out: Path = args.out_dir.resolve()
    setup_logging(out / "logs", "eval_predict_centroids")

    coord_scale = args.coord_scale
    if coord_scale is None:
        s0 = out / "stage0_report.json"
        if s0.exists():
            coord_scale = float(json.loads(s0.read_text(encoding="utf-8"))
                                .get("coord_scale", 1.0))
        else:
            coord_scale = 1.0
            logging.warning("no stage0_report.json and no --coord-scale; assuming "
                            "coord_scale=1.0. Gate 0e must confirm this before any "
                            "number is believed.")
    logging.info("coord_scale=%.4f", coord_scale)

    strata = json.loads((out / "strata.json").read_text(encoding="utf-8"))
    keys = sorted(k for g in strata["strata"].values() for k in g)
    if args.key:
        selected = [args.key]
    elif args.array_index is not None:
        if args.array_index >= len(keys):
            logging.error("array index %d out of range (%d keys)",
                          args.array_index, len(keys))
            return 1
        selected = [keys[args.array_index]]
    else:
        selected = keys

    preds = out / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    results = []
    for key in selected:
        manifest = json.loads(
            (out / "anchors" / f"{key}.json").read_text(encoding="utf-8"))
        width, height = int(manifest["width"]), int(manifest["height"])
        clip = out / "clips" / f"{key}.mkv"
        if not clip.exists():
            logging.error("[%s] no clip at %s -- run Stage A first", key, clip)
            continue
        for name, model_dir in MODELS.items():
            staged = out / "staged" / name
            info = stage_model(model_dir, staged)
            logging.info("[%s/%s] anchor_part=%r scale=%s stride=%s",
                         key, name, info["anchor_part"], info["scale"],
                         info["output_stride"])
            out_slp = preds / f"{key}_{name}.slp"
            try:
                run = run_predict(staged, clip, out_slp, args.batch_size,
                                  args.device, args.dry_run, args.k_cap)
            except SystemExit as exc:
                logging.error("[%s/%s] %s", key, name, exc)
                continue
            if args.dry_run:
                continue
            df = slp_to_points(out_slp, coord_scale, width, height, args.k_cap)
            df.insert(0, "model", name)
            df.insert(0, "key", key)
            saved = save_table(df, preds / f"{key}_{name}.parquet")
            n_frames = max(int(df["clip_frame"].nunique()), 1)
            per_frame = len(df) / n_frames
            logging.info("[%s/%s] %d peaks over %d frames (%.2f/frame) -> %s in "
                         "%.1f min", key, name, len(df), n_frames, per_frame,
                         saved.name, run.get("seconds", 0) / 60)
            results.append({"key": key, "model": name, "n_peaks": int(len(df)),
                            "per_frame": per_frame, "coord_scale": coord_scale,
                            **info,
                            **{k: v for k, v in run.items() if k != "command"}})
    if not results:
        return 0 if args.dry_run else 1
    tag = args.key or (f"idx{args.array_index}" if args.array_index is not None
                       else "all")
    write_json(results, preds / f"stage_b_stats_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
