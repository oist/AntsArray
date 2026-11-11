#!/usr/bin/env python3
"""A/B test of the centered-instance crop size, without retraining.

The production centered-instance model was trained on 640 px crops centred on the
ArUco tag. Labelled antenna tips sit at a p99 of 313 px (Chebyshev) from the tag, so
about 0.9 % of instances have a node outside the crop. sleap-nn accepts a different
crop at inference (``--crop_size``; ``--crop-size`` at TRT export) because the UNet is
fully convolutional. This script measures whether the model still behaves at a larger
crop, before the pipeline is changed to export one.

Every arm is ground-truth anchored (see group_labelling/predict_gt_anchored.py): only
the centered-instance model is passed, so each crop is cut around a labelled
instance's own anchor and one prediction comes back per label. The centroid model
contributes nothing, and the arms differ in exactly one thing, the crop size.

What it reports, per crop:

  far tips        antenna_L3 / antenna_R_3 error, binned by how far the labelled tip
                  sits from the tag. Tips beyond 320 px are unreachable at 640, so
                  this is where a larger crop has to win.
  wrong ant       the crop is centred on the labelled tag, so the predicted tag must
                  land there; a tag > 20 px off means the model posed another ant
                  (``on_nbr``: it landed on a neighbour's tag).
  non-regression  every node within 256 px of the tag, which each crop sees in full.
                  A larger crop must leave these alone; a rise in gross errors
                  (> 20 px) on dense frames would mean the model is drifting onto a
                  neighbour that the wider crop now shows.
  per node        the full table, for reference.

Caveat: the production model trained on ch06 (ch06_full_v2) plus a reviewed subset of
ch01-05 (ch01to05_reviewed31), so ch06 scores optimistically and ch01-05 is mostly
unseen; score them separately (one --out-dir each) to tell the two apart. The
comparison between crops is fair either way: during training any keypoint outside the
640 crop was dropped, so the far tips this test is about were never fitted at the
positions they occupy here.

Run it with the sleap-nn uv tool's Python, which carries sleap_io and scipy::

    & "C:\\Program Files\\uv-tools\\tools\\sleap-nn\\Scripts\\python.exe" `
        makoto_play\\crop_size_ab.py `
        --model D:\\crop_ab\\model\\260803_topdown.centered_instance `
        --out-dir D:\\crop_ab\\run "D:\\crop_ab\\labels\\*.pkg.slp"

Existing predictions are reused, so a re-run only re-scores (``--overwrite`` redoes
them). Output: ``<out-dir>/report.json``, ``arrays.npz`` with every per-instance
array, sleap-nn logs under ``logs/``, and this script's console output in
``crop_size_ab_log_<timestamp>.txt``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sleap_io as sio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "group_labelling"))
from topdown_common import (  # noqa: E402
    DEFAULT_ANCHOR_NODE,
    iter_frame_pairs,
    match_point,
    node_names,
    resolve_nodes,
    stats_of,
    stem_of,
)

TIP_NODES = ["antenna_L3", "antenna_R_3"]
BASELINE_CROP = 640
INNER_REACH = 256          # px from the tag; every crop under test sees these in full
REACH_BINS = [(0, 256), (256, 320), (320, 352), (352, 384), (384, float("inf"))]
DENSE_MIN = 10             # user instances per frame
PCK = [5.0, 10.0, 20.0]
GROSS_PX = 20.0
MATCH_MAX_DIST = 100.0
PLAIN_FW = "torch_dataset"

_LOG_FH = None


def log(msg: str = "") -> None:
    print(msg, flush=True)
    if _LOG_FH:
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()


# ------------------------------------------------------------------ model staging


def stage_model(model_dir: Path, staged: Path) -> int:
    """Copy the config with the training-node cache undone; link the checkpoint.

    Same patch as predict_gt_anchored.py, but the checkpoint is hard-linked instead
    of symlinked: Windows refuses symlinks without developer mode. Returns the
    backbone's max_stride, which every crop must be a multiple of.
    """
    config = yaml.safe_load((model_dir / "training_config.yaml").read_text())
    data = config.get("data_config", {})
    data["data_pipeline_fw"] = PLAIN_FW
    data["cache_img_path"] = None
    data["use_existing_imgs"] = False

    staged.mkdir(parents=True, exist_ok=True)
    (staged / "training_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    link = staged / "best.ckpt"
    if not link.exists():
        try:
            os.link(model_dir / "best.ckpt", link)
        except OSError:
            shutil.copy2(model_dir / "best.ckpt", link)

    backbone = config["model_config"]["backbone_config"]
    return int(next(v for v in backbone.values() if v)["max_stride"])


# --------------------------------------------------------------------- prediction


def find_sleap_nn(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    beside = Path(sys.executable).parent / ("sleap-nn.exe" if os.name == "nt" else "sleap-nn")
    if beside.is_file():
        return beside
    found = shutil.which("sleap-nn")
    if not found:
        raise SystemExit("[ERR] sleap-nn not found; pass --sleap-nn")
    return Path(found)


def predict(sleap_nn: Path, staged: Path, src: Path, out: Path, crop: int, args) -> float:
    """One GT-anchored run. sleap-nn's own output goes to logs/, not the console."""
    command = [
        str(sleap_nn), "predict",
        "--data_path", str(src),
        "--model_paths", str(staged),
        "--output_path", str(out),
        "--crop_size", str(crop),
        "--peak_threshold", "0.0",   # any higher and GT-anchored frames come back empty
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--only_labeled_frames",
    ]
    run_log = args.out_dir / "logs" / f"{stem_of(src)}_crop{crop}.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with run_log.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not out.is_file():
        raise SystemExit(f"[ERR] sleap-nn predict failed ({result.returncode}); see {run_log}")
    return time.time() - started


# ------------------------------------------------------------------------ scoring


def nearest_other(anchors: np.ndarray, position: int, point: np.ndarray) -> float:
    """Distance from ``point`` to the closest anchor of any *other* labelled ant."""
    others = np.delete(anchors, position, axis=0)
    distances = np.linalg.norm(others - point, axis=1) if len(others) else np.array([])
    distances = distances[~np.isnan(distances)]
    return float(distances.min()) if len(distances) else float("nan")


def partner_of(pair, position: int):
    """(points, instance) of the prediction cut from this label's crop, or None.

    sleap-nn returns GT-anchored predictions in label order (checked on chunk03:
    1788 of 1789 anchors within 20 px by position at crop 640), so position IS the
    correspondence. Proximity matching would hide exactly what this test looks for:
    a crop that locked onto a neighbour more than MATCH_MAX_DIST away would score as
    "unmatched" instead of as an error. Proximity is only the fallback for a frame
    whose prediction count differs from its label count.
    """
    if len(pair.pred_instances) == len(pair.instances):
        return pair.pred_points[position], pair.pred_instances[position]
    matched = pair.match(position)
    return None if matched is None else (matched.points, matched.instance)


def collect(gt: sio.Labels, pred: sio.Labels, anchor_idx: int) -> dict:
    """Per labelled instance: truth, prediction, per-node score, frame density, and how
    close the predicted anchor came to a neighbour's labelled anchor."""
    n_nodes = len(gt.skeletons[0].nodes)
    rows = {"gt": [], "pred": [], "score": [], "density": [], "neighbour": [],
            "video": [], "frame": [], "position": []}
    for pair in iter_frame_pairs(gt, pred, anchor_idx, MATCH_MAX_DIST):
        anchors = np.array([match_point(points, anchor_idx) for points in pair.gt_points])
        for position in range(len(pair.instances)):
            rows["gt"].append(pair.gt_points[position])
            rows["density"].append(len(pair.instances))
            rows["video"].append(pair.video_idx)
            rows["frame"].append(int(pair.frame.frame_idx))
            rows["position"].append(position)
            partner = partner_of(pair, position)
            if partner is None:
                rows["pred"].append(np.full((n_nodes, 2), np.nan))
                rows["score"].append(np.full(n_nodes, np.nan))
                rows["neighbour"].append(np.nan)
                continue
            points, instance = partner
            rows["pred"].append(points)
            rows["score"].append(instance.numpy(scores=True)[:, 2])
            rows["neighbour"].append(nearest_other(anchors, position, match_point(points, anchor_idx)))
    return {key: np.array(value) for key, value in rows.items()}


def reach_of(gt_points: np.ndarray, anchor_idx: int) -> np.ndarray:
    """(N, nodes) Chebyshev distance of each labelled node from the crop centre.

    The centre is the anchor, or the visible-node mean when the anchor is hidden --
    the same point sleap-nn crops around.
    """
    centres = np.array([match_point(points, anchor_idx) for points in gt_points])
    return np.abs(gt_points - centres[:, None, :]).max(axis=2)


def score_package(src: Path, preds: dict[int, Path]) -> tuple[list[str], int, dict]:
    gt = sio.load_slp(str(src))
    names = node_names(gt)
    anchor_idx = resolve_nodes(gt, [DEFAULT_ANCHOR_NODE], "gt")[0]
    per_crop = {crop: collect(gt, sio.load_slp(str(path)), anchor_idx) for crop, path in preds.items()}

    first = next(iter(per_crop.values()))
    for crop, rows in per_crop.items():
        if not np.allclose(rows["gt"], first["gt"], equal_nan=True):
            raise SystemExit(f"[ERR] {src.name}: crop {crop} paired a different label order")
    return names, anchor_idx, per_crop


# ----------------------------------------------------------------------- reporting

STATS_HEADER = f"{'n':>6}{'median':>8}{'p90':>8}{'pck@5':>8}{'pck@10':>8}{'>20px':>8}"


def fmt_stats(s: dict) -> str:
    if not s.get("n"):
        return f"{0:>6}" + " " * 40
    gross = 1.0 - s["pck@20"]
    return (f"{s['n']:>6}{s['median']:>8.2f}{s['p90']:>8.2f}"
            f"{s['pck@5']:>8.3f}{s['pck@10']:>8.3f}{gross:>8.3f}")


def tip_block(err: dict, score: dict, reach: np.ndarray, tip_idx: list[int], crops: list[int]) -> dict:
    log("\n=== antenna tips (antenna_L3, antenna_R_3) by labelled reach from the tag ===")
    log(f"{'reach px':<12}{'crop':>6}{STATS_HEADER}{'score':>8}")
    out = {}
    tip_reach = reach[:, tip_idx]
    for lo, hi in REACH_BINS:
        in_bin = (tip_reach > lo) & (tip_reach <= hi) if lo else (tip_reach <= hi)
        label = f"{lo}-{hi:g}" if hi != float("inf") else f">{lo}"
        for crop in crops:
            s = stats_of(err[crop][:, tip_idx][in_bin], PCK)
            mean_score = float(np.nanmean(score[crop][:, tip_idx][in_bin])) if s.get("n") else float("nan")
            log(f"{label:<12}{crop:>6}{fmt_stats(s)}{mean_score:>8.3f}")
            out[f"{label}@{crop}"] = {**s, "mean_score": round(mean_score, 4)}
        log("")
    return out


def density_groups(density: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [("all", np.ones(len(density), bool)),
            (f"dense>={DENSE_MIN}", density >= DENSE_MIN),
            (f"sparse<{DENSE_MIN}", density < DENSE_MIN)]


def swap_block(err: dict, neighbour: dict, density: np.ndarray, anchor_idx: int,
               crops: list[int]) -> dict:
    """How often the crop's prediction is not the ant the crop was centred on.

    The crop is centred on the labelled tag, so the predicted tag should land at the
    centre. ``off`` counts predictions whose tag is more than GROSS_PX away;
    ``on_neighbour`` is the subset that landed on another labelled ant's tag instead.
    """
    log(f"\n=== crop centred on one ant, prediction of another (tag > {GROSS_PX:g} px off) ===")
    log(f"{'frames':<12}{'crop':>6}{'n':>7}{'off':>7}{'off %':>8}{'on_nbr':>8}{'on_nbr %':>10}")
    out = {}
    for label, rows in density_groups(density):
        for crop in crops:
            anchor_err = err[crop][rows, anchor_idx]
            valid = ~np.isnan(anchor_err)
            off = valid & (anchor_err > GROSS_PX)
            on_nbr = off & (neighbour[crop][rows] <= GROSS_PX)
            n = int(valid.sum())
            rec = {"n": n, "off": int(off.sum()), "on_neighbour": int(on_nbr.sum()),
                   "off_rate": round(off.sum() / max(n, 1), 5),
                   "on_neighbour_rate": round(on_nbr.sum() / max(n, 1), 5)}
            log(f"{label:<12}{crop:>6}{n:>7}{rec['off']:>7}{100 * rec['off_rate']:>8.2f}"
                f"{rec['on_neighbour']:>8}{100 * rec['on_neighbour_rate']:>10.2f}")
            out[f"{label}@{crop}"] = rec
        log("")
    return out


def regression_block(err: dict, reach: np.ndarray, density: np.ndarray, anchor_idx: int,
                     crops: list[int]) -> dict:
    log(f"=== non-regression: every non-anchor node within {INNER_REACH} px of the tag ===")
    log(f"{'frames':<12}{'crop':>6}{STATS_HEADER}{'dMean':>8}{'worse5':>8}{'better5':>8}")
    inner = reach <= INNER_REACH
    inner[:, anchor_idx] = False
    base = err[BASELINE_CROP]
    out = {}
    for label, rows in density_groups(density):
        mask = inner & rows[:, None]
        for crop in crops:
            s = stats_of(err[crop][mask], PCK)
            delta = (err[crop] - base)[mask]
            delta = delta[~np.isnan(delta)]
            d = {"mean_delta": float(delta.mean()) if len(delta) else float("nan"),
                 "frac_worse_5px": float((delta > 5).mean()) if len(delta) else float("nan"),
                 "frac_better_5px": float((delta < -5).mean()) if len(delta) else float("nan")}
            log(f"{label:<12}{crop:>6}{fmt_stats(s)}{d['mean_delta']:>8.2f}"
                f"{d['frac_worse_5px']:>8.4f}{d['frac_better_5px']:>8.4f}")
            out[f"{label}@{crop}"] = {**s, **{k: round(v, 5) for k, v in d.items()}}
        log("")
    log("dMean / worse5 / better5 are paired against crop 640 on the same node.")
    return out


NODE_GROUPS = {
    "tag": ["aruco"],
    "body": ["occiput", "petiole", "gaster_tip"],
    "antenna base": ["antenna_L_1", "antenna_R_1"],
    "antenna mid": ["antenna L_2", "antenna_R_2"],
    "antenna tip": ["antenna_L3", "antenna_R_3"],
}
GOOD_PX = 10.0


def flip_block(err: dict, names: list[str], crops: list[int]) -> dict:
    """Per 1000 instances: node predictions that flip between good and bad vs crop 640.

    ``lost`` = good (<= 10 px) at 640 but gross (> 20 px) at the other crop;
    ``gained`` = the reverse. Easier to weigh than a shift in a percentage: it says how
    many body points per thousand ants a crop change breaks and how many it fixes.
    """
    folded = [name.strip().lower().replace(" ", "_") for name in names]
    base = err[BASELINE_CROP]
    n_inst = len(base)
    log(f"\n=== good<->bad flips vs crop {BASELINE_CROP}, per 1000 instances "
        f"(good <= {GOOD_PX:g} px, bad > {GROSS_PX:g} px) ===")
    log(f"{'nodes':<14}" + "".join(f"{'crop ' + str(c):>18}" for c in crops if c != BASELINE_CROP))
    log(f"{'':<14}" + "".join(f"{'lost':>9}{'gained':>9}" for c in crops if c != BASELINE_CROP))
    out = {}
    for group, members in [*NODE_GROUPS.items(), ("all", None)]:
        idx = list(range(len(names))) if members is None else [
            folded.index(m.lower().replace(" ", "_")) for m in members]
        cells = []
        for crop in crops:
            if crop == BASELINE_CROP:
                continue
            a, b = base[:, idx], err[crop][:, idx]
            lost = int(((a <= GOOD_PX) & (b > GROSS_PX)).sum())
            gained = int(((a > GROSS_PX) & (b <= GOOD_PX)).sum())
            out[f"{group}@{crop}"] = {"lost": lost, "gained": gained,
                                      "lost_per_1000": round(1000 * lost / n_inst, 2),
                                      "gained_per_1000": round(1000 * gained / n_inst, 2)}
            cells.append(f"{1000 * lost / n_inst:>9.1f}{1000 * gained / n_inst:>9.1f}")
        log(f"{group:<14}" + "".join(cells))
    return out


def node_block(err: dict, names: list[str], crops: list[int]) -> dict:
    log("\n=== per node, all instances ===")
    log(f"{'node':<14}{'crop':>6}{STATS_HEADER}")
    out = {}
    for idx, name in enumerate(names):
        for crop in crops:
            s = stats_of(err[crop][:, idx], PCK)
            log(f"{name.strip():<14}{crop:>6}{fmt_stats(s)}")
            out[f"{name.strip()}@{crop}"] = s
    return out


# --------------------------------------------------------------------------- main


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        hits = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        paths.extend(Path(hit) for hit in hits)
    missing = [p for p in paths if not p.is_file()]
    if not paths or missing:
        raise SystemExit(f"[ERR] no label packages, or missing: {missing}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="Label packages (globs are expanded).")
    parser.add_argument("--model", type=Path, required=True, help="Centered-instance model dir.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--crops", default="640,704,768",
                        help="Comma-separated crop sizes; 640 (the trained size) is always included.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Frames per batch, NOT crops: a dense frame is ~40 crops of up to 768 px.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sleap-nn", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Redo existing predictions.")
    return parser


def run_predictions(args, packages: list[Path], crops: list[int]) -> dict:
    staged = args.out_dir / "model"
    max_stride = stage_model(args.model, staged)
    bad = [c for c in crops if c % max_stride]
    if bad:
        raise SystemExit(f"[ERR] crops {bad} are not multiples of max_stride {max_stride}")
    sleap_nn = find_sleap_nn(args.sleap_nn)
    log(f"[OK] staged {args.model} at {staged} (max_stride {max_stride}); sleap-nn {sleap_nn}")

    paths, seconds = {}, {crop: 0.0 for crop in crops}
    for crop in crops:
        for src in packages:
            out = args.out_dir / f"crop{crop}" / f"{stem_of(src)}.pred.slp"
            out.parent.mkdir(parents=True, exist_ok=True)
            paths[(src, crop)] = out
            if out.is_file() and not args.overwrite:
                log(f"[skip] crop {crop} {src.name}: prediction exists")
                continue
            log(f"[..] crop {crop} {src.name}")
            took = predict(sleap_nn, staged, src, out, crop, args)
            seconds[crop] += took
            log(f"[OK] crop {crop} {src.name} in {took / 60:.1f} min")
    return {"paths": paths, "seconds": seconds}


def main(argv: list[str] | None = None) -> int:
    global _LOG_FH
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FH = (args.out_dir / f"crop_size_ab_log_{stamp}.txt").open("w", encoding="utf-8")

    crops = sorted({BASELINE_CROP, *(int(c) for c in args.crops.split(",") if c.strip())})
    packages = expand_inputs(args.inputs)
    log(f"crop_size_ab {stamp}: crops {crops}, {len(packages)} packages, batch {args.batch_size}")

    runs = run_predictions(args, packages, crops)

    merged: dict[int, dict[str, list]] = {crop: {} for crop in crops}
    names = anchor_idx = None
    for pkg_idx, src in enumerate(packages):
        names, anchor_idx, per_crop = score_package(src, {c: runs["paths"][(src, c)] for c in crops})
        log(f"[OK] scored {src.name}: {len(per_crop[BASELINE_CROP]['gt'])} instances")
        for crop, rows in per_crop.items():
            rows["package"] = np.full(len(rows["gt"]), pkg_idx)
            for key, value in rows.items():
                merged[crop].setdefault(key, []).append(value)
    arrays = {crop: {k: np.concatenate(v) for k, v in rows.items()} for crop, rows in merged.items()}

    gt_points = arrays[BASELINE_CROP]["gt"]
    density = arrays[BASELINE_CROP]["density"]
    reach = reach_of(gt_points, anchor_idx)
    err = {c: np.linalg.norm(gt_points - arrays[c]["pred"], axis=-1) for c in crops}
    score = {c: arrays[c]["score"] for c in crops}
    unmatched = {c: int(np.isnan(arrays[c]["pred"]).all(axis=(1, 2)).sum()) for c in crops}
    log(f"\n{len(gt_points)} labelled instances; unmatched per crop {unmatched}")
    tip_idx = [names.index(n) for n in TIP_NODES]

    report = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "crops": crops,
        "n_instances": int(len(gt_points)),
        "unmatched": unmatched,
        "predict_seconds_this_run": {str(c): round(s, 1) for c, s in runs["seconds"].items()},
        "tips_by_reach": tip_block(err, score, reach, tip_idx, crops),
        "wrong_ant": swap_block(err, {c: arrays[c]["neighbour"] for c in crops}, density,
                                anchor_idx, crops),
        "non_regression": regression_block(err, reach, density, anchor_idx, crops),
        "flips": flip_block(err, names, crops),
        "per_node": node_block(err, names, crops),
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    ident = arrays[BASELINE_CROP]
    np.savez_compressed(args.out_dir / "arrays.npz", gt=gt_points, density=density, reach=reach,
                        names=np.array(names), packages=np.array([str(p) for p in packages]),
                        package=ident["package"], video=ident["video"], frame=ident["frame"],
                        position=ident["position"],
                        **{f"pred_{c}": arrays[c]["pred"] for c in crops},
                        **{f"score_{c}": score[c] for c in crops},
                        **{f"neighbour_{c}": arrays[c]["neighbour"] for c in crops})
    log(f"\n[OK] report -> {args.out_dir / 'report.json'}; arrays -> {args.out_dir / 'arrays.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
