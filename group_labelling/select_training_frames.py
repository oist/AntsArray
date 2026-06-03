#!/usr/bin/env python3
"""Select training frames from a per-frame inventory CSV.

Reads the output of ``build_training_inventory.py`` and emits
``selected_frames.csv`` with ``(camera, frame_idx, stratum, ...)`` rows for
Stage 3 to bundle into SLEAP labeling packages.

Selection strategy
------------------
Per-camera quotas with stratified greedy sampling and a temporal min-gap
(default 500 frames). Strata are applied in priority order so rare strata
(problematic, crowding) fill first and don't get crowded out by abundant
strata (density bins). A frame is tagged with the first stratum that
selects it; later strata skip already-selected frames.

For foraging cams, frames with zero ArUco AND zero SLEAP are pre-filtered
out before stratification - empty frames waste quota.

If a stratum pool runs out before its sub-quota, the leftover is rolled
into the next stratum, and finally into a "fill" stratum that samples
from any remaining useful frames. If even that pool can't satisfy
min-gap, the script retries with progressively smaller gaps
(500 -> 200 -> 100 -> 50 -> 0).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

NEST_CAMS = {"cam01", "cam02", "cam04", "cam05", "cam06", "cam07", "cam09", "cam10"}

QUOTAS: dict[str, int] = {
    # Nest standard
    "cam01": 100, "cam04": 100, "cam06": 100, "cam07": 100, "cam09": 100, "cam10": 100,
    # Nest high-uncertainty
    "cam02": 120, "cam05": 120,
    # Foraging dense
    "cam14": 60, "cam24": 60,
    # Foraging high-uncertainty
    "cam21": 80, "cam22": 80, "cam25": 80,
    # Foraging standard
    "cam03": 40, "cam11": 40, "cam15": 40, "cam19": 40, "cam20": 40, "cam23": 40,
    # Foraging sparse - small but non-zero per user request
    "cam08": 20, "cam12": 20, "cam13": 20, "cam16": 20, "cam17": 20, "cam18": 20,
}

# Sub-quota weights - applied in this order. Rare strata first.
STRATA_WEIGHTS: "OrderedDict[str, float]" = OrderedDict([
    ("problematic",      0.15),
    ("crowding_or_busy", 0.15),
    ("edge",             0.10),
    ("density_high",     0.15),
    ("density_mid",      0.15),
    ("density_low",      0.10),
    ("motion_fast",      0.10),
    ("motion_slow",      0.10),
])

GAP_DECAY = (500, 200, 100, 50, 0)
DEFAULT_SEED = 20260520


@dataclass(frozen=True)
class CamStats:
    unm_sleap_p95: float
    score_low_thr: float
    min_pair_p10: float
    n_sleap_p90: float
    n_aruco_p25: float
    n_aruco_p75: float
    mean_speed_p20: float
    mean_speed_p80: float


def cam_stats(df: pd.DataFrame) -> CamStats:
    score_med = float(df["mean_kp_score"].median(skipna=True))
    return CamStats(
        unm_sleap_p95=float(df["n_unmatched_sleap"].quantile(0.95)),
        score_low_thr=score_med - 0.05 if np.isfinite(score_med) else 0.5,
        min_pair_p10=float(df["min_pair_dist"].quantile(0.10)),
        n_sleap_p90=float(df["n_sleap"].quantile(0.90)),
        n_aruco_p25=float(df["n_aruco"].quantile(0.25)),
        n_aruco_p75=float(df["n_aruco"].quantile(0.75)),
        mean_speed_p20=float(df["mean_speed"].quantile(0.20)),
        mean_speed_p80=float(df["mean_speed"].quantile(0.80)),
    )


def stratum_mask(df: pd.DataFrame, stratum: str, stats: CamStats, is_nest: bool) -> pd.Series:
    if stratum == "problematic":
        return (
            (df["n_unmatched_sleap"] > stats.unm_sleap_p95)
            | (df["n_duplicate_sleap"] >= 1)
            | (df["n_low_score_inst"] >= 2)
            | (df["mean_kp_score"] < stats.score_low_thr)
        )
    if stratum == "crowding_or_busy":
        if is_nest:
            return df["min_pair_dist"] <= stats.min_pair_p10
        return df["n_sleap"] >= stats.n_sleap_p90
    if stratum == "edge":
        return df["n_near_edge_aruco"] >= 1
    if stratum == "density_high":
        return df["n_aruco"] >= stats.n_aruco_p75
    if stratum == "density_mid":
        return (df["n_aruco"] > stats.n_aruco_p25) & (df["n_aruco"] < stats.n_aruco_p75)
    if stratum == "density_low":
        return (df["n_aruco"] >= 1) & (df["n_aruco"] <= stats.n_aruco_p25)
    if stratum == "motion_fast":
        return df["mean_speed"] >= stats.mean_speed_p80
    if stratum == "motion_slow":
        return (df["mean_speed"] <= stats.mean_speed_p20) & df["mean_speed"].notna()
    raise ValueError(f"unknown stratum: {stratum}")


def greedy_sample_with_gap(
    candidate_idx: np.ndarray,
    sub_quota: int,
    already_selected: dict[int, str],
    min_gap: int,
    rng: np.random.Generator,
) -> list[int]:
    """Pick up to `sub_quota` frames from candidate_idx s.t. each pick is >= min_gap
    away from any already-selected frame in the same camera."""
    if sub_quota <= 0 or candidate_idx.size == 0:
        return []
    order = rng.permutation(candidate_idx)
    if already_selected:
        existing = np.fromiter(already_selected.keys(), dtype=np.int64)
        existing.sort()
    else:
        existing = np.array([], dtype=np.int64)

    picked: list[int] = []
    picked_arr = np.array([], dtype=np.int64)

    for f in order:
        if len(picked) >= sub_quota:
            break
        if f in already_selected:
            continue
        if existing.size and np.min(np.abs(existing - f)) < min_gap:
            continue
        if picked_arr.size and np.min(np.abs(picked_arr - f)) < min_gap:
            continue
        picked.append(int(f))
        picked_arr = np.append(picked_arr, int(f))
    return picked


def select_for_camera(
    df_cam: pd.DataFrame,
    quota: int,
    rng: np.random.Generator,
) -> dict[int, str]:
    cam = df_cam["camera"].iloc[0]
    is_nest = cam in NEST_CAMS

    # Frames absent from the source .slp cannot be packaged for labeling.
    # frame_in_slp added in inventory script — fall back to "all True" for
    # backward compatibility with pre-fix inventory CSVs.
    if "frame_in_slp" in df_cam.columns:
        in_slp = df_cam["frame_in_slp"].astype(bool)
        n_absent = int((~in_slp).sum())
        if n_absent:
            logging.info("[%s] excluding %d frame(s) not present in source .slp",
                         cam, n_absent)
        df_cam = df_cam[in_slp].reset_index(drop=True)

    if is_nest:
        useful = df_cam
    else:
        useful = df_cam[(df_cam["n_aruco"] >= 1) | (df_cam["n_sleap"] >= 1)]

    if useful.empty:
        logging.warning("[%s] no useful frames after filtering - skipping", cam)
        return {}

    stats = cam_stats(useful)
    logging.info(
        "[%s] useful=%d (of %d)  unm_sleap_p95=%.1f  score_low_thr=%.2f  "
        "min_pair_p10=%.0f  n_sleap_p90=%.0f  density_q=(%.0f/%.0f)  speed_q=(%.2f/%.2f)",
        cam, len(useful), len(df_cam),
        stats.unm_sleap_p95, stats.score_low_thr, stats.min_pair_p10,
        stats.n_sleap_p90, stats.n_aruco_p25, stats.n_aruco_p75,
        stats.mean_speed_p20, stats.mean_speed_p80,
    )

    selected: dict[int, str] = {}
    sub_quotas = {s: int(round(quota * w)) for s, w in STRATA_WEIGHTS.items()}
    diff = quota - sum(sub_quotas.values())
    if diff != 0:
        first = next(iter(sub_quotas))
        sub_quotas[first] += diff

    leftover = 0
    for stratum, sub_q in sub_quotas.items():
        target = sub_q + leftover
        if target <= 0:
            continue
        mask = stratum_mask(useful, stratum, stats, is_nest)
        candidates = useful.loc[mask, "frame_idx"].to_numpy(dtype=np.int64)
        picked: list[int] = []
        for gap in GAP_DECAY:
            picked = greedy_sample_with_gap(candidates, target, selected, gap, rng)
            if len(picked) >= target or gap == 0:
                break
        for f in picked:
            selected[f] = stratum
        leftover = max(0, target - len(picked))
        logging.debug("[%s]   %-18s target=%d picked=%d leftover=%d",
                      cam, stratum, target, len(picked), leftover)

    remaining = quota - len(selected)
    if remaining > 0:
        candidates = useful["frame_idx"].to_numpy(dtype=np.int64)
        for gap in GAP_DECAY:
            picked = greedy_sample_with_gap(candidates, remaining, selected, gap, rng)
            if len(picked) >= remaining or gap == 0:
                break
        for f in picked:
            selected[f] = "fill"

    logging.info("[%s] selected %d/%d frames", cam, len(selected), quota)
    return selected


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True,
                   help="inventory_master.csv from build_training_inventory.py")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Output path (default: <inventory dir>/selected_frames.csv)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--cameras", type=str, default=None,
                   help="Comma-separated cam ids to limit to")
    args = p.parse_args()

    inv_path: Path = args.inventory.resolve()
    out_path: Path = (args.out_csv or inv_path.parent / "selected_frames.csv").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / f"select_training_frames_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_path, encoding="utf-8")],
    )
    logging.info("inventory=%s", inv_path)
    logging.info("out=%s", out_path)
    logging.info("seed=%d", args.seed)

    df = pd.read_csv(inv_path)
    logging.info("loaded %d rows, %d cameras", len(df), df["camera"].nunique())

    rng = np.random.default_rng(args.seed)
    wanted = {c.strip() for c in args.cameras.split(",")} if args.cameras else None

    all_rows: list[pd.DataFrame] = []
    for cam in sorted(df["camera"].unique()):
        if wanted is not None and cam not in wanted:
            continue
        quota = QUOTAS.get(cam)
        if quota is None:
            logging.warning("[%s] no quota defined - skipping", cam)
            continue
        df_cam = df[df["camera"] == cam].sort_values("frame_idx").reset_index(drop=True)
        selected = select_for_camera(df_cam, quota, rng)
        if not selected:
            continue
        sel_df = df_cam[df_cam["frame_idx"].isin(selected)].copy()
        sel_df["stratum"] = sel_df["frame_idx"].map(selected)
        all_rows.append(sel_df[[
            "camera", "frame_idx", "stratum",
            "n_aruco", "n_sleap", "n_unmatched_sleap",
            "mean_kp_score", "mean_speed", "min_pair_dist",
        ]])

    if not all_rows:
        logging.error("no frames selected")
        return 1

    out = pd.concat(all_rows, ignore_index=True).sort_values(
        ["camera", "frame_idx"]
    ).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    logging.info("wrote %d rows to %s", len(out), out_path)

    by_cam = out.groupby("camera").size()
    logging.info("per-camera counts:\n%s", by_cam.to_string())
    logging.info("per-stratum counts:\n%s",
                 out["stratum"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
