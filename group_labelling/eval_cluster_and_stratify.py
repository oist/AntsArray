#!/usr/bin/env python3
"""Stage C: clusters, exact cancellation, the disagreement set X, HT sampling.

This is the heart of the design (EVAL_HARNESS_DESIGN.md section 2).

For each anchor frame and operating point:
  1. thin each model to the operating point on its OWN score quantile
  2. symmetric 50 px edge exclusion, identical for both models
  3. single-linkage components of D_A0 union D_all6 at R_link = 25 px
  4. inside each component, optimal assignment between the two models at R_pair
  5. a component where every detection pairs is a CANCELLING cluster: the two
     models present the same point set, so it contributes exactly zero to both
     dFN and dFP whatever is really in it -- queen, untagged worker, unread tag
     and shared hallucination all drop out with coefficient zero
  6. components with >=1 model-exclusive detection are the disagreement set X,
     the only population carrying signal and the only one a human ever sees

X is partitioned H1..H5 by which model is exclusive and by ArUco tag support,
then sampled with RECORDED inclusion probabilities. Tag support sets the
sampling RATE only -- every sampled cluster gets a human verdict, so the
Horvitz-Thompson estimator is design-unbiased regardless of any ArUco property.
That is the precise sense in which the primary estimator is ArUco-free.

    python group_labelling/eval_cluster_and_stratify.py \
        --out-dir /work/ReiterU/centroid_eval --key block01_cam10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import (  # noqa: E402
    EDGE_MARGIN,
    MEGA_CLUSTER,
    R_LINK,
    R_PAIR_ENVELOPE,
    R_PAIR_PRIMARY,
    in_interior,
    match_lsap,
    save_table,
    setup_logging,
    single_linkage_clusters,
    write_json,
)

# Published cam10 detection counts per frame; pi_native reproduces these.
NATIVE_PER_FRAME = {"A0": 23.04, "all6": 22.22}
FRONTIER_MULTIPLIERS = (0.7, 0.85, 1.15, 1.3)
DENSITY_RADIUS = 120.0
DENSITY_BINS = (1, 4)          # -> bins 0-1, 2-4, 5+
TAG_SUPPORT_RADIUS = 30.0
# Default sampling fractions per H-stratum; Stage 0h retunes from measured E|X|.
DEFAULT_FRACTIONS = {"H1": 0.35, "H2": 0.95, "H3": 0.50, "H4": 0.08, "H5": 0.20}


def thin_to_rate(df: pd.DataFrame, target_per_frame: float,
                 n_frames: int) -> pd.DataFrame:
    """Keep the highest-scoring detections until the camera-run hits the target
    rate. The monotone knob is the within-model score quantile (section 6): a
    shared nominal threshold is meaningless across a converted legacy model and
    a native sleap-nn one."""
    keep = int(round(target_per_frame * n_frames))
    if keep >= len(df):
        return df.copy()
    return df.nlargest(keep, "score", keep="first").copy()


def density_bin(pts: np.ndarray, radius: float = DENSITY_RADIUS) -> np.ndarray:
    """Neighbour count within `radius`, computed on the UNION of both models'
    detections. Model-derived but identical for both, so it labels the stratum
    without biasing the paired difference.

    The spec prefers background-subtraction blob density; that needs the
    per-camera temporal median and is not built here. This is the documented
    fallback. ArUco reference count is deliberately NOT used: tag recall
    collapses in piles, so it under-counts density exactly where density is
    highest.
    """
    if len(pts) == 0:
        return np.empty(0, np.int64)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    return ((d <= radius).sum(axis=1) - 1).astype(np.int64)


def load_reference(ref_h5: Path) -> dict[int, np.ndarray]:
    """source_frame -> (n,4) [x, y, tier, bit_err], for tag-support strata only."""
    if not ref_h5.exists():
        return {}
    with h5py.File(ref_h5, "r") as h:
        frame = h["det_frame"][:]
        centre = h["det_center"][:]
        tier = (h["det_tier"][:] if "det_tier" in h
                else np.zeros(len(frame), np.uint8))
        bit = h["det_bit_err"][:]
    out: dict[int, np.ndarray] = {}
    for f in np.unique(frame):
        m = frame == f
        out[int(f)] = np.column_stack([centre[m], tier[m], bit[m]]).astype(np.float64)
    return out


def inclusion_prob(key: str, frame: int, cluster: int,
                   fraction: float) -> tuple[float, bool]:
    """Deterministic Bernoulli sampling with a recorded probability.

    SHA1 of the cluster identity rather than an RNG, so the sample is
    reproducible, resumable and auditable: re-running never redraws, and every
    included cluster's identity can be re-derived from the manifest.
    """
    h = hashlib.sha1(f"{key}/{frame}/{cluster}".encode("utf-8")).hexdigest()
    u = int(h[:12], 16) / float(1 << 48)
    return fraction, bool(u < fraction)


def cluster_frame(pa: np.ndarray, sa: np.ndarray, pb: np.ndarray,
                  sb: np.ndarray, r_pair: float, r_link: float) -> list[dict]:
    """Cluster one frame's union and classify each component.

    `n_excl_a` / `n_excl_b` are the model-exclusive counts; a component with
    both zero cancels exactly and never reaches a human.
    """
    if len(pa) == 0 and len(pb) == 0:
        return []
    union = np.vstack([pa, pb]) if len(pa) and len(pb) else (
        pa if len(pa) else pb)
    labels = single_linkage_clusters(union, r_link)
    na = len(pa)
    dens = density_bin(union)

    out = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        ai = idx[idx < na]
        bi = idx[idx >= na] - na
        m = match_lsap(pa[ai], pb[bi], r_pair)
        n_excl_a, n_excl_b = len(m.unmatched_a), len(m.unmatched_b)
        out.append({
            "cluster": int(lab),
            "n_a": int(len(ai)), "n_b": int(len(bi)),
            "n_paired": int(len(m.pairs)),
            "n_excl_a": int(n_excl_a), "n_excl_b": int(n_excl_b),
            "cancels": bool(n_excl_a == 0 and n_excl_b == 0),
            "mega": bool(len(idx) > MEGA_CLUSTER),
            "cx": float(union[idx, 0].mean()), "cy": float(union[idx, 1].mean()),
            "x0": float(union[idx, 0].min()), "x1": float(union[idx, 0].max()),
            "y0": float(union[idx, 1].min()), "y1": float(union[idx, 1].max()),
            "density": int(dens[idx].max()),
            "pair_disp": float(m.cost.mean()) if len(m.cost) else float("nan"),
            "score_a": float(sa[ai].mean()) if len(ai) else float("nan"),
            "score_b": float(sb[bi].mean()) if len(bi) else float("nan"),
        })
    return out


def h_stratum(rec: dict, n_strict_tags: int, n_soft_tags: int) -> str:
    """H1..H5 per section 5.2. Tag support sets the sampling rate only."""
    if n_strict_tags > 0:
        return "H4"
    if rec["n_excl_a"] > 0 and rec["n_excl_b"] > 0:
        return "H3"
    if n_soft_tags > 0:
        return "H5"
    return "H1" if rec["n_excl_a"] > 0 else "H2"


def process_camera(key: str, out: Path, r_pair: float,
                   fractions: dict[str, float],
                   native: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    preds = out / "preds"
    dfs = {}
    for model in ("A0", "all6"):
        for ext in (".parquet", ".csv"):
            p = preds / f"{key}_{model}{ext}"
            if p.exists():
                dfs[model] = (pd.read_parquet(p) if ext == ".parquet"
                              else pd.read_csv(p))
                break
        else:
            raise FileNotFoundError(f"no predictions for {key}/{model}")

    manifest = json.loads((out / "anchors" / f"{key}.json").read_text("utf-8"))
    width, height = int(manifest["width"]), int(manifest["height"])
    clip_index = json.loads(
        (out / "clips" / f"{key}_index.json").read_text("utf-8"))
    to_source = {int(r["clip_frame"]): int(r["source_frame"]) for r in clip_index}
    n_frames = max(len(clip_index), 1)

    ref = load_reference(out / "aruco" / f"{key}_ref.h5")

    # Operating points. pi_match equalises the detection budget, so Cobs = 0 by
    # construction and one statistic characterises which model spends an equal
    # budget better.
    pi_match_rate = float(np.sqrt(native["A0"] * native["all6"]))
    points = {"pi_native": {"A0": native["A0"], "all6": native["all6"]},
              "pi_match": {"A0": pi_match_rate, "all6": pi_match_rate}}
    for mult in FRONTIER_MULTIPLIERS:
        points[f"pi_frontier_{mult:g}"] = {"A0": pi_match_rate * mult,
                                           "all6": pi_match_rate * mult}

    rows: list[dict] = []
    summary: dict = {}
    for pi_name, targets in points.items():
        thinned = {m: thin_to_rate(dfs[m], targets[m], n_frames)
                   for m in ("A0", "all6")}
        by_frame = {m: dict(tuple(thinned[m].groupby("clip_frame")))
                    for m in ("A0", "all6")}
        cobs = (len(thinned["A0"]) - len(thinned["all6"])) / n_frames
        n_disagree = 0
        for cf in range(n_frames):
            ga = by_frame["A0"].get(cf)
            gb = by_frame["all6"].get(cf)
            pa = (ga[["x", "y"]].to_numpy(np.float64) if ga is not None
                  else np.empty((0, 2)))
            pb = (gb[["x", "y"]].to_numpy(np.float64) if gb is not None
                  else np.empty((0, 2)))
            sa = (ga["score"].to_numpy(np.float64) if ga is not None
                  else np.empty(0))
            sb = (gb["score"].to_numpy(np.float64) if gb is not None
                  else np.empty(0))
            # Symmetric edge exclusion. Asymmetric would charge a correct edge
            # detection as a false positive.
            ka = in_interior(pa, width, height, EDGE_MARGIN)
            kb = in_interior(pb, width, height, EDGE_MARGIN)
            pa, sa, pb, sb = pa[ka], sa[ka], pb[kb], sb[kb]

            src = to_source.get(cf, -1)
            tags = ref.get(src, np.empty((0, 4)))
            for rec in cluster_frame(pa, sa, pb, sb, r_pair, R_LINK):
                if rec["cancels"]:
                    continue
                n_disagree += 1
                n_strict = n_soft = 0
                if len(tags):
                    half = max(rec["x1"] - rec["x0"], rec["y1"] - rec["y0"]) / 2
                    d = np.hypot(tags[:, 0] - rec["cx"], tags[:, 1] - rec["cy"])
                    near = d <= TAG_SUPPORT_RADIUS + half
                    n_strict = int((near & (tags[:, 3] == 0)).sum())
                    n_soft = int((near & (tags[:, 3] == 1)).sum())
                hs = h_stratum(rec, n_strict, n_soft)
                frac, included = inclusion_prob(key, cf, rec["cluster"],
                                                fractions[hs])
                rows.append({
                    "key": key, "operating_point": pi_name, "r_pair": r_pair,
                    "clip_frame": cf, "source_frame": src, **rec,
                    "n_strict_tags": n_strict, "n_soft_tags": n_soft,
                    "h_stratum": hs, "p_incl": frac, "sampled": included,
                    "density_bin": int(np.digitize(rec["density"], DENSITY_BINS)),
                })
        summary[pi_name] = {
            "n_A0": int(len(thinned["A0"])), "n_all6": int(len(thinned["all6"])),
            "per_frame_A0": len(thinned["A0"]) / n_frames,
            "per_frame_all6": len(thinned["all6"]) / n_frames,
            "cobs": cobs, "n_disagreement_clusters": n_disagree,
            "disagreement_per_frame": n_disagree / n_frames,
        }
        logging.info("[%s/%s r=%g] Cobs=%+.3f/frame  X=%d (%.2f/frame)",
                     key, pi_name, r_pair, cobs, n_disagree,
                     n_disagree / n_frames)

    df = pd.DataFrame(rows)
    summary.update({"key": key, "r_pair": r_pair, "n_frames": n_frames})
    if len(df):
        sampled = df[df["sampled"]]
        summary["n_sampled_total"] = int(len(sampled))
        summary["n_sampled_primary"] = int(
            sampled["operating_point"].isin(["pi_native", "pi_match"]).sum())
        summary["h_counts"] = (df[df["operating_point"] == "pi_match"]
                               ["h_stratum"].value_counts().to_dict())
        summary["mega_fraction"] = float(df["mega"].mean())
    return df, summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--key", type=str, default=None)
    p.add_argument("--r-pair", type=float, default=R_PAIR_PRIMARY)
    p.add_argument("--r-pair-envelope", action="store_true",
                   help=f"also run {R_PAIR_ENVELOPE} for the sensitivity envelope")
    p.add_argument("--fractions", type=str, default=None,
                   help='JSON, e.g. \'{"H1":0.4,"H2":1.0}\'')
    p.add_argument("--native", type=str, default=None,
                   help='JSON, e.g. \'{"A0":23.04,"all6":22.22}\'')
    args = p.parse_args()

    out: Path = args.out_dir.resolve()
    setup_logging(out / "logs", "eval_cluster_and_stratify")

    fractions = dict(DEFAULT_FRACTIONS)
    if args.fractions:
        fractions.update(json.loads(args.fractions))
    native = dict(NATIVE_PER_FRAME)
    if args.native:
        native.update(json.loads(args.native))
    logging.info("fractions=%s native=%s", fractions, native)

    strata = json.loads((out / "strata.json").read_text(encoding="utf-8"))
    keys = [args.key] if args.key else sorted(
        k for g in strata["strata"].values() for k in g)
    radii = list(R_PAIR_ENVELOPE) if args.r_pair_envelope else [args.r_pair]

    cdir = out / "clusters"
    cdir.mkdir(parents=True, exist_ok=True)
    all_summ = []
    for key in keys:
        for r_pair in radii:
            try:
                df, summ = process_camera(key, out, r_pair, fractions, native)
            except FileNotFoundError as exc:
                logging.error("[%s] %s", key, exc)
                continue
            except Exception:
                logging.exception("[%s] clustering failed", key)
                continue
            if len(df):
                save_table(df, cdir / f"{key}_rpair{r_pair:g}.parquet")
            all_summ.append(summ)
    if not all_summ:
        return 1
    write_json(all_summ, cdir / f"cluster_summary_{args.key or 'all'}.json")

    tot = sum(s.get("n_sampled_primary", 0) for s in all_summ)
    logging.info("clusters sampled for adjudication at pi_native+pi_match: %d", tot)
    if tot > 3000:
        logging.warning("adjudication load %d exceeds the ~2,500-verdict budget; "
                        "lower the H-fractions (spec Open Question 1)", tot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
