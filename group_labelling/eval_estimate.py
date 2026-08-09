#!/usr/bin/env python3
"""Stage E: Horvitz-Thompson estimators, variances, brackets, decision.

Two modes, and the difference matters:

  WITHOUT verdicts (adjudication/verdicts.jsonl absent) it reports only what is
  computable with no human input and no reference: Cobs, the disagreement
  inventory and the H-stratum breakdown. The primary statistic Delta is NOT
  estimated and no decision is emitted. This is the expected state before the
  reviewers have worked.

  WITH verdicts it estimates dFP and dFN by Horvitz-Thompson over the
  adjudicated clusters, computes the variances, brackets the can't-tells, and
  evaluates decision gate G-C.

Delta is design-unbiased regardless of every ArUco property, because every term
in the HT sum comes from a human verdict and cancelling clusters contribute
exactly zero (EVAL_HARNESS_DESIGN.md section 2). Nothing here may put an
unmatched-centroid count into a numerator or denominator; eval_report.py
enforces that at output time.

Verdict schema, one JSON object per line:
  {"key":"block01_cam10","operating_point":"pi_match","clip_frame":0,
   "cluster":3,"k":2,"tp_A0":2,"tp_all6":1,"cant_tell":false,"flags":["queen"]}

    python group_labelling/eval_estimate.py --out-dir /work/ReiterU/centroid_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import (  # noqa: E402
    R_PAIR_PRIMARY,
    save_table,
    setup_logging,
    write_json,
)

DELTA_CAP = 0.05        # equivalence margin, detections/frame (section 8)
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260805


def load_clusters(out: Path, r_pair: float) -> pd.DataFrame:
    parts = []
    for p in sorted((out / "clusters").glob(f"*_rpair{r_pair:g}.parquet")):
        parts.append(pd.read_parquet(p))
    for p in sorted((out / "clusters").glob(f"*_rpair{r_pair:g}.csv")):
        parts.append(pd.read_csv(p))
    if not parts:
        raise FileNotFoundError(f"no cluster tables for r_pair={r_pair:g}")
    return pd.concat(parts, ignore_index=True)


def load_verdicts(out: Path) -> pd.DataFrame | None:
    path = out / "adjudication" / "verdicts.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()
            if line.strip()]
    return pd.DataFrame(rows) if rows else None


def attach_strata(df: pd.DataFrame, strata: dict) -> pd.DataFrame:
    lut = {k: s for s, keys in strata["strata"].items() for k in keys}
    df = df.copy()
    df["stratum"] = df["key"].map(lut)
    return df


def cobs_by_camera(summaries: list[dict], point: str) -> pd.DataFrame:
    rows = []
    for s in summaries:
        if point not in s:
            continue
        rows.append({"key": s["key"], "r_pair": s["r_pair"],
                     "n_frames": s["n_frames"], **s[point]})
    return pd.DataFrame(rows)


def ht_delta(merged: pd.DataFrame, n_frames_by_key: dict[str, int],
             bracket: str = "point") -> pd.DataFrame:
    """Per-camera HT estimates of dFP and dFN.

    dFP(c) = (1/n_frames) * sum_k (1/p_k) * [FP_A0(k) - FP_all6(k)]
    dFN(c) = (1/n_frames) * sum_k (1/p_k) * [TP_A0(k) - TP_all6(k)]

    with FP_m(k) = |D_m in k| - TP_m(k) from the human verdict. Ratio of sums,
    never mean of per-frame ratios.

    `bracket` handles can't-tell clusters: "point" drops them, and
    "favourable"/"unfavourable" push them to the endpoint that most/least
    favours centroid_all6. The gate reads the unfavourable endpoint.
    """
    rows = []
    for key, g in merged.groupby("key"):
        n_frames = max(n_frames_by_key.get(key, 1), 1)
        ct = g["cant_tell"].fillna(False).astype(bool) if "cant_tell" in g \
            else pd.Series(False, index=g.index)
        usable = g[~ct]

        tp_a0 = usable["tp_A0"].to_numpy(float)
        tp_b = usable["tp_all6"].to_numpy(float)
        fp_a0 = usable["n_a"].to_numpy(float) - tp_a0
        fp_b = usable["n_b"].to_numpy(float) - tp_b
        w = 1.0 / usable["p_incl"].to_numpy(float)

        d_fp = float((w * (fp_a0 - fp_b)).sum()) / n_frames
        d_fn = float((w * (tp_a0 - tp_b)).sum()) / n_frames

        if bracket != "point" and bool(ct.any()):
            amb = g[ct]
            wa = 1.0 / amb["p_incl"].to_numpy(float)
            excl_a = amb["n_excl_a"].to_numpy(float)
            excl_b = amb["n_excl_b"].to_numpy(float)
            if bracket == "unfavourable":
                # Worst case for all6: every ambiguous A0-exclusive is a real
                # ant A0 found, every all6-exclusive is a false positive.
                d_fn += float((wa * excl_a).sum()) / n_frames
                d_fp -= float((wa * excl_b).sum()) / n_frames
            else:
                d_fp += float((wa * excl_a).sum()) / n_frames
                d_fn -= float((wa * excl_b).sum()) / n_frames

        rows.append({"key": key, "stratum": g["stratum"].iloc[0],
                     "n_frames": n_frames, "n_adjudicated": int(len(g)),
                     "n_cant_tell": int(ct.sum()),
                     "cant_tell_rate": float(ct.mean()) if len(g) else 0.0,
                     "d_fp": d_fp, "d_fn": d_fn, "delta": d_fp,
                     "bracket": bracket})
    return pd.DataFrame(rows)


def camera_cluster_bootstrap(per_cam: pd.DataFrame, col: str = "delta",
                             n_rep: int = N_BOOTSTRAP
                             ) -> tuple[float, float, float]:
    """Resample camera-runs with replacement within stratum. Frames inside a
    camera are autocorrelated, so the camera-run is the cluster unit."""
    vals = per_cam[col].to_numpy(float)
    weights = per_cam["n_frames"].to_numpy(float)
    if len(vals) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(vals), size=(n_rep, len(vals)))
    boots = (vals[idx] * weights[idx]).sum(axis=1) / weights[idx].sum(axis=1)
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)),
            float(boots.std(ddof=1)))


def replicate_variance(merged: pd.DataFrame, n_frames_by_key: dict[str, int],
                       col: str = "delta") -> dict:
    """Interpenetrating-replicate variance.

    The only estimator including systematic-sampling PHASE, i.e. the only answer
    to "would a one-frame stride shift change the verdict?". Needs a `replicate`
    column carried through from the anchor manifest.
    """
    if "replicate" not in merged.columns:
        return {"available": False,
                "reason": "no replicate column on the adjudicated clusters"}
    ests = []
    for _r, g in merged.groupby("replicate"):
        pc = ht_delta(g, n_frames_by_key)
        if len(pc):
            ests.append(float(np.average(pc[col], weights=pc["n_frames"])))
    if len(ests) < 2:
        return {"available": False, "reason": f"only {len(ests)} replicate(s)"}
    arr = np.asarray(ests, float)
    return {"available": True, "n_replicates": len(ests), "estimates": ests,
            "mean": float(arr.mean()),
            "se": float(np.sqrt(arr.var(ddof=1) / len(arr)))}


def decide(delta_lo: float, delta_hi: float, cap: float = DELTA_CAP) -> dict:
    """Gate G-C only. G0/G-Q/G-R/G-N live in eval_report.py, which owns the
    prerequisites and the queen and retrain arms and can override this."""
    if delta_lo > cap:
        return {"branch": "SHIP centroid_all6",
                "why": f"95% CI [{delta_lo:+.4f}, {delta_hi:+.4f}] lies entirely "
                       f"above +{cap}: at an equal detection budget A0 wastes "
                       f"more of it on non-ants."}
    if delta_hi < -cap:
        return {"branch": "KEEP A0",
                "why": f"95% CI [{delta_lo:+.4f}, {delta_hi:+.4f}] lies entirely "
                       f"below -{cap}."}
    if delta_lo >= -cap and delta_hi <= cap:
        return {"branch": "EQUIVALENT CAPABILITY -> KEEP A0 AND RETUNE",
                "why": f"95% CI [{delta_lo:+.4f}, {delta_hi:+.4f}] lies inside "
                       f"+/-{cap}. A config change on the incumbent strictly "
                       f"dominates shipping a checkpoint trained with settings "
                       f"later found broken."}
    return {"branch": "INDETERMINATE",
            "why": f"95% CI [{delta_lo:+.4f}, {delta_hi:+.4f}] straddles the "
                   f"+/-{cap} margin. Ship nothing; report the frontier."}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--r-pair", type=float, default=R_PAIR_PRIMARY)
    p.add_argument("--operating-point", type=str, default="pi_match")
    p.add_argument("--delta-cap", type=float, default=DELTA_CAP)
    args = p.parse_args()

    out: Path = args.out_dir.resolve()
    setup_logging(out / "logs", "eval_estimate")
    est = out / "estimates"
    est.mkdir(parents=True, exist_ok=True)

    strata = json.loads((out / "strata.json").read_text(encoding="utf-8"))
    clusters = attach_strata(load_clusters(out, args.r_pair), strata)
    summaries: list[dict] = []
    for sp in sorted((out / "clusters").glob("cluster_summary_*.json")):
        summaries.extend(json.loads(sp.read_text(encoding="utf-8")))
    n_frames_by_key = {s["key"]: s["n_frames"] for s in summaries}

    # ---- always computable: reference-free and human-free ------------------
    cobs = cobs_by_camera(summaries, args.operating_point)
    if len(cobs):
        cobs["stratum"] = cobs["key"].map(
            {k: s for s, ks in strata["strata"].items() for k in ks})
        save_table(cobs, est / "cobs.parquet")
        for s, g in cobs.groupby("stratum"):
            logging.info("[%s] Cobs=%+.4f/frame over %d camera-runs, "
                         "disagreement=%.3f/frame", s,
                         float(np.average(g["cobs"], weights=g["n_frames"])),
                         len(g),
                         float(np.average(g["disagreement_per_frame"],
                                          weights=g["n_frames"])))

    at_point = clusters[clusters["operating_point"] == args.operating_point]
    inventory = (at_point.groupby(["stratum", "h_stratum"])
                 .agg(n_clusters=("cluster", "size"),
                      n_sampled=("sampled", "sum"),
                      mean_p_incl=("p_incl", "mean"),
                      mega_rate=("mega", "mean"))
                 .reset_index())
    save_table(inventory, est / "disagreement_inventory.parquet")
    logging.info("disagreement inventory:\n%s", inventory.to_string(index=False))

    verdicts = load_verdicts(out)
    if verdicts is None:
        logging.warning(
            "no adjudication/verdicts.jsonl -- the PRIMARY statistic Delta is "
            "NOT estimated and no decision is emitted. %d clusters are sampled "
            "and awaiting human verdicts.", int(at_point["sampled"].sum()))
        write_json({
            "state": "AWAITING_ADJUDICATION",
            "operating_point": args.operating_point,
            "r_pair": args.r_pair,
            "n_sampled_clusters": int(at_point["sampled"].sum()),
            "n_disagreement_clusters": int(len(at_point)),
            "note": ("Delta requires human verdicts. Cobs and the disagreement "
                     "inventory are reference-free and are reported above."),
        }, est / "primary.json")
        return 0

    # ---- with verdicts: the primary estimator ------------------------------
    key_cols = ["key", "operating_point", "clip_frame", "cluster"]
    merged = at_point.merge(verdicts, on=key_cols, how="inner",
                            suffixes=("", "_v"))
    if len(merged) == 0:
        logging.error("verdicts present but none join to the sampled clusters "
                      "on %s", key_cols)
        return 1
    logging.info("%d verdicts joined to sampled clusters", len(merged))

    results: dict[str, dict] = {}
    for bracket in ("point", "favourable", "unfavourable"):
        per_cam = ht_delta(merged, n_frames_by_key, bracket)
        save_table(per_cam, est / f"per_camera_{bracket}.parquet")
        for s, g in per_cam.groupby("stratum"):
            lo, hi, se = camera_cluster_bootstrap(g)
            pooled = float(np.average(g["delta"], weights=g["n_frames"]))
            results[f"{s}/{bracket}"] = {
                "stratum": s, "bracket": bracket, "n_cameras": int(len(g)),
                "delta": pooled, "ci_lo": lo, "ci_hi": hi, "se_bootstrap": se,
                "d_fn": float(np.average(g["d_fn"], weights=g["n_frames"])),
                "cant_tell_rate": float(g["cant_tell_rate"].mean()),
            }
            logging.info("[%s/%s] Delta=%+.4f CI[%+.4f,%+.4f] n_cam=%d "
                         "cant_tell=%.1f%%", s, bracket, pooled, lo, hi,
                         len(g), 100 * g["cant_tell_rate"].mean())

    rep = replicate_variance(merged, n_frames_by_key)
    write_json(rep, est / "replicate_variance.json")

    # The decision reads S3 at the unfavourable endpoint with the LARGEST of the
    # available variances (section 8).
    s3 = results.get("S3/unfavourable")
    if s3 is None:
        logging.error("no S3 estimate -- cannot evaluate the decision rule")
        return 1
    se = s3["se_bootstrap"]
    if rep.get("available") and np.isfinite(rep.get("se", np.nan)):
        se = max(se, rep["se"])
    lo, hi = s3["delta"] - 1.96 * se, s3["delta"] + 1.96 * se
    decision = decide(lo, hi, args.delta_cap)
    decision.update({
        "note": ("Gate G-C only. G0 prerequisites, G-Q queen and G-R retrain "
                 "floor are evaluated in eval_report.py and can override this."),
        "stratum": "S3", "bracket": "unfavourable",
        "delta": s3["delta"], "ci_lo": lo, "ci_hi": hi,
        "se_used": se, "se_bootstrap": s3["se_bootstrap"],
        "se_replicate": rep.get("se"), "delta_cap": args.delta_cap,
    })
    write_json({"results": results, "replicate_variance": rep,
                "decision_G_C": decision}, est / "primary.json")
    logging.info("G-C: %s -- %s", decision["branch"], decision["why"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
