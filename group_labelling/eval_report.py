#!/usr/bin/env python3
"""Stage F: the report, with the honesty rules enforced in code, not by habit.

Three rules raise rather than footnote, because a footnote is not a control:

  R1 NO POOLING ACROSS STRATA. S1 (same camera, different unseen footage),
     S2 (six holdout cameras, n=6) and S3 (the decision stratum) are never
     aggregated. There is no "overall" column. assert_no_pooling() raises.

  R2 NO UNMATCHED-CENTROID COUNT IN ANY RATIO. An unmatched centroid may be the
     queen, an untagged worker, a blur-failed tag read, or a genuine false
     positive, and nothing in this dataset distinguishes them. Any ratio built
     from one is uninterpretable in absolute terms; only the paired DIFFERENCE
     between the two models is. assert_no_unmatched_ratio() raises.

  R3 NO ABSOLUTE PRECISION OUTSIDE S4, which is the only stratum with labelled
     ants.

Contract: group_labelling/EVAL_HARNESS_DESIGN.md sections 7, 8, 10.

    python group_labelling/eval_report.py --out-dir /work/ReiterU/centroid_eval
    python group_labelling/eval_report.py --out-dir /tmp/x --self-test
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

from eval_common import setup_logging, write_json  # noqa: E402

BANNERS = {
    "S1": ("**SAME CAMERA, DIFFERENT UNSEEN FOOTAGE.** Camera-identity overlap "
           "only, not memorisation -- the ch01-06 training footage (20260515 "
           "13-09-56) has been deleted and no block01/02/03 frame was in "
           "training. Still optimistic. **No gate reads S1.**"),
    "S2": ("**UNDERPOWERED, n = 6.** Reported as six individual camera values. "
           "No stratum-level confidence interval is computed, because six "
           "cameras do not support one."),
    "S3": ("**THE DECISION STRATUM.** `n_blocks = 2, n_colonies = 1, "
           "n_recording_days = 1 (20260515)`. \"Never seen\" means "
           "never-seen-cameras-within-one-recording-day. block02 and block03 "
           "are reported side by side; there is no pooled block-level CI, "
           "because n = 2 admits none."),
    "S4": "**CALIBRATION.** Never merged into S1-S3.",
}

FORBIDDEN_RATIO_TERMS = ("unmatched_sleap", "unmatched_centroid", "n_excl_a",
                         "n_excl_b", "precision")

MANDATORY_SENTENCES = [
    "centroid_all6 warm-starts from A0's own best.ckpt (both "
    "pretrained_backbone_weights and pretrained_head_weights point at it) and "
    "early-stopped at epoch 40 with a checkpoint from roughly epoch 1, under "
    "settings later found broken (gaussian_noise_p 1.0, lr 3e-5, rotation "
    "+/-15 deg). It must not be described as \"improved by six datasets\".",

    "Ants missed by BOTH models are invisible to this harness by construction. "
    "They cancel identically in the paired difference and no instrument in this "
    "dataset sees them, so no absolute recall claim may be made from these "
    "numbers.",

    "The ArUco TagRecall arm is CORROBORATIVE. Its bias terms (filler, "
    "easy-ant attenuation of unknown magnitude, ghost-steal, edge asymmetry) "
    "sum to the same order as the effect they would measure. No gate reads it.",

    "Colony and day generalisation is unbounded and unestimable here: all three "
    "blocks are one colony, one tag set, one lighting rig, one recording day.",

    "This harness stops at stage 1. It does not measure whether A0's extra "
    "detections produce ghost tracks, nor whether centroid_all6's extra misses "
    "are recovered by track interpolation.",
]


class HonestyError(RuntimeError):
    """Raised when the report is asked to emit something uninterpretable."""


def assert_no_pooling(df: pd.DataFrame, context: str) -> None:
    """R1. Any table reaching output carries a stratum and must not mix."""
    if "stratum" not in df.columns:
        raise HonestyError(
            f"{context}: table has no 'stratum' column, so it cannot be shown "
            f"without risking a cross-stratum aggregate. Add one, or split it.")
    present = sorted(set(df["stratum"].dropna()))
    if len(present) > 1:
        raise HonestyError(
            f"{context}: refusing to emit a table spanning strata {present}. "
            f"S1 is optimistic, S2 has n=6, S3 is the decision stratum -- "
            f"pooling them produces a number with no defensible meaning.")


def assert_no_unmatched_ratio(name: str, numerator: str,
                              denominator: str) -> None:
    """R2. Reject any ratio whose numerator or denominator is an
    unmatched-centroid count."""
    for term in FORBIDDEN_RATIO_TERMS:
        if term in numerator.lower() or term in denominator.lower():
            raise HonestyError(
                f"{name}: refusing to compute {numerator}/{denominator}. An "
                f"unmatched centroid may be the queen, an untagged worker, a "
                f"blur-failed tag read or a genuine false positive, and nothing "
                f"here distinguishes them. Only the paired DIFFERENCE between "
                f"the two models is interpretable.")


def fmt(x, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and not np.isfinite(x):
        return "n/a"
    return f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def section_cobs(cobs: pd.DataFrame, stratum: str) -> str:
    g = cobs[cobs["stratum"] == stratum]
    if not len(g):
        return f"_No {stratum} cameras._\n"
    assert_no_pooling(g, f"Cobs/{stratum}")
    w = g["n_frames"].to_numpy(float)
    lines = [
        "| camera | frames | dets/frame A0 | dets/frame all6 | Cobs | X/frame |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in g.sort_values("key").iterrows():
        lines.append(f"| {r['key']} | {int(r['n_frames'])} | "
                     f"{r['per_frame_A0']:.2f} | {r['per_frame_all6']:.2f} | "
                     f"{r['cobs']:+.3f} | {r['disagreement_per_frame']:.3f} |")
    if stratum != "S2" and len(g) > 1:
        lines.append(
            f"| **{stratum} (ratio of sums)** | {int(w.sum())} | "
            f"{np.average(g['per_frame_A0'], weights=w):.2f} | "
            f"{np.average(g['per_frame_all6'], weights=w):.2f} | "
            f"{np.average(g['cobs'], weights=w):+.3f} | "
            f"{np.average(g['disagreement_per_frame'], weights=w):.3f} |")
    else:
        lines += ["", "_Six individual camera values; no stratum row, by design._"]
    return "\n".join(lines) + "\n"


def build_report(out: Path) -> str:
    est = out / "estimates"
    strata = json.loads((out / "strata.json").read_text(encoding="utf-8"))
    primary = {}
    if (est / "primary.json").exists():
        primary = json.loads((est / "primary.json").read_text(encoding="utf-8"))

    cobs = None
    for ext in (".parquet", ".csv"):
        p = est / f"cobs{ext}"
        if p.exists():
            cobs = pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
            break

    L: list[str] = []
    A = L.append
    A("# Centroid model evaluation — A0 vs centroid_all6")
    A("")
    A("Harness: **W1.1 PDA-C / TagProbe**. Contract: `EVAL_HARNESS_DESIGN.md`.")
    A("")
    A("## Status")
    A("")
    state = primary.get("state")
    if state == "AWAITING_ADJUDICATION":
        A(f"> **AWAITING ADJUDICATION.** {primary['n_sampled_clusters']} of "
          f"{primary['n_disagreement_clusters']} disagreement clusters are "
          f"sampled and waiting on human verdicts. The primary statistic "
          f"**Delta is not estimated** and **no ship/keep decision is made**. "
          f"Everything below is reference-free and human-free.")
    elif "decision_G_C" in primary:
        d = primary["decision_G_C"]
        A(f"> **Gate G-C: {d['branch']}**")
        A(">")
        A(f"> {d['why']}")
        A(">")
        A(f"> Delta = {fmt(d['delta'])} detections/frame, 95% CI "
          f"[{fmt(d['ci_lo'])}, {fmt(d['ci_hi'])}], SE {d['se_used']:.4f} "
          f"(max of bootstrap {d['se_bootstrap']:.4f} and replicate "
          f"{fmt(d.get('se_replicate'))}), read at the **unfavourable** bracket "
          f"endpoint on **S3 only**.")
        A(">")
        A(f"> {d['note']}")
    else:
        A("> No estimates yet.")
    A("")

    A("## Strata")
    A("")
    for s in ("S1", "S2", "S3"):
        n = len(strata["strata"].get(s, []))
        A(f"- **{s}** ({n} camera-runs) — {BANNERS[s]}")
    A("")
    A("There is deliberately no \"overall\" row anywhere in this report; "
      "`eval_report.assert_no_pooling` raises if one is attempted.")
    A("")

    if cobs is not None and len(cobs):
        A("## Cobs — the reference-free observable")
        A("")
        A("`Cobs = (|D_A0| - |D_all6|) / frames = dFN + dFP`, exactly observed, "
          "with no reference and no human involved. At `pi_match` it is 0 by "
          "construction. `X/frame` is the disagreement-cluster rate — the only "
          "population that carries signal and the only one a human ever sees.")
        for s in ("S1", "S2", "S3"):
            A("")
            A(f"### {s}")
            A("")
            A(BANNERS[s])
            A("")
            A(section_cobs(cobs, s))

    if "results" in primary:
        A("## Primary — Delta at pi_match (the only gated capability statistic)")
        A("")
        A("`Delta = dFP(pi_match) = -dFN(pi_match)`, Horvitz-Thompson over "
          "adjudicated clusters. Design-unbiased regardless of every ArUco "
          "property: every term comes from a human verdict, and cancelling "
          "clusters contribute exactly zero. `Delta > 0` means that at an equal "
          "detection budget A0 wastes more of it on non-ants.")
        A("")
        A("| stratum | bracket | cameras | Delta | 95% CI | dFN | can't-tell |")
        A("|---|---|---|---|---|---|---|")
        for _k, r in sorted(primary["results"].items()):
            A(f"| {r['stratum']} | {r['bracket']} | {r['n_cameras']} | "
              f"{fmt(r['delta'])} | [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] | "
              f"{fmt(r['d_fn'])} | {100 * r['cant_tell_rate']:.1f}% |")
        A("")
        A("Gates read the **unfavourable** endpoint. A can't-tell rate above "
          "10% in any stratum makes that stratum interval-only and its gate "
          "outcome `UNRESOLVED`.")
        A("")

    A("## What these numbers do not say")
    A("")
    for s in MANDATORY_SENTENCES:
        A(f"- {s}")
    A("")
    A("Full limitation list: `EVAL_HARNESS_DESIGN.md` section 10.")
    A("")
    return "\n".join(L)


def self_test() -> int:
    ok = True
    try:
        assert_no_pooling(pd.DataFrame({"stratum": ["S1", "S3"]}), "test")
        ok = False
        logging.error("R1 did NOT raise on a cross-stratum table")
    except HonestyError:
        logging.info("R1 pooling guard raises as intended")
    try:
        assert_no_pooling(pd.DataFrame({"key": ["a"]}), "test")
        ok = False
        logging.error("R1 did NOT raise on a table with no stratum column")
    except HonestyError:
        logging.info("R1 raises when the stratum column is missing")
    try:
        assert_no_unmatched_ratio("test", "n_unmatched_sleap", "n_aruco")
        ok = False
        logging.error("R2 did NOT raise on an unmatched-centroid ratio")
    except HonestyError:
        logging.info("R2 unmatched-centroid guard raises as intended")
    try:
        assert_no_pooling(pd.DataFrame({"stratum": ["S3", "S3"]}), "test")
        logging.info("R1 permits a single-stratum table")
    except HonestyError:
        ok = False
        logging.error("R1 wrongly raised on a single-stratum table")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--self-test", action="store_true",
                   help="verify the honesty guards actually raise")
    args = p.parse_args()

    out: Path = args.out_dir.resolve()
    setup_logging(out / "logs", "eval_report")

    if args.self_test:
        return self_test()

    report = build_report(out)
    rdir = out / "report"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "report.md").write_text(report, encoding="utf-8")

    est = out / "estimates" / "primary.json"
    decision: dict = {}
    if est.exists():
        primary = json.loads(est.read_text(encoding="utf-8"))
        decision = primary.get(
            "decision_G_C", {"branch": primary.get("state", "NO_ESTIMATE")})
    write_json(decision, rdir / "decision.json")
    logging.info("wrote %s (%d chars); decision=%s", rdir / "report.md",
                 len(report), decision.get("branch"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
