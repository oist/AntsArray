#!/usr/bin/env python3
"""Per-node localisation error of a centered-instance model, plus a convention check.

Scored from ground-truth-anchored predictions (see predict_gt_anchored.py), so every
labelled instance has exactly one prediction and the centroid model contributes
nothing to the number. That is what is wanted when comparing centered-instance arms:
an arm should be neither rewarded nor punished for a detection stage every arm
shares.

The headline number for the review experiment is the antenna-base error, since
antenna_L_1 and antenna_R_1 are the nodes a review actually corrects. The full
per-node table is reported alongside, so a gain there can be checked against a
regression elsewhere.

PCK is reported at several thresholds because the confidence-map target is about
10px wide at sigma 2.5 / output_stride 4: a 5px threshold sits well inside one
target and 20px spans two, so the pair brackets the scale that matters.

THE CONVENTION CHECK. A per-node distance says *how far* the model is from the
labels; it cannot say whether the gap is a systematic labelling convention or just
noise. The body-frame block answers that. Each instance defines its own frame from
the ``--body-axis`` nodes -- gaster_tip to occiput by default, origin at the head,
where the antennae attach -- and the review nodes are re-expressed as ``along`` and
``perp`` offsets, in pixels and in body lengths.

``perp`` is the convention indicator. A convention difference puts a node
consistently to one side of the body axis and keeps doing so at every posture, so it
shows up as a shift in median ``perp`` that survives normalisation. ``along`` does
not work for this: it moves with posture by more than the effect being measured
(6.6px between the sparse and dense regimes of one and the same labeller), so
reading it would chase biology instead of labelling drift. Normalising by body
length removes apparent-size differences between cameras, which would otherwise
register as a convention difference whenever two label sets cover different cameras.

Example::

    python score_node_error.py --gt arms_cam24/..._REVIEWED_test.pkg.slp \
        --pred eval/final_v2.pred.slp --label final_v2 --report eval/final_v2.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sleap_io as sio

from topdown_common import (
    DEFAULT_ANCHOR_NODE,
    DEFAULT_BODY_AXIS,
    DEFAULT_REVIEW_NODES,
    body_frame_table,
    compare_to_predictions,
    node_names,
    parse_floats,
    resolve_nodes,
    stats_of,
    write_report,
)

DEFAULT_PCK = "3,5,10,20"
CONVENTION_KEYS = ("along_px", "perp_px", "along_norm", "perp_norm")


def convention_block(comparison, node_idx: list[int], axis_idx: tuple[int, int], names: list[str]):
    """Body-frame offsets of the review nodes, for the labels and for the model.

    ``delta_median`` is labels minus predictions: a convention difference shows as a
    ``perp`` delta that is large next to the same node's ``sd`` and that barely moves
    when read in body lengths rather than in pixels.
    """
    labels = body_frame_table(comparison.gt, node_idx, axis_idx, names)
    model = body_frame_table(comparison.pred, node_idx, axis_idx, names)

    delta = {}
    for name, row in labels["per_node"].items():
        other = model["per_node"][name]
        delta[name] = {
            key: round(row[key]["median"] - other[key]["median"], 4)
            for key in CONVENTION_KEYS
            if row.get(key, {}).get("n") and other.get(key, {}).get("n")
        }
    return {
        "body_axis": [names[axis_idx[0]], names[axis_idx[1]]],
        "labels": labels,
        "predictions": model,
        "delta_median": delta,
        "note": (
            "perp is the convention indicator; along varies with posture and would "
            "chase biology instead of labelling drift."
        ),
    }


def print_convention(block: dict) -> None:
    axis = " -> ".join(name.strip() for name in block["body_axis"])
    print(f"\n=== body-frame convention ===  axis {axis}, origin at the anterior node")
    header = (
        f"{'node':<16}{'source':<8}{'perp px':>10}{'perp/len':>10}"
        f"{'along px':>10}{'along/len':>11}"
    )
    print(header)
    print("-" * len(header))
    for name in block["labels"]["per_node"]:
        for source, side in (("labels", block["labels"]), ("model", block["predictions"])):
            row = side["per_node"][name]
            if not row.get("perp_px", {}).get("n"):
                continue
            print(
                f"{name.strip():<16}{source:<8}{row['perp_px']['median']:>10.2f}"
                f"{row['perp_norm']['median']:>10.3f}{row['along_px']['median']:>10.2f}"
                f"{row['along_norm']['median']:>11.3f}"
            )
        gap = block["delta_median"].get(name) or {}
        if gap:
            print(
                f"{'':<16}{'delta':<8}{gap.get('perp_px', 0.0):>10.2f}"
                f"{gap.get('perp_norm', 0.0):>10.3f}{gap.get('along_px', 0.0):>10.2f}"
                f"{gap.get('along_norm', 0.0):>11.3f}"
            )
    print("delta = labels minus model. A convention gap shows in perp and survives /len.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gt", type=Path, required=True, help="Labelled test package.")
    parser.add_argument("--pred", type=Path, required=True, help="GT-anchored predictions on it.")
    parser.add_argument("--label", default=None, help="Name for this arm in the report.")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--anchor-node", default=DEFAULT_ANCHOR_NODE)
    parser.add_argument("--review-nodes", default=DEFAULT_REVIEW_NODES)
    parser.add_argument("--match-max-dist", type=float, default=100.0)
    parser.add_argument("--pck-thresholds", default=DEFAULT_PCK)
    parser.add_argument(
        "--body-axis",
        default=DEFAULT_BODY_AXIS,
        help="posterior,anterior node pair defining each instance's own frame. The "
        "origin is the anterior node, where the antennae attach.",
    )
    parser.add_argument(
        "--convention-nodes",
        default=None,
        help="Nodes to express in the body frame. Defaults to --review-nodes.",
    )
    parser.add_argument(
        "--no-convention", action="store_true", help="Skip the body-frame block entirely."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = parse_floats(args.pck_thresholds)

    gt = sio.load_slp(str(args.gt))
    pred = sio.load_slp(str(args.pred))
    names = node_names(gt)

    anchor_idx = resolve_nodes(gt, [args.anchor_node], "gt")[0]
    comparison = compare_to_predictions(gt, pred, anchor_idx, args.match_max_dist)
    if not len(comparison.error):
        raise SystemExit(f"[ERR] no comparable instances in {args.gt}")
    if comparison.n_unmatched:
        print(f"[!!] {comparison.n_unmatched} ground-truth instances had no prediction")

    errors = comparison.error
    review_idx = resolve_nodes(gt, args.review_nodes.split(","), "gt")
    other_idx = [i for i in range(len(names)) if i not in review_idx]
    headline = {
        "label": args.label or args.gt.stem,
        "gt": str(args.gt),
        "pred": str(args.pred),
        "n_instances": int(len(errors)),
        "n_unmatched": comparison.n_unmatched,
        "antenna_base": stats_of(errors[:, review_idx].ravel(), thresholds),
        "all_other_nodes": stats_of(errors[:, other_idx].ravel(), thresholds),
        "per_node": {name: stats_of(errors[:, idx], thresholds) for idx, name in enumerate(names)},
    }

    print(f"\n=== {headline['label']} ===  {headline['n_instances']} instances")
    header = f"{'node':<14}{'n':>7}{'median':>9}{'mean':>9}{'p90':>9}" + "".join(
        f"{'pck@' + f'{t:g}':>10}" for t in thresholds
    )
    print(header)
    print("-" * len(header))
    for name, stats in headline["per_node"].items():
        if not stats.get("n"):
            continue
        print(
            f"{name.strip():<14}{stats['n']:>7}{stats['median']:>9.2f}{stats['mean']:>9.2f}"
            f"{stats['p90']:>9.2f}" + "".join(f"{stats[f'pck@{t:g}']:>10.3f}" for t in thresholds)
        )
    base = headline["antenna_base"]
    print(
        f"\nANTENNA BASE   median {base['median']:.2f}px  mean {base['mean']:.2f}px  "
        f"p90 {base['p90']:.2f}px  "
        + "  ".join(f"pck@{t:g} {base[f'pck@{t:g}']:.3f}" for t in thresholds)
    )

    if not args.no_convention:
        axis = resolve_nodes(gt, args.body_axis.split(","), "gt")
        if len(axis) != 2:
            raise SystemExit(f"[ERR] --body-axis wants exactly two nodes, got {args.body_axis!r}")
        wanted = (args.convention_nodes or args.review_nodes).split(",")
        block = convention_block(
            comparison, resolve_nodes(gt, wanted, "gt"), (axis[0], axis[1]), names
        )
        headline["convention"] = block
        print_convention(block)

    write_report(args.report, headline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
