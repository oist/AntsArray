#!/usr/bin/env python3
"""A/B-test instance-selection strategies offline, by replaying the ch06 review.

Which instances are worth reviewing can normally only be answered by reviewing them,
retraining and measuring -- so one selection rule cannot be compared against another
without spending the human time twice. ch06 is the exception: it exists in both a
collaborator and a reviewed version, and 5,512 instances pair up between them. That
means a review can be *replayed*. Select N instances by some rule, swap in their
reviewed labels, retrain, and score against held-out reviewed cam24. No human is
involved, so any number of rules can be compared at equal budget.

Every arm trains on the same frames and the same number of instances. The only thing
that varies is which N instances carry corrected labels, which isolates label quality
from dataset size and composition. Two arms bound the result:

  floor_n0      pure collaborator labels, nothing corrected
  ceiling_all   every paired instance corrected

WHAT THIS HAS ALREADY SETTLED, AND WHY IT IS STILL HERE. Ranking by model-vs-label
disagreement does not work: replayed at equal budget it correlated with the
corrections actually made at r=0.03 (Pearson +0.025), indistinguishable from random,
and it must not be reinstated as the selector's default. The oracle arm -- which
ranks by the correction each instance actually received, and so is unavailable in
real use -- beat random by about 23%. That gap is the reason this file survives: it
is the only way to test a *new* selection rule against a real review without paying
for the review twice. Nothing here belongs to the routine loop.

**The scoring model must never have seen reviewed ch06.** reviewed260 was trained on
it, so its disagreement with the collaborator labels is precisely the set of
corrections it has already memorised; using it would let ``disagreement`` cheat and
would invalidate the whole comparison. The intended scorer is ``ci_A2_ch01to05``:
trained on different cameras and on unreviewed labels, so it cannot leak the
corrections. Its unreviewed antenna convention does not matter here, because ranking
looks only at aruco/occiput/petiole/gaster_tip, where collaborator and reviewed
labels already agree to a median of 0.00px.

Example::

    python simulate_review_ch06.py \
        --collab flat/..._chunk06_..._recorrected.pkg.slp \
        --reviewed flat/..._chunk06_..._recorrected_REVIEWED.slp \
        --pred pred_ch06/..._chunk06_....pred.slp \
        --manifest chunk_manifest.csv --holdout-camera cam24 \
        --test-slp arms_cam24/..._REVIEWED_test.pkg.slp \
        --strategies random,disagreement,lowconf,diverse,oracle --budgets 600 \
        --out-dir /work/.../selection_ab --arms-json /work/.../selection_ab/arms.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sleap_io as sio

from topdown_common import (
    DEFAULT_ANCHOR_NODE,
    DEFAULT_RANK_NODES,
    DEFAULT_REVIEW_NODES,
    antenna_indices,
    chunk_number,
    classify_points,
    disagreement,
    iter_frame_pairs,
    map_videos_to_cameras,
    match_point,
    pair_label_versions,
    parse_weighted_nodes,
    read_manifest,
    recorded_args,
    resolve_nodes,
    save_package,
    write_report,
)

STRATEGIES = ("random", "disagreement", "lowconf", "diverse", "oracle")


def correction_magnitude(
    collab_points: np.ndarray, reviewed_points: np.ndarray, review_idx: list[int]
) -> tuple[float, int]:
    """How much this instance actually changed: mean base shift, and un-hidden nodes.

    This is the oracle signal. It is computable only because the review already
    happened, so it exists to bound the experiment and is never available in real use.
    """
    shifts, unhidden = [], 0
    for node_idx in review_idx:
        before, after = collab_points[node_idx], reviewed_points[node_idx]
        if np.isnan(before).any() and not np.isnan(after).any():
            unhidden += 1
        elif not np.isnan(before).any() and not np.isnan(after).any():
            shifts.append(float(np.linalg.norm(before - after)))
    return (float(np.mean(shifts)) if shifts else 0.0), unhidden


def gather(
    collab: sio.Labels,
    pred: sio.Labels,
    pairs: dict[tuple[int, int, int], object],
    camera_of_video: dict[int, str],
    args: argparse.Namespace,
) -> list[dict]:
    rank_names, rank_weights = parse_weighted_nodes(args.rank_nodes)
    rank_idx = resolve_nodes(collab, rank_names, "collab")
    anchor_idx = resolve_nodes(collab, [args.anchor_node], "collab")[0]
    review_idx = resolve_nodes(collab, args.review_nodes.split(","), "collab")
    antenna_idx = antenna_indices(collab)

    records: list[dict] = []
    stats: Counter = Counter()

    for pair in iter_frame_pairs(collab, pred, anchor_idx, args.match_max_dist):
        camera = camera_of_video[pair.video_idx]
        height, width = pair.frame.video.shape[1:3]
        frame_idx = int(pair.frame.frame_idx)

        for position in range(len(pair.instances)):
            stats["total"] += 1
            points = pair.gt_points[position]
            if (
                classify_points(
                    points, antenna_idx, (height, width), args.edge_margin, args.max_hidden
                )
                is not None
            ):
                stats["excluded_truncated"] += 1
                continue
            partner = pairs.get((pair.video_idx, frame_idx, position))
            if partner is None:
                stats["unpaired_with_reviewed"] += 1
                continue

            matched = pair.match(position)
            if matched is None:
                stats["no_prediction"] += 1
                score, confidence = None, 0.0
            else:
                score, _, _ = disagreement(points, matched.points, rank_idx, rank_weights)
                confidence = float(np.mean(np.asarray(matched.instance.points["score"])[rank_idx]))
            shift, unhidden = correction_magnitude(points, np.asarray(partner.numpy()), review_idx)

            records.append(
                {
                    "camera": camera,
                    "video_idx": pair.video_idx,
                    "frame_idx": frame_idx,
                    "position": position,
                    "disagreement": 0.0 if score is None else round(score, 3),
                    "confidence": round(confidence, 4),
                    "oracle_shift": round(shift, 3),
                    "oracle_unhidden": unhidden,
                    "anchor_xy": [round(float(v), 2) for v in match_point(points, anchor_idx)],
                }
            )

    print(f"[..] {dict(stats)}  paired+eligible={len(records)}")
    return records


def rank(records: list[dict], strategy: str, args: argparse.Namespace) -> list[dict]:
    """Order the eligible pool by one selection rule, best first."""
    if strategy == "random":
        rng = np.random.default_rng(args.seed)
        return [records[i] for i in rng.permutation(len(records))]
    if strategy == "disagreement":
        return sorted(records, key=lambda r: -r["disagreement"])
    if strategy == "lowconf":
        # Prefer instances the model gets wrong *and* is unsure about: a confident
        # error is more often a label defect than a gap in what the model has learnt.
        return sorted(
            records,
            key=lambda r: -(r["disagreement"] * (1.0 - r["confidence"]) ** args.conf_alpha),
        )
    if strategy == "oracle":
        return sorted(
            records, key=lambda r: -(r["oracle_shift"] + args.unhide_weight * r["oracle_unhidden"])
        )
    if strategy == "diverse":
        # Same ranking as `disagreement`, but each pick suppresses its temporal
        # neighbours in the same camera: adjacent frames of one camera are nearly
        # identical, so a second pick from the same moment teaches almost nothing.
        chosen, deferred = [], []
        used: dict[str, list[int]] = defaultdict(list)
        for record in sorted(records, key=lambda r: -r["disagreement"]):
            if any(
                abs(record["frame_idx"] - seen) < args.diversity_gap
                for seen in used[record["camera"]]
            ):
                deferred.append(record)
            else:
                used[record["camera"]].append(record["frame_idx"])
                chosen.append(record)
        return chosen + deferred
    raise SystemExit(f"[ERR] unknown strategy {strategy!r}; choose from {STRATEGIES}")


def build_arm(
    collab: sio.Labels,
    pairs: dict[tuple[int, int, int], object],
    selected: list[dict],
    holdout: set[str],
    camera_of_video: dict[int, str],
    out: Path,
) -> dict:
    """The full training split, with exactly the selected instances corrected."""
    corrected = {(r["video_idx"], r["frame_idx"], r["position"]) for r in selected}
    skeleton = collab.skeletons[0]
    frames, videos, n_corrected = [], [], 0

    for frame in collab:
        video_idx = collab.videos.index(frame.video)
        if camera_of_video[video_idx] in holdout:
            continue
        instances = []
        for position, instance in enumerate(frame.user_instances):
            key = (video_idx, int(frame.frame_idx), position)
            if key in corrected and key in pairs:
                instances.append(sio.Instance(points=pairs[key].points.copy(), skeleton=skeleton))
                n_corrected += 1
            else:
                instances.append(instance)
        if not instances:
            continue
        if frame.video not in videos:
            videos.append(frame.video)
        frames.append(
            sio.LabeledFrame(video=frame.video, frame_idx=frame.frame_idx, instances=instances)
        )

    save_package(frames, videos, [skeleton], out)
    return {
        "train": [str(out)],
        "n_frames": len(frames),
        "n_user_instances": sum(len(f.instances) for f in frames),
        "n_corrected": n_corrected,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--collab", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True, help="GT-anchored preds on --collab.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test-slp", type=Path, required=True, help="Reviewed holdout test set.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arms-json", type=Path, required=True)
    parser.add_argument("--holdout-camera", default="cam24")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--budgets", default="600")
    parser.add_argument("--skip-bounds", action="store_true", help="Do not build floor/ceiling.")
    parser.add_argument("--report", type=Path, default=None)

    parser.add_argument("--rank-nodes", default=DEFAULT_RANK_NODES)
    parser.add_argument("--anchor-node", default=DEFAULT_ANCHOR_NODE)
    parser.add_argument("--review-nodes", default=DEFAULT_REVIEW_NODES)
    parser.add_argument("--match-max-dist", type=float, default=100.0)
    parser.add_argument("--match-gate", type=float, default=15.0, help="collab<->reviewed pairing.")
    parser.add_argument(
        "--min-video-jaccard",
        type=float,
        default=0.8,
        help="Labelled-frame overlap required to accept a collab<->reviewed video pairing.",
    )
    parser.add_argument("--edge-margin", type=float, default=100.0)
    parser.add_argument("--max-hidden", type=int, default=3)
    parser.add_argument(
        "--conf-alpha", type=float, default=1.0, help="Exponent on (1 - confidence) for lowconf."
    )
    parser.add_argument(
        "--unhide-weight",
        type=float,
        default=10.0,
        help="Pixels an un-hidden node is worth in the oracle ranking.",
    )
    parser.add_argument("--diversity-gap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    holdout = {args.holdout_camera}

    manifest = read_manifest(args.manifest)
    collab = sio.load_slp(str(args.collab))
    reviewed = sio.load_slp(str(args.reviewed))
    pred = sio.load_slp(str(args.pred))

    camera_of_video = map_videos_to_cameras(collab, manifest[chunk_number(args.collab)], "collab")
    anchor_idx = resolve_nodes(collab, [args.anchor_node], "collab")[0]
    pairs = pair_label_versions(
        collab, reviewed, anchor_idx, args.match_gate, args.min_video_jaccard
    )
    print(f"[OK] paired collaborator<->reviewed instances: {len(pairs)}")

    records = gather(collab, pred, pairs, camera_of_video, args)
    pool = [r for r in records if r["camera"] not in holdout]
    corrections = sum(1 for r in pool if r["oracle_shift"] > 5 or r["oracle_unhidden"])
    print(
        f"[OK] pool: {len(pool)} instances on {sorted({r['camera'] for r in pool})} "
        f"(holdout {sorted(holdout)}); {corrections} carry a real correction"
    )

    arms: dict[str, dict] = {}
    if not args.skip_bounds:
        for name, chosen in (("floor_n0", []), ("ceiling_all", pool)):
            arms[name] = build_arm(
                collab, pairs, chosen, holdout, camera_of_video, args.out_dir / f"{name}.pkg.slp"
            )
            print(f"[OK] {name:<28} corrected {arms[name]['n_corrected']:>5}")

    for strategy in strategies:
        ordered = rank(pool, strategy, args)
        for budget in budgets:
            name = f"{strategy}_n{budget}"
            arms[name] = build_arm(
                collab,
                pairs,
                ordered[:budget],
                holdout,
                camera_of_video,
                args.out_dir / f"{name}.pkg.slp",
            )
            arms[name].update(strategy=strategy, budget=budget)
            hit = sum(1 for r in ordered[:budget] if r["oracle_shift"] > 5 or r["oracle_unhidden"])
            arms[name]["n_real_corrections"] = hit
            print(
                f"[OK] {name:<28} corrected {arms[name]['n_corrected']:>5}  "
                f"of which a real correction: {hit} ({100.0 * hit / max(budget, 1):.1f}%)"
            )

    payload = {
        "arms": arms,
        "test_gold": [str(args.test_slp)],
        "pool_size": len(pool),
        "pool_real_corrections": corrections,
        "holdout_cameras": sorted(holdout),
        "scoring_predictions": str(args.pred),
        "args": recorded_args(args),
    }
    write_report(args.arms_json, payload, quiet=True)
    write_report(args.report, {**payload, "records": records}, quiet=True)
    print(f"\n[OK] {len(arms)} arm(s) -> {args.arms_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
