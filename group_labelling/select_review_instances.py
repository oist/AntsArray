#!/usr/bin/env python3
"""Pick the next review batch out of the unreviewed chunks, dense frames first.

The training unit of a top-down centered-instance model is the *instance*, not the
frame: each sample is a 640px crop around one anchor. Nothing requires every ant in
a frame to be labelled, so instances can be selected one at a time, and the ones left
out simply become image context carrying no supervision. That is what makes a
targeted review affordable.

What needs reviewing is narrow. Matching the collaborator ch06 against the reviewed
ch06 over 5,521 instances, median per-node displacement was 0.00px on aruco, occiput,
petiole and gaster_tip, and 8.54 / 8.13px on antenna_L_1 and antenna_R_1 -- moved by
more than 5px in ~64% of instances against ~5-23% for every other node. With sigma
2.5 at output_stride 4 the confidence-map target is about 10px wide, so an 8.5px
offset is nearly a full target width: a systematic convention difference the model
will learn faithfully rather than noise it will average out. Per-instance review
scope is therefore just the two antenna base points, plus any antennae marked hidden
while actually visible.

Two selection modes, and the default is not the interesting-sounding one:

``frame`` (default)
    Take whole dense frames, densest first. Measured on ch06 the
    collaborator/reviewed difference is a regime switch rather than a gradient: at
    16+ instances per frame 72-78% of antenna bases moved by a mean of ~10px, below
    it only 23-43% moved by ~3.8px. Within a camera density then predicts nothing
    further (Spearman -0.004 on cam06), so the threshold carries the whole signal and
    there is nothing left to rank on inside the dense regime.

``instance``
    Rank individual instances by model-vs-label disagreement on the reliable nodes
    (aruco, occiput, petiole, gaster_tip; the antenna bases are excluded because they
    differ by convention rather than by difficulty). Replaying the ch06 review
    offline put that ranking's correlation with the corrections actually made at
    r=0.03 -- indistinguishable from random. It is kept as the comparison baseline
    and must not become the default again.

Both modes first apply, in this order:

  1. Camera diversity, enforced per camera. ch06 covers cam06/cam10/cam24 only;
     ch01-05 holds the other 22 cameras, and cross-camera generalisation is the known
     weak point. Cameras are chunk-disjoint, verified against the manifest.
  2. The truncation rule from ``topdown_common.classify_points``. The rig is a 5x5
     array with ~10% overlap so edge ants are physically cut off, but 45% of
     near-edge instances still carry a complete skeleton -- position alone over-drops.
  3. Density stratification (``instance`` mode). ch06 runs at a median of 25
     instances/frame while the sparse ch01-05 cameras average 2.1, and a model shown
     only crowded scenes learns only crowded scenes.

Predictions come from predict_gt_anchored.py, which crops on the ground-truth anchor
so exactly one prediction exists per labelled instance.

Output is a .pkg.slp holding only the selected instances with images embedded, plus a
JSON manifest recording camera, frame_idx, density bucket and disagreement score per
instance, an order CSV, and an effort estimate so the review budget can be chosen
from evidence rather than guessed.

Selection is deterministic -- no randomness anywhere, and every ordering is a stable
sort over a fixed input order -- so the same arguments over the same inputs reproduce
the same package.

Example::

    python select_review_instances.py \
        --flat-dir flat/ --pred-dir pred/ --manifest chunk_manifest.csv \
        --chunks 1,2,3,4,5 --total-budget 3000 \
        --out review/ch01to05_dense_n3000.pkg.slp --report review/selection.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sleap_io as sio
from sleap_io.model.suggestions import SuggestionFrame

from topdown_common import (
    DEFAULT_ANCHOR_NODE,
    DEFAULT_RANK_NODES,
    DEFAULT_REVIEW_NODES,
    antenna_indices,
    chunk_number,
    classify_points,
    disagreement,
    edge_margin_of,
    iter_frame_pairs,
    map_videos_to_cameras,
    match_point,
    node_names,
    parse_floats,
    parse_weighted_nodes,
    read_manifest,
    recorded_args,
    resolve_nodes,
    save_package,
    stem_of,
)


def density_bucket(n_instances: int, edges: tuple[float, float]) -> str:
    if n_instances <= edges[0]:
        return "sparse"
    if n_instances <= edges[1]:
        return "medium"
    return "dense"


def gather(
    labels: sio.Labels, src: Path, pred: sio.Labels, cameras: dict[str, set[int]], args
) -> tuple[list[dict], dict]:
    """Score and describe every user instance in one chunk."""
    camera_of_video = map_videos_to_cameras(labels, cameras, src.name)

    rank_names, rank_weights = parse_weighted_nodes(args.rank_nodes)
    rank_idx = resolve_nodes(labels, rank_names, src.name)
    anchor_idx = resolve_nodes(labels, [args.anchor_node], src.name)[0]
    review_idx = resolve_nodes(labels, args.review_nodes.split(","), src.name)
    antenna_idx = antenna_indices(labels)
    names = node_names(labels)

    records: list[dict] = []
    stats: Counter = Counter()
    chunk = chunk_number(src)

    for pair in iter_frame_pairs(labels, pred, anchor_idx, args.match_max_dist, src.name):
        camera = camera_of_video[pair.video_idx]
        height, width = pair.frame.video.shape[1:3]
        bucket = density_bucket(len(pair.instances), args.density_edges)

        for position, instance in enumerate(pair.instances):
            points = pair.gt_points[position]
            stats["total"] += 1
            reason = classify_points(
                points, antenna_idx, (height, width), args.edge_margin, args.max_hidden
            )
            if reason is not None:
                stats[f"excluded_{reason}"] += 1
                continue

            visible = ~np.isnan(points).any(axis=1)
            matched = pair.match(position)
            score, per_node, n_scored = (None, {}, 0)
            if matched is not None:
                score, per_node, n_scored = disagreement(
                    points, matched.points, rank_idx, rank_weights
                )
            usable = matched is not None and n_scored >= args.min_scored_nodes
            if not usable:
                stats["unmatched"] += 1

            records.append(
                {
                    "src_file": str(src),
                    "chunk": chunk,
                    "camera": camera,
                    "video_idx": pair.video_idx,
                    "frame_idx": int(pair.frame.frame_idx),
                    "src_instance_index": position,
                    "track": instance.track.name if instance.track is not None else None,
                    "anchor_xy": [
                        None if np.isnan(v) else round(float(v), 2)
                        for v in match_point(points, anchor_idx)
                    ],
                    "frame_n_user_instances": len(pair.instances),
                    "density_bucket": bucket,
                    "score": None if score is None else round(score, 3),
                    "score_per_node": {names[k]: v for k, v in per_node.items()},
                    "n_scored_nodes": n_scored,
                    "matched": usable,
                    "match_dist": None if matched is None else round(matched.distance, 2),
                    "edge_distance": round(edge_margin_of(points, (height, width)), 1),
                    "n_hidden": int((~visible).sum()),
                    "n_hidden_antenna": int((~visible[antenna_idx]).sum()),
                    "n_hidden_review_nodes": int((~visible[review_idx]).sum()),
                }
            )

    stats["eligible"] = len(records)
    return records, {
        "src": str(src),
        "chunk": chunk,
        "cameras": sorted(set(camera_of_video.values())),
        "counts": dict(stats),
        "node_names": names,
    }


# ------------------------------------------------------------------- allocation


def rank_key(record: dict, unmatched_first: bool) -> tuple:
    """Sort key within a camera: most disagreement first.

    An instance with no usable prediction is a total model failure rather than a
    small displacement, so it has no comparable score and is placed as a block --
    at the top by default, since the model got nothing right there. Ties fall back on
    the stability of Python's sort, which preserves the order instances were gathered
    in, so the result is reproducible without an artificial tiebreaker.
    """
    unmatched_rank = (
        (0 if unmatched_first else 1) if not record["matched"] else (1 if unmatched_first else 0)
    )
    return (unmatched_rank, -(record["score"] or 0.0))


def allocate_instances(records: list[dict], args) -> list[dict]:
    """``instance`` mode: density floor, per-camera quota, then redistribute leftovers.

    Cameras are visited round-robin inside every pass, so a budget that runs out
    mid-pass truncates evenly across cameras instead of alphabetically. Pass 3
    matters because several ch01-05 cameras hold barely 20 instances in total and
    cannot fill any useful quota; without it their unspent budget is simply lost.
    """
    by_camera: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_camera[record["camera"]].append(record)
    unmatched_first = args.unmatched == "top"
    for camera in by_camera:
        by_camera[camera].sort(key=lambda r: rank_key(r, unmatched_first))

    cameras = sorted(by_camera)
    budget = (
        args.total_budget if args.total_budget is not None else args.per_camera_quota * len(cameras)
    )
    taken: dict[str, set[int]] = {camera: set() for camera in cameras}
    selected: list[dict] = []

    def queue_density_floor(camera: str) -> list[int]:
        wanted: list[int] = []
        per_bucket: Counter = Counter()
        for position, record in enumerate(by_camera[camera]):
            bucket = record["density_bucket"]
            if per_bucket[bucket] < args.density_min_per_bucket:
                per_bucket[bucket] += 1
                wanted.append(position)
        return wanted

    def queue_remaining(camera: str, limit: int | None) -> list[int]:
        wanted = [p for p in range(len(by_camera[camera])) if p not in taken[camera]]
        return wanted if limit is None else wanted[:limit]

    def drain(queues: dict[str, list[int]], pass_no: int) -> None:
        cursors = {camera: 0 for camera in queues}
        while len(selected) < budget:
            progressed = False
            for camera in cameras:
                if len(selected) >= budget:
                    break
                queue = queues.get(camera, [])
                while cursors[camera] < len(queue) and queue[cursors[camera]] in taken[camera]:
                    cursors[camera] += 1
                if cursors[camera] >= len(queue):
                    continue
                position = queue[cursors[camera]]
                cursors[camera] += 1
                taken[camera].add(position)
                record = dict(by_camera[camera][position])
                record["rank_in_camera"] = position
                record["selected_in_pass"] = pass_no
                selected.append(record)
                progressed = True
            if not progressed:
                return

    drain({camera: queue_density_floor(camera) for camera in cameras}, 1)
    drain(
        {
            camera: queue_remaining(camera, max(0, args.per_camera_quota - len(taken[camera])))
            for camera in cameras
        },
        2,
    )
    if not args.no_redistribute:
        drain({camera: queue_remaining(camera, None) for camera in cameras}, 3)
    return selected


def allocate_frames(records: list[dict], args) -> list[dict]:
    """``frame`` mode: take whole dense frames, highest yield first.

    Taking every eligible instance of a chosen frame is what makes this cheap. The
    reviewer opens one crowded frame and fixes ~20 ants, instead of opening 20 frames
    for one ant each -- and frame-opening was about half the estimated review time.

    Frames are emitted in selection order, so the reviewer can work top-down and stop
    at any point with the most valuable work already done.
    """
    by_frame: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for record in records:
        by_frame[(record["src_file"], record["video_idx"], record["frame_idx"])].append(record)

    by_camera: dict[str, list] = defaultdict(list)
    for key, group in by_frame.items():
        if group[0]["frame_n_user_instances"] >= args.min_frame_density:
            by_camera[group[0]["camera"]].append((key, group))
    if not by_camera:
        raise SystemExit(
            f"[ERR] no frame holds >= {args.min_frame_density} instances; "
            f"lower --min-frame-density"
        )
    # Densest frames first: they carry the most correctable ants per frame opened.
    for camera in by_camera:
        by_camera[camera].sort(key=lambda kv: (-len(kv[1]), kv[0]))

    cameras = sorted(by_camera)
    budget = args.total_budget if args.total_budget is not None else len(records)
    cursors = {camera: 0 for camera in cameras}
    selected: list[dict] = []
    frame_rank = 0

    while len(selected) < budget:
        progressed = False
        for camera in cameras:
            if len(selected) >= budget:
                break
            cursor = cursors[camera]
            if cursor >= len(by_camera[camera]):
                continue
            cursors[camera] += 1
            _, group = by_camera[camera][cursor]
            # A frame is never split: a half-reviewed frame costs the same to open as
            # a fully reviewed one, so the final frame may overshoot the budget.
            for record in sorted(group, key=lambda r: r["src_instance_index"])[: args.max_per_frame]:
                picked = dict(record)
                picked["rank_in_camera"] = cursor
                picked["frame_rank"] = frame_rank
                picked["selected_in_pass"] = 1
                selected.append(picked)
            frame_rank += 1
            progressed = True
        if not progressed:
            break
    return selected


# ---------------------------------------------------------------------- output


def write_package(
    selected: list[dict], sources: dict[str, sio.Labels], out: Path, group_size: int = 20
) -> dict:
    """One package holding only the selected instances, images embedded.

    Every frame is also registered as a SLEAP suggestion, in selection order. Without
    that the priority ordering is invisible: the GUI walks frames per video by
    frame_idx, so the order of ``labeled_frames`` never reaches the reviewer. The
    suggestion list *is* the GUI's navigation order, so it is the only channel that
    carries "do this one first".

    SLEAP persists one integer per suggestion, ``group``. It is used here for the
    priority band (``group_size`` frames each), so the GUI clusters the queue into
    "first 20 frames", "next 20", and progress is visible at a glance.
    """
    reference = next(iter(sources.values())).skeletons[0]
    reference_names = [node.name for node in reference.nodes]
    for path, labels in sources.items():
        names = node_names(labels)
        if names != reference_names:
            raise SystemExit(
                f"[ERR] skeleton mismatch, cannot merge: {Path(path).name} has {names}, "
                f"expected {reference_names}"
            )

    grouped: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for record in selected:
        grouped[(record["src_file"], record["video_idx"], record["frame_idx"])].append(
            record["src_instance_index"]
        )

    # Insertion order is selection order, and it is deliberately not re-sorted: the
    # GUI walks frames in file order, so emitting them by priority lets the reviewer
    # start at the top and stop whenever the budget runs out.
    frames, videos, placement = [], [], {}
    for (src_file, video_idx, frame_idx), positions in grouped.items():
        labels = sources[src_file]
        video = labels.videos[video_idx]
        source_frame = next(f for f in labels if f.video is video and int(f.frame_idx) == frame_idx)
        user = list(source_frame.user_instances)
        chosen = []
        for out_position, position in enumerate(sorted(positions)):
            instance = user[position]
            # Re-point at one shared skeleton: a merged package carries exactly one,
            # and the equality check above proved the node order is identical.
            instance.skeleton = reference
            chosen.append(instance)
            placement[(src_file, video_idx, frame_idx, position)] = (len(frames), out_position)
        if video not in videos:
            videos.append(video)
        frames.append(sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=chosen))

    for record in selected:
        record["out_frame_index"], record["out_instance_index"] = placement[
            (
                record["src_file"],
                record["video_idx"],
                record["frame_idx"],
                record["src_instance_index"],
            )
        ]

    suggestions = [
        SuggestionFrame(
            video=frame.video, frame_idx=frame.frame_idx, metadata={"group": rank // group_size}
        )
        for rank, frame in enumerate(frames)
    ]

    started = time.time()
    save_package(frames, videos, [reference], out, suggestions=suggestions)
    return {
        "path": str(out),
        "n_frames": len(frames),
        "n_instances": len(selected),
        "n_videos": len(videos),
        "n_suggestions": len(suggestions),
        "suggestion_group_size": group_size,
        "bytes": out.stat().st_size,
        "seconds": round(time.time() - started, 1),
    }


def write_order_sheet(selected: list[dict], args) -> Path:
    """The review queue as a CSV, one row per frame, in the order the GUI presents it.

    The suggestion list carries the order inside SLEAP, but not how much work is left.
    This sheet does: cumulative instances and cumulative hours per frame, so a
    stopping point can be chosen against the clock rather than by feel.
    """
    path = args.out.parent / f"{stem_of(args.out)}_order.csv"

    per_frame: dict[int, list[dict]] = defaultdict(list)
    for record in selected:
        per_frame[record["out_frame_index"]].append(record)

    seconds = args.sec_per_op[0] if args.sec_per_op else 5.0
    overhead = args.sec_per_frame[0] if args.sec_per_frame else 15.0
    n_base = len(args.review_nodes.split(","))

    cumulative_instances, cumulative_seconds = 0, 0.0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "review_order", "suggestion_group", "camera", "chunk", "frame_idx",
                "instances_to_fix", "hidden_antennae", "cum_instances", "cum_hours",
            ]
        )
        for order, frame_index in enumerate(sorted(per_frame)):
            group = per_frame[frame_index]
            hidden = sum(r["n_hidden_antenna"] for r in group)
            cumulative_instances += len(group)
            cumulative_seconds += (
                args.expected_base_fix_rate * n_base * len(group) + hidden
            ) * seconds + overhead
            writer.writerow(
                [
                    order,
                    order // args.suggestion_group_size,
                    group[0]["camera"],
                    group[0]["chunk"],
                    group[0]["frame_idx"],
                    len(group),
                    hidden,
                    cumulative_instances,
                    f"{cumulative_seconds / 3600.0:.2f}",
                ]
            )
    return path


def effort(selected: list[dict], args) -> dict:
    """Estimated review cost, tabulated so a budget can be set from measurement."""
    n_frames = len({(r["src_file"], r["video_idx"], r["frame_idx"]) for r in selected}) or 1
    hidden = sum(r["n_hidden_antenna"] for r in selected)
    n_base = len(args.review_nodes.split(","))

    worst_case_ops = n_base * len(selected) + hidden
    expected_ops = args.expected_base_fix_rate * n_base * len(selected) + hidden

    table = [
        {
            "sec_per_keypoint": seconds,
            "sec_per_frame_overhead": overhead,
            "hours_expected": round((expected_ops * seconds + n_frames * overhead) / 3600.0, 2),
            "hours_worst_case": round((worst_case_ops * seconds + n_frames * overhead) / 3600.0, 2),
        }
        for seconds in args.sec_per_op
        for overhead in args.sec_per_frame
    ]
    return {
        "n_instances": len(selected),
        "n_frames_to_open": n_frames,
        "instances_per_frame": round(len(selected) / n_frames, 2),
        "hidden_antenna_nodes_to_unhide": hidden,
        "keypoint_ops_worst_case": worst_case_ops,
        "keypoint_ops_expected": round(expected_ops, 1),
        "expected_base_fix_rate": args.expected_base_fix_rate,
        "time_table": table,
        "note": (
            "Worst case assumes both antenna bases are repositioned on every instance. "
            "Expected applies --expected-base-fix-rate, the measured ch06 rate at which a "
            "base moved more than 5px. Time the first 20 instances, then re-read this "
            "table to choose a budget from evidence."
        ),
    }


def summarise(selected: list[dict], records: list[dict]) -> dict:
    per_camera: dict[str, Counter] = defaultdict(Counter)
    for record in selected:
        per_camera[record["camera"]]["n"] += 1
        per_camera[record["camera"]][record["density_bucket"]] += 1
    scores = [r["score"] for r in selected if r["score"] is not None]
    available = Counter(r["camera"] for r in records)
    return {
        "n_eligible": len(records),
        "n_selected": len(selected),
        "n_cameras": len(per_camera),
        "per_camera": {
            camera: {**counts, "available": available[camera]}
            for camera, counts in sorted(per_camera.items())
        },
        "per_density_bucket": dict(Counter(r["density_bucket"] for r in selected)),
        "per_pass": dict(Counter(r["selected_in_pass"] for r in selected)),
        "score": {
            "min": round(min(scores), 3) if scores else None,
            "median": round(float(np.median(scores)), 3) if scores else None,
            "max": round(max(scores), 3) if scores else None,
            "n_unmatched_selected": sum(1 for r in selected if not r["matched"]),
        },
    }


def print_report(report: dict, out_path: Path, report_path: Path) -> None:
    summary, cost, package = report["summary"], report["effort"], report["package"]
    print("\n=== selection ===")
    print(
        f"  eligible {summary['n_eligible']}  ->  selected {summary['n_selected']} "
        f"across {summary['n_cameras']} cameras"
    )
    print(f"  density buckets: {summary['per_density_bucket']}")
    print(f"  passes (1=density floor, 2=quota, 3=redistributed): {summary['per_pass']}")
    print(
        f"  disagreement px:  min {summary['score']['min']}  "
        f"median {summary['score']['median']}  max {summary['score']['max']}  "
        f"unmatched picked {summary['score']['n_unmatched_selected']}"
    )
    print(f"\n{'camera':<10}{'picked':>8}{'avail':>8}{'sparse':>8}{'medium':>8}{'dense':>8}")
    for camera, counts in summary["per_camera"].items():
        print(
            f"{camera:<10}{counts['n']:>8}{counts['available']:>8}"
            f"{counts.get('sparse', 0):>8}{counts.get('medium', 0):>8}{counts.get('dense', 0):>8}"
        )

    print("\n=== estimated review effort ===")
    print(
        f"  {cost['n_instances']} instances over {cost['n_frames_to_open']} frames "
        f"({cost['instances_per_frame']} per frame)"
    )
    print(
        f"  keypoint ops: {cost['keypoint_ops_expected']} expected, "
        f"{cost['keypoint_ops_worst_case']} worst case "
        f"({cost['hidden_antenna_nodes_to_unhide']} hidden antennae to un-hide)"
    )
    print(f"  {'sec/keypoint':>14}{'sec/frame':>11}{'hours':>9}{'hours worst':>13}")
    for row in cost["time_table"]:
        print(
            f"  {row['sec_per_keypoint']:>14g}{row['sec_per_frame_overhead']:>11g}"
            f"{row['hours_expected']:>9.2f}{row['hours_worst_case']:>13.2f}"
        )

    print(
        f"\n[OK] package  -> {out_path} ({package['n_frames']} frames, "
        f"{package['bytes'] / 1e9:.2f} GB, {package['seconds']}s)"
    )
    print(
        f"     {package['n_suggestions']} SLEAP suggestions in review order, "
        f"grouped every {package['suggestion_group_size']} frames"
    )
    print(f"[OK] manifest -> {report_path}")
    print(f"[OK] order    -> {package['order_csv']}")


# ------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--flat-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True, help="From predict_gt_anchored.py.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Review package (.pkg.slp).")
    parser.add_argument("--report", type=Path, required=True, help="JSON manifest.")
    parser.add_argument("--chunks", default="1,2,3,4,5")

    budget = parser.add_argument_group("budget")
    budget.add_argument(
        "--mode",
        choices=("frame", "instance"),
        default="frame",
        help="frame: take whole dense frames, highest yield first (far cheaper to "
        "review, and the dense regime is where the corrections are). instance: rank "
        "individual instances by model disagreement -- measured on ch06 to correlate "
        "with the actual corrections at only r=0.03, so it is kept for comparison.",
    )
    budget.add_argument(
        "--min-frame-density",
        type=int,
        default=16,
        help="frame mode: skip frames below this many instances. At 16+ per frame "
        "72-78%% of antenna bases needed a ~10px fix; below it, 23-43%% needed ~3.8px.",
    )
    budget.add_argument(
        "--max-per-frame",
        type=int,
        default=100,
        help="frame mode: cap instances taken from one frame. The default takes all.",
    )
    budget.add_argument(
        "--suggestion-group-size",
        type=int,
        default=20,
        help="Frames per SLEAP suggestion group. The GUI clusters the review queue by "
        "this integer, so it reads as progress bands through the priority order.",
    )
    budget.add_argument("--per-camera-quota", type=int, default=40)
    budget.add_argument(
        "--total-budget",
        type=int,
        default=None,
        help="Hard cap on selected instances. Default: quota x number of cameras.",
    )
    budget.add_argument(
        "--density-min-per-bucket",
        type=int,
        default=3,
        help="instance mode: floor taken from every density bucket a camera contains.",
    )
    budget.add_argument(
        "--no-redistribute",
        action="store_true",
        help="Leave the quota of instance-starved cameras unspent instead of reallocating it.",
    )

    scoring = parser.add_argument_group("scoring")
    scoring.add_argument("--rank-nodes", default=DEFAULT_RANK_NODES, help="name:weight,...")
    scoring.add_argument("--anchor-node", default=DEFAULT_ANCHOR_NODE)
    scoring.add_argument(
        "--review-nodes",
        default=DEFAULT_REVIEW_NODES,
        help="Nodes the reviewer repositions; drives the effort estimate, not the ranking.",
    )
    scoring.add_argument("--match-max-dist", type=float, default=100.0)
    scoring.add_argument("--min-scored-nodes", type=int, default=2)
    scoring.add_argument(
        "--unmatched",
        choices=("top", "bottom"),
        default="top",
        help="Where instances the model failed on entirely sit in the ranking.",
    )
    scoring.add_argument(
        "--max-unmatched-frac",
        type=float,
        default=0.10,
        help="Abort above this share of unmatched instances: at that level labels and "
        "predictions are misaligned rather than the model being bad.",
    )

    shaping = parser.add_argument_group("filtering and stratification")
    shaping.add_argument("--edge-margin", type=float, default=100.0)
    shaping.add_argument("--max-hidden", type=int, default=3)
    shaping.add_argument(
        "--density-edges",
        default="5,15",
        help="sparse <= first < medium <= second < dense, in instances per frame.",
    )

    estimate = parser.add_argument_group("effort estimate")
    estimate.add_argument("--expected-base-fix-rate", type=float, default=0.64)
    estimate.add_argument("--sec-per-op", default="3,5,8")
    estimate.add_argument("--sec-per-frame", default="10,20")
    return parser


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    edges = parse_floats(args.density_edges)
    if len(edges) != 2 or edges[0] >= edges[1]:
        raise SystemExit(
            f"[ERR] --density-edges needs two increasing values, got {args.density_edges!r}"
        )
    args.density_edges = (edges[0], edges[1])
    args.sec_per_op = parse_floats(args.sec_per_op)
    args.sec_per_frame = parse_floats(args.sec_per_frame)
    return args


def main(argv: list[str] | None = None) -> int:
    args = resolve_args(build_parser().parse_args(argv))
    wanted_chunks = {int(c) for c in args.chunks.split(",") if c.strip()}
    manifest = read_manifest(args.manifest)

    files = sorted(
        p
        for p in args.flat_dir.glob("*.slp")
        if not p.name.startswith(".") and chunk_number(p) in wanted_chunks
    )
    if not files:
        raise SystemExit(f"[ERR] no chunk {sorted(wanted_chunks)} packages in {args.flat_dir}")

    records: list[dict] = []
    per_file: list[dict] = []
    sources: dict[str, sio.Labels] = {}

    for src in files:
        pred_path = args.pred_dir / f"{stem_of(src)}.pred.slp"
        if not pred_path.is_file():
            raise SystemExit(f"[ERR] missing predictions for {src.name}: {pred_path}")
        print(f"[..] {src.name}", flush=True)

        labels = sio.load_slp(str(src))
        chunk_records, info = gather(
            labels, src, sio.load_slp(str(pred_path)), manifest[chunk_number(src)], args
        )
        records.extend(chunk_records)
        per_file.append(info)
        sources[str(src)] = labels
        counts = info["counts"]
        print(
            f"     {counts.get('total', 0)} instances, {counts.get('eligible', 0)} eligible, "
            f"{counts.get('unmatched', 0)} unmatched, cameras {','.join(info['cameras'])}"
        )

    if not records:
        raise SystemExit("[ERR] no eligible instances survived filtering")

    unmatched = sum(1 for r in records if not r["matched"])
    fraction = unmatched / len(records)
    if fraction > args.max_unmatched_frac:
        raise SystemExit(
            f"[ERR] {unmatched}/{len(records)} ({100 * fraction:.1f}%) instances have no usable "
            f"prediction, above --max-unmatched-frac {args.max_unmatched_frac:g}. Ground-truth "
            f"anchoring should match nearly everything, so check that --pred-dir was produced "
            f"from these exact packages and that predict ran with --peak_threshold 0.0."
        )

    selected = (
        allocate_frames(records, args) if args.mode == "frame" else allocate_instances(records, args)
    )
    package = write_package(selected, sources, args.out, args.suggestion_group_size)
    package["order_csv"] = str(write_order_sheet(selected, args))
    report = {
        "args": recorded_args(args),
        "per_file": per_file,
        "package": package,
        "summary": summarise(selected, records),
        "effort": effort(selected, args),
        # Manifest order matches package order, so row N describes what the reviewer
        # sees Nth and a partial review can be read off by truncating the list.
        "selection": sorted(selected, key=lambda r: (r["out_frame_index"], r["out_instance_index"])),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print_report(report, args.out, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
