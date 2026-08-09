#!/usr/bin/env python3
"""Cut the finished part of a review package out into a training set.

A review package is worked through in priority order and saved back whenever the
reviewer stops, so at any moment it is part corrected and part untouched. Training on
the whole file would be worse than useless: the untouched instances still carry the
old antenna convention, which is exactly the systematic error being removed, and
early on they outnumber the corrected ones several times over. Only finished frames
may be used.

Which frames those are is detected rather than typed in. A frame counts as reviewed
when at least ``--min-changed-fraction`` of its antenna bases moved by more than
``--changed-threshold`` px against the package that was handed out. On real data the
separation is wide -- reviewed frames run about 76% of bases moved, untouched ones
under 1% -- so the rule has a large margin and a miscounted stopping point cannot
quietly poison the training set.

Every instance of a reviewed frame is kept, including ones the reviewer left alone:
those were looked at and judged correct, which is a label rather than an omission.

The two files are two *versions* of the same labels, so their videos are paired by
best labelled-frame overlap and never by index -- see
``topdown_common.match_videos_by_overlap``. A Windows GUI save rewrites video paths
to ``Z:/...`` and de-duplicates embedded videos by filename (6 collapsing to 2 has
been observed), so position in ``labels.videos`` survives nothing. ``frame_idx`` is
the key.

Example::

    python extract_reviewed_batch.py \
        --reviewed review_ch01to05/ch01to05_dense_n3000.slp \
        --original review_ch01to05/ch01to05_dense_n3000.pkg.slp \
        --out /work/.../reviewed_batch1.pkg.slp --report /work/.../reviewed_batch1.json
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sleap_io as sio

from topdown_common import (
    DEFAULT_ANCHOR_NODE,
    DEFAULT_REVIEW_NODES,
    index_user_instances,
    instance_array,
    match_frame,
    match_videos_by_overlap,
    resolve_nodes,
    save_package,
    videos_of,
    write_report,
)


def frame_change(
    original: list, reviewed: list, anchor_idx: int, review_idx: list[int], gate: float
) -> tuple[int, int, list[float]]:
    """(bases compared, bases un-hidden, displacements) for one frame."""
    if not original or not reviewed:
        return 0, 0, []
    before = instance_array(original)
    after = instance_array(reviewed)

    compared = unhidden = 0
    shifts: list[float] = []
    for position, (other, _) in match_frame(before, after, anchor_idx, gate).items():
        for node_idx in review_idx:
            a, b = before[position][node_idx], after[other][node_idx]
            if np.isnan(a).any() and not np.isnan(b).any():
                unhidden += 1
            elif not np.isnan(a).any() and not np.isnan(b).any():
                compared += 1
                shifts.append(float(np.linalg.norm(a - b)))
    return compared, unhidden, shifts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reviewed", type=Path, required=True, help="The saved review file.")
    parser.add_argument("--original", type=Path, required=True, help="The package handed out.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--anchor-node", default=DEFAULT_ANCHOR_NODE)
    parser.add_argument("--review-nodes", default=DEFAULT_REVIEW_NODES)
    parser.add_argument("--match-gate", type=float, default=15.0)
    parser.add_argument("--changed-threshold", type=float, default=5.0)
    parser.add_argument(
        "--min-changed-fraction",
        type=float,
        default=0.20,
        help="Fraction of a frame's antenna bases that must have moved for the frame "
        "to count as reviewed. Reviewed frames sit near 0.76, untouched under 0.01.",
    )
    parser.add_argument(
        "--min-video-jaccard",
        type=float,
        default=0.8,
        help="Labelled-frame overlap required to accept a reviewed<->original video "
        "pairing. Below it these are not two versions of the same labels.",
    )
    parser.add_argument(
        "--frames",
        default=None,
        help="Override detection with an explicit 0-based range into the review order, "
        "e.g. '0-19'. Use only if the detector disagrees with what was actually done.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    original = sio.load_slp(str(args.original))
    reviewed = sio.load_slp(str(args.reviewed))

    anchor_idx = resolve_nodes(original, [args.anchor_node], "original")[0]
    review_idx = resolve_nodes(original, args.review_nodes.split(","), "original")
    video_map = match_videos_by_overlap(reviewed, original, args.min_video_jaccard)
    original_by_key = index_user_instances(original)
    reviewed_by_key = index_user_instances(reviewed)

    explicit = None
    if args.frames:
        lo, _, hi = args.frames.partition("-")
        explicit = range(int(lo), int(hi or lo) + 1)

    kept, stats, per_video = [], Counter(), Counter()
    shifts_all: list[float] = []
    detail = []

    for order, frame in enumerate(reviewed):
        video_idx = reviewed.videos.index(frame.video)
        frame_idx = int(frame.frame_idx)
        partner = video_map.get(video_idx)
        compared, unhidden, shifts = frame_change(
            [] if partner is None else original_by_key.get((partner, frame_idx), []),
            reviewed_by_key.get((video_idx, frame_idx), []),
            anchor_idx,
            review_idx,
            args.match_gate,
        )
        strong = sum(1 for s in shifts if s > args.changed_threshold)
        fraction = strong / compared if compared else 0.0
        is_reviewed = (
            order in explicit if explicit is not None else fraction >= args.min_changed_fraction
        )
        detail.append(
            {
                "review_order": order,
                "frame_idx": frame_idx,
                "n_instances": len(frame.user_instances),
                "bases_compared": compared,
                "bases_moved": strong,
                "fraction_moved": round(fraction, 4),
                "un_hidden": unhidden,
                "counted_as_reviewed": bool(is_reviewed),
            }
        )
        if not is_reviewed:
            continue
        kept.append(frame)
        stats["frames"] += 1
        stats["instances"] += len(frame.user_instances)
        stats["bases_moved"] += strong
        stats["un_hidden"] += unhidden
        per_video[f"video{video_idx}"] += len(frame.user_instances)
        shifts_all.extend(s for s in shifts if s > args.changed_threshold)

    if not kept:
        raise SystemExit(
            "[ERR] no frame passed the reviewed test: either nothing has been corrected "
            "yet, or --min-changed-fraction is set too high"
        )

    started = time.time()
    save_package(kept, videos_of(kept, reviewed), reviewed.skeletons, args.out)

    orders = [d["review_order"] for d in detail if d["counted_as_reviewed"]]
    report = {
        "reviewed": str(args.reviewed),
        "original": str(args.original),
        "out": str(args.out),
        "n_frames": stats["frames"],
        "n_instances": stats["instances"],
        "review_order_range": [min(orders), max(orders)],
        # A gap means a frame inside the worked range failed the test -- worth a look
        # before training, since it may be a frame that genuinely needed no change.
        "contiguous_from_start": orders == list(range(len(orders))),
        "antenna_bases_moved": stats["bases_moved"],
        "un_hidden": stats["un_hidden"],
        "displacement": {
            "median": round(float(np.median(shifts_all)), 3) if shifts_all else None,
            "mean": round(float(np.mean(shifts_all)), 3) if shifts_all else None,
            "p90": round(float(np.percentile(shifts_all, 90)), 3) if shifts_all else None,
        },
        "per_video_instances": dict(per_video),
        "video_map_reviewed_to_original": {str(k): v for k, v in sorted(video_map.items())},
        "bytes": args.out.stat().st_size,
        "seconds": round(time.time() - started, 1),
        "frames": detail,
    }

    print(f"[OK] reviewed frames detected: {stats['frames']} of {len(reviewed)}")
    print(
        f"     review order {report['review_order_range']}, "
        f"contiguous from the start: {report['contiguous_from_start']}"
    )
    print(
        f"     {stats['instances']} instances, {stats['bases_moved']} antenna bases moved "
        f"(median {report['displacement']['median']}px), {stats['un_hidden']} un-hidden"
    )
    print(f"[OK] package -> {args.out} ({report['bytes'] / 1e9:.2f} GB, {report['seconds']}s)")

    write_report(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
