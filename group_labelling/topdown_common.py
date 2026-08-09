#!/usr/bin/env python3
"""Shared primitives for the top-down centered-instance tooling.

Every entry point beside this file needs some of the same handful of operations --
resolve a node name, map a video to a camera, pair labels with predictions, decide
whether an instance is truncated, measure per-node error. Each script that grew its
own copy eventually grew a different answer, and at least one of those differences
produced numbers that looked plausible and were not. They live here so there is
exactly one definition of each.

Sections, in the order a run uses them:

  skeleton nodes      irregular names, resolved forgivingly but never rewritten
  manifest + cameras  video -> camera by exact labelled-frame set
  video alignment     TWO relations that must not be confused (see below)
  instance matching   Hungarian assignment on the anchor node, per frame
  truncation rule     edge margin AND hidden-node count, never either alone
  error + PCK         per-node displacement against ground-truth-anchored predictions
  body frame          along/perp offsets normalised by body length
  io                  package writing and JSON reports, spelled the same everywhere

THE TWO VIDEO RELATIONS. ``align_videos`` and ``match_videos_by_overlap`` look
interchangeable and are not:

  predictions vs labels    containment.  A prediction can only exist where a label
                           anchored it, so the prediction's frame set is a subset of
                           the source's -- never equal. Use ``align_videos``.
  two label versions       best overlap. A review may leave a frame with no
                           instances or add one to a frame that had none, so neither
                           side contains the other. On the real ch06 pair that is
                           Jaccard 0.990, not containment. Use
                           ``match_videos_by_overlap``.

Conflating them is what broke this once: containment matched nothing where the two
sides merely overlapped, and the run went on to score one ant against another ant's
prediction.

``frame_idx`` is the alignment key throughout, never the video index. A Windows GUI
save rewrites video paths to ``Z:/...`` (unreadable on Linux) and de-duplicates
embedded videos by filename -- 6 collapsing to 2 has been observed -- so position in
``labels.videos`` survives nothing.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import sleap_io as sio
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------- skeleton nodes

N_ANTENNA = 6
CHUNK_RE = re.compile(r"chunk(\d+)", re.IGNORECASE)

DEFAULT_ANCHOR_NODE = "aruco"
DEFAULT_REVIEW_NODES = "antenna_L_1,antenna_R_1"
DEFAULT_RANK_NODES = "aruco:2.0,occiput:1.0,petiole:1.0,gaster_tip:1.0"
DEFAULT_BODY_AXIS = "gaster_tip,occiput"


def normalise(name: str) -> str:
    """Fold a node name for *comparison only*; never write the result back.

    The skeleton's names carry deliberate irregularities: ``'aruco '`` has a
    trailing space, ``'antenna L_2'`` uses a space where its siblings use an
    underscore, and ``'antenna_L3'`` drops one. Warm start loads a state dict by
    head-channel name, so tidying them renames channels and silently breaks it.
    Everything here compares forgivingly and stores the original.
    """
    return name.strip().lower().replace(" ", "_")


def node_names(labels: sio.Labels) -> list[str]:
    """The skeleton's node names, verbatim and in order."""
    return [node.name for node in labels.skeletons[0].nodes]


def resolve_nodes(labels: sio.Labels, wanted: list[str], label: str) -> list[int]:
    """Indices of ``wanted`` in the skeleton, matched through ``normalise``."""
    names = [normalise(name) for name in node_names(labels)]
    indices = []
    for want in wanted:
        key = normalise(want)
        if key not in names:
            raise SystemExit(f"[ERR] {label}: no node {want!r} in skeleton {names}")
        indices.append(names.index(key))
    return indices


def parse_weighted_nodes(spec: str) -> tuple[list[str], list[float]]:
    """``'aruco:2.0,occiput:1.0'`` -> (names, weights); a bare name weighs 1.0."""
    names, weights = [], []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, weight = item.partition(":")
        names.append(name)
        weights.append(float(weight) if weight else 1.0)
    if not names:
        raise SystemExit("[ERR] empty node spec")
    return names, weights


def antenna_indices(labels: sio.Labels) -> list[int]:
    """Every antenna node, found by substring so all six spellings are caught."""
    return [i for i, name in enumerate(node_names(labels)) if "antenna" in name.lower()]


def parse_floats(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


# ------------------------------------------------------------ manifest + cameras


def camera_of(video_name: str) -> str:
    """'cam01_cam0_2026-05-15-13-09-56_000.avi' -> 'cam01'."""
    return video_name.split("_", 1)[0]


def chunk_number(path: Path) -> int:
    match = CHUNK_RE.search(path.name)
    if not match:
        raise SystemExit(f"[ERR] cannot read a chunk number from {path.name}")
    return int(match.group(1))


def read_manifest(path: Path) -> dict[int, dict[str, set[int]]]:
    """chunk -> camera -> set of labelled frame indices, from chunk_manifest.csv."""
    by_chunk: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            by_chunk[int(row["chunk"])][camera_of(row["video"])].add(int(row["frame_idx"]))
    return {chunk: dict(cams) for chunk, cams in by_chunk.items()}


def map_videos_to_cameras(labels: sio.Labels, cameras: dict[str, set[int]], label: str) -> dict:
    """Match each video to a camera by its exact set of labelled frame indices.

    By content, not by position: video order inside a package is not guaranteed to
    survive a GUI re-save, and a wrong pairing would attribute every instance of one
    camera to another while still producing a complete-looking report.
    """
    frames_by_video: dict[int, set[int]] = defaultdict(set)
    for frame in labels:
        frames_by_video[labels.videos.index(frame.video)].add(int(frame.frame_idx))

    mapping: dict[int, str] = {}
    unclaimed = dict(cameras)
    for video_idx, indices in sorted(frames_by_video.items()):
        exact = [cam for cam, want in unclaimed.items() if want == indices]
        if len(exact) == 1:
            mapping[video_idx] = exact[0]
            del unclaimed[exact[0]]
            continue
        # No unique exact match: report the closest camera so a mismatch is
        # diagnosable rather than silently pairing frames with the wrong camera.
        scored = sorted(
            (len(indices & want) / max(len(indices | want), 1), cam)
            for cam, want in unclaimed.items()
        )
        best = scored[-1] if scored else (0.0, "?")
        raise SystemExit(
            f"[ERR] {label}: video{video_idx} ({len(indices)} frames) matched "
            f"{len(exact)} cameras exactly; closest is {best[1]} at Jaccard {best[0]:.3f}. "
            f"Unclaimed: {sorted(unclaimed)}"
        )
    return mapping


# --------------------------------------------------------------- video alignment


def labelled_frame_sets(labels: sio.Labels, user_only: bool = True) -> dict[int, set[int]]:
    """video index -> the frame indices it carries (optionally: that carry a label)."""
    sets: dict[int, set[int]] = defaultdict(set)
    for frame in labels:
        if user_only and not frame.user_instances:
            continue
        sets[labels.videos.index(frame.video)].add(int(frame.frame_idx))
    return sets


def align_videos(src: sio.Labels, pred: sio.Labels, label: str) -> dict[int, int]:
    """CONTAINMENT: source video index -> prediction video index.

    For predictions against the labels they were generated from. A prediction's
    frame set is a *subset* of its source's, not an equal: ground-truth anchoring can
    only emit a frame that had a user instance to anchor on, and each chunk carries a
    handful of frames holding predictions only. So each prediction video is assigned
    to the one source video that contains it, and a source video with no counterpart
    simply scores as unmatched rather than aborting the run.

    Do NOT use this to pair two versions of the same labels -- see
    ``match_videos_by_overlap``.
    """
    src_sets = labelled_frame_sets(src, user_only=True)
    pred_sets = labelled_frame_sets(pred, user_only=False)
    mapping: dict[int, int] = {}

    for pred_idx, have in sorted(pred_sets.items()):
        contained = [s for s, want in src_sets.items() if s not in mapping and have <= want]
        if len(contained) == 1:
            mapping[contained[0]] = pred_idx
            continue
        scored = sorted(
            (len(have & want) / max(len(have | want), 1), s)
            for s, want in src_sets.items()
            if s not in mapping
        )
        best = scored[-1] if scored else (0.0, -1)
        raise SystemExit(
            f"[ERR] {label}: prediction video{pred_idx} ({len(have)} frames) is contained "
            f"in {len(contained)} unclaimed source videos; closest is source video{best[1]} "
            f"at Jaccard {best[0]:.3f}. Predictions do not correspond to these labels."
        )
    return mapping


def match_videos_by_overlap(
    before: sio.Labels, after: sio.Labels, min_jaccard: float
) -> dict[int, int]:
    """BEST OVERLAP: ``before`` video index -> ``after`` video index.

    For two *versions* of the same labels -- a package and its reviewed re-save.
    ``align_videos`` cannot be used here. It assumes one side's frame set is
    contained in the other's, which holds for predictions but not for two
    independently edited label versions: a review may leave a frame with no
    instances, or add one to a frame that had none. On the real ch06 pair that shows
    up as Jaccard 0.990 rather than containment. Best-overlap with a floor is the
    right relation, and uniqueness is still required so a genuine mismatch cannot
    pass silently.
    """
    before_sets, after_sets = labelled_frame_sets(before), labelled_frame_sets(after)
    mapping: dict[int, int] = {}
    for after_idx, have in sorted(after_sets.items()):
        scored = sorted(
            (len(have & want) / max(len(have | want), 1), before_idx)
            for before_idx, want in before_sets.items()
            if before_idx not in mapping
        )
        if not scored or scored[-1][0] < min_jaccard:
            best = scored[-1] if scored else (0.0, -1)
            raise SystemExit(
                f"[ERR] video{after_idx} ({len(have)} labelled frames) matches no source "
                f"video above Jaccard {min_jaccard:g}; best was video{best[1]} at "
                f"{best[0]:.3f}. These are not two versions of the same labels."
            )
        if len(scored) > 1 and scored[-2][0] >= min_jaccard:
            raise SystemExit(
                f"[ERR] video{after_idx} matches two source videos above Jaccard "
                f"{min_jaccard:g} ({scored[-1]}, {scored[-2]}); pairing is ambiguous."
            )
        mapping[scored[-1][1]] = after_idx
    return mapping


# ------------------------------------------------------------- instance matching


def instance_array(instances: list, n_nodes: int = 0) -> np.ndarray:
    """(n_instances, n_nodes, 2) of coordinates; NaN where a node is hidden."""
    if not instances:
        return np.zeros((0, n_nodes, 2))
    return np.array([np.asarray(inst.numpy()) for inst in instances])


def match_point(points: np.ndarray, anchor_idx: int) -> np.ndarray:
    """Anchor node if visible, else the NaN-ignoring mean of the visible nodes.

    Mirrors sleap-nn's own centroid fallback, so an instance is matched at the same
    location its crop was generated around.
    """
    anchor = points[anchor_idx]
    if not np.isnan(anchor).any():
        return anchor
    visible = points[~np.isnan(points).any(axis=1)]
    return visible.mean(axis=0) if len(visible) else np.array([np.nan, np.nan])


def match_frame(
    gt_points: np.ndarray, pred_points: np.ndarray, anchor_idx: int, max_dist: float
) -> dict[int, tuple[int, float]]:
    """Pair instances across two sets by anchor proximity, one frame at a time.

    Ground-truth anchoring already makes the correspondence near-exact, but the
    pairing is computed rather than assumed from array position, so that a video
    mis-alignment surfaces as unmatched instances instead of silently scoring one
    ant against another ant's prediction.
    """
    if not len(gt_points) or not len(pred_points):
        return {}
    gt_anchors = np.array([match_point(p, anchor_idx) for p in gt_points])
    pred_anchors = np.array([match_point(p, anchor_idx) for p in pred_points])
    cost = np.linalg.norm(gt_anchors[:, None, :] - pred_anchors[None, :, :], axis=-1)
    cost = np.nan_to_num(cost, nan=1e9)
    rows, cols = linear_sum_assignment(cost)
    return {
        int(r): (int(c), float(cost[r, c])) for r, c in zip(rows, cols) if cost[r, c] <= max_dist
    }


def index_predictions(pred: sio.Labels) -> dict[tuple[int, int], list]:
    """(video index, frame_idx) -> predicted instances."""
    by_key: dict[tuple[int, int], list] = {}
    for frame in pred:
        key = (pred.videos.index(frame.video), int(frame.frame_idx))
        by_key.setdefault(key, []).extend(frame.predicted_instances)
    return by_key


def index_user_instances(labels: sio.Labels) -> dict[tuple[int, int], list]:
    """(video index, frame_idx) -> user instances."""
    by_key: dict[tuple[int, int], list] = defaultdict(list)
    for frame in labels:
        by_key[(labels.videos.index(frame.video), int(frame.frame_idx))].extend(
            frame.user_instances
        )
    return dict(by_key)


class Match(NamedTuple):
    """One ground-truth instance paired with its prediction."""

    points: np.ndarray
    instance: object
    distance: float


class FramePair(NamedTuple):
    """One labelled frame together with the predictions that landed on it."""

    frame: object
    video_idx: int
    instances: list
    gt_points: np.ndarray
    pred_points: np.ndarray
    pred_instances: list
    pairing: dict[int, tuple[int, float]]

    def match(self, position: int) -> Match | None:
        """The prediction paired with the instance at ``position``, if any."""
        matched = self.pairing.get(position)
        if matched is None:
            return None
        index, distance = matched
        return Match(self.pred_points[index], self.pred_instances[index], distance)


def iter_frame_pairs(
    labels: sio.Labels,
    pred: sio.Labels,
    anchor_idx: int,
    max_dist: float,
    label: str = "predictions",
) -> Iterator[FramePair]:
    """Walk labelled frames, each already paired against its predictions.

    Frames with no user instance are skipped: ground-truth anchoring has nothing to
    crop around there, so they carry no comparison.
    """
    video_map = align_videos(labels, pred, label)
    pred_by_key = index_predictions(pred)
    n_nodes = len(labels.skeletons[0].nodes)

    for frame in labels:
        video_idx = labels.videos.index(frame.video)
        instances = list(frame.user_instances)
        if not instances:
            continue
        gt_points = instance_array(instances, n_nodes)
        partner = video_map.get(video_idx)
        predicted = [] if partner is None else pred_by_key.get((partner, int(frame.frame_idx)), [])
        pred_points = instance_array(predicted, n_nodes)
        yield FramePair(
            frame=frame,
            video_idx=video_idx,
            instances=instances,
            gt_points=gt_points,
            pred_points=pred_points,
            pred_instances=predicted,
            pairing=match_frame(gt_points, pred_points, anchor_idx, max_dist),
        )


def pair_label_versions(
    before: sio.Labels,
    after: sio.Labels,
    anchor_idx: int,
    gate: float,
    min_jaccard: float,
) -> dict[tuple[int, int, int], object]:
    """(before video_idx, frame_idx, position) -> the corresponding ``after`` instance.

    Paired by anchor proximity inside a frame rather than by position, because an
    edit may add, drop or reorder instances. Videos are paired by overlap, not by
    index: see ``match_videos_by_overlap``.
    """
    video_map = match_videos_by_overlap(before, after, min_jaccard)
    after_by_key = index_user_instances(after)

    pairs: dict[tuple[int, int, int], object] = {}
    for frame in before:
        video_idx = before.videos.index(frame.video)
        partner = video_map.get(video_idx)
        if partner is None:
            continue
        others = after_by_key.get((partner, int(frame.frame_idx)), [])
        instances = list(frame.user_instances)
        if not others or not instances:
            continue
        left = instance_array(instances)
        right = instance_array(others)
        for position, (other_idx, _) in match_frame(left, right, anchor_idx, gate).items():
            pairs[(video_idx, int(frame.frame_idx), position)] = others[other_idx]
    return pairs


# --------------------------------------------------------------- truncation rule


def edge_margin_of(points: np.ndarray, hw: tuple[int, int]) -> float:
    """Distance from the visible-node bounding box to the nearest image boundary."""
    height, width = hw
    seen = points[~np.isnan(points).any(axis=1)]
    return float(
        min(
            seen[:, 0].min(),
            seen[:, 1].min(),
            width - seen[:, 0].max(),
            height - seen[:, 1].max(),
        )
    )


def classify_points(
    points: np.ndarray,
    antenna_idx: list[int],
    hw: tuple[int, int],
    edge_margin: float,
    max_hidden: int,
) -> str | None:
    """Rule name rejecting this (n_nodes, 2) NaN-for-hidden array, or None to keep it.

    The combined rule, and both halves are load-bearing:

      1. near an image edge AND missing at least ``max_hidden`` nodes -> truncated
      2. interior but with all six antenna nodes hidden               -> label defect

    Neither test works alone. Across the six labelling chunks 29.6% of instances sit
    within 100px of a boundary, but **45% of those carry a complete skeleton** -- so
    filtering on position alone discards ~2,500 perfectly good ants. Conversely,
    hidden-node count alone cannot separate a genuinely truncated ant from a crowded
    interior one whose antennae the labeller chose to hide, and that interior case is
    the harder and more valuable data.

    Rule 2 is deliberately narrow: only 0.6% of instances are interior with all six
    antennae hidden, because all-antennae-hidden is overwhelmingly an edge symptom.
    It is kept because such an instance teaches the model to predict nothing at six
    locations on a fully-visible ant, which is actively wrong -- sleap-nn renders a
    hidden node as an all-zero confidence map rather than as missing data.
    """
    visible = ~np.isnan(points).any(axis=1)
    if not visible.any():
        return "no_visible_nodes"

    n_hidden = int((~visible).sum())
    n_hidden_antenna = int((~visible[antenna_idx]).sum())
    near_edge = edge_margin_of(points, hw) < edge_margin

    if near_edge and n_hidden >= max_hidden:
        return "edge_and_hidden"
    if not near_edge and n_hidden_antenna >= N_ANTENNA:
        return "centre_no_antennae"
    return None


# ------------------------------------------------------------------ error + PCK


def disagreement(
    gt: np.ndarray, pred: np.ndarray, rank_idx: list[int], weights: list[float]
) -> tuple[float | None, dict[int, float], int]:
    """Weighted mean displacement over the nodes visible in both.

    Ranked on the reliable nodes only -- aruco, occiput, petiole, gaster_tip.
    antenna_L_1/antenna_R_1 are deliberately excluded: they disagree by convention
    rather than by difficulty, so they say nothing about which instances are hard.
    """
    per_node, used_weights, values = {}, [], []
    for node_idx, weight in zip(rank_idx, weights):
        a, b = gt[node_idx], pred[node_idx]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        distance = float(np.linalg.norm(a - b))
        per_node[node_idx] = round(distance, 3)
        values.append(distance)
        used_weights.append(weight)
    if not values:
        return None, per_node, 0
    return float(np.average(values, weights=used_weights)), per_node, len(values)


class Comparison(NamedTuple):
    """Aligned ground-truth and predicted coordinates, plus their displacement.

    ``error`` is (n_instances, n_nodes); ``gt`` and ``pred`` are
    (n_instances, n_nodes, 2). Every array is NaN wherever a node is not comparable,
    and all three share one row order, so a body-frame check reads exactly the
    instances that were scored.
    """

    error: np.ndarray
    gt: np.ndarray
    pred: np.ndarray
    n_unmatched: int


def compare_to_predictions(
    gt: sio.Labels, pred: sio.Labels, anchor_idx: int, max_dist: float
) -> Comparison:
    """Per-node displacement of ground-truth-anchored predictions against labels.

    Ground-truth anchoring gives every labelled instance exactly one prediction, so
    the centroid model contributes nothing to the number. That is what is wanted when
    comparing centered-instance arms: an arm should be neither rewarded nor punished
    for a detection stage every arm shares.
    """
    n_nodes = len(gt.skeletons[0].nodes)
    errors, truth, guess, unmatched = [], [], [], 0
    blank = np.full((n_nodes, 2), np.nan)
    for pair in iter_frame_pairs(gt, pred, anchor_idx, max_dist):
        for position in range(len(pair.instances)):
            truth.append(pair.gt_points[position])
            matched = pair.match(position)
            if matched is None:
                unmatched += 1
                errors.append(np.full(n_nodes, np.nan))
                guess.append(blank)
                continue
            errors.append(np.linalg.norm(pair.gt_points[position] - matched.points, axis=-1))
            guess.append(matched.points)
    if not errors:
        return Comparison(
            np.zeros((0, n_nodes)), np.zeros((0, n_nodes, 2)), np.zeros((0, n_nodes, 2)), unmatched
        )
    return Comparison(np.array(errors), np.array(truth), np.array(guess), unmatched)


def stats_of(values: np.ndarray, thresholds: list[float]) -> dict:
    """Median / mean / p90 and PCK at each threshold, over the non-NaN entries.

    PCK is reported at several thresholds because the confidence-map target is about
    10px wide at sigma 2.5 / output_stride 4: a 5px threshold sits well inside one
    target and 20px spans two, so the pair brackets the scale that matters.
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if not len(values):
        return {"n": 0}
    return {
        "n": int(len(values)),
        "median": round(float(np.median(values)), 3),
        "mean": round(float(values.mean()), 3),
        "p90": round(float(np.percentile(values, 90)), 3),
        **{f"pck@{t:g}": round(float((values <= t).mean()), 4) for t in thresholds},
    }


# ------------------------------------------------------------------- body frame


def body_frame(points: np.ndarray, axis_idx: tuple[int, int]) -> tuple | None:
    """(origin, forward unit vector, normal, body length) for one instance.

    ``axis_idx`` is (posterior, anterior) -- by default (gaster_tip, occiput) -- and
    the origin is the anterior node, because the nodes this describes are the antenna
    bases, which attach at the head. Returns None when either axis node is hidden or
    the two coincide.
    """
    tail, head = points[axis_idx[0]], points[axis_idx[1]]
    if np.isnan(tail).any() or np.isnan(head).any():
        return None
    forward = head - tail
    length = float(np.linalg.norm(forward))
    if length <= 0:
        return None
    unit = forward / length
    # A fixed 90-degree rotation. Which physical side gets the positive sign does not
    # matter; that it is the *same* rotation everywhere does, because that is what
    # puts the L and R nodes on opposite signs and turns a convention flip into a
    # sign or magnitude change rather than into noise.
    normal = np.array([-unit[1], unit[0]])
    return head, unit, normal, length


def body_frame_offsets(
    points: np.ndarray, node_idx: list[int], axis_idx: tuple[int, int]
) -> dict[int, dict] | None:
    """Per-node along/perp offset in the instance's own body frame.

    Reported in pixels and in body lengths. ``perp`` is the convention indicator: a
    labelling-convention difference moves a node sideways relative to the body axis
    and keeps doing so at every posture. ``along`` cannot serve that purpose -- it
    varies with posture by more than the effect being measured (6.6px between the
    sparse and dense regimes of one and the same labeller), so ranking on it would
    chase biology instead of labelling drift.

    Normalising by body length removes apparent-size differences between cameras and
    between ants, which would otherwise register as a convention difference whenever
    two label sets happen to cover different cameras.
    """
    frame = body_frame(points, axis_idx)
    if frame is None:
        return None
    origin, unit, normal, length = frame

    out: dict[int, dict] = {}
    for idx in node_idx:
        point = points[idx]
        if np.isnan(point).any():
            continue
        offset = point - origin
        along, perp = float(offset @ unit), float(offset @ normal)
        out[idx] = {
            "along_px": along,
            "perp_px": perp,
            "along_norm": along / length,
            "perp_norm": perp / length,
            "body_length_px": length,
        }
    return out


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=float)
    return {
        "n": int(len(array)),
        "median": round(float(np.median(array)), 4),
        "mean": round(float(array.mean()), 4),
        "sd": round(float(array.std(ddof=1)), 4) if len(array) > 1 else 0.0,
    }


def body_frame_table(
    points: np.ndarray, node_idx: list[int], axis_idx: tuple[int, int], names: list[str]
) -> dict:
    """Aggregate ``body_frame_offsets`` over an (n_instances, n_nodes, 2) stack."""
    collected: dict[int, dict[str, list[float]]] = {idx: defaultdict(list) for idx in node_idx}
    n_framed = 0
    for instance in points:
        offsets = body_frame_offsets(instance, node_idx, axis_idx)
        if offsets is None:
            continue
        n_framed += 1
        for idx, values in offsets.items():
            for key, value in values.items():
                collected[idx][key].append(value)

    return {
        "n_instances": int(len(points)),
        "n_with_body_frame": n_framed,
        "per_node": {
            names[idx]: {key: describe(values) for key, values in sorted(collected[idx].items())}
            for idx in node_idx
        },
    }


# --------------------------------------------------------------------------- io


def stem_of(path: Path) -> str:
    """Strip the double extension: ``foo.pkg.slp`` -> ``foo``."""
    name = path.name
    if name.endswith(".pkg.slp"):
        return name[: -len(".pkg.slp")]
    return path.stem


def videos_of(frames: list, labels: sio.Labels) -> list:
    """The videos actually referenced by ``frames``, in the source's own order."""
    return [video for video in labels.videos if any(f.video is video for f in frames)]


def save_package(
    frames: list, videos: list, skeletons: list, path: Path, suggestions: list | None = None
) -> None:
    """Write a self-contained package: images embedded, source videos not restored.

    ``restore_original_videos=False`` keeps the embedded images as the backend. The
    original paths are frequently a Windows ``Z:/...`` mount that no compute node can
    read, so restoring them produces a file that loads and then cannot be trained on.
    """
    labels = sio.Labels(labeled_frames=frames, videos=videos, skeletons=skeletons)
    if suggestions is not None:
        labels.suggestions = suggestions
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(labels, str(path), embed=True, restore_original_videos=False, verbose=False)


def write_report(path: Path | None, payload: dict | list, quiet: bool = False) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    if not quiet:
        print(f"[OK] report -> {path}")


def recorded_args(args) -> dict:
    """An argparse namespace, made storable in a JSON report."""
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
