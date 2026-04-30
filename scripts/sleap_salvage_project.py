#!/usr/bin/env python3
"""Build, split, and merge SLEAP rescue projects from problematic-frame CSVs.

The default build path uses the flattened ``*_sleap_data.h5`` files referenced by the
CSV because loading every full source ``.slp`` can be very slow. It reads the skeleton
from a sibling ``.slp`` once, reconstructs predicted instances for only the requested
frames, and writes SLEAP projects with video references rewritten to usable paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import sleap_io as sio
from sleap_io.io.slp import read_skeletons
from sleap_io.model.instance import Instance, PredictedInstance
from sleap_io.model.labeled_frame import LabeledFrame
from sleap_io.model.skeleton import Skeleton
from sleap_io.model.suggestions import SuggestionFrame
from sleap_io.model.video import Video


DEFAULT_PATH_MAPS = (("/home/sam-reiter/bucket", "Z:/"),)


@dataclass(frozen=True)
class ProblemRow:
    row_index: int
    source_h5: Path
    source_slp: Path
    video_path: Path
    frame_idx: int
    raw: dict[str, str]


def parse_path_maps(specs: Iterable[str] | None) -> list[tuple[str, str]]:
    maps = list(DEFAULT_PATH_MAPS)
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"Path map must be OLD=NEW, got: {spec}")
        old, new = spec.split("=", 1)
        maps.append((old.rstrip("/\\"), new.rstrip("/\\")))
    return maps


def mapped_path(value: str, path_maps: list[tuple[str, str]]) -> Path:
    text = value.strip()
    for old, new in path_maps:
        if text.startswith(old):
            rest = text[len(old) :].lstrip("/\\")
            text = f"{new}/{rest}" if rest else new
            break
    return Path(text.replace("/", "\\"))


def h5_to_slp_path(path: Path) -> Path:
    if path.name.endswith("_sleap_data.h5"):
        return path.with_name(path.name[: -len("_sleap_data.h5")] + ".slp")
    if path.suffix.lower() == ".slp":
        return path
    return path.with_suffix(".slp")


def read_problem_csv(
    csv_path: Path,
    path_maps: list[tuple[str, str]],
    limit_rows: int | None = None,
) -> list[ProblemRow]:
    rows: list[ProblemRow] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"sleap", "video", "frame"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for i, raw in enumerate(reader):
            if limit_rows is not None and len(rows) >= limit_rows:
                break
            source_h5 = mapped_path(raw["sleap"], path_maps)
            rows.append(
                ProblemRow(
                    row_index=i,
                    source_h5=source_h5,
                    source_slp=h5_to_slp_path(source_h5),
                    video_path=mapped_path(raw["video"], path_maps),
                    frame_idx=int(raw["frame"]),
                    raw=raw,
                )
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def validate_problem_rows(rows: list[ProblemRow]) -> tuple[list[dict], list[dict]]:
    resolved: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()

    for row in rows:
        key = (str(row.source_h5), str(row.video_path), row.frame_idx)
        duplicate = key in seen
        seen.add(key)

        record = {
            **row.raw,
            "row_index": row.row_index,
            "source_h5_win": str(row.source_h5),
            "source_slp_win": str(row.source_slp),
            "video_win": str(row.video_path),
            "frame_idx": row.frame_idx,
            "source_h5_exists": row.source_h5.exists(),
            "source_slp_exists": row.source_slp.exists(),
            "video_exists": row.video_path.exists(),
            "duplicate_key": duplicate,
        }
        resolved.append(record)

        if (
            not record["source_h5_exists"]
            or not record["source_slp_exists"]
            or not record["video_exists"]
            or duplicate
        ):
            unmatched.append(record)

    return resolved, unmatched


def load_canonical_skeleton(rows: list[ProblemRow], skeleton_slp: Path | None) -> Skeleton:
    source = skeleton_slp or rows[0].source_slp
    skeletons = read_skeletons(str(source))
    if not skeletons:
        raise ValueError(f"No skeletons found in {source}")
    return skeletons[0]


def make_video(video_path: Path, open_video: bool = True) -> Video:
    if open_video:
        try:
            return Video.from_filename(str(video_path), keep_open=False)
        except Exception as exc:
            print(
                f"warning: could not open video metadata for {video_path}: {exc}",
                file=sys.stderr,
            )
    return Video(filename=str(video_path), open_backend=False)


def clone_instance(
    inst: Instance | PredictedInstance,
    skeleton: Skeleton,
    predicted_map: dict[int, PredictedInstance] | None = None,
) -> Instance | PredictedInstance:
    if type(inst) is PredictedInstance:
        cloned = PredictedInstance(
            points=inst.points.copy(),
            skeleton=skeleton,
            score=inst.score,
            track=inst.track,
            tracking_score=inst.tracking_score,
            from_predicted=None,
        )
        if predicted_map is not None:
            predicted_map[id(inst)] = cloned
        return cloned

    from_predicted = None
    if inst.from_predicted is not None and predicted_map is not None:
        from_predicted = predicted_map.get(id(inst.from_predicted))
    return Instance(
        points=inst.points.copy(),
        skeleton=skeleton,
        track=inst.track,
        tracking_score=inst.tracking_score,
        from_predicted=from_predicted,
    )


def clone_labeled_frame(
    lf: LabeledFrame,
    video: Video,
    skeleton: Skeleton,
) -> LabeledFrame:
    predicted_map: dict[int, PredictedInstance] = {}
    cloned_instances: list[Instance | PredictedInstance] = []

    for inst in lf.instances:
        if type(inst) is PredictedInstance:
            cloned_instances.append(clone_instance(inst, skeleton, predicted_map))
    for inst in lf.instances:
        if type(inst) is not PredictedInstance:
            cloned_instances.append(clone_instance(inst, skeleton, predicted_map))

    return LabeledFrame(
        video=video,
        frame_idx=lf.frame_idx,
        instances=cloned_instances,
    )


def skeleton_neighbors(skeleton: Skeleton) -> dict[int, list[int]]:
    neighbors = {i: [] for i in range(len(skeleton))}
    for edge in skeleton.edges:
        src = skeleton.index(edge.source)
        dst = skeleton.index(edge.destination)
        neighbors[src].append(dst)
        neighbors[dst].append(src)
    return neighbors


def hidden_offset(node_idx: int, offset_px: float) -> np.ndarray:
    angle = 2.399963229728653 * (node_idx + 1)
    return np.array([math.cos(angle), math.sin(angle)]) * offset_px


def place_hidden_points_near_visible(
    points: np.ndarray,
    skeleton: Skeleton,
    offset_px: float = 8.0,
    mode: str = "nearest",
) -> tuple[np.ndarray, np.ndarray]:
    """Give invisible points finite nearby coordinates while keeping visibility false."""
    placed = np.asarray(points, dtype=np.float64).copy()
    visible = np.isfinite(placed[:, 0]) & np.isfinite(placed[:, 1])
    if mode == "nan" or visible.all():
        return placed, visible

    visible_points = placed[visible]
    if len(visible_points):
        centroid = np.nanmean(visible_points, axis=0)
    else:
        centroid = np.array([0.0, 0.0], dtype=np.float64)

    neighbors = skeleton_neighbors(skeleton)
    for node_idx in np.where(~visible)[0].tolist():
        anchor: np.ndarray | None = None
        if mode == "nearest":
            frontier = list(neighbors.get(node_idx, []))
            seen = {node_idx}
            while frontier and anchor is None:
                next_frontier: list[int] = []
                for other_idx in frontier:
                    if other_idx in seen:
                        continue
                    seen.add(other_idx)
                    if visible[other_idx]:
                        anchor = placed[other_idx]
                        break
                    next_frontier.extend(neighbors.get(other_idx, []))
                frontier = next_frontier

        if anchor is None:
            anchor = centroid
        placed[node_idx] = anchor + hidden_offset(node_idx, offset_px)

    return placed, visible


def set_instance_visibility(inst: Instance | PredictedInstance, visible: np.ndarray) -> None:
    inst.points["visible"] = visible
    inst.points["complete"] = visible


def promote_predictions(
    lf: LabeledFrame,
    hidden_placement: str = "nearest",
    hidden_offset_px: float = 8.0,
) -> None:
    existing = len(lf.user_instances)
    if existing:
        return
    for pred in list(lf.predicted_instances):
        points, visible = place_hidden_points_near_visible(
            pred.numpy(),
            pred.skeleton,
            offset_px=hidden_offset_px,
            mode=hidden_placement,
        )
        user = Instance.from_numpy(
            points_data=points,
            skeleton=pred.skeleton,
            track=pred.track,
            tracking_score=pred.tracking_score,
            from_predicted=pred,
        )
        set_instance_visibility(user, visible)
        lf.instances.append(user)


def labels_from_h5_frames(
    grouped_rows: dict[Path, list[ProblemRow]],
    skeleton: Skeleton,
    promote: bool,
    open_videos: bool,
    hidden_placement: str,
    hidden_offset_px: float,
) -> tuple[sio.Labels, list[dict[str, object]]]:
    labels = sio.Labels(skeletons=[skeleton])
    video_by_path: dict[Path, Video] = {}
    missing_frames: list[dict[str, object]] = []

    for source_i, (h5_path, rows) in enumerate(sorted(grouped_rows.items())):
        frames = sorted({row.frame_idx for row in rows})
        print(
            f"[{source_i + 1}/{len(grouped_rows)}] reading {h5_path.name} "
            f"for {len(frames)} frames",
            file=sys.stderr,
        )
        with h5py.File(h5_path, "r") as f:
            data = f["sleap_data"][:]

        selected = data[np.isin(data["Frame"], np.array(frames, dtype=data["Frame"].dtype))]
        video_path = rows[0].video_path
        video = video_by_path.setdefault(video_path, make_video(video_path, open_videos))

        for frame_idx in frames:
            frame_data = selected[selected["Frame"] == frame_idx]
            if len(frame_data) == 0:
                missing_frames.append(
                    {
                        "source_h5_win": str(h5_path),
                        "video_win": str(video_path),
                        "frame_idx": frame_idx,
                        "reason": "frame_not_found_in_h5",
                    }
                )
                continue

            instances: list[PredictedInstance] = []
            for inst_id in sorted(np.unique(frame_data["Instance"]).tolist()):
                inst_rows = frame_data[frame_data["Instance"] == inst_id]
                points = np.full((len(skeleton), 2), np.nan, dtype=np.float64)
                scores = np.zeros((len(skeleton),), dtype=np.float64)
                for rec in inst_rows:
                    bp = int(rec["Bodypoint"])
                    if 0 <= bp < len(skeleton):
                        points[bp, 0] = float(rec["X"])
                        points[bp, 1] = float(rec["Y"])
                        score = float(rec["Score_node"])
                        scores[bp] = 0.0 if math.isnan(score) else score
                finite_scores = scores[np.isfinite(scores) & (scores > 0)]
                inst_score = float(np.mean(finite_scores)) if len(finite_scores) else 0.0
                placed_points, visible = place_hidden_points_near_visible(
                    points,
                    skeleton,
                    offset_px=hidden_offset_px,
                    mode=hidden_placement,
                )
                pred = PredictedInstance.from_numpy(
                    points_data=placed_points,
                    skeleton=skeleton,
                    point_scores=scores,
                    score=inst_score,
                )
                set_instance_visibility(pred, visible)
                instances.append(pred)

            lf = LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
            if promote:
                promote_predictions(
                    lf,
                    hidden_placement=hidden_placement,
                    hidden_offset_px=hidden_offset_px,
                )
            labels.append(lf)
            labels.suggestions.append(
                SuggestionFrame(
                    video=video,
                    frame_idx=frame_idx,
                    metadata={"source_h5": str(h5_path), "reason": "problem_csv"},
                )
            )

    labels.update()
    return labels, missing_frames


def labels_from_slp_frames(
    grouped_rows: dict[Path, list[ProblemRow]],
    skeleton: Skeleton,
    promote: bool,
    open_videos: bool,
    hidden_placement: str,
    hidden_offset_px: float,
) -> tuple[sio.Labels, list[dict[str, object]]]:
    labels = sio.Labels(skeletons=[skeleton])
    video_by_path: dict[Path, Video] = {}
    missing_frames: list[dict[str, object]] = []

    for source_i, (slp_path, rows) in enumerate(sorted(grouped_rows.items())):
        print(
            f"[{source_i + 1}/{len(grouped_rows)}] loading {slp_path.name}",
            file=sys.stderr,
        )
        source = sio.load_slp(str(slp_path), open_videos=False)
        if not source.skeleton.matches(skeleton, require_same_order=False):
            raise ValueError(f"Skeleton mismatch in {slp_path}")

        source_video = source.videos[0]
        video_path = rows[0].video_path
        video = video_by_path.setdefault(video_path, make_video(video_path, open_videos))

        for row in sorted(rows, key=lambda r: r.frame_idx):
            matches = source.find(source_video, row.frame_idx)
            if not matches:
                missing_frames.append(
                    {
                        "source_slp_win": str(slp_path),
                        "video_win": str(video_path),
                        "frame_idx": row.frame_idx,
                        "reason": "frame_not_found_in_slp",
                    }
                )
                continue
            lf = clone_labeled_frame(matches[0], video=video, skeleton=skeleton)
            if promote:
                promote_predictions(
                    lf,
                    hidden_placement=hidden_placement,
                    hidden_offset_px=hidden_offset_px,
                )
            labels.append(lf)
            labels.suggestions.append(
                SuggestionFrame(
                    video=video,
                    frame_idx=row.frame_idx,
                    metadata={"source_slp": str(slp_path), "reason": "problem_csv"},
                )
            )

    labels.update()
    return labels, missing_frames


def command_inspect(args: argparse.Namespace) -> int:
    rows = read_problem_csv(
        args.problem_csv,
        parse_path_maps(args.path_map),
        limit_rows=args.limit_rows,
    )
    resolved, unmatched = validate_problem_rows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.source_h5)].append(row)

    out_dir = args.out_dir or args.problem_csv.parent
    write_csv(out_dir / "manifest_resolved.csv", resolved, list(resolved[0].keys()))
    if unmatched:
        write_csv(out_dir / "unmatched_rows.csv", unmatched, list(unmatched[0].keys()))

    summary = {
        "rows": len(rows),
        "unique_sources": len(grouped),
        "unique_videos": len({str(row.video_path) for row in rows}),
        "min_frames_per_source": min(len(v) for v in grouped.values()) if grouped else 0,
        "max_frames_per_source": max(len(v) for v in grouped.values()) if grouped else 0,
        "unmatched_rows": len(unmatched),
    }
    write_json(out_dir / "inspect_summary.json", summary)
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 1 if unmatched else 0


def command_build_master(args: argparse.Namespace) -> int:
    rows = read_problem_csv(
        args.problem_csv,
        parse_path_maps(args.path_map),
        limit_rows=args.limit_rows,
    )
    resolved, unmatched = validate_problem_rows(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out.parent / "manifest_resolved.csv", resolved, list(resolved[0].keys()))
    if unmatched:
        write_csv(args.out.parent / "unmatched_rows.csv", unmatched, list(unmatched[0].keys()))
        raise RuntimeError(f"Refusing to build with {len(unmatched)} unresolved CSV rows.")
    if args.dry_run:
        print("dry run complete; no SLP written", file=sys.stderr)
        return 0

    skeleton = load_canonical_skeleton(rows, args.skeleton_slp)
    if args.source_mode == "h5":
        grouped = defaultdict(list)
        for row in rows:
            grouped[row.source_h5].append(row)
        labels, missing_frames = labels_from_h5_frames(
            grouped,
            skeleton=skeleton,
            promote=args.promote_predictions,
            open_videos=not args.no_open_videos,
            hidden_placement=args.hidden_placement,
            hidden_offset_px=args.hidden_offset_px,
        )
    else:
        grouped = defaultdict(list)
        for row in rows:
            grouped[row.source_slp].append(row)
        labels, missing_frames = labels_from_slp_frames(
            grouped,
            skeleton=skeleton,
            promote=args.promote_predictions,
            open_videos=not args.no_open_videos,
            hidden_placement=args.hidden_placement,
            hidden_offset_px=args.hidden_offset_px,
        )

    if missing_frames:
        write_csv(
            args.out.parent / "missing_frames.csv",
            missing_frames,
            list(missing_frames[0].keys()),
        )
        raise RuntimeError(f"Refusing to save with {len(missing_frames)} missing frames.")

    labels.provenance["sleap_salvage_project"] = {
        "problem_csv": str(args.problem_csv),
        "source_mode": args.source_mode,
        "promote_predictions": args.promote_predictions,
        "hidden_placement": args.hidden_placement,
        "hidden_offset_px": args.hidden_offset_px,
        "n_rows": len(rows),
    }
    labels.save(str(args.out), embed="source", verbose=not args.quiet)

    summary = summarize_labels(labels)
    summary.update({"output": str(args.out), "source_mode": args.source_mode})
    write_json(args.out.parent / "build_master_summary.json", summary)
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0


def video_identity(video: Video) -> str:
    source = getattr(video, "original_video", None) or getattr(video, "source_video", None)
    filename = source.filename if source is not None else video.filename
    if isinstance(filename, list):
        filename = filename[0]
    return Path(str(filename)).name


def clone_labels_subset(
    labels: sio.Labels,
    frames: list[LabeledFrame],
    skeleton: Skeleton | None = None,
) -> sio.Labels:
    skeleton = skeleton or labels.skeleton
    out = sio.Labels(skeletons=[skeleton])
    for lf in sorted(frames, key=lambda x: (video_identity(x.video), x.frame_idx)):
        out.append(clone_labeled_frame(lf, video=lf.video, skeleton=skeleton))
    frame_keys = {(lf.video, lf.frame_idx) for lf in frames}
    out.suggestions = [
        SuggestionFrame(video=sf.video, frame_idx=sf.frame_idx, metadata=dict(sf.metadata))
        for sf in labels.suggestions
        if (sf.video, sf.frame_idx) in frame_keys
    ]
    out.update()
    return out


def embed_frames_with_context(
    frames: list[LabeledFrame],
    context_frames: int,
) -> list[tuple[Video, int]]:
    embed: set[tuple[Video, int]] = set()
    for lf in frames:
        video_len = len(lf.video)
        start = max(0, lf.frame_idx - context_frames)
        stop = min(video_len - 1, lf.frame_idx + context_frames)
        for frame_idx in range(start, stop + 1):
            embed.add((lf.video, frame_idx))
    return sorted(embed, key=lambda item: (video_identity(item[0]), item[1]))


def add_context_suggestions(
    labels: sio.Labels,
    frames: list[LabeledFrame],
    context_frames: int,
) -> None:
    if context_frames <= 0:
        return

    target_keys = {(lf.video, lf.frame_idx) for lf in frames}
    existing = {(sf.video, sf.frame_idx) for sf in labels.suggestions}

    for video, frame_idx in embed_frames_with_context(frames, context_frames):
        key = (video, frame_idx)
        if key in existing:
            continue
        role = "target" if key in target_keys else "context"
        labels.suggestions.append(
            SuggestionFrame(
                video=video,
                frame_idx=frame_idx,
                metadata={"role": role, "context_frames_each_side": context_frames},
            )
        )
        existing.add(key)


def summarize_labels(labels: sio.Labels) -> dict[str, int]:
    return {
        "labeled_frames": len(labels.labeled_frames),
        "videos": len(labels.videos),
        "skeletons": len(labels.skeletons),
        "suggestions": len(labels.suggestions),
        "predicted_instances": sum(len(lf.predicted_instances) for lf in labels),
        "user_instances": sum(len(lf.user_instances) for lf in labels),
    }


def command_split(args: argparse.Namespace) -> int:
    labels = sio.load_slp(str(args.master), open_videos=args.package)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames_by_video: dict[str, list[LabeledFrame]] = defaultdict(list)
    for lf in labels:
        frames_by_video[video_identity(lf.video)].append(lf)

    chunks: list[list[LabeledFrame]] = [[] for _ in range(args.chunks)]
    counts = [0] * args.chunks
    for _, frames in sorted(frames_by_video.items()):
        target = min(range(args.chunks), key=lambda i: counts[i])
        chunks[target].extend(frames)
        counts[target] += len(frames)

    manifest_rows: list[dict[str, object]] = []
    for i, frames in enumerate(chunks, start=1):
        chunk_labels = clone_labels_subset(labels, frames)
        suffix = ".pkg.slp" if args.package else ".slp"
        out_path = args.out_dir / f"{args.prefix}_chunk{i:02d}{suffix}"
        if args.package:
            if args.context_as_suggestions:
                add_context_suggestions(chunk_labels, frames, args.context_frames)
            embed = embed_frames_with_context(frames, args.context_frames)
        else:
            embed = "source"
        chunk_labels.save(str(out_path), embed=embed, verbose=not args.quiet)
        embedded_count = len(embed) if isinstance(embed, list) else 0
        for lf in frames:
            manifest_rows.append(
                {
                    "chunk": i,
                    "chunk_file": str(out_path),
                    "video": video_identity(lf.video),
                    "frame_idx": lf.frame_idx,
                    "predicted_instances": len(lf.predicted_instances),
                    "user_instances": len(lf.user_instances),
                    "context_frames_each_side": args.context_frames if args.package else 0,
                }
            )
        print(
            f"wrote {out_path} ({len(frames)} labeled frames"
            + (f", {embedded_count} embedded video frames" if args.package else "")
            + ")",
            file=sys.stderr,
        )

    write_csv(
        args.out_dir / "chunk_manifest.csv",
        manifest_rows,
        [
            "chunk",
            "chunk_file",
            "video",
            "frame_idx",
            "predicted_instances",
            "user_instances",
            "context_frames_each_side",
        ],
    )
    write_json(
        args.out_dir / "split_summary.json",
        {
            "master": str(args.master),
            "chunks": args.chunks,
            "package": args.package,
            "context_frames": args.context_frames if args.package else 0,
            "frames_per_chunk": counts,
        },
    )
    return 0


def labels_frame_index(labels: sio.Labels) -> dict[tuple[str, int], LabeledFrame]:
    return {(video_identity(lf.video), lf.frame_idx): lf for lf in labels}


def command_merge(args: argparse.Namespace) -> int:
    master = sio.load_slp(str(args.master), open_videos=False)
    merged = clone_labels_subset(master, list(master.labeled_frames))
    index = labels_frame_index(merged)
    seen: dict[tuple[str, int], Path] = {}
    report: list[dict[str, object]] = []

    files = sorted(args.corrected_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern} in {args.corrected_dir}")

    for path in files:
        labels = sio.load_slp(str(path), open_videos=False)
        for lf in labels:
            key = (video_identity(lf.video), lf.frame_idx)
            status = "merged"
            if key not in index:
                status = "not_in_master"
            elif key in seen and args.duplicate_policy == "keep-first":
                status = "duplicate_skipped"
            elif key in seen and args.duplicate_policy == "error":
                raise RuntimeError(f"Duplicate returned frame {key} in {path} and {seen[key]}")
            else:
                target = index[key]
                target.instances = clone_labeled_frame(
                    lf,
                    video=target.video,
                    skeleton=merged.skeleton,
                ).instances
                seen[key] = path

            report.append(
                {
                    "file": str(path),
                    "video": key[0],
                    "frame_idx": key[1],
                    "status": status,
                    "predicted_instances": len(lf.predicted_instances),
                    "user_instances": len(lf.user_instances),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.save(str(args.out), embed="source", verbose=not args.quiet)

    training_labels = clone_labels_subset(merged, list(merged.labeled_frames))
    training_labels.remove_predictions(clean=True)
    if args.training_out:
        training_labels.save(str(args.training_out), embed="source", verbose=not args.quiet)

    write_csv(
        args.out.parent / "merge_report.csv",
        report,
        ["file", "video", "frame_idx", "status", "predicted_instances", "user_instances"],
    )
    summary = {
        "corrected_files": len(files),
        "master_frames": len(master.labeled_frames),
        "returned_unique_frames": len(seen),
        "merged": summarize_labels(merged),
        "training_user_only": summarize_labels(training_labels),
        "out": str(args.out),
        "training_out": str(args.training_out) if args.training_out else None,
    }
    write_json(args.out.parent / "merge_summary.json", summary)
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common_csv = argparse.ArgumentParser(add_help=False)
    common_csv.add_argument("problem_csv", type=Path)
    common_csv.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Additional OLD=NEW path rewrite. Default includes /home/sam-reiter/bucket=Z:/",
    )
    common_csv.add_argument("--limit-rows", type=int, default=None)

    p = sub.add_parser("inspect", parents=[common_csv], help="Audit CSV path resolution.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.set_defaults(func=command_inspect)

    p = sub.add_parser("build-master", parents=[common_csv], help="Build rescue master SLP.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-mode", choices=["h5", "slp"], default="h5")
    p.add_argument("--skeleton-slp", type=Path, default=None)
    p.add_argument("--promote-predictions", action="store_true")
    p.add_argument(
        "--hidden-placement",
        choices=["nearest", "centroid", "nan"],
        default="nearest",
        help=(
            "Where to place invisible/unpredicted keypoints while keeping them hidden. "
            "'nearest' uses the nearest visible skeleton neighbor, 'centroid' uses the "
            "instance centroid, and 'nan' preserves old NaN behavior."
        ),
    )
    p.add_argument(
        "--hidden-offset-px",
        type=float,
        default=8.0,
        help="Pixel offset used when placing hidden keypoints near a visible anchor.",
    )
    p.add_argument("--no-open-videos", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=command_build_master)

    p = sub.add_parser("split", help="Split a master rescue SLP into colleague chunks.")
    p.add_argument("master", type=Path)
    p.add_argument("--chunks", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="problem_frames")
    p.add_argument("--package", action="store_true", help="Embed selected frames as .pkg.slp chunks.")
    p.add_argument(
        "--context-frames",
        type=int,
        default=0,
        help=(
            "When packaging, also embed this many neighboring frames before and after "
            "each labeled frame. These frames are available for visual checking but "
            "are not added as annotation targets."
        ),
    )
    p.add_argument(
        "--context-as-suggestions",
        action="store_true",
        help=(
            "Also add embedded context frames as empty SLEAP suggestions so they are "
            "visible in the GUI's suggestion navigation. Only original problem frames "
            "have labels/predictions."
        ),
    )
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=command_split)

    p = sub.add_parser("merge", help="Merge corrected chunks back into one SLP.")
    p.add_argument("--master", type=Path, required=True)
    p.add_argument("--corrected-dir", type=Path, required=True)
    p.add_argument("--pattern", default="*.slp")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--training-out", type=Path, default=None)
    p.add_argument(
        "--duplicate-policy",
        choices=["keep-last", "keep-first", "error"],
        default="keep-last",
    )
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=command_merge)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
