# -*- coding: utf-8 -*-
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

__all__ = ["get_complete_tracks"]


def get_complete_tracks(
    output_path: str | Path | None,
    aruco_detection: pd.DataFrame,
    sleap_detection: pd.DataFrame,
    video_file: str | Path | None = None,
    *,
    num_frames: int | None = None,
    anchor_bodypoint: int = 0,
    visualize: bool = False,
    video_out_path: str | Path | None = None,
    harvest_crops: bool = False,
    crops_output_dir: str | Path | None = None,
    crop_size: int = 128,
    max_distance: float = 90.0,
    lost_track_max_frames: int = 120,
    lost_track_max_distance: float | None = None,
    lost_track_aruco_max_distance: float | None = None,
    aruco_sleap_max_distance: float | None = None,
    start_frame: int = 0,
    harvest_interval: int = 5,
    # ---------------- NEW: debug visualization controls (no tracking logic changes)
    debug_viz: bool = False,
    debug_show_aruco: bool = False,
    debug_show_sleap: bool = False,
    debug_show_track_output: bool = False,
    debug_show_sleap_anchor_only: bool = False,
    debug_show_aruco_raw: bool = False,  # NEW: show pre-filter (raw) ArUco detections

    debug_layout: str = "stack",  # "stack" (single annotated view) or "tiles" (3-panel)
    debug_window_prefix: str = "Tracking",
    debug_resize: Tuple[int, int] = (1080, 720),
    stream_output: bool = False,
    output_batch_rows: int = 200_000,
) -> List[Dict[int, Dict[int, Tuple[float, float]]]]:
    """
    Combine ArUco + SLEAP detections into tracks and optionally write a parquet output.

    Debug visualization (NEW, optional; tracking logic unchanged)
    ------------------------------------------------------------
    - debug_viz: if True, render additional overlays of:
        * raw ArUco detections (post filter step, i.e., same aruco_arr used for tracking)
        * raw SLEAP detections (optionally anchor-only)
        * tracking output (existing overlay)
    - debug_layout:
        * "stack": one image with all enabled layers
        * "tiles": three separate views composed into one image (Aruco | SLEAP | Tracks)
    Notes:
      - Requires video_file (same as visualize/video_out/harvest)
      - Uses only copies/overlays; does not affect assignment, spawning, or containers.
    """
    if aruco_sleap_max_distance is None:
        aruco_sleap_max_distance = max_distance
    if lost_track_max_distance is None:
        lost_track_max_distance = max_distance * 2.0
    if lost_track_aruco_max_distance is None:
        lost_track_aruco_max_distance = max(lost_track_max_distance * 2.0, max_distance * 4.0)

    use_video = video_file is not None
    stream_output = bool(stream_output and output_path is not None)
    if stream_output and use_video:
        raise ValueError("stream_output is only supported for non-video tracking runs.")

    # --------------------------------------------------------- sanity checks
    if not use_video and (
        visualize or harvest_crops or video_out_path is not None or debug_viz
    ):
        raise ValueError(
            "`video_file` must be provided when `visualize`, `harvest_crops`, "
            "`video_out_path`, or `debug_viz` are enabled."
        )

    # --------------------------------------------------------- I/O prep – group detections
    if not isinstance(aruco_detection, pd.DataFrame):
        raise TypeError(f"aruco_detection must be a DataFrame (got {type(aruco_detection)})")
    if not isinstance(sleap_detection, pd.DataFrame):
        raise TypeError(f"sleap_detection must be a DataFrame (got {type(sleap_detection)})")

    if (not aruco_detection.empty) and ("Frame" not in aruco_detection.columns):
        raise ValueError("aruco_detection is missing required column 'Frame'")
    if (not sleap_detection.empty) and ("Frame" not in sleap_detection.columns):
        raise ValueError("sleap_detection is missing required column 'Frame'")

    sleap_has_cam = False
    sleap_instance_min = 0
    sleap_instance_stride = 0
    if not sleap_detection.empty and "Cam" in sleap_detection.columns:
        cam = sleap_detection["Cam"]
        inst = sleap_detection["Instance"]
        if cam.notna().all() and inst.notna().all():
            sleap_has_cam = True
            sleap_instance_min = int(inst.min())
            sleap_instance_stride = int(inst.max()) - sleap_instance_min + 1

    grouped_aruco = {f: g for f, g in aruco_detection.groupby("Frame")} if not aruco_detection.empty else {}
    grouped_sleap = {f: g for f, g in sleap_detection.groupby("Frame")} if not sleap_detection.empty else {}

    # --------------------------------------------------------- obtain frame count & video capture
    cap: Optional[cv2.VideoCapture]
    if use_video:
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_file}")
        inferred_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if num_frames is None:
            num_frames = inferred_frames
        else:
            num_frames = int(num_frames)
            if num_frames != inferred_frames:
                num_frames = inferred_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        if num_frames is None:
            if sleap_detection.empty and aruco_detection.empty:
                raise ValueError("Detection DataFrames are empty – cannot infer frame count.")
            num_frames = (
                max(
                    int(sleap_detection["Frame"].max()) if not sleap_detection.empty else -1,
                    int(aruco_detection["Frame"].max()) if not aruco_detection.empty else -1,
                )
                + 1
            )
        else:
            num_frames = int(num_frames)
        cap = None

    if start_frame < 0 or start_frame >= num_frames:
        raise ValueError(f"start_frame={start_frame} is out of bounds for num_frames={num_frames}")

    # --------------------------------------------------------- optional writer
    writer: Optional[cv2.VideoWriter] = None
    if use_video and video_out_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))

    # --------------------------------------------------------- containers
    all_pos: List[Dict[int, Dict[int, Tuple[float, float]]]] = [] if stream_output else [{} for _ in range(num_frames)]
    prev_frame_pos: Dict[int, Dict[int, Tuple[float, float]]] = {}
    last_seen_tracks: Dict[int, Tuple[int, Dict[int, Tuple[float, float]]]] = {}

    parquet_writer = None
    parquet_schema = None
    output_rows: list[Tuple[int, int, int, float, float]] = []
    output_path_obj = Path(output_path) if output_path is not None else None

    if stream_output:
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:
            raise RuntimeError("stream_output requires pyarrow") from exc

        assert output_path_obj is not None
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        parquet_schema = pa.schema(
            [
                ("Frame", pa.int64()),
                ("TrackID", pa.int64()),
                ("Bodypoint", pa.int64()),
                ("X", pa.float64()),
                ("Y", pa.float64()),
            ],
            metadata={b"num_frames": str(int(num_frames)).encode("utf-8")},
        )
        parquet_writer = pq.ParquetWriter(str(output_path_obj), parquet_schema)

    def _flush_output_rows() -> None:
        nonlocal output_rows, parquet_writer
        if not output_rows:
            return
        if parquet_writer is None or parquet_schema is None:
            return
        import pyarrow as pa  # type: ignore

        df = pd.DataFrame(output_rows, columns=["Frame", "TrackID", "Bodypoint", "X", "Y"])
        table = pa.Table.from_pandas(df, schema=parquet_schema, preserve_index=False)
        parquet_writer.write_table(table)
        output_rows = []

    def _record_frame_rows(frame_idx: int, frame_pos: Dict[int, Dict[int, Tuple[float, float]]]) -> None:
        if not stream_output:
            return
        for tid, nodes in frame_pos.items():
            for bp, (x, y) in nodes.items():
                output_rows.append((int(frame_idx), int(tid), int(bp), float(x), float(y)))
        if len(output_rows) >= int(output_batch_rows):
            _flush_output_rows()

    if harvest_crops and crops_output_dir is not None and use_video:
        Path(crops_output_dir).mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------- colour util
    def _id2bgr(tid: int) -> Tuple[int, int, int]:
        rng = np.random.default_rng(tid)
        r, g, b = (int(x) for x in rng.integers(60, 255, size=3))
        return (b, g, r)

    # ---------------- NEW: debug drawing helpers (purely visual)

    def _draw_aruco_debug(img: np.ndarray, aruco_arr: np.ndarray, *, stage: str = "post-filter") -> None:
        """
        aruco_arr: (N,3) -> columns [Instance, X, Y]
        stage: "raw" or "post-filter" (labeling only)
        """
        if aruco_arr.size == 0:
            cv2.putText(
                img,
                f"ArUco {stage}: none",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (200, 200, 200),
                2,
            )
            return

        for tag_id, x, y in aruco_arr:
            if not (np.isfinite(tag_id) and np.isfinite(x) and np.isfinite(y)):
                continue
            xi, yi = int(round(x)), int(round(y))
            cv2.circle(img, (xi, yi), 8, (0, 255, 255), 2)
            cv2.putText(
                img,
                f"tag{int(tag_id)}",
                (xi - 10, yi - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            img,
            f"ArUco ({stage}): {aruco_arr.shape[0]}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
        )

    def _draw_sleap_debug(img: np.ndarray, s_df: pd.DataFrame, sleap_anchor: np.ndarray) -> None:
        if s_df.empty:
            cv2.putText(img, "SLEAP: none", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
            return

        if debug_show_sleap_anchor_only:
            pts = sleap_anchor[:, 1:3] if sleap_anchor.size else np.empty((0, 2))
            # anchor instance id in col0
            for inst, x, y in sleap_anchor:
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(img, (int(round(x)), int(round(y))), 7, (255, 0, 255), -1)
                    cv2.putText(
                        img,
                        f"inst{int(inst)}(a{anchor_bodypoint})",
                        (int(round(x)) + 10, int(round(y)) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 255),
                        2,
                    )
            cv2.putText(img, f"SLEAP anchor-only: {len(pts)}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)
            return

        # full bodypoints
        for inst_id, inst_df in s_df.groupby("Instance"):
            col = _id2bgr(int(inst_id) + 10_000)  # keep distinct from track colours
            for bp, x, y in inst_df[["Bodypoint", "X", "Y"]].itertuples(index=False):
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(img, (int(round(x)), int(round(y))), 5, col, -1)
                    if int(bp) == anchor_bodypoint:
                        cv2.circle(img, (int(round(x)), int(round(y))), 10, (255, 0, 255), 2)
                        cv2.putText(
                            img,
                            f"inst{int(inst_id)}",
                            (int(round(x)) + 10, int(round(y)) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (255, 0, 255),
                            2,
                        )
        cv2.putText(img, f"SLEAP (all bps): {len(s_df)}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)

    def _draw_tracks(img: np.ndarray, frame_idx: int) -> None:
        for tid, nodes in all_pos[frame_idx].items():
            colour = _id2bgr(tid)
            for (x, y) in nodes.values():
                if not np.isnan(x) and not np.isnan(y):
                    cv2.circle(img, (int(x), int(y)), 8, colour, -1)
            if anchor_bodypoint in nodes:
                ax, ay = nodes[anchor_bodypoint]
                if not np.isnan(ax) and not np.isnan(ay):
                    cv2.putText(
                        img,
                        str(tid),
                        (int(ax) + 8, int(ay) + 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3,
                        colour,
                        3,
                    )

    def _compose_debug_tiles(
        base: np.ndarray,
        raw_aruco_arr: np.ndarray,
        aruco_arr: np.ndarray,
        s_df: pd.DataFrame,
        sleap_anchor: np.ndarray,
        frame_idx: int,
    ) -> np.ndarray:

        # Three side-by-side views: ArUco | SLEAP | Tracks
        h, w = base.shape[:2]
        a = base.copy()
        s = base.copy()
        t = base.copy()

        if debug_show_aruco_raw:
            _draw_aruco_debug(a, raw_aruco_arr, stage="raw")
        if debug_show_aruco:
            _draw_aruco_debug(a, aruco_arr, stage="post-filter")

        if debug_show_sleap:
            _draw_sleap_debug(s, s_df, sleap_anchor)
        if debug_show_track_output:
            _draw_tracks(t, frame_idx)

        # headers
        cv2.putText(a, "ArUco", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(s, "SLEAP", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(t, "Tracks", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        tiled = np.concatenate([a, s, t], axis=1)
        cv2.putText(
            tiled,
            f"Frame {frame_idx}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        return tiled

    # ======================================================== main loop
    for frame_idx in tqdm(range(start_frame, num_frames), unit="frame"):
        frame_pos = {} if stream_output else all_pos[frame_idx]
        # -------------------------------- load frame (if available)
        if use_video:
            ok, frame = cap.read()
            if not ok:
                break
            img = frame.copy()
        else:
            img = None

        # ---------------- detections for this frame
        a_df = grouped_aruco.get(frame_idx, pd.DataFrame())
        s_df = grouped_sleap.get(frame_idx, pd.DataFrame())

        raw_aruco_arr = (
            a_df[["Instance", "X", "Y"]].to_numpy(float) if not a_df.empty else np.empty((0, 3))
        )
        aruco_arr = raw_aruco_arr.copy()
        raw_aruco_by_tag: Dict[int, List[Tuple[float, float]]] = {}
        for tag_id, ax, ay in raw_aruco_arr:
            if not (np.isfinite(tag_id) and np.isfinite(ax) and np.isfinite(ay)):
                continue
            raw_aruco_by_tag.setdefault(int(tag_id), []).append((float(ax), float(ay)))

        if s_df.empty:
            s_instance_keys = np.empty((0,), dtype=np.int64)
            sleap_anchor = np.empty((0, 3))
        else:
            if sleap_has_cam:
                s_instance_keys = (
                    s_df["Cam"].to_numpy(dtype=np.int64, copy=False) * sleap_instance_stride
                    + s_df["Instance"].to_numpy(dtype=np.int64, copy=False)
                    - sleap_instance_min
                )
            else:
                s_instance_keys = s_df["Instance"].to_numpy(dtype=np.int64, copy=False)

            anchor_mask = s_df["Bodypoint"].to_numpy(dtype=np.int64, copy=False) == anchor_bodypoint
            if np.any(anchor_mask):
                sleap_anchor = np.column_stack(
                    [
                        s_instance_keys[anchor_mask],
                        s_df.loc[anchor_mask, "X"].to_numpy(float),
                        s_df.loc[anchor_mask, "Y"].to_numpy(float),
                    ]
                )
            else:
                sleap_anchor = np.empty((0, 3))

        # ------- ArUco too far from any anchor -> drop
        #
        # All tracking decisions below should use this post-filter ArUco set.
        # Keeping a separate raw set is useful for debug overlays, but using raw
        # detections for TrackID keep-alive lets camera-specific false positives
        # revive a tag after a gap and block the real tag from being picked up.
        if len(aruco_arr) and len(sleap_anchor):
            diff = sleap_anchor[:, 1:3][None, :, :] - aruco_arr[:, 1:3][:, None, :]
            keep = np.linalg.norm(diff, axis=-1).min(axis=1) <= aruco_sleap_max_distance
            aruco_arr = aruco_arr[keep]
        elif len(aruco_arr):
            aruco_arr = np.empty((0, 3))

        inst_aruco_id: Dict[int, int] = {}
        aruco_inst_candidates: Dict[int, List[Tuple[float, int]]] = {}
        if len(aruco_arr) and len(sleap_anchor):
            diff = aruco_arr[:, 1:3][None, :, :] - sleap_anchor[:, 1:3][:, None, :]
            dists_to_tags = np.linalg.norm(diff, axis=-1)
            nearest_tag = np.argmin(dists_to_tags, axis=1)
            nearest_dist = dists_to_tags[np.arange(len(sleap_anchor)), nearest_tag]
            for s_idx, tag_idx in enumerate(nearest_tag):
                if nearest_dist[s_idx] <= aruco_sleap_max_distance:
                    inst_id = int(sleap_anchor[s_idx, 0])
                    tag_id = int(aruco_arr[tag_idx, 0])
                    inst_aruco_id[inst_id] = tag_id
                    aruco_inst_candidates.setdefault(tag_id, []).append(
                        (float(nearest_dist[s_idx]), inst_id)
                    )

        used_tag: set[int] = set()
        used_inst: set[int] = set()
        sleap_anchor_xy_by_inst: Dict[int, Tuple[float, float]] = {
            int(inst_id): (float(x), float(y)) for inst_id, x, y in sleap_anchor
        }
        aruco_by_tag: Dict[int, List[Tuple[float, float]]] = {}
        for tag_id, ax, ay in aruco_arr:
            if not (np.isfinite(tag_id) and np.isfinite(ax) and np.isfinite(ay)):
                continue
            aruco_by_tag.setdefault(int(tag_id), []).append((float(ax), float(ay)))

        def _nodes_for_instance(inst_id: int) -> Dict[int, Tuple[float, float]]:
            inst_df = s_df[s_instance_keys == inst_id]
            return {
                int(b): (float(x), float(y))
                for b, x, y in inst_df[["Bodypoint", "X", "Y"]].itertuples(index=False)
            }

        def _assign_sleap_instance(tid: int, inst_id: int) -> bool:
            node_dict = _nodes_for_instance(inst_id)
            if anchor_bodypoint not in node_dict:
                return False
            frame_pos[tid] = node_dict
            last_seen_tracks[tid] = (frame_idx, node_dict)
            used_inst.add(inst_id)
            tag_id = inst_aruco_id.get(inst_id)
            if tag_id == int(tid):
                used_tag.add(int(tid))
            return True

        def _nearest_raw_tag_xy(
            tid: int,
            prev_xy: Tuple[float, float],
            distance_limit: float,
        ) -> Tuple[float, float] | None:
            candidates = raw_aruco_by_tag.get(int(tid), [])
            if not candidates:
                return None
            prev_arr = np.asarray(prev_xy, dtype=float)
            ax, ay = min(
                candidates,
                key=lambda xy: float(np.linalg.norm(np.asarray(xy) - prev_arr)),
            )
            dist = float(np.linalg.norm(np.asarray((ax, ay)) - prev_arr))
            if dist > float(distance_limit):
                return None
            return (float(ax), float(ay))

        # ---------------- update existing tracks
        if frame_idx > start_frame:
            prev_tracks: list[tuple[int, Dict[int, Tuple[float, float]], int, float]] = []
            seen_tids: set[int] = set()

            previous_pos = prev_frame_pos if stream_output else all_pos[frame_idx - 1]
            for tid, prev_nodes in previous_pos.items():
                prev_tracks.append((int(tid), prev_nodes, 1, float(max_distance)))
                seen_tids.add(int(tid))

            lost_frames = max(0, int(lost_track_max_frames))
            if lost_frames > 0:
                for tid, (last_frame, prev_nodes) in list(last_seen_tracks.items()):
                    tid = int(tid)
                    if tid in seen_tids:
                        continue
                    age = frame_idx - int(last_frame)
                    if 1 < age <= lost_frames:
                        prev_tracks.append(
                            (tid, prev_nodes, age, float(lost_track_max_distance))
                        )

            # ArUco identity is primary. If an existing TrackID's tag is visible
            # and matched to a SLEAP instance, use that instance even if SLEAP
            # continuity had drifted to a different ant. Duplicate same-ID tags
            # are common near camera borders; in that case, keep the candidate
            # consistent with the current track and ignore one-frame far dupes.
            for tid, prev_nodes, _age, _distance_limit in prev_tracks:
                if tid in frame_pos:
                    continue
                cands = [
                    (dist, inst_id)
                    for dist, inst_id in aruco_inst_candidates.get(int(tid), [])
                    if inst_id not in used_inst
                ]
                if not cands:
                    continue
                prev_xy = (
                    prev_nodes[anchor_bodypoint]
                    if anchor_bodypoint in prev_nodes
                    else None
                )
                raw_near_prev = (
                    _nearest_raw_tag_xy(
                        int(tid),
                        prev_xy,
                        float(lost_track_aruco_max_distance),
                    )
                    if prev_xy is not None
                    else None
                )
                if len(cands) == 1:
                    inst_id = cands[0][1]
                    anchor_xy = sleap_anchor_xy_by_inst.get(inst_id)
                    if raw_near_prev is not None and anchor_xy is not None:
                        prev_arr = np.asarray(prev_xy, dtype=float)
                        candidate_dist = float(
                            np.linalg.norm(np.asarray(anchor_xy) - prev_arr)
                        )
                        if candidate_dist > float(lost_track_aruco_max_distance):
                            continue
                elif anchor_bodypoint in prev_nodes:
                    prev_xy_arr = np.asarray(prev_nodes[anchor_bodypoint], dtype=float)
                    ranked: list[tuple[float, float, int]] = []
                    for tag_dist, cand_inst_id in cands:
                        anchor_xy = sleap_anchor_xy_by_inst.get(cand_inst_id)
                        if anchor_xy is None:
                            continue
                        prev_dist = float(np.linalg.norm(np.asarray(anchor_xy) - prev_xy_arr))
                        ranked.append((prev_dist, float(tag_dist), cand_inst_id))
                    near_prev = [
                        item
                        for item in ranked
                        if item[0] <= float(lost_track_aruco_max_distance)
                    ]
                    if not near_prev:
                        continue
                    _, _, inst_id = min(near_prev, key=lambda x: (x[0], x[1]))
                else:
                    continue
                _assign_sleap_instance(int(tid), inst_id)

            # If the same raw ArUco tag is still visible near the previous
            # position but SLEAP is missing at that border, keep the track alive
            # with the ArUco anchor only. This does not allow far raw detections
            # to revive a stale ID.
            for tid, prev_nodes, _age, _distance_limit in prev_tracks:
                if tid in frame_pos or anchor_bodypoint not in prev_nodes:
                    continue
                if int(tid) in used_tag:
                    continue
                raw_xy = _nearest_raw_tag_xy(
                    int(tid),
                    prev_nodes[anchor_bodypoint],
                    float(lost_track_aruco_max_distance),
                )
                if raw_xy is None:
                    continue
                frame_pos[int(tid)] = {anchor_bodypoint: raw_xy}
                last_seen_tracks[int(tid)] = (
                    frame_idx,
                    {anchor_bodypoint: raw_xy},
                )
                used_tag.add(int(tid))

            # Then claim the distance-consistent cases: a nearby SLEAP instance
            # whose nearest ArUco tag is the same as the existing TrackID.
            for tid, prev_nodes, _age, distance_limit in prev_tracks:
                if tid in frame_pos:
                    continue
                if anchor_bodypoint not in prev_nodes:
                    continue
                prev_xy = prev_nodes[anchor_bodypoint]
                if not len(sleap_anchor):
                    continue
                dists = np.linalg.norm(sleap_anchor[:, 1:3] - prev_xy, axis=1)
                cands: list[tuple[float, int]] = []
                for i, dist in enumerate(dists):
                    inst_id = int(sleap_anchor[i, 0])
                    if inst_id in used_inst:
                        continue
                    if dist <= distance_limit and inst_aruco_id.get(inst_id) == int(tid):
                        cands.append((float(dist), inst_id))
                if cands:
                    _, inst_id = min(cands, key=lambda x: x[0])
                    _assign_sleap_instance(int(tid), inst_id)

            # Then bridge isolated SLEAP continuity only when ArUco identity is
            # absent. A visible same-ID tag elsewhere, or a conflicting tag on a
            # nearby SLEAP instance, should not be overridden by SLEAP motion.
            for tid, prev_nodes, _age, distance_limit in prev_tracks:
                if tid in frame_pos or anchor_bodypoint not in prev_nodes:
                    continue
                if int(tid) in aruco_by_tag:
                    continue
                if not len(sleap_anchor):
                    continue
                prev_xy = prev_nodes[anchor_bodypoint]
                dists = np.linalg.norm(sleap_anchor[:, 1:3] - prev_xy, axis=1)
                cands: list[tuple[float, int]] = []
                for i, dist in enumerate(dists):
                    inst_id = int(sleap_anchor[i, 0])
                    if inst_id in used_inst:
                        continue
                    candidate_tag_id = inst_aruco_id.get(inst_id)
                    if candidate_tag_id is not None and candidate_tag_id != int(tid):
                        continue
                    if dist <= distance_limit:
                        cands.append((float(dist), inst_id))
                if len(cands) == 1:
                    _, inst_id = cands[0]
                    _assign_sleap_instance(int(tid), inst_id)

            # Finally, if the matching post-filter ArUco tag is visible but no
            # SLEAP instance could be assigned to this TrackID, keep the track
            # alive with anchor_bodypoint only. "Post-filter" means the tag was
            # near some SLEAP anchor in this frame; raw ArUco detections cannot
            # revive a track here.
            for tid, prev_nodes, age, _distance_limit in prev_tracks:
                if tid in frame_pos or anchor_bodypoint not in prev_nodes:
                    continue
                if int(tid) in used_tag or int(tid) not in aruco_by_tag:
                    continue
                prev_xy = prev_nodes[anchor_bodypoint]
                candidates = aruco_by_tag[int(tid)]
                ax, ay = min(
                    candidates,
                    key=lambda xy: float(np.linalg.norm(np.asarray(xy) - np.asarray(prev_xy))),
                )
                aruco_dist = float(np.linalg.norm(np.asarray((ax, ay)) - np.asarray(prev_xy)))
                if aruco_dist > float(lost_track_aruco_max_distance):
                    continue
                frame_pos[int(tid)] = {anchor_bodypoint: (ax, ay)}
                last_seen_tracks[int(tid)] = (
                    frame_idx,
                    {anchor_bodypoint: (ax, ay)},
                )
                used_tag.add(int(tid))

        # ---------------- spawn new tracks from unused ArUco
        for i, (tag_id, ax, ay) in enumerate(aruco_arr):
            tag_id = int(tag_id)
            if tag_id in used_tag or tag_id in frame_pos:
                continue
            if len(sleap_anchor) == 0:
                continue

            dists = np.linalg.norm(sleap_anchor[:, 1:3] - (ax, ay), axis=1)
            j = int(np.argmin(dists))
            if dists[j] > aruco_sleap_max_distance:
                continue

            inst_id = int(sleap_anchor[j, 0])
            if inst_id in used_inst:
                continue

            inst_df = s_df[s_instance_keys == inst_id]
            node_dict = {
                int(b): (float(x), float(y))
                for b, x, y in inst_df[["Bodypoint", "X", "Y"]].itertuples(index=False)
            }
            if anchor_bodypoint not in node_dict:
                continue

            recent = last_seen_tracks.get(tag_id)
            if recent is not None:
                last_frame, prev_nodes = recent
                age = frame_idx - int(last_frame)
                if age <= lost_frames and anchor_bodypoint in prev_nodes:
                    distance_limit = (
                        float(max_distance)
                        if age <= 1
                        else float(lost_track_max_distance)
                    )
                    prev_xy = prev_nodes[anchor_bodypoint]
                    spawn_dist = float(
                        np.linalg.norm(
                            np.asarray(node_dict[anchor_bodypoint]) - np.asarray(prev_xy)
                        )
                    )
                    if spawn_dist > distance_limit:
                        continue

            used_inst.add(inst_id)
            used_tag.add(tag_id)
            tid = tag_id
            frame_pos[tid] = node_dict
            last_seen_tracks[tid] = (frame_idx, node_dict)

        lost_frames = max(0, int(lost_track_max_frames))
        if lost_frames > 0:
            stale_tids = [
                tid
                for tid, (last_frame, _nodes) in last_seen_tracks.items()
                if frame_idx - int(last_frame) > lost_frames
            ]
            for tid in stale_tids:
                last_seen_tracks.pop(tid, None)

        # ---------------- crops
        if (
            harvest_crops
            and crops_output_dir is not None
            and use_video
            and frame_idx % harvest_interval == 0
        ):
            half = crop_size // 2
            for tid, nodes in all_pos[frame_idx].items():
                if anchor_bodypoint not in nodes:
                    continue
                cx, cy = map(int, map(round, nodes[anchor_bodypoint]))
                xmin, ymin = max(cx - half, 0), max(cy - half, 0)
                xmax, ymax = min(xmin + crop_size, img.shape[1]), min(ymin + crop_size, img.shape[0])
                crop = img[ymin:ymax, xmin:xmax]
                if crop.shape[:2] != (crop_size, crop_size):
                    crop = cv2.resize(crop, (crop_size, crop_size))
                out_dir = Path(crops_output_dir) / str(tid)
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / f"{frame_idx}_{uuid.uuid4().hex}.png"), crop)

        # ---------------- visualisation (existing + debug)
        if use_video and (visualize or writer is not None or debug_viz):
            base = img.copy()

            # Existing output overlay (kept, unchanged)
            disp_tracks = base.copy()
            _draw_tracks(disp_tracks, frame_idx)
            cv2.putText(
                disp_tracks,
                f"Frame {frame_idx}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 255, 255),
                3,
            )

            if debug_viz:
                if debug_layout not in {"stack", "tiles"}:
                    raise ValueError("debug_layout must be 'stack' or 'tiles'")

                if debug_layout == "tiles":
                    disp = _compose_debug_tiles(base, raw_aruco_arr, aruco_arr, s_df, sleap_anchor, frame_idx)

                else:
                    # stack: single frame with multiple overlays
                    disp = base.copy()

                    if debug_show_aruco_raw:
                        _draw_aruco_debug(disp, raw_aruco_arr, stage="raw")
                    if debug_show_aruco:
                        _draw_aruco_debug(disp, aruco_arr, stage="post-filter")

                    if debug_show_sleap:
                        _draw_sleap_debug(disp, s_df, sleap_anchor)
                    if debug_show_track_output:
                        _draw_tracks(disp, frame_idx)

                    cv2.putText(
                        disp,
                        f"Frame {frame_idx}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                    )
            else:
                # Preserve original behavior: show/write tracking overlay only
                disp = disp_tracks

            if visualize or debug_viz:
                win = f"{debug_window_prefix}"
                cv2.imshow(win, cv2.resize(disp, debug_resize))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer is not None:
                # If debug_viz enabled, write the debug composite; otherwise original overlay.
                writer.write(disp if debug_viz else disp_tracks)

        if stream_output:
            _record_frame_rows(frame_idx, frame_pos)
            prev_frame_pos = frame_pos

    # ======================================================== cleanup
    if use_video and cap is not None:
        cap.release()
        if visualize or debug_viz:
            cv2.destroyAllWindows()
        if writer is not None:
            writer.release()

    # ---------------- save tracks
    if stream_output:
        _flush_output_rows()
        if parquet_writer is not None:
            parquet_writer.close()
    elif output_path is not None:
        rows: list[Tuple[int, int, int, float, float]] = []
        for f, posdict in enumerate(all_pos):
            for tid, nodes in posdict.items():
                for bp, (x, y) in nodes.items():
                    rows.append((f, tid, bp, x, y))

        df = pd.DataFrame(rows, columns=["Frame", "TrackID", "Bodypoint", "X", "Y"])
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)

        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore

            table = pa.Table.from_pandas(df, preserve_index=False)
            md = dict(table.schema.metadata or {})
            md[b"num_frames"] = str(int(num_frames)).encode("utf-8")
            table = table.replace_schema_metadata(md)
            pq.write_table(table, str(outp))
        except Exception:
            df.to_parquet(outp)

    return all_pos
