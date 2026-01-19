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
    max_distance: float = 50.0,
    aruco_sleap_max_distance: float | None = None,
    start_frame: int = 0,
    harvest_interval: int = 5,
    # ---------------- NEW: debug visualization controls (no tracking logic changes)
    debug_viz: bool = False,
    debug_show_aruco: bool = False,
    debug_show_sleap: bool = False,
    debug_show_track_output: bool = False,
    debug_show_sleap_anchor_only: bool = False,
    debug_layout: str = "stack",  # "stack" (single annotated view) or "tiles" (3-panel)
    debug_window_prefix: str = "Tracking",
    debug_resize: Tuple[int, int] = (1080, 720),
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

    use_video = video_file is not None

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
    all_pos: List[Dict[int, Dict[int, Tuple[float, float]]]] = [{} for _ in range(num_frames)]
    tracked_anchor_xy: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}

    if harvest_crops and crops_output_dir is not None and use_video:
        Path(crops_output_dir).mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------- colour util
    def _id2bgr(tid: int) -> Tuple[int, int, int]:
        rng = np.random.default_rng(tid)
        r, g, b = (int(x) for x in rng.integers(60, 255, size=3))
        return (b, g, r)

    # ---------------- NEW: debug drawing helpers (purely visual)
    def _draw_points(
        img: np.ndarray,
        pts_xy: np.ndarray,
        *,
        label: str,
        colour: Tuple[int, int, int],
        radius: int = 6,
        thickness: int = -1,
    ) -> None:
        if pts_xy.size == 0:
            return
        for x, y in pts_xy:
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(img, (int(round(x)), int(round(y))), radius, colour, thickness)
        cv2.putText(
            img,
            label,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            colour,
            3,
        )

    def _draw_aruco_debug(img: np.ndarray, aruco_arr: np.ndarray) -> None:
        # aruco_arr: (N,3) -> columns [Instance, X, Y]
        if aruco_arr.size == 0:
            cv2.putText(img, "ArUco: none", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 2)
            return
        pts = aruco_arr[:, 1:3]
        for tag_id, x, y in aruco_arr:
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(img, (int(round(x)), int(round(y))), 8, (0, 255, 255), 2)
                cv2.putText(
                    img,
                    f"tag{int(tag_id)}",
                    (int(round(x)) + 10, int(round(y)) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 255),
                    2,
                )
        cv2.putText(img, f"ArUco (post-filter): {len(pts)}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

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

    def _compose_debug_tiles(base: np.ndarray, aruco_arr: np.ndarray, s_df: pd.DataFrame, sleap_anchor: np.ndarray, frame_idx: int) -> np.ndarray:
        # Three side-by-side views: ArUco | SLEAP | Tracks
        h, w = base.shape[:2]
        a = base.copy()
        s = base.copy()
        t = base.copy()

        if debug_show_aruco:
            _draw_aruco_debug(a, aruco_arr)
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

        aruco_arr = (
            a_df[["Instance", "X", "Y"]].to_numpy(float) if not a_df.empty else np.empty((0, 3))
        )

        if s_df.empty:
            sleap_anchor = np.empty((0, 3))
        else:
            s_anchor = s_df[s_df["Bodypoint"] == anchor_bodypoint]
            sleap_anchor = s_anchor[["Instance", "X", "Y"]].to_numpy(float)

        # ------- ArUco too far from any anchor → drop
        if len(aruco_arr) and len(sleap_anchor):
            diff = sleap_anchor[:, 1:3][None, :, :] - aruco_arr[:, 1:3][:, None, :]
            keep = np.linalg.norm(diff, axis=-1).min(axis=1) <= aruco_sleap_max_distance
            aruco_arr = aruco_arr[keep]
        elif len(sleap_anchor) == 0:
            aruco_arr = np.empty((0, 3))

        used_tag: set[int] = set()
        used_inst: set[int] = set()

        # ---------------- update existing tracks
        if frame_idx > start_frame:
            for tid, prev_nodes in all_pos[frame_idx - 1].items():
                if anchor_bodypoint not in prev_nodes:
                    continue
                prev_xy = prev_nodes[anchor_bodypoint]
                assigned_inst = None

                if len(sleap_anchor):
                    dists = np.linalg.norm(sleap_anchor[:, 1:3] - prev_xy, axis=1)
                    cands = [
                        i
                        for i in np.where(dists <= max_distance)[0]
                        if int(sleap_anchor[i, 0]) not in used_inst
                    ]
                    if len(cands) == 1:
                        i = cands[0]
                        assigned_inst = int(sleap_anchor[i, 0])
                        used_inst.add(assigned_inst)

                if assigned_inst is not None:
                    inst_df = s_df[s_df["Instance"] == assigned_inst]
                    node_dict = {
                        int(b): (float(x), float(y))
                        for b, x, y in inst_df[["Bodypoint", "X", "Y"]].itertuples(index=False)
                    }
                    if anchor_bodypoint in node_dict:
                        all_pos[frame_idx][tid] = node_dict
                        tracked_anchor_xy.setdefault(tid, []).append(
                            (frame_idx, node_dict[anchor_bodypoint])
                        )

        # ---------------- spawn new tracks from unused ArUco
        for i, (tag_id, ax, ay) in enumerate(aruco_arr):
            if i in used_tag:
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
            used_inst.add(inst_id)

            inst_df = s_df[s_df["Instance"] == inst_id]
            node_dict = {
                int(b): (float(x), float(y))
                for b, x, y in inst_df[["Bodypoint", "X", "Y"]].itertuples(index=False)
            }
            if anchor_bodypoint not in node_dict:
                continue

            tid = int(tag_id)
            all_pos[frame_idx][tid] = node_dict
            tracked_anchor_xy.setdefault(tid, []).append(
                (frame_idx, node_dict[anchor_bodypoint])
            )

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
                    disp = _compose_debug_tiles(base, aruco_arr, s_df, sleap_anchor, frame_idx)
                else:
                    # stack: single frame with multiple overlays
                    disp = base.copy()

                    if debug_show_aruco:
                        _draw_aruco_debug(disp, aruco_arr)
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

    # ======================================================== cleanup
    if use_video and cap is not None:
        cap.release()
        if visualize or debug_viz:
            cv2.destroyAllWindows()
        if writer is not None:
            writer.release()

    # ---------------- save tracks
    if output_path is not None:
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
