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
    anchor_bodypoint: int = 0,               # body‑point used for linking
    visualize: bool = False,
    video_out_path: str | Path | None = None,
    harvest_crops: bool = False,
    crops_output_dir: str | Path | None = None,
    crop_size: int = 128,
    max_distance: float = 50.0,
    aruco_sleap_max_distance: float | None = None,
    start_frame: int = 0,
    harvest_interval: int = 5,
) -> List[Dict[int, Dict[int, Tuple[float, float]]]]:
   

    if aruco_sleap_max_distance is None:
        aruco_sleap_max_distance = max_distance

    use_video = video_file is not None

    # --------------------------------------------------------- sanity checks
    if not use_video and (visualize or harvest_crops or video_out_path is not None):
        raise ValueError(
            "`video_file` must be provided when `visualize`, `harvest_crops`, or "
            "`video_out_path` are enabled."
        )

    # --------------------------------------------------------- I/O prep – group detections
    grouped_aruco = {f: g for f, g in aruco_detection.groupby("Frame")}
    grouped_sleap = {f: g for f, g in sleap_detection.groupby("Frame")}

    # --------------------------------------------------------- obtain frame count & video capture
    cap: Optional[cv2.VideoCapture]
    if use_video:
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_file}")
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        # derive from detections; +1 because frames are assumed 0‑based
        if sleap_detection.empty and aruco_detection.empty:
            raise ValueError("Detection DataFrames are empty – cannot infer frame count.")
        num_frames = (
            max(
                sleap_detection["Frame"].max() if not sleap_detection.empty else -1,
                aruco_detection["Frame"].max() if not aruco_detection.empty else -1,
            )
            + 1
        )
 
        cap = None

    # --------------------------------------------------------- optional writer
    writer: Optional[cv2.VideoWriter] = None
    if use_video and video_out_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(video_out_path), fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    # --------------------------------------------------------- containers
    all_pos: List[Dict[int, Dict[int, Tuple[float, float]]]] = [
        {} for _ in range(num_frames)
    ]
    tracked_anchor_xy: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}

    if harvest_crops and crops_output_dir is not None and use_video:
        Path(crops_output_dir).mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------- colour util
    def _id2bgr(tid: int) -> Tuple[int, int, int]:
        rng = np.random.default_rng(tid)
        r, g, b = (int(x) for x in rng.integers(60, 255, size=3))
        return (b, g, r)

    # ======================================================== main loop
    for frame_idx in tqdm(range(start_frame, num_frames), unit="frame"):
        # -------------------------------- load frame (if available)
        if use_video:
            ok, frame = cap.read()
            if not ok:
                break
            img = frame.copy()
        else:
            img = None  # placeholder; used only if crops/visualisation requested

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
                cx, cy = map(int, map(round, nodes[anchor_bodypoint]))
                xmin, ymin = max(cx - half, 0), max(cy - half, 0)
                xmax, ymax = min(xmin + crop_size, img.shape[1]), min(
                    ymin + crop_size, img.shape[0]
                )
                crop = img[ymin:ymax, xmin:xmax]
                if crop.shape[:2] != (crop_size, crop_size):
                    crop = cv2.resize(crop, (crop_size, crop_size))
                out_dir = Path(crops_output_dir) / str(tid)
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / f"{frame_idx}_{uuid.uuid4().hex}.png"), crop)

        # ---------------- visualisation
        if use_video and (visualize or writer is not None):
            disp = img.copy()
            for tid, nodes in all_pos[frame_idx].items():
                colour = _id2bgr(tid)
                for (x, y) in nodes.values():
                    cv2.circle(disp, (int(x), int(y)), 8, colour, -1)
                ax, ay = nodes[anchor_bodypoint]
                cv2.putText(
                    disp,
                    str(tid),
                    (int(ax) + 8, int(ay) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    colour,
                    3,
                )
            cv2.putText(
                disp,
                f"Frame {frame_idx}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 255, 255),
                3,
            )

            if visualize:
                cv2.imshow("Tracking", cv2.resize(disp, (1080, 720)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if writer is not None:
                writer.write(disp)

    # ======================================================== cleanup
    if use_video:
        cap.release()
        if visualize:
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
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(output_path)

    return all_pos
