"""Conservative temporal ID gap-filling along SLEAP tracklets.

The ArUco detector identifies each ant *independently* per frame.
When the marker is temporarily unreadable (perspective distortion,
occlusion, blur), the frame has no ArUco ID even though SLEAP still
tracks the ant's body.

This module fills **interior gaps only** with strict safety rules:
- Only fill a gap when **the same ID appears on both sides**
- Never correct an existing detection
- Cap gap length (default 10 frames)
- Never fill across track breaks, merges, or competing IDs
- Prefer no label over a guessed label

The result is a modified ArUco DataFrame that can be fed directly into
``tracking_utils.get_complete_tracks()`` as a drop-in replacement.

Usage:
    from tracking.temporal_id import fill_interior_gaps

    aruco_filled = fill_interior_gaps(aruco_df, sleap_df)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Tracklet:
    """A contiguous run of a SLEAP instance with spatial continuity."""

    sleap_instance_id: int
    frames: list[int] = field(default_factory=list)
    positions: list[tuple[float, float]] = field(default_factory=list)
    aruco_ids: list[int] = field(default_factory=list)


def build_sleap_tracklets(
    sleap_df: pd.DataFrame,
    anchor_bodypoint: int = 0,
    max_gap: int = 5,
    max_distance: float = 50.0,
) -> list[Tracklet]:
    """Build contiguous tracklets via frame-to-frame nearest-neighbor linking.

    SLEAP instance IDs are typically per-frame (not persistent across
    frames), so this function links detections across frames by spatial
    proximity rather than instance ID.

    For each frame, each unlinked detection is matched to the nearest
    active tracklet endpoint within *max_distance* pixels.  Unmatched
    detections start new tracklets.  Tracklets that haven't been extended
    for more than *max_gap* frames are closed.

    Parameters
    ----------
    sleap_df : DataFrame
        Columns: Frame, Instance, Bodypoint, X, Y, ...
    anchor_bodypoint : int
        Which bodypoint to use as the spatial anchor.
    max_gap : int
        Maximum frame gap before closing a tracklet.
    max_distance : float
        Maximum displacement (pixels) to link across frames.
    """
    if sleap_df.empty:
        return []

    anchor = sleap_df[sleap_df["Bodypoint"] == anchor_bodypoint].copy()
    anchor = anchor.dropna(subset=["X", "Y"])
    if anchor.empty:
        return []

    # Group by frame, sorted
    frames_sorted = sorted(anchor["Frame"].unique())
    by_frame: dict[int, np.ndarray] = {}
    for f, grp in anchor.groupby("Frame"):
        by_frame[int(f)] = grp[["Instance", "X", "Y"]].values

    # Active tracklets: list of (Tracklet, last_frame)
    active: list[tuple[Tracklet, int]] = []
    finished: list[Tracklet] = []
    next_id = 0

    for fi in frames_sorted:
        pts = by_frame[fi]  # (N, 3): instance, x, y

        # Close expired tracklets
        still_active = []
        for t, last_f in active:
            if fi - last_f > max_gap:
                if len(t.frames) >= 2:
                    finished.append(t)
            else:
                still_active.append((t, last_f))
        active = still_active

        if len(pts) == 0:
            continue

        used_tracklet = [False] * len(active)
        used_point = [False] * len(pts)

        if active:
            # Build cost matrix: distance from each point to each tracklet endpoint
            track_positions = np.array([t.positions[-1] for t, _ in active])  # (M, 2)
            point_positions = pts[:, 1:3]  # (N, 2)
            # (N, M) distance matrix
            dx = point_positions[:, 0:1] - track_positions[:, 0:1].T
            dy = point_positions[:, 1:2] - track_positions[:, 1:2].T
            dists = np.sqrt(dx**2 + dy**2)

            # Greedy matching: assign closest pairs first
            while True:
                valid = dists.copy()
                valid[used_point, :] = np.inf
                valid[:, used_tracklet] = np.inf
                valid[valid > max_distance] = np.inf

                if np.all(np.isinf(valid)):
                    break

                min_idx = np.unravel_index(np.argmin(valid), valid.shape)
                pi, ti = int(min_idx[0]), int(min_idx[1])

                # Extend tracklet
                t, _ = active[ti]
                t.frames.append(int(fi))
                t.positions.append((float(pts[pi, 1]), float(pts[pi, 2])))
                active[ti] = (t, int(fi))

                used_point[pi] = True
                used_tracklet[ti] = True

        # Start new tracklets for unmatched points
        for pi in range(len(pts)):
            if not used_point[pi]:
                t = Tracklet(sleap_instance_id=next_id)
                next_id += 1
                t.frames.append(int(fi))
                t.positions.append((float(pts[pi, 1]), float(pts[pi, 2])))
                active.append((t, int(fi)))

    # Close remaining active tracklets
    for t, _ in active:
        if len(t.frames) >= 2:
            finished.append(t)

    return finished


def _match_aruco_to_tracklets(
    tracklets: list[Tracklet],
    aruco_df: pd.DataFrame,
    max_distance: float = 50.0,
) -> None:
    """For each tracklet frame, find the nearest ArUco detection and store its ID.

    Modifies tracklets in place (fills ``aruco_ids``).
    """
    if aruco_df.empty:
        for t in tracklets:
            t.aruco_ids = [-1] * len(t.frames)
        return

    aruco_by_frame: dict[int, np.ndarray] = {}
    for frame, grp in aruco_df.groupby("Frame"):
        aruco_by_frame[int(frame)] = grp[["Instance", "X", "Y"]].values

    for t in tracklets:
        t.aruco_ids = []
        for fi, (fx, fy) in zip(t.frames, t.positions):
            arr = aruco_by_frame.get(fi)
            if arr is None or len(arr) == 0:
                t.aruco_ids.append(-1)
                continue

            dists = np.sqrt((arr[:, 1] - fx) ** 2 + (arr[:, 2] - fy) ** 2)
            best = int(np.argmin(dists))
            if dists[best] <= max_distance:
                t.aruco_ids.append(int(arr[best, 0]))
            else:
                t.aruco_ids.append(-1)


def _find_interior_gaps(aruco_ids: list[int], max_gap_length: int = 10) -> list[dict]:
    """Find interior gaps flanked by the same ID on both sides.

    Returns list of {"fill_id": int, "start": int, "end": int} where
    start/end are indices into aruco_ids (inclusive range of -1 entries
    to fill).
    """
    fills: list[dict] = []
    n = len(aruco_ids)
    i = 0

    while i < n:
        if aruco_ids[i] >= 0:
            # Found a detection — look for a gap after it
            left_id = aruco_ids[i]
            left_idx = i

            # Scan forward through the gap
            j = i + 1
            while j < n and aruco_ids[j] < 0:
                j += 1

            if j < n and aruco_ids[j] >= 0:
                right_id = aruco_ids[j]
                gap_length = j - left_idx - 1

                if left_id == right_id and 0 < gap_length <= max_gap_length:
                    fills.append({
                        "fill_id": left_id,
                        "start": left_idx + 1,
                        "end": j - 1,
                    })

            i = j if j < n else n
        else:
            i += 1

    return fills


def fill_interior_gaps(
    aruco_df: pd.DataFrame,
    sleap_df: pd.DataFrame,
    max_distance: float = 50.0,
    anchor_bodypoint: int = 0,
    max_gap_frames: int = 10,
    max_tracklet_gap: int = 5,
) -> pd.DataFrame:
    """Conservative interior-gap filling along SLEAP tracklets.

    Only fills a gap when **the same ID appears on both sides** of the
    gap within the same tracklet.  Never corrects existing detections.
    Never propagates one-sided.  Never fills gaps longer than
    *max_gap_frames*.

    Parameters
    ----------
    aruco_df : DataFrame
        Standard ArUco output: Frame, Instance, X, Y, Confidence.
    sleap_df : DataFrame
        SLEAP output: Frame, Instance, Bodypoint, X, Y, ...
    max_distance : float
        Max ArUco-SLEAP spatial match distance (px).
    anchor_bodypoint : int
        SLEAP bodypoint for spatial anchoring.
    max_gap_frames : int
        Maximum gap length to fill (in tracklet entries, not video frames).
    max_tracklet_gap : int
        Maximum frame gap within a SLEAP tracklet before splitting.

    Returns
    -------
    DataFrame
        Same columns as *aruco_df* plus ``propagation_source``:
        ``"direct"`` (original), ``"gap_filled"`` (interior fill).
    """
    if aruco_df.empty or sleap_df.empty:
        out = aruco_df.copy()
        if not out.empty:
            out["propagation_source"] = "direct"
        return out

    # 1. Build tracklets
    tracklets = build_sleap_tracklets(
        sleap_df,
        anchor_bodypoint=anchor_bodypoint,
        max_gap=max_tracklet_gap,
        max_distance=max_distance,
    )

    # 2. Match ArUco IDs to tracklet frames
    _match_aruco_to_tracklets(tracklets, aruco_df, max_distance=max_distance)

    # 3. Find interior gaps and fill
    out = aruco_df.copy()
    out["propagation_source"] = "direct"

    existing = set()
    for _, row in out.iterrows():
        existing.add((int(row["Frame"]), int(row["Instance"])))

    new_rows: list[dict] = []
    n_fills = 0
    n_gaps_found = 0

    for t in tracklets:
        gaps = _find_interior_gaps(t.aruco_ids, max_gap_length=max_gap_frames)
        n_gaps_found += len(gaps)

        for gap in gaps:
            fill_id = gap["fill_id"]
            for idx in range(gap["start"], gap["end"] + 1):
                fi = t.frames[idx]
                fx, fy = t.positions[idx]
                key = (fi, fill_id)
                if key not in existing:
                    new_rows.append({
                        "Frame": fi,
                        "Instance": fill_id,
                        "X": fx,
                        "Y": fy,
                        "Confidence": 0.5,  # lower confidence for gap-fills
                        "propagation_source": "gap_filled",
                    })
                    existing.add(key)
                    n_fills += 1

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        out = pd.concat([out, new_df], ignore_index=True)
        out = out.sort_values("Frame").reset_index(drop=True)

    return out


# Keep backward-compatible alias
def propagate_ids_majority(
    aruco_df: pd.DataFrame,
    sleap_df: pd.DataFrame,
    max_distance: float = 50.0,
    anchor_bodypoint: int = 0,
    min_consensus: float = 0.5,
    max_gap: int = 5,
) -> pd.DataFrame:
    """Backward-compatible wrapper. Now uses conservative gap-filling."""
    return fill_interior_gaps(
        aruco_df, sleap_df,
        max_distance=max_distance,
        anchor_bodypoint=anchor_bodypoint,
        max_gap_frames=10,
        max_tracklet_gap=max_gap,
    )
