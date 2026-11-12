"""Historical pilot and shared manual-cup input utilities.

The active full-run protocol is tracking.single_ant.full_cup_tracking. It
uses the native colony engine and does not apply the historical pilot's stops.

This protocol is independent of the colony tracker. It reuses its lossless
ArUco reader, homography projection, bodypoint-0 anchor and parquet columns.
All decisions use physical millimetres; exported coordinates use 0.016 mm/px,
the colony analysis convention. CupID, rather than a possibly repeated tag,
is the unique TrackID. No inferred arenas or left/right split are used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from tracking.colony.map_combine import apply_homography, aruco_h5_to_long_df_full

OUTPUT_MM_PER_PX = .016
PROTOCOL = "manual_cup_single_ant_v1"


@dataclass(frozen=True)
class Settings:
    id_frame_fraction: float = .60
    duplicate_view_distance_mm: float = 2.
    aruco_sleap_distance_mm: float = 1.44
    min_anchor_score: float = .20
    min_pose_nodes: int = 4
    strong_pose_nodes: int = 8
    strong_pose_score: float = .50
    multi_hits: int = 15
    multi_window: int = 24
    bridge_step_mm_per_frame: float = 4.
    multiplicity_policy: str = "supported"
    pose_duplicate_anchor_mm: float = 1.5
    pose_duplicate_node_mm: float = .35
    pose_duplicate_median_mm: float = .25
    pose_duplicate_shared_nodes: int = 6


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cups(path):
    df = pd.read_csv(path)
    df = df.loc[df.semantic_label.eq("cup")].copy().reset_index(drop=True)
    if df.empty or df.name.duplicated().any():
        raise ValueError("Expected uniquely named, manually annotated cups")
    if not df["shape"].eq("circle").all():
        raise ValueError("This protocol currently requires circular cups")
    cups = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        scale = float(row.mm_per_pixel)
        center = np.array([row.tracking_center_x_px, row.tracking_center_y_px]) * scale
        radius = float(row.radius_px) * scale
        if not np.isfinite([*center, radius, scale]).all() or min(radius, scale) <= 0:
            raise ValueError(f"Invalid cup geometry: {row.name}")
        cups.append(dict(cup_id=i, name=row.name, center_mm=center.tolist(), radius_mm=radius))
    return cups


def assign_cups(xy, cups):
    """An anchor must be inside exactly one cup; overlapping boundaries are unknown."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    centers = np.array([c["center_mm"] for c in cups])
    radius = np.array([c["radius_mm"] for c in cups])
    result = np.full(len(xy), -1, np.int32)
    for start in range(0, len(xy), 100_000):
        points = xy[start:start + 100_000]
        inside = np.linalg.norm(points[:, None] - centers, axis=2) <= radius
        one = inside.sum(axis=1) == 1
        result[start:start + len(points)][one] = np.argmax(inside[one], axis=1) + 1
    return result


def validate_timing(manifest):
    """Reuse the verified trigger-frame contract, never filename arming times."""
    if manifest["synchronization"]["method"] != "common hardware trigger/block IDs":
        raise ValueError("A verified common-trigger detection manifest is required")
    total = int(manifest["n_frames"])
    for cam, info in manifest["cameras"].items():
        diag = manifest["synchronization"]["diagnostics"][cam]
        frames = int(info["frames"])
        interrupted = diag.get("interrupted", False)
        if (info["offset"] != 0 or not 0 < frames <= total or diag["frames"] != frames
                or (not interrupted and (frames != total or diag["first_block_id"] != 0
                                         or diag["last_block_id"] != frames - 1))
                or (interrupted and (cam not in manifest["synchronization"].get("alignment_assumed_cameras", [])
                                     or diag.get("telemetry_available") is not False))):
            raise ValueError(f"Incompatible synchronization contract: {cam}")
        files = sorted((f for f in manifest["files"] if f["camera"] == cam),
                       key=lambda f: f["index"])
        stop = 0
        for index, f in enumerate(files):
            if f["index"] != index or f["frame_start"] != stop or f["local_start"] != stop:
                raise ValueError(f"Noncontiguous source chunks: {cam}")
            stop += f["frames"]
        if stop != frames:
            raise ValueError(f"Incomplete source camera: {cam}")


def read_window(manifest, cups, start, stop, settings):
    """Read raw detections, keeping camera/instance keys and entire skeletons."""
    validate_timing(manifest)
    a_parts, s_parts = [], []
    scale = 1. / float(manifest["board_px_per_mm"])
    sources = []
    for f in manifest["files"]:
        lo, hi = max(start, f["frame_start"]), min(stop, f["frame_start"] + f["frames"])
        if hi <= lo:
            continue
        print(f"READ {f['camera']} chunk {f['index']:03d} frames [{lo}, {hi})", flush=True)
        with h5py.File(f["aruco"]) as h:
            if "aruco_detections" not in h:
                raise ValueError(f"Lossless ArUco detections required: {f['aruco']}")
            if int(h.attrs["num_frames"]) != f.get("aruco_frames", f["frames"]):
                raise ValueError(f"Changed detection frame count: {f['aruco']}")
            a = aruco_h5_to_long_df_full(h, frame_offset=f["frame_start"])
        a = a.loc[a.Frame.ge(lo) & a.Frame.lt(hi)].copy()
        H = np.asarray(manifest["homographies"][f["camera_id"]])
        a[["X", "Y"]] = apply_homography(a[["X", "Y"]].to_numpy(), H) * scale
        a["Cam"] = f["camera_id"]
        a["CupID"] = assign_cups(a[["X", "Y"]].to_numpy(), cups)
        a_parts.append(a.loc[a.CupID.gt(0)])
        with h5py.File(f["sleap"]) as h:
            if int(h.attrs["frame_count"]) != f["frames"]:
                raise ValueError(f"Changed SLEAP frame count: {f['sleap']}")
            s = pd.DataFrame.from_records(h["sleap_data"][:])
        s["Frame"] = s.Frame.astype(np.int64) + f["frame_start"]
        s = s.loc[s.Frame.ge(lo) & s.Frame.lt(hi)].copy()
        s[["X", "Y"]] = apply_homography(s[["X", "Y"]].to_numpy(), H) * scale
        s["Cam"] = f["camera_id"]
        s_parts.append(s)
        sources.append({k: f[k] for k in ("camera", "index", "frame_start", "frames", "aruco", "sleap")})
    if not a_parts or not s_parts:
        raise ValueError("No complete source files in requested frame window")
    a = pd.concat(a_parts, ignore_index=True)
    s = pd.concat(s_parts, ignore_index=True)
    finite = np.isfinite(s[["X", "Y"]]).all(axis=1)
    s = s.loc[finite & s.Score_node.ge(settings.min_anchor_score)].copy()
    keys = ["Frame", "Cam", "Instance"]
    if s.duplicated(keys + ["Bodypoint"]).any():
        raise ValueError("Repeated bodypoints inside a camera/instance")
    stats = s.groupby(keys, sort=False).agg(NNodes=("Bodypoint", "size"), MedianScore=("Score_node", "median"))
    poses = s.loc[s.Bodypoint.eq(0), keys + ["X", "Y", "Score_node"]].merge(stats, on=keys)
    poses["CupID"] = assign_cups(poses[["X", "Y"]].to_numpy(), cups)
    poses = poses.loc[poses.CupID.gt(0) & poses.NNodes.ge(settings.min_pose_nodes)].copy()
    poses["Strong"] = (poses.NNodes.ge(settings.strong_pose_nodes)
                       & poses.Score_node.ge(settings.strong_pose_score)
                       & poses.MedianScore.ge(settings.strong_pose_score))
    poses["PoseKey"] = np.arange(len(poses), dtype=np.int64)
    nodes = s.merge(poses[keys + ["CupID", "PoseKey"]], on=keys, how="inner", validate="many_to_one")
    return a, poses, nodes, sources


def view_groups(xy, cams, distance):
    """Merge nearby views with at most one detection from each camera per group.

    Two instances in the SAME camera always remain two instances, including
    two tags with the same ID. Distance is checked against every group member
    so a chain of overlapping views cannot merge spatially separate ants.
    """
    xy, cams = np.asarray(xy), np.asarray(cams)
    groups = []
    for i in np.argsort(cams, kind="stable"):
        candidates = [(float(np.linalg.norm(xy[g] - xy[i], axis=1).max()), j)
                      for j, g in enumerate(groups) if cams[i] not in cams[g]]
        candidates = [(d, j) for d, j in candidates if d <= distance]
        if candidates:
            groups[min(candidates)[1]].append(int(i))
        else:
            groups.append([int(i)])
    return [np.asarray(g, dtype=int) for g in groups]


def deduplicate_poses(poses, nodes, settings):
    """Collapse near-identical predictions of one pose inside the SAME camera.

    Six corresponding landmarks must coincide within 0.35 mm and median
    landmark error must be <=0.25 mm. Anchor proximity alone is insufficient.
    Whole skeletons are retained, never averaged or assembled from two poses.
    """
    candidates = poses.loc[poses.duplicated(["Frame", "Cam", "CupID"], keep=False)]
    candidate_nodes = nodes.loc[nodes.PoseKey.isin(candidates.PoseKey)]
    by_key = {int(key): g.set_index("Bodypoint") for key, g in candidate_nodes.groupby("PoseKey")}
    aliases = []
    for _, group in candidates.groupby(["Frame", "Cam", "CupID"], sort=False):
        kept = []
        for row in group.sort_values(["NNodes", "MedianScore", "PoseKey"], ascending=[False, False, True]).itertuples(index=False):
            duplicate = None
            for other in kept:
                if row.CupID != other.CupID or np.hypot(row.X - other.X, row.Y - other.Y) > settings.pose_duplicate_anchor_mm:
                    continue
                a, b = by_key[int(row.PoseKey)], by_key[int(other.PoseKey)]
                common = a.index.intersection(b.index)
                if len(common) < settings.pose_duplicate_shared_nodes:
                    continue
                distance = np.linalg.norm(a.loc[common, ["X", "Y"]].to_numpy()
                                          - b.loc[common, ["X", "Y"]].to_numpy(), axis=1)
                if ((distance <= settings.pose_duplicate_node_mm).sum() >= settings.pose_duplicate_shared_nodes
                        and np.median(distance) <= settings.pose_duplicate_median_mm):
                    duplicate = (other, distance)
                    break
            if duplicate is None:
                kept.append(row)
            else:
                other, distance = duplicate
                aliases.append(dict(Frame=int(row.Frame), Cam=int(row.Cam), CupID=int(row.CupID),
                                    RemovedPoseKey=int(row.PoseKey), KeptPoseKey=int(other.PoseKey),
                                    SharedNodes=len(distance), MedianNodeDistanceMM=float(np.median(distance))))
    aliases = pd.DataFrame(aliases, columns=["Frame", "Cam", "CupID", "RemovedPoseKey", "KeptPoseKey", "SharedNodes", "MedianNodeDistanceMM"])
    return poses.loc[~poses.PoseKey.isin(aliases.RemovedPoseKey)].copy(), aliases


def frame_evidence(detections, start, stop, settings):
    """Count physical instances without constructing a DataFrame per frame."""
    count = np.zeros(stop - start, np.int16)
    single_id = np.full(stop - start, -1, np.int32)
    if detections.empty:
        return count, single_id
    order = np.argsort(detections.Frame.to_numpy(), kind="stable")
    frame = detections.Frame.to_numpy(dtype=np.int64)[order] - start
    xy = detections[["X", "Y"]].to_numpy()[order]
    cam = detections.Cam.to_numpy()[order]
    starts = np.r_[0, np.flatnonzero(np.diff(frame)) + 1]
    ends = np.r_[starts[1:], len(frame)]
    sizes = ends - starts
    count[frame[starts]] = 1
    double = starts[sizes == 2]
    same_view = cam[double] == cam[double + 1]
    apart = np.linalg.norm(xy[double] - xy[double + 1], axis=1) > settings.duplicate_view_distance_mm
    count[frame[double]] = 1 + (same_view | apart).astype(np.int16)
    for lo, hi in zip(starts[sizes > 2], ends[sizes > 2]):
        count[frame[lo]] = len(view_groups(xy[lo:hi], cam[lo:hi], settings.duplicate_view_distance_mm))
    if "Instance" in detections:
        ids = detections.Instance.to_numpy(dtype=np.int32)[order]
        low, high = np.minimum.reduceat(ids, starts), np.maximum.reduceat(ids, starts)
        unique = (low == high) & (count[frame[starts]] == 1)
        single_id[frame[starts[unique]]] = low[unique]
    return count, single_id


def supported_multiple_tags(aruco, poses, settings):
    """Two tags need two distinct strong postures, not two views of one ant."""
    from scipy.optimize import linear_sum_assignment
    if len(aruco) < 2 or len(poses) < 2:
        return False
    axy, pxy = aruco[["X", "Y"]].to_numpy(), poses[["X", "Y"]].to_numpy()
    aa = view_groups(axy, aruco.Cam.to_numpy(), settings.duplicate_view_distance_mm)
    pp = view_groups(pxy, poses.Cam.to_numpy(), settings.duplicate_view_distance_mm)
    if min(len(aa), len(pp)) < 2:
        return False
    distance = np.array([[np.linalg.norm(axy[g][:, None] - pxy[h][None], axis=2).min()
                          for h in pp] for g in aa])
    # Maximize the number of supported one-to-one pairs before distance.
    cost = np.where(distance <= settings.aruco_sleap_distance_mm, distance, 1e6)
    rr, cc = linear_sum_assignment(cost)
    return int((cost[rr, cc] < 1e6).sum()) >= 2


def confirmed_multiple(mask, settings):
    return bool(np.any(np.convolve(np.asarray(mask, int), np.ones(settings.multi_window, int), mode="full")
                       >= settings.multi_hits))


def first_persistent_frame(mask, settings):
    mask = np.asarray(mask, bool)
    hits = np.convolve(mask.astype(int), np.ones(settings.multi_window, int), mode="full")[:len(mask)]
    confirmed = np.flatnonzero(hits >= settings.multi_hits)
    if not len(confirmed):
        return None
    end = int(confirmed[0])
    start = max(0, end - settings.multi_window + 1)
    return start + int(np.flatnonzero(mask[start:end + 1])[0])


def supported_foreign_ids(aruco, poses, resident_id, settings):
    """A foreign tag needs its own pose, not the resident's associated pose."""
    if aruco.empty or poses.empty:
        return set()
    axy, pxy = aruco[["X", "Y"]].to_numpy(), poses[["X", "Y"]].to_numpy()
    groups = view_groups(pxy, poses.Cam.to_numpy(), settings.duplicate_view_distance_mm)
    distances = np.array([[np.linalg.norm(axy[i] - pxy[g], axis=1).min() for g in groups]
                          for i in range(len(axy))])
    ids = aruco.Instance.to_numpy()
    resident_groups = np.any(distances[ids == resident_id] <= settings.aruco_sleap_distance_mm, axis=0)
    foreign = set()
    for i in np.flatnonzero(ids != resident_id):
        if np.any((distances[i] <= settings.aruco_sleap_distance_mm) & ~resident_groups):
            foreign.add(int(ids[i]))
    return foreign


def tag_has_skeleton(aruco, poses, settings):
    """Same-frame nearest-anchor support, without dropping unpaired resident reads."""
    from scipy.spatial import cKDTree
    if aruco.empty or poses.empty:
        return np.zeros(len(aruco), bool)
    spacing = 3 * settings.aruco_sleap_distance_mm + 1
    tree = cKDTree(np.c_[poses.Frame.to_numpy() * spacing, poses[["X", "Y"]].to_numpy()])
    distance, _ = tree.query(np.c_[aruco.Frame.to_numpy() * spacing, aruco[["X", "Y"]].to_numpy()],
                             distance_upper_bound=settings.aruco_sleap_distance_mm)
    return np.isfinite(distance)


def census(cups, aruco, poses, start, stop, settings):
    n = stop - start
    result, summaries = [], []
    for cup in cups:
        cid = cup["cup_id"]
        a = aruco.loc[aruco.CupID.eq(cid)]
        count, single_id = frame_evidence(a, start, stop, settings)
        p = poses.loc[poses.CupID.eq(cid)]
        s_count, _ = frame_evidence(p, start, stop, settings)
        strong_count, _ = frame_evidence(p.loc[p.Strong], start, stop, settings)
        support = tag_has_skeleton(a, p, settings)
        frequencies = a.loc[support].groupby("Instance").Frame.nunique().sort_index()
        dominant = int(frequencies.idxmax()) if len(frequencies) else -1
        resident_count, _ = frame_evidence(a.loc[a.Instance.eq(dominant)], start, stop, settings)
        raw_fraction = float((single_id == dominant).mean()) if dominant >= 0 else 0.
        multiple_s = strong_count > 1
        supported_a = np.zeros(n, bool)
        candidate_frames = np.flatnonzero((count > 1) & (s_count > 1)) + start
        if len(candidate_frames):
            ag = {int(fr): g for fr, g in a.loc[a.Frame.isin(candidate_frames)].groupby("Frame")}
            pg = {int(fr): g for fr, g in p.loc[p.Frame.isin(candidate_frames)].groupby("Frame")}
            for fr in candidate_frames:
                supported_a[fr - start] = supported_multiple_tags(ag[fr], pg[fr], settings)
        foreign_masks = {}
        foreign_frames = a.loc[a.Instance.ne(dominant), "Frame"].unique()
        if len(foreign_frames):
            ag = {int(fr): g for fr, g in a.loc[a.Frame.isin(foreign_frames)].groupby("Frame")}
            pg = {int(fr): g for fr, g in p.loc[p.Frame.isin(foreign_frames)].groupby("Frame")}
            for fr in foreign_frames:
                if int(fr) not in pg:
                    continue
                for foreign_id in supported_foreign_ids(ag[int(fr)], pg[int(fr)], dominant, settings):
                    foreign_masks.setdefault(foreign_id, np.zeros(n, bool))[int(fr) - start] = True
        foreign_supported = np.logical_or.reduce(list(foreign_masks.values())) if foreign_masks else np.zeros(n, bool)
        fraction = float(((resident_count == 1) & ~foreign_supported).mean())
        events = []
        for label, mask in [("multiple_posture_supported_aruco_instances", supported_a),
                            ("multiple_supported_sleap_instances", multiple_s)]:
            event = first_persistent_frame(mask, settings)
            if event is not None:
                events.append((event + start, label))
        for foreign_id, mask in foreign_masks.items():
            event = first_persistent_frame(mask, settings)
            if event is not None:
                events.append((event + start, f"persistent_skeleton_supported_foreign_ID_{foreign_id}"))
        cutoff, event_reason = min(events) if events else (None, "")
        reasons = []
        if not np.any(count):
            reasons.append("no_aruco_ant_detected")
        if dominant < 0:
            reasons.append("no_skeleton_supported_tag_for_identity")
        if settings.multiplicity_policy == "raw" and np.any(count > 1):
            reasons.append("multiple_aruco_instances")
        if cutoff is not None and cutoff <= start:
            reasons.append("multiple_ants_at_start:" + event_reason)
        if fraction <= settings.id_frame_fraction:
            reasons.append("single_ID_fraction_not_over_60_percent")
        summaries.append(dict(CupID=cid, Cup=cup["name"], ArucoID=dominant,
                              Frames=n, SingleIDFraction=fraction,
                              TagAndSkeletonFrameFraction=float(a.loc[support & a.Instance.eq(dominant)].Frame.nunique() / n),
                              RawSingleIDFraction=raw_fraction,
                              ZeroArucoFrames=int((count == 0).sum()),
                              MultipleArucoFrames=int((count > 1).sum()),
                              MultipleSupportedArucoFrames=int(supported_a.sum()),
                              MultipleStrongSleapFrames=int(multiple_s.sum()),
                              SupportedForeignTagFrames=int(foreign_supported.sum()),
                              FlaggedMultiAnt=cutoff is not None, StopFrame=cutoff, EventReason=event_reason,
                              Eligible=not reasons, Reason=";".join(reasons) or
                              ("eligible_until_multi_ant_event" if cutoff is not None else "eligible")))
        result.append(pd.DataFrame(dict(Frame=np.arange(start, stop), CupID=cid,
                                        ArucoCount=count, SingleArucoID=single_id,
                                        SleapCount=s_count, StrongSleapCount=strong_count,
                                        MultipleSupportedAruco=supported_a,
                                        SupportedForeignTag=foreign_supported,
                                        TrackingStopped=np.arange(start, stop) >= (cutoff if cutoff is not None else stop))))
        print("CENSUS", summaries[-1], flush=True)
    return pd.DataFrame(summaries), pd.concat(result, ignore_index=True)


def track_cup(cup, tag_id, aruco, poses, counts, start, stop, settings, mask_tags=None):
    """Use nearest tag/anchor pairs, then unique contained SLEAP through tag gaps.

    There is no fixed 120-frame tag-loss timeout: a unique observed ant can be
    followed throughout its cup. Empty or ambiguous frames remain missing.
    Pose gaps are never filled with repeated coordinates or synthetic skeletons.
    """
    cid = cup["cup_id"]
    ag = {int(fr): d for fr, d in aruco.loc[aruco.CupID.eq(cid)].groupby("Frame", sort=False)}
    sg = {int(fr): d for fr, d in poses.loc[poses.CupID.eq(cid)].groupby("Frame", sort=False)}
    c = counts.loc[counts.CupID.eq(cid)].set_index("Frame")
    previous = None
    rows = []
    for fr in range(start, stop):
        a, p = ag.get(fr), sg.get(fr)
        if mask_tags is not None and mask_tags[fr - start]:
            a = None
        row = dict(Frame=fr, CupID=cid, TrackID=cid, ArucoID=tag_id, PoseKey=-1,
                   TrackX=np.nan, TrackY=np.nan, ArucoX=np.nan, ArucoY=np.nan,
                   SleapAnchorX=np.nan, SleapAnchorY=np.nan, ArucoCam=-1, SleapCam=-1,
                   CameraID=-1, MatchDistanceMM=np.nan, PositionSource="missing", NNodes=0,
                   ExtraTagRead=bool(a is not None and len(a) and a.Instance.ne(tag_id).any()))
        has_a = a is not None and len(a) > 0
        has_p = p is not None and len(p) > 0
        raw_veto = settings.multiplicity_policy == "raw" and c.at[fr, "ArucoCount"] > 1
        if has_a and settings.multiplicity_policy == "supported" and not a.Instance.eq(tag_id).any():
            if not c.at[fr, "SupportedForeignTag"]:
                a = None  # unsupported foreign decodes do not block a posture bridge
                has_a = False
        if has_a and settings.multiplicity_policy == "supported" and a.Instance.eq(tag_id).any():
            a = a.loc[a.Instance.eq(tag_id)]
            # Even if an extra same-ID detection is a false decode, its
            # ambiguous frame is not resolved by choosing an arbitrary tag.
            raw_veto |= len(view_groups(a[["X", "Y"]].to_numpy(), a.Cam.to_numpy(),
                                        settings.duplicate_view_distance_mm)) > 1
        if c.at[fr, "TrackingStopped"]:
            row["PositionSource"] = "stopped_multi_ant"
        elif raw_veto or c.at[fr, "StrongSleapCount"] > 1:
            row["PositionSource"] = "multiple_instances"
        elif has_a and not a.Instance.eq(tag_id).all():
            row["PositionSource"] = "conflicting_tag"
        elif has_a:
            selected = None
            if has_p:
                distances = np.linalg.norm(a[["X", "Y"]].to_numpy()[:, None]
                                           - p[["X", "Y"]].to_numpy()[None], axis=2)
                ai, pi = np.unravel_index(np.argmin(distances), distances.shape)
                if distances[ai, pi] <= settings.aruco_sleap_distance_mm:
                    selected = p.iloc[pi]
                    tag = a.iloc[ai]
                    row["MatchDistanceMM"] = float(distances[ai, pi])
            if selected is None:
                # Retain actual tag positions without inventing a posture.
                if previous is None:
                    tag = a.iloc[0]
                else:
                    tag = a.iloc[np.argmin(np.linalg.norm(a[["X", "Y"]].to_numpy() - previous[1], axis=1))]
            row.update(TrackX=float(tag.X), TrackY=float(tag.Y), ArucoX=float(tag.X),
                       ArucoY=float(tag.Y), ArucoCam=int(tag.Cam), CameraID=int(tag.Cam),
                       PositionSource="aruco_only" if selected is None else "aruco_sleap")
            if selected is not None:
                row.update(PoseKey=int(selected.PoseKey), SleapAnchorX=float(selected.X),
                           SleapAnchorY=float(selected.Y), SleapCam=int(selected.Cam), NNodes=int(selected.NNodes))
        elif has_p and c.at[fr, "SleapCount"] == 1:
            # Views of a unique ant: keep a whole pose, prefer completeness,
            # then the previous camera and node confidence, rather than mix nodes.
            previous_cam = previous[2] if previous is not None else -1
            selected = max(p.itertuples(index=False), key=lambda v:
                           (v.NNodes, int(v.Cam == previous_cam), v.MedianScore, -v.PoseKey))
            xy = np.array([selected.X, selected.Y])
            allowed = (previous is None or np.linalg.norm(xy - previous[1])
                       <= settings.bridge_step_mm_per_frame * (fr - previous[0]))
            if allowed:
                row.update(TrackX=float(selected.X), TrackY=float(selected.Y),
                           SleapAnchorX=float(selected.X), SleapAnchorY=float(selected.Y),
                           PoseKey=int(selected.PoseKey), SleapCam=int(selected.Cam),
                           CameraID=int(selected.Cam), NNodes=int(selected.NNodes),
                           PositionSource="sleap_bridge")
            else:
                row["PositionSource"] = "bridge_step_rejected"
        elif has_p:
            row["PositionSource"] = "ambiguous_sleap"
        if np.isfinite(row["TrackX"]):
            xy = np.array([row["TrackX"], row["TrackY"]])
            if np.linalg.norm(xy - cup["center_mm"]) > cup["radius_mm"] + 1e-8:
                raise AssertionError("Attempted position outside the manually labeled cup")
            previous = (fr, xy, row["SleapCam"])
        rows.append(row)
    return pd.DataFrame(rows)


def export_tracks(frames, nodes, path):
    selected = frames.loc[frames.PoseKey.ge(0)]
    df = selected.merge(nodes[["PoseKey", "Bodypoint", "X", "Y", "Score_node"]], on="PoseKey", validate="one_to_many")
    tag_only = frames.loc[frames.PositionSource.eq("aruco_only")].copy()
    tag_only["Bodypoint"] = 0
    tag_only["X"], tag_only["Y"] = tag_only.TrackX, tag_only.TrackY
    tag_only["Score_node"] = np.nan
    df = pd.concat([df, tag_only], ignore_index=True).sort_values(["Frame", "TrackID", "Bodypoint"])
    for col in ("X", "Y", "TrackX", "TrackY", "ArucoX", "ArucoY", "SleapAnchorX", "SleapAnchorY"):
        df[col] = df[col] / OUTPUT_MM_PER_PX
    df.to_parquet(path, index=False)


def select_pilot_cups(qualifications, poses, count):
    """Include high coverage, low coverage, and an overlapping-camera cup."""
    good = qualifications.loc[qualifications.Eligible].sort_values("SingleIDFraction")
    if good.empty:
        return []
    cameras = poses.groupby("CupID").Cam.nunique()
    order = [int(good.iloc[0].CupID), int(good.iloc[-1].CupID)]
    overlap = good.loc[good.CupID.map(cameras).fillna(0).gt(1)]
    if len(overlap):
        order.append(int(overlap.iloc[len(overlap) // 2].CupID))
    order += [int(x) for x in good.iloc[np.linspace(0, len(good) - 1, min(count * 2, len(good)), dtype=int)].CupID]
    return list(dict.fromkeys(order))[:count]


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise FileExistsError("Use a new output directory; existing pilot results are preserved")
    m = json.loads(args.manifest.read_text())
    cups = load_cups(args.regions)
    settings = Settings(multiplicity_policy=args.multiplicity_policy,
                        multi_window=int(round(m["fps"])), multi_hits=int(np.floor(.60 * m["fps"])) + 1)
    start, stop = args.start_frame, args.stop_frame
    if not 0 <= start < stop <= m["n_frames"]:
        raise ValueError("Requested frame window is outside the detection recording")
    meta = json.loads(args.metadata.read_text())
    if not np.isclose(float(meta["mm_per_pixel"]), 1. / m["board_px_per_mm"]):
        raise ValueError("Manual labels and source homographies have different coordinate scales")
    if Path(meta["homographies"]).name != Path(m["homography_source"]).name:
        raise ValueError("Manual labels use a different homography")
    if sha256(args.homography) != m["homography_sha256"]:
        raise ValueError("Source homography differs from verified detection manifest")
    contract = dict(protocol=PROTOCOL, settings=asdict(settings), cups=cups, start_frame=start,
                    stop_frame=stop, fps=m["fps"], full_recording_frames=m["n_frames"],
                    qualification_scope="pilot window only; not full-recording approval",
                    regions_sha256=sha256(args.regions), metadata_sha256=sha256(args.metadata),
                    detection_manifest_sha256=sha256(args.manifest), homography_sha256=sha256(args.homography),
                    source_block=m["block"], output_mm_per_px=OUTPUT_MM_PER_PX,
                    synchronization=m["synchronization"], script_sha256=sha256(__file__))
    write_json(output / "run.json", contract)
    for source, name in [(args.regions, "manual_regions.csv"), (args.metadata, "panorama_metadata.json")]:
        (output / name).write_bytes(source.read_bytes())
    if args.cached_observations:
        cache = args.cached_observations
        prior = json.loads((cache / "run.json").read_text())
        for key in ["start_frame", "stop_frame", "regions_sha256", "metadata_sha256", "detection_manifest_sha256", "homography_sha256"]:
            if prior[key] != contract[key]:
                raise ValueError(f"Cached raw observations have different {key}")
        for key in ["min_anchor_score", "min_pose_nodes", "strong_pose_nodes", "strong_pose_score"]:
            if prior["settings"][key] != contract["settings"][key]:
                raise ValueError(f"Cached pose filtering differs: {key}")
        a = pd.read_parquet(cache / "aruco_mm.parquet")
        raw_poses = cache / "poses_raw_mm.parquet"
        poses = pd.read_parquet(raw_poses if raw_poses.exists() else cache / "poses_mm.parquet")
        nodes = pd.read_parquet(cache / "nodes_mm.parquet")
        sources = json.loads((cache / "sources.json").read_text())
        write_json(output / "observation_cache.json", dict(path=str(cache),
                   files={name: sha256(cache / name) for name in ["aruco_mm.parquet", "poses_mm.parquet", "nodes_mm.parquet"]}))
    else:
        a, poses, nodes, sources = read_window(m, cups, start, stop, settings)
    write_json(output / "sources.json", sources)
    poses.to_parquet(output / "poses_raw_mm.parquet", index=False)
    poses, aliases = deduplicate_poses(poses, nodes, settings)
    aliases.to_csv(output / "duplicate_pose_predictions.csv", index=False)
    print(f"Collapsed {len(aliases)} near-identical same-camera pose predictions", flush=True)
    a.to_parquet(output / "aruco_mm.parquet", index=False)
    poses.to_parquet(output / "poses_mm.parquet", index=False)
    nodes.to_parquet(output / "nodes_mm.parquet", index=False)
    q, counts = census(cups, a, poses, start, stop, settings)
    q.to_csv(output / "cup_qualification.csv", index=False)
    counts.to_parquet(output / "frame_counts.parquet", index=False)
    requested = ([c["cup_id"] for c in cups if c["name"] in args.cups]
                 if args.cups else select_pilot_cups(q, poses, args.pilot_cups))
    if args.cups and len(requested) != len(set(args.cups)):
        raise ValueError("Unknown cup name requested")
    outputs, summaries = [], []
    for cup in cups:
        cid = cup["cup_id"]
        if cid not in requested:
            continue
        qualification = q.loc[q.CupID.eq(cid)].iloc[0]
        if not qualification.Eligible:
            print(f"SKIP {cup['name']}: {qualification.Reason}", flush=True)
            continue
        track = track_cup(cup, int(qualification.ArucoID), a, poses, counts, start, stop, settings)
        outputs.append(track)
        sources_count = track.PositionSource.value_counts().to_dict()
        summaries.append(dict(cup=cup["name"], cup_id=cid, aruco_id=int(qualification.ArucoID),
                              single_id_fraction=float(qualification.SingleIDFraction),
                              position_fraction=float(track.TrackX.notna().mean()),
                              pose_fraction=float(track.PoseKey.ge(0).mean()),
                              position_sources={str(k): int(v) for k, v in sources_count.items()}))
        print("TRACK", summaries[-1], flush=True)
    if outputs:
        frames = pd.concat(outputs, ignore_index=True)
        frames.to_parquet(output / "pilot_frames_mm.parquet", index=False)
        export_tracks(frames, nodes, output / "pilot_tracks.parquet")
    write_json(output / "PILOT_COMPLETE.json", dict(tracked=summaries,
               eligible_cups=int(q.Eligible.sum()), labeled_cups=len(cups),
               flagged_cups=int(q.FlaggedMultiAnt.sum()),
               stop_events=[dict(cup=r.Cup, frame=int(r.StopFrame), reason=r.EventReason)
                            for r in q.loc[q.FlaggedMultiAnt].itertuples(index=False)],
               requested_cups=requested, full_run_submitted=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Verified raw-detection/synchronization manifest")
    p.add_argument("--regions", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--homography", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--stop-frame", type=int, default=14400)
    p.add_argument("--pilot-cups", type=int, default=5)
    p.add_argument("--cups", nargs="+")
    p.add_argument("--multiplicity-policy", choices=["supported", "raw"], default="supported",
                   help="Supported: persistent second-ant evidence; raw: reject any extra tag decode")
    p.add_argument("--cached-observations", type=Path, help="Reuse identically projected raw observations from a prior pilot")
    run(p.parse_args())


if __name__ == "__main__":
    main()
