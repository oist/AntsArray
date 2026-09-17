"""Small, cached, frame-resolved contact review clips; no sleep recomputation.

Trial geometry is a proximity proxy, not proof of physical contact. Original
binary detections and full-recording bout onsets remain separately available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import interaction_analysis_utils as ia
from analysis import return_sleep_utils as rs
from tracking.colony.interaction_one_chunk import DEFAULT_MICRO_DISTANCE_MM


GEOMETRIES = ("Antenna to any node (legacy)", "Antenna to body nodes", "Body to body nodes")


def detection_bouts(frames, *, gap_frames: int, min_hits: int = 1) -> pd.DataFrame:
    """Merge distinct pair-detection frames, not directed rows or playback visits."""
    frames = np.unique(np.asarray(frames, dtype=np.int64))
    columns = ["start_frame", "end_frame", "n_detection_frames"]
    if not len(frames):
        return pd.DataFrame(columns=columns)
    starts = np.r_[0, np.flatnonzero(np.diff(frames) > gap_frames) + 1]
    ends = np.r_[starts[1:] - 1, len(frames) - 1]
    result = pd.DataFrame(dict(start_frame=frames[starts], end_frame=frames[ends],
                               n_detection_frames=ends - starts + 1))
    return result[result.n_detection_frames >= min_hits].reset_index(drop=True)


def pair_geometry(focal: np.ndarray, partner: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distances and endpoint node IDs for F x 10 x 2 global pose coordinates."""
    delta = focal[:, :, None, :] - partner[:, None, :, :]
    d2 = np.square(delta).sum(axis=-1)
    d2[~np.isfinite(d2)] = np.inf
    a, b = np.indices((10, 10))
    masks = [(a >= 4) | (b >= 4), ((a >= 4) & (b < 4)) | ((a < 4) & (b >= 4)), (a < 4) & (b < 4)]
    distances, endpoints = [], []
    for mask in masks:
        selected = np.where(mask[None], d2, np.inf).reshape(len(focal), 100)
        arg = selected.argmin(axis=1)
        distances.append(np.sqrt(selected[np.arange(len(focal)), arg]))
        endpoints.append(np.stack([arg // 10, arg % 10], axis=1))
    shared = np.isfinite(focal).all(axis=2) & np.isfinite(partner).all(axis=2)
    identical = (shared.sum(axis=1) >= 4) & (((np.square(focal - partner).sum(axis=2) < 0.01 ** 2) | ~shared).all(axis=1))
    return np.stack(distances, axis=1), np.stack(endpoints, axis=1).astype(np.int8), identical


def build_review_clip(block_dir: Path, *, side: str, target: int, return_frame: int,
                      start_frame: int, stop_frame: int, fps: float = 24.0) -> Path:
    import pyarrow.dataset as ds

    block_dir = Path(block_dir)
    all_chunks = ia.resolve_chunks(block_dir / "interactions", block_dir / "tracks", chunks="all", side=side, fps=fps)
    chunks = [c for c in all_chunks if c.chunk_global_frame_offset < stop_frame
              and c.chunk_global_frame_offset + c.chunk_frame_count > start_frame]
    if not chunks:
        raise ValueError("No tracked chunks cover the requested clip")
    manifest = dict(version=1, side=side, target=target, return_frame=return_frame,
                    start_frame=start_frame, stop_frame=stop_frame, fps=fps, mm_per_pixel=0.016,
                    sources=[rs.fingerprint(p) for c in chunks for p in (c.track_path, c.interaction_path)],
                    code=rs.fingerprint(Path(__file__)))
    root = block_dir / "stitched" / "analysis_cache" / "contact_review" / rs.cache_key(manifest)
    if (root / "manifest.json").is_file():
        return root
    root.mkdir(parents=True, exist_ok=True)
    poses, raw = [], []
    for c in chunks:
        offset = c.chunk_global_frame_offset
        predicate = (ds.field("Frame") >= start_frame - offset) & (ds.field("Frame") < stop_frame - offset)
        pose = ds.dataset(c.track_path, format="parquet").to_table(
            columns=["Frame", "TrackID", "Bodypoint", "X", "Y", "TrackX", "TrackY"], filter=predicate,
        ).to_pandas()
        pose.Frame += offset
        poses.append(pose)
        contact = ds.dataset(c.interaction_path, format="parquet").to_table(
            filter=predicate,
        ).to_pandas()
        contact.Frame += offset
        raw.append(contact)
    pose = pd.concat(poses, ignore_index=True).dropna(subset=["Frame", "TrackID", "Bodypoint"])
    pose = pose[pose.Bodypoint.between(0, 9)].drop_duplicates()
    if pose.duplicated(["Frame", "TrackID", "Bodypoint"]).any():
        raise ValueError("Conflicting poses share Frame/TrackID/Bodypoint; inspect tracking before contact review")
    ids = np.sort(pose.TrackID.unique()).astype(np.int64)
    if target not in ids:
        raise ValueError(f"T{target} has no pose in this clip")
    n = stop_frame - start_frame
    xy = np.full((n, len(ids), 10, 2), np.nan, dtype=np.float32)
    anchors = np.full((n, len(ids), 2), np.nan, dtype=np.float32)
    f = pose.Frame.to_numpy(np.int64) - start_frame
    ant = np.searchsorted(ids, pose.TrackID.to_numpy(np.int64))
    node = pose.Bodypoint.to_numpy(np.int64)
    xy[f, ant, node] = pose[["X", "Y"]].to_numpy(np.float32)
    anchors[f, ant] = pose[["TrackX", "TrackY"]].to_numpy(np.float32)
    target_index = int(np.searchsorted(ids, target))
    distances = np.full((n, len(ids), len(GEOMETRIES)), np.inf, dtype=np.float32)
    endpoints = np.zeros((n, len(ids), len(GEOMETRIES), 2), dtype=np.int8)
    duplicate = np.zeros((n, len(ids)), dtype=bool)
    available = np.zeros((n, len(ids)), dtype=bool)
    centers = np.linalg.norm(anchors - anchors[:, target_index:target_index+1], axis=-1) * manifest["mm_per_pixel"]
    for index, ant_id in enumerate(ids):
        if ant_id == target:
            continue
        distances[:, index], endpoints[:, index], duplicate[:, index] = pair_geometry(xy[:, target_index], xy[:, index])
        available[:, index] = (np.isfinite(xy[:, target_index, :4]).all(axis=2).sum(axis=1) >= 3) & (np.isfinite(xy[:, index, :4]).all(axis=2).sum(axis=1) >= 3)
    distances *= manifest["mm_per_pixel"]
    distances[~(centers <= 8)] = np.inf
    np.savez_compressed(root / "geometry.npz", ids=ids, xy=xy, anchors=anchors, distances=distances,
                        endpoints=endpoints, duplicate=duplicate, available=available)
    raw = pd.concat(raw, ignore_index=True)
    raw.to_parquet(root / "all_raw_contacts.parquet", index=False)
    raw = raw[(raw.antenna_track_id == target) | (raw.body_track_id == target)]
    raw.to_parquet(root / "raw_contacts.parquet", index=False)
    bouts, _ = rs.load_contact_bouts(all_chunks, rs.ReturnSettings(fps=fps),
                                    block_dir / "stitched" / "analysis_cache" / "return_sleep")
    bouts[(bouts.start_frame < stop_frame) & (bouts.end_frame >= start_frame)].to_parquet(root / "all_analysis_bouts.parquet", index=False)
    bouts = bouts[((bouts.ant_a == target) | (bouts.ant_b == target)) & (bouts.start_frame < stop_frame)
                  & (bouts.end_frame >= start_frame)].copy()
    bouts["partner_id"] = np.where(bouts.ant_a == target, bouts.ant_b, bouts.ant_a)
    bouts.to_parquet(root / "analysis_bouts.parquet", index=False)
    manifest["raw_directed_rows"] = len(raw)
    manifest["unique_pair_frames"] = len(raw.assign(partner_id=np.where(raw.antenna_track_id == target, raw.body_track_id, raw.antenna_track_id)).drop_duplicates(["Frame", "partner_id"]))
    manifest["exact_duplicate_pose_frames"] = int(duplicate.sum())
    # Write the completion marker last; an interrupted preparation is rebuilt.
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


class ContactReview:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.start = self.manifest["start_frame"]
        self.stop = self.manifest["stop_frame"]
        self.target = self.manifest["target"]
        self.fps = self.manifest["fps"]
        with np.load(self.root / "geometry.npz") as data:
            for name in data.files:
                setattr(self, name, data[name])
        self.target_index = int(np.searchsorted(self.ids, self.target))
        self.analysis_bouts = pd.read_parquet(self.root / "analysis_bouts.parquet")
        background_path = self.root / "all_analysis_bouts.parquet"
        self.background_bouts = pd.read_parquet(background_path) if background_path.is_file() else pd.DataFrame()
        self.background_hits = {}
        if (self.root / "all_raw_contacts.parquet").is_file():
            background = pd.read_parquet(self.root / "all_raw_contacts.parquet")
            background["a"] = np.minimum(background.antenna_track_id, background.body_track_id)
            background["b"] = np.maximum(background.antenna_track_id, background.body_track_id)
            background = background.drop_duplicates(["Frame", "a", "b"])
            self.background_hits = {int(f): g[["a", "b"]].to_numpy(np.int64) for f, g in background.groupby("Frame")}
        raw = pd.read_parquet(self.root / "raw_contacts.parquet")
        self.raw_hits = np.zeros(self.available.shape, dtype=bool)
        partners = np.where(raw.antenna_track_id == self.target, raw.body_track_id, raw.antenna_track_id)
        indices = np.searchsorted(self.ids, partners)
        keep = (indices < len(self.ids))
        keep &= self.ids[np.minimum(indices, len(self.ids)-1)] == partners
        self.raw_hits[raw.Frame.to_numpy(np.int64)[keep] - self.start, indices[keep]] = True
        self.apply()

    def apply(self, *, geometry=0, distance_mm=DEFAULT_MICRO_DISTANCE_MM, gap_seconds=2.0, min_hits=1, exclude_duplicates=True):
        if geometry not in range(len(GEOMETRIES)) or not np.isfinite(distance_mm) or distance_mm <= 0:
            raise ValueError("Choose a geometry and a positive finite distance")
        if not np.isfinite(gap_seconds) or not 0 < gap_seconds <= 30 or int(min_hits) != min_hits or min_hits < 1:
            raise ValueError("Gap must be >0 and <=30 s; minimum hits must be a positive integer")
        self.settings = dict(geometry=geometry, distance_mm=distance_mm, gap_seconds=gap_seconds,
                             min_hits=min_hits, exclude_duplicates=exclude_duplicates)
        self.geometry = geometry
        self.hits = self.distances[:, :, geometry] <= distance_mm
        if exclude_duplicates:
            self.hits &= ~self.duplicate
        gap = max(1, round(gap_seconds * self.fps))
        observed = self.available.copy()
        if geometry != 2:
            antenna_valid = np.isfinite(self.xy[:, :, 4:]).all(axis=3).sum(axis=2) >= 3
            observed &= antenna_valid & antenna_valid[:, self.target_index:self.target_index+1]
        rows = []
        for index, ant in enumerate(self.ids):
            if ant == self.target:
                continue
            bouts = detection_bouts(np.flatnonzero(self.hits[:, index]) + self.start, gap_frames=gap)
            for row in bouts.to_dict("records"):
                local = row["start_frame"] - self.start
                prior_observed = local >= gap and observed[local-gap:local, index].all()
                row.update(partner_id=int(ant), kept=row["n_detection_frames"] >= min_hits,
                           observed_separation=bool(prior_observed),
                           counted=bool(prior_observed and row["n_detection_frames"] >= min_hits))
                rows.append(row)
        self.trial_bouts = pd.DataFrame(rows, columns=["start_frame", "end_frame", "n_detection_frames", "partner_id",
                                                     "kept", "observed_separation", "counted"])

    def export(self) -> Path:
        root = self.root / "reviews" / rs.cache_key(self.settings)
        root.mkdir(parents=True, exist_ok=True)
        self.trial_bouts.to_csv(root / "trial_bouts.csv", index=False)
        (root / "settings.json").write_text(json.dumps(self.settings, indent=2) + "\n")
        return root


class AllPairContactReview:
    """All unordered pairs in an existing pose clip, independent of its focal ID."""

    def __init__(self, root: Path):
        self.source = ContactReview(root)
        for name in ("root", "manifest", "start", "stop", "target", "fps", "ids", "xy", "anchors"):
            setattr(self, name, getattr(self.source, name))
        key = rs.cache_key(dict(version=1, pose=rs.fingerprint(self.root / "geometry.npz"),
                                scale=self.manifest["mm_per_pixel"]))
        path = self.root / f"all_pair_geometry_{key}.npz"
        if not path.is_file():
            self._build_geometry(path)
        with np.load(path) as data:
            for name in data.files:
                setattr(self, name, data[name])
        body = np.isfinite(self.xy[:, :, :4]).all(axis=3).sum(axis=2) >= 3
        antenna = np.isfinite(self.xy[:, :, 4:]).all(axis=3).sum(axis=2) >= 3
        a, b = self.pair_indices.T
        self.body_observed = body[:, a] & body[:, b]
        self.antenna_observed = antenna[:, a] & antenna[:, b]
        self.apply()

    def _build_geometry(self, path):
        pairs = []
        scale = self.manifest["mm_per_pixel"]
        n = len(self.xy)
        for a in range(len(self.ids)):
            for b in range(a+1, len(self.ids)):
                d2 = np.square(self.anchors[:, a] - self.anchors[:, b]).sum(axis=1) * scale**2
                if (d2 <= 8**2).any():
                    pairs.append((a, b))
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        distances = np.full((n, len(pairs), 3), np.inf, dtype=np.float32)
        endpoints = np.zeros((n, len(pairs), 3, 2), dtype=np.int8)
        duplicate = np.zeros((n, len(pairs)), dtype=bool)
        for index, (a, b) in enumerate(pairs):
            d2 = np.square(self.anchors[:, a] - self.anchors[:, b]).sum(axis=1) * scale**2
            frames = np.flatnonzero(d2 <= 8**2)
            d, e, dup = pair_geometry(self.xy[frames, a], self.xy[frames, b])
            distances[frames, index] = d * scale
            endpoints[frames, index] = e
            duplicate[frames, index] = dup
            if (index+1) % 100 == 0:
                print(f"All-pair geometry: {index+1}/{len(pairs)}", flush=True)
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, pair_indices=pairs, distances=distances, endpoints=endpoints, duplicate=duplicate)
        temp.replace(path)

    def apply(self, *, geometry=0, distance_mm=DEFAULT_MICRO_DISTANCE_MM, gap_seconds=2.0,
              min_hits=1, exclude_duplicates=True, require_observed_separation=False):
        # Reuse the focal implementation's validation and retain its matching settings.
        self.source.apply(geometry=geometry, distance_mm=distance_mm, gap_seconds=gap_seconds,
                          min_hits=min_hits, exclude_duplicates=exclude_duplicates)
        self.settings = {**self.source.settings, "all_pairs": True,
                         "require_observed_separation": require_observed_separation}
        self.geometry = geometry
        self.hits = self.distances[:, :, geometry] <= distance_mm
        if exclude_duplicates:
            self.hits &= ~self.duplicate
        observed = self.body_observed.copy()
        if geometry != 2:
            observed &= self.antenna_observed
        gap = max(1, round(gap_seconds*self.fps))
        rows = []
        for index, (a, b) in enumerate(self.pair_indices):
            bouts = detection_bouts(np.flatnonzero(self.hits[:, index]) + self.start, gap_frames=gap)
            missing = np.r_[0, np.cumsum(~observed[:, index])]
            for row in bouts.to_dict("records"):
                local = row["start_frame"] - self.start
                verified = local >= gap and missing[local] == missing[local-gap]
                kept = row["n_detection_frames"] >= min_hits
                row.update(pair_index=index, ant_a=int(self.ids[a]), ant_b=int(self.ids[b]), kept=kept,
                           observed_separation=bool(verified),
                           counted=bool(kept and local >= gap and (verified or not require_observed_separation)))
                rows.append(row)
        columns = ["start_frame", "end_frame", "n_detection_frames", "pair_index", "ant_a", "ant_b", "kept", "observed_separation", "counted"]
        self.trial_bouts = pd.DataFrame(rows, columns=columns).sort_values("start_frame").reset_index(drop=True)
        for name in ("kept", "observed_separation", "counted"):
            self.trial_bouts[name] = self.trial_bouts[name].astype(bool)

    def export(self):
        root = self.root / "reviews" / rs.cache_key(self.settings)
        root.mkdir(parents=True, exist_ok=True)
        self.trial_bouts.to_csv(root / "trial_bouts.csv", index=False)
        (root / "settings.json").write_text(json.dumps(self.settings, indent=2)+"\n")
        return root
