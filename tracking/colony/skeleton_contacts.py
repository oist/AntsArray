"""Shared finished-skeleton geometry for live review and production detection."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import shapely
from shapely.geometry import GeometryCollection, MultiLineString, MultiPoint
from shapely.ops import nearest_points
from shapely.strtree import STRtree


CONTACT_GEOMETRY = "all_skeleton_segments_and_nodes"
SKELETON_EDGES = ((0, 1), (0, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9))


def validate_distance(raw):
    try:
        value = float(raw)
    except (ValueError, TypeError):
        raise ValueError("Distance must be a finite number >= 0") from None
    if not np.isfinite(value) or value < 0:
        raise ValueError("Distance must be a finite number >= 0")
    return value


def validate_scale(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("mm_per_pixel must be positive and finite")
    return value


@dataclass(frozen=True)
class PairDistance:
    ant_a: int
    ant_b: int
    distance_mm: float
    point_a: tuple[float, float] | None
    point_b: tuple[float, float] | None

    def is_hit(self, threshold):
        return np.isfinite(self.distance_mm) and self.distance_mm <= validate_distance(threshold)


def skeleton_geometries(poses, *, edges=SKELETON_EDGES):
    geometries = {}
    for track_id, xy in sorted(poses.items()):
        xy = np.asarray(xy, dtype=float)
        valid = np.isfinite(xy).all(axis=1)
        if not valid.any():
            geometries[int(track_id)] = None
            continue
        segments = [xy[[a, b]] for a, b in edges if valid[a] and valid[b]
                    and not np.array_equal(xy[a], xy[b])]
        parts = [MultiPoint(xy[valid])]
        if segments:
            parts.append(MultiLineString(segments))
        geometries[int(track_id)] = GeometryCollection(parts)
    return geometries


def skeleton_pair_distances(poses, *, mm_per_pixel, edges=SKELETON_EDGES):
    """All unordered pair distances and closest points in tracking coordinates."""
    scale = validate_scale(mm_per_pixel)
    geometries = skeleton_geometries(poses, edges=edges)
    rows = []
    for a, b in combinations(geometries, 2):
        ga, gb = geometries[a], geometries[b]
        if ga is None or gb is None:
            rows.append(PairDistance(a, b, np.nan, None, None))
            continue
        pa, pb = nearest_points(ga, gb)
        rows.append(PairDistance(a, b, float(pa.distance(pb))*scale,
                                 tuple(pa.coords[0]), tuple(pb.coords[0])))
    return sorted(rows, key=lambda row: (not np.isfinite(row.distance_mm),
                                        row.distance_mm if np.isfinite(row.distance_mm) else 0,
                                        row.ant_a, row.ant_b))


def skeleton_contact_pairs(poses, *, distance_mm, mm_per_pixel, edges=SKELETON_EDGES):
    """Exact distance hits, once per pair. Spatial indexing is not a center cutoff."""
    threshold = validate_distance(distance_mm)
    scale = validate_scale(mm_per_pixel)
    geometries = {a: g for a, g in skeleton_geometries(poses, edges=edges).items() if g is not None}
    if len(geometries) < 2:
        return []
    ids = np.array(list(geometries), dtype=np.int64)
    shapes = np.array(list(geometries.values()), dtype=object)
    tree = STRtree(shapes)
    a, b = tree.query(shapes, predicate="dwithin", distance=threshold/scale)
    unique = a < b
    a, b = a[unique], b[unique]
    distances = shapely.distance(shapes[a], shapes[b]) * scale
    return sorted((int(ids[i]), int(ids[j]), float(d)) for i, j, d in zip(a, b, distances) if d <= threshold)
