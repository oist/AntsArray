#!/usr/bin/env python3
"""Read A100 ArUco tag IDs from close-up photos of dead ants and loose tags.

Photos are not like the arena videos: a tag is 25-300 px wide, glare and curled film are common,
and OpenCV's own decoder rejects many real tags. So every quad OpenCV finds, accepted or rejected,
is read again here: warped to an 8x8 grid (white margin ring, black border ring, 4x4 bits),
thresholded, and the bits matched against every dictionary code in all four rotations. Each tag
is then re-detected in a crop around it at several scales, with and without CLAHE, and every
read within 1 bit of a code is a vote.

  ok          the winning ID has >= 2 exact votes and other IDs have < 25 % of its votes
  check       the winner has only 1-bit-off votes, a single vote, a strong competitor, or is
              read mirrored (a tag seen from behind through its clear film)
  unreadable  a black-bordered square in a white margin with no read within 1 bit (read by eye)

Mirror images: IDs 0 6 21 38 47 67 92 97 are their own mirror image, so they read the same from
behind; 8/22 and 10/88 are each other's mirror image 1 bit off. A read votes mirrored only when
the mirrored bits match a code strictly better than the bits as seen.

The npz `patterns` array is 1 = white; reference bits come from aruco.generateImageMarker instead.
"""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

CELL = 12                 # px per cell in the warped grid
PROC_LONG_SIDE = 2048     # first detection pass at this long side, the second at twice it
MIN_SIDE = 12             # px at full resolution; smaller quads are bit blobs or dust
MAX_ASPECT = 1.8          # longest / shortest quad side
CAND_BORDER, CAND_MARGIN = 0.8, 0.6     # a quad worth re-reading
VOTE_BORDER = 0.85        # a read may vote when >= 17 of the 20 border cells are black
# Speckled UV prints read part of the black border as white; a complete white margin ring around
# the quad is strong evidence of a tag, so the border may then drop to 12 of 20 cells.
WEAK_BORDER, FULL_MARGIN = 0.6, 0.9
SHOW_BORDER, SHOW_MARGIN, SHOW_SIDE = 0.8, 0.9, 16    # an unread quad reported as "unreadable"
REFINE_SIDES = (60, 100, 160)           # marker side (px) a crop is scaled to
COMPETITOR_FRAC = 0.25
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
DEFAULT_NPZ = (Path(__file__).resolve().parents[2] / "aruco_detection" / "custom_dicts"
               / "custom_4x4_A100_d4_20260410_103938.npz")


@dataclass(frozen=True)
class Decoder:
    dictionary: aruco.Dictionary
    refs: np.ndarray            # (n ids, 4 rotations, 4, 4), 1 = black
    full: aruco.ArucoDetector   # whole photo
    crop: aruco.ArucoDetector   # crop around one tag


@dataclass(frozen=True)
class Grid:
    bits: np.ndarray            # inner 4x4, 1 = black
    border_black: float         # fraction of the 20 border-ring cells that are black
    margin_white: float         # fraction of the 28 margin-ring cells that are white


@dataclass(frozen=True)
class Cand:
    quad: np.ndarray
    side: float
    grid: Grid


@dataclass(frozen=True)
class TagRead:
    tag_id: int | None
    status: str                 # ok | check | unreadable
    d: int | None               # 0 exact, 1 one bit off
    votes: int
    alt: str                    # competing IDs, "41x2 7x1"
    x: float
    y: float
    side: float
    quad: np.ndarray            # full-resolution corners, clockwise
    mirrored: bool = False      # read from behind


def _params(min_rate):
    p = aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 43, 8
    p.minMarkerPerimeterRate, p.maxMarkerPerimeterRate = min_rate, 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.maxErroneousBitsInBorderRate = 0.6
    return p


def load_decoder(npz_path=DEFAULT_NPZ):
    data = np.load(str(npz_path), allow_pickle=True)
    d = aruco.Dictionary()
    d.bytesList = data["bytesList"]
    d.markerSize = 4
    d.maxCorrectionBits = int(data["max_correction_bits"])
    n = d.bytesList.shape[0]
    base = [(aruco.generateImageMarker(d, i, 6, borderBits=1)[1:5, 1:5] < 128).astype(np.uint8)
            for i in range(n)]
    refs = np.stack([np.stack([np.rot90(b, k) for k in range(4)]) for b in base])
    return Decoder(d, refs, aruco.ArucoDetector(d, _params(0.01)), aruco.ArucoDetector(d, _params(0.2)))


def clockwise(quad):
    """Corners in OpenCV's marker order (positive shoelace area with y down)."""
    q = np.asarray(quad, np.float32).reshape(4, 2)
    x, y = q[:, 0], q[:, 1]
    area = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return q if area >= 0 else q[::-1].copy()


def side_lengths(quad):
    return np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)


def warp(gray, quad, cell=CELL):
    """8x8-cell view: one margin cell around the 6x6 marker that quad outlines."""
    dst = np.float32([[cell, cell], [7 * cell, cell], [7 * cell, 7 * cell], [cell, 7 * cell]])
    m = cv2.getPerspectiveTransform(np.float32(quad).reshape(4, 2), dst)
    return cv2.warpPerspective(gray, m, (8 * cell, 8 * cell), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def read_grid(gray, quad, cell=CELL):
    w = warp(gray, quad, cell)
    thr, _ = cv2.threshold(w[cell:7 * cell, cell:7 * cell], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cell // 4
    # median of each cell's centre: a glare speck or blur halo covering < half of it does not flip it
    cells = w.reshape(8, cell, 8, cell)[:, k:cell - k, :, k:cell - k].transpose(0, 2, 1, 3)
    black = np.median(cells.reshape(8, 8, -1), axis=2) < thr
    marker = black[1:7, 1:7]
    border = np.concatenate([marker[0], marker[-1], marker[1:-1, 0], marker[1:-1, -1]])
    margin = np.concatenate([black[0], black[-1], black[1:-1, 0], black[1:-1, -1]])
    return Grid(marker[1:5, 1:5].astype(np.uint8), float(border.mean()), float(1.0 - margin.mean()))


def nearest(bits, refs):
    """(id, distance, distance to the nearest other id, rotation) over all codes and rotations."""
    dist = (refs != np.asarray(bits, np.uint8)[None, None]).sum(axis=(2, 3))
    per_id = dist.min(axis=1)
    best = int(per_id.argmin())
    d2 = int(np.partition(per_id, 1)[1]) if per_id.size > 1 else 16
    return best, int(per_id[best]), d2, int(dist[best].argmin())


def rotation_of(bits, refs, tag_id):
    """Rotation k with bits ~ rot90(reference of tag_id, k), to draw the reference as seen."""
    return int((refs[tag_id] != np.asarray(bits, np.uint8)[None]).sum(axis=(1, 2)).argmin())


def _resize(img, f):
    if abs(f - 1.0) < 1e-6:
        return img
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)


def _framed(g, border):
    """Black border ring inside a white margin ring (see WEAK_BORDER)."""
    return g.border_black >= border or (g.border_black >= WEAK_BORDER and g.margin_white >= FULL_MARGIN)


def _candidate(gray, quad):
    sides = side_lengths(quad)
    if sides.min() < MIN_SIDE or sides.max() > MAX_ASPECT * sides.min():
        return None
    g = read_grid(gray, quad)
    if not _framed(g, CAND_BORDER) or g.margin_white < CAND_MARGIN:
        return None
    return Cand(quad, float(sides.mean()), g)


def candidate_quads(gray, dec):
    """Accepted and rejected quads of two whole-photo passes that look like a bordered tag.
    A tag whose black border runs into an ant leg or body is not a quad to OpenCV and is missed;
    the reviewer adds such tags to review.csv by hand (README)."""
    base = PROC_LONG_SIDE / max(gray.shape)
    cands = []
    for f in (base, 2 * base):
        corners, _, rejected = dec.full.detectMarkers(_resize(gray, f))
        for c in (*corners, *rejected):
            cand = _candidate(gray, clockwise(c.reshape(4, 2) / f))
            if cand is not None:
                cands.append(cand)
    return cands


def cluster(cands):
    """Group quads of the same tag; the best-bordered, largest quad represents the group."""
    order = sorted(cands, key=lambda c: (-round(c.grid.border_black + c.grid.margin_white, 1), -c.side))
    groups = []
    for c in order:
        ctr = c.quad.mean(axis=0)
        for rep, members in groups:
            if np.linalg.norm(rep.quad.mean(axis=0) - ctr) < 0.5 * rep.side:
                members.append(c)
                break
        else:
            groups.append((c, [c]))
    return groups


def _vote(votes, grid, refs):
    """votes[(id, distance, mirrored)] += 1 for a framed read within 1 bit of a code."""
    if not _framed(grid, VOTE_BORDER):
        return
    tag_id, d, _, _ = nearest(grid.bits, refs)
    m_id, m_d, _, _ = nearest(np.fliplr(grid.bits), refs)
    if d <= 1 and d <= m_d:
        votes[(tag_id, d, False)] += 1
    elif m_d <= 1:
        votes[(m_id, m_d, True)] += 1


def refine(gray, rep, members, dec):
    """Votes from the whole-photo reads plus re-detections in a crop at several scales."""
    votes = Counter()
    for c in members:
        _vote(votes, c.grid, dec.refs)
    cx, cy = rep.quad.mean(axis=0)
    half = int(1.6 * rep.side) + 4
    x0, y0 = max(0, int(cx) - half), max(0, int(cy) - half)
    crop = gray[y0:int(cy) + half, x0:int(cx) + half]
    for img in (crop, CLAHE.apply(crop)):
        for target in REFINE_SIDES:
            f = target / rep.side
            g = _resize(img, f)
            ctr = np.array([(cx - x0) * f, (cy - y0) * f])
            corners, _, rejected = dec.crop.detectMarkers(g)
            for c in (*corners, *rejected):
                q = clockwise(c.reshape(4, 2))
                s = side_lengths(q).mean()
                if 0.6 * target <= s <= 1.6 * target and np.linalg.norm(q.mean(axis=0) - ctr) < 0.5 * target:
                    _vote(votes, read_grid(g, q), dec.refs)
    return votes


def summarize(votes):
    """(tag id, status, d, votes for it, competitors, mirrored) from the vote counter."""
    if not votes:
        return None, "unreadable", None, 0, "", False
    per_id, exact, mirrored = Counter(), Counter(), Counter()
    for (i, d, m), n in votes.items():
        per_id[i] += n
        exact[i] += n if d == 0 else 0
        mirrored[i] += n if m else 0
    best = max(per_id, key=lambda i: (exact[i], per_id[i]))
    others = sum(n for i, n in per_id.items() if i != best)
    is_mirror = 2 * mirrored[best] > per_id[best]
    ok = exact[best] >= 2 and others < COMPETITOR_FRAC * per_id[best] and not is_mirror
    alt = " ".join((["mirrored"] if is_mirror else []) +
                   [f"{i}x{n}" for i, n in per_id.most_common() if i != best])
    return best, "ok" if ok else "check", 0 if exact[best] else 1, per_id[best], alt, is_mirror


def _shown(rep):
    g = rep.grid
    return g.border_black >= SHOW_BORDER and g.margin_white >= SHOW_MARGIN and rep.side >= SHOW_SIDE


def _dedupe(reads):
    """One read per physical tag: drop repeats of an ID and unread quads on a read tag. Two real
    tags, even touching, sit >= 1.4 marker sides apart (marker + white margins), so 0.8 sides
    only catches quads of the same tag."""
    kept = []
    for r in sorted(reads, key=lambda r: (r.tag_id is None, -r.votes)):
        near = [k for k in kept if np.hypot(k.x - r.x, k.y - r.y) < 0.8 * max(k.side, r.side)]
        if any(k.tag_id == r.tag_id or r.tag_id is None for k in near):
            continue
        kept.append(r)
    return sorted(kept, key=lambda r: (round(r.y / 50), r.x))


def decode_photo(gray, dec):
    """Every tag in a grayscale photo as TagRead, top to bottom."""
    reads = []
    for rep, members in cluster(candidate_quads(gray, dec)):
        tag_id, status, d, n, alt, mirrored = summarize(refine(gray, rep, members, dec))
        if status == "unreadable" and not _shown(rep):
            continue
        cx, cy = rep.quad.mean(axis=0)
        reads.append(TagRead(tag_id, status, d, n, alt, float(cx), float(cy), rep.side, rep.quad, mirrored))
    return _dedupe(reads)
