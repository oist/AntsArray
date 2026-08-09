#!/usr/bin/env python3
"""Shared core for the W1.1 PDA-C / TagProbe centroid evaluation harness.

Design contract: group_labelling/EVAL_HARNESS_DESIGN.md

This module owns the pieces every stage needs and that must not drift between
stages: camera discovery, stratum assignment, deterministic anchor sampling,
the ArUco detector configurations, and the optimal-assignment matcher.

WHY A NEW MATCHER. build_training_inventory.compute_sleap_match_metrics uses
``d2.argmin(axis=1)`` followed by a radius test. That is many-to-one: several
SLEAP anchors may claim the same ArUco tag, ``n_matched`` counts each of them
while ``n_unmatched_aruco`` de-duplicates via ``set()``. So TP + FN != |A| and
recall can exceed 1, and the inflation grows with detection density -- i.e. it
systematically flatters whichever model emits more peaks, which is precisely
the axis this study measures. ``match_lsap`` below is exclusive by construction.
Only the two constants (match radius, edge margin) carry over from that module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:  # scipy is present in the sleap-nn env; keep the import failure legible
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "eval_common requires scipy. Use the sleap-nn interpreter:\n"
        "  /apps/unit/ReiterU/sleap-nn/0.3.1/tools/sleap-nn/bin/python"
    ) from exc

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------

BASLER_ROOT = Path("/bucket/ReiterU/Ants/basler/20260515")
ARUCO_DICT_NPZ = Path(
    "/bucket/ReiterU/Ants/aruco_dicts/custom_4x4_A100_d4_20260410_103938.npz"
)
MODEL_A0 = Path("/bucket/ReiterU/Ants/SLEAP_files/Simple_skeleton_nn/250408_141245.centroid")
MODEL_ALL6 = Path(
    "/bucket/ReiterU/Ants/SLEAP_files/Group_labelling/20260803_topdown/models/centroid_all6"
)
MODELS = {"A0": MODEL_A0, "all6": MODEL_ALL6}

BLOCKS = ("block01", "block02", "block03")
CAM_RE = re.compile(r"^(cam\d{2})_")

# Held-out cameras of the ch01-06 arms (build_topdown_arms.DEFAULT_HOLDOUT,
# cross-checked against arms.json "holdout_cameras").
HOLDOUT_CAMERAS = ("cam10", "cam15", "cam21", "cam22", "cam23", "cam25")

# Sampling (spec section 5).
N_REPLICATES = 6
ANCHORS_PER_REPLICATE = 100
BURST_HALF_WIDTH = 3          # anchor drags {f-3 .. f+3}
CORE_HALF_WIDTH = 2           # persistence is judged on {f-2 .. f+2}

# Matching / geometry (spec sections 6.1, 6.3). Full-resolution pixels.
R_PAIR_PRIMARY = 8.0
R_PAIR_ENVELOPE = (5.0, 8.0, 12.0)
R_LINK = 25.0                 # single-linkage cluster radius
R_REF_ENVELOPE = (20.0, 30.0)  # R_tight is measured in Stage 0 and prepended
EDGE_MARGIN = 50.0            # DEFAULT_EDGE_MARGIN, carried over
MATCH_RADIUS_DEFAULT = 30.0   # DEFAULT_MATCH_RADIUS, carried over
BIT_MARGIN_TOL = 6.0          # set-difference tolerance for ECR0 vs ECR1
MEGA_CLUSTER = 8              # |kappa| above which a cluster is flagged MEGA

# Reference tier bitfield (spec section 4).
T_EXACT = 1 << 0
T_CORR = 1 << 1
T_ROSTER = 1 << 2
T_INTERIOR = 1 << 3
T_PERSIST3 = 1 << 4
T_PERSIST5 = 1 << 5
T_GEOM = 1 << 6
T_MOBILE = 1 << 7

# Named reporting tiers. T_ALL additionally requires (EXACT or CORR), which is
# handled in eval_build_reference because it is a disjunction.
TIER_MASKS = {
    "T_ALL": T_ROSTER | T_INTERIOR,
    "T_EXACT": T_ROSTER | T_INTERIOR | T_EXACT,
    "T_STRICT": T_ROSTER | T_INTERIOR | T_EXACT | T_PERSIST3 | T_GEOM | T_MOBILE,
    "T_ULTRA": (T_ROSTER | T_INTERIOR | T_EXACT | T_PERSIST3 | T_PERSIST5
                | T_GEOM | T_MOBILE),
}

ROSTER_FRACTION = 0.005       # (id, side) must appear in >=0.5% of burst frames
ROSTER_SWEEP = (0.002, 0.005, 0.010)
V_MAX_CLAMP = (8.0, 60.0)
MOBILE_Q90_MIN = 100.0        # 90th pct pairwise displacement over the sample
TWO_COLONY_SEP = 200.0        # same-frame duplicate-ID separation
TWO_COLONY_RATE = 0.01
TWO_COLONY_AMBIGUOUS = 0.002


# --------------------------------------------------------------------------
# Detector configurations (spec section 3.2)
# --------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def detector_configs():
    """Return (DET_EXACT, DET_CORR).

    Identical apart from error_correction_rate. DET_CORR permits the single bit
    correction the dictionary was designed for (min_distance=4 > 2 *
    max_correction_bits=2), so a 1-bit correction decodes uniquely and
    correctly; DET_EXACT requires a 16-of-16 match. The set difference is the
    bit margin -- a measured purity axis rather than a hidden filter.
    """
    sys.path.insert(0, str(_repo_root()))
    from run_aruco import DetectorConfig  # noqa: E402  (path-dependent import)

    base = dict(
        corner_refinement="contour",
        adaptive_thresh_constant=3.0,
        adaptive_thresh_win_min=10,
        adaptive_thresh_win_max=40,
        adaptive_thresh_win_step=10,
        min_marker_perimeter_rate=0.03,
        max_marker_perimeter_rate=4.0,
        polygonal_approx_accuracy_rate=0.03,
        max_erroneous_bits_in_border_rate=0.35,
        min_otsu_std_dev=5.0,
    )
    return (
        DetectorConfig(error_correction_rate=0.0, **base),
        DetectorConfig(error_correction_rate=1.0, **base),
    )


def build_detectors():
    """Instantiate (exact, corrected) cv2 ArucoDetectors on the custom dict."""
    sys.path.insert(0, str(_repo_root()))
    from run_aruco import build_aruco_detector, load_custom_aruco_dict  # noqa: E402

    cfg_exact, cfg_corr = detector_configs()
    adict = load_custom_aruco_dict(ARUCO_DICT_NPZ)
    return (
        build_aruco_detector(cfg_exact, aruco_dict=adict),
        build_aruco_detector(cfg_corr, aruco_dict=adict),
    )


# --------------------------------------------------------------------------
# Camera discovery and strata
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraRun:
    block: str
    cam: str
    video: Path
    n_frames: int = -1

    @property
    def key(self) -> str:
        return f"{self.block}_{self.cam}"

    @property
    def stratum(self) -> str:
        """S1/S2/S3 per spec section 5.1, with the provenance correction.

        NOTE (measured 2026-08-05): the ch01-06 training frames came from the
        13-09-56 recording that block00's *_000 symlinks point at, and that
        recording has been deleted -- all 25 symlinks dangle. No frame of
        block01/02/03 was in training. S1 therefore denotes camera-identity
        overlap (same physical camera, different unseen footage), NOT footage
        memorisation. See EVAL_HARNESS_DESIGN.md "Provenance correction".
        """
        if self.block != "block01":
            return "S3"
        return "S2" if self.cam in HOLDOUT_CAMERAS else "S1"


def discover_camera_runs(blocks: Sequence[str] = BLOCKS,
                         cameras: Sequence[str] | None = None) -> list[CameraRun]:
    """Find one .avi per (block, camNN). Skips non-camera videos (global_*)."""
    runs: list[CameraRun] = []
    for block in blocks:
        bdir = BASLER_ROOT / block
        if not bdir.is_dir():
            logging.warning("missing block dir: %s", bdir)
            continue
        by_cam: dict[str, Path] = {}
        for vid in sorted(bdir.glob("*.avi")):
            m = CAM_RE.match(vid.name)
            if not m:
                continue  # global_* panorama, not a colony camera
            cam = m.group(1)
            if cam in by_cam:
                logging.warning("[%s] duplicate video for %s: %s (keeping %s)",
                                block, cam, vid.name, by_cam[cam].name)
                continue
            by_cam[cam] = vid
        for cam in sorted(by_cam):
            if cameras and cam not in cameras:
                continue
            runs.append(CameraRun(block=block, cam=cam, video=by_cam[cam]))
    return runs


# --------------------------------------------------------------------------
# Deterministic interpenetrating systematic sampling (spec section 5)
# --------------------------------------------------------------------------

def replicate_phase(block: str, cam: str, r: int, stride: int) -> int:
    """Stable per-(camera, replicate) phase offset. SHA1 rather than hash() so
    it reproduces across machines and processes (hash() is salted)."""
    h = hashlib.sha1(f"{block}/{cam}/{r}".encode("utf-8")).hexdigest()
    return int(h, 16) % max(1, stride)


def sample_anchors(block: str, cam: str, n_frames: int,
                   n_replicates: int = N_REPLICATES,
                   per_replicate: int = ANCHORS_PER_REPLICATE
                   ) -> tuple[np.ndarray, np.ndarray, int]:
    """Six interpenetrating systematic subsamples over the whole recording.

    Returns (anchors, replicate_id, n_shifted), anchors unique and sorted.
    Collisions between replicates are resolved by shifting one forward a frame
    (counted in n_shifted) so each replicate keeps its full count and the
    across-replicate spread stays a valid design-based variance.

    Uniform stride over the entire recording is mandatory: the colony sleeps in
    clusters at night and disperses by day, so any contiguous window samples a
    single behavioural regime.
    """
    lo = BURST_HALF_WIDTH
    hi = n_frames - 1 - BURST_HALF_WIDTH
    if hi <= lo:
        return np.empty(0, np.int64), np.empty(0, np.int8), 0
    span = hi - lo + 1
    stride = max(1, span // per_replicate)

    taken: set[int] = set()
    anchors: list[int] = []
    reps: list[int] = []
    n_shifted = 0
    for r in range(n_replicates):
        phase = replicate_phase(block, cam, r, stride)
        for k in range(per_replicate):
            f = lo + phase + k * stride
            if f > hi:
                continue
            while f in taken and f <= hi:
                f += 1
                n_shifted += 1
            if f > hi:
                continue
            taken.add(f)
            anchors.append(f)
            reps.append(r)
    order = np.argsort(anchors, kind="stable")
    return (np.asarray(anchors, np.int64)[order],
            np.asarray(reps, np.int8)[order],
            n_shifted)


def burst_frames(anchors: Iterable[int], half: int = BURST_HALF_WIDTH
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Expand anchors into their burst frames.

    Returns (frames, anchor_of_frame), sorted by frame and de-duplicated, so an
    overlapping burst (possible when the stride is tiny) is decoded once.
    """
    seen: dict[int, int] = {}
    for a in anchors:
        for f in range(int(a) - half, int(a) + half + 1):
            seen.setdefault(f, int(a))
    frames = np.fromiter(sorted(seen), np.int64, count=len(seen))
    owner = np.fromiter((seen[int(f)] for f in frames), np.int64, count=len(seen))
    return frames, owner


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

@dataclass
class MatchResult:
    pairs: np.ndarray            # (P, 2) int64, indices into a and b
    unmatched_a: np.ndarray      # (Ua,) int64
    unmatched_b: np.ndarray      # (Ub,) int64
    cost: np.ndarray             # (P,) float64, Euclidean distance per pair
    second_best: np.ndarray      # (P,) float64, next feasible b for that a
    ambiguous: np.ndarray        # (P,) bool, second_best also within radius


def match_lsap(pts_a: np.ndarray, pts_b: np.ndarray, radius: float) -> MatchResult:
    """Exclusive optimal assignment between two point sets, gated at `radius`.

    Guarantees, unlike the greedy argmin matcher it replaces:
      len(pairs) + len(unmatched_a) == len(pts_a)
      len(pairs) + len(unmatched_b) == len(pts_b)
    so TP + FN == |reference| exactly and recall can never exceed 1.

    `second_best` is the distance from a to its next nearest b; `ambiguous`
    marks pairs whose runner-up is also inside the radius, i.e. the assignment
    could plausibly have gone elsewhere. Callers report those as a bracket
    rather than trusting them silently.
    """
    pts_a = np.asarray(pts_a, np.float64).reshape(-1, 2)
    pts_b = np.asarray(pts_b, np.float64).reshape(-1, 2)
    na, nb = len(pts_a), len(pts_b)
    empty_f = np.empty(0, np.float64)
    if na == 0 or nb == 0:
        return MatchResult(np.empty((0, 2), np.int64),
                           np.arange(na, dtype=np.int64),
                           np.arange(nb, dtype=np.int64),
                           empty_f, empty_f, np.empty(0, bool))

    d = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=2)
    # Infeasible entries stay finite (the solver needs a feasible completion)
    # but exceed any achievable feasible total, so the optimum never prefers
    # an out-of-radius pair over an in-radius one.
    big = float(d.max()) * (na + nb) + 1.0
    cost = np.where(d <= radius, d, big)
    ri, ci = linear_sum_assignment(cost)

    keep = d[ri, ci] <= radius
    pi, pj = ri[keep], ci[keep]
    pair_cost = d[pi, pj]

    if len(pi):
        masked = d[pi].copy()
        masked[np.arange(len(pi)), pj] = np.inf
        second = masked.min(axis=1) if nb > 1 else np.full(len(pi), np.inf)
    else:
        second = empty_f

    matched_a = set(pi.tolist())
    matched_b = set(pj.tolist())
    return MatchResult(
        pairs=(np.stack([pi, pj], axis=1).astype(np.int64) if len(pi)
               else np.empty((0, 2), np.int64)),
        unmatched_a=np.array([i for i in range(na) if i not in matched_a],
                             np.int64),
        unmatched_b=np.array([j for j in range(nb) if j not in matched_b],
                             np.int64),
        cost=pair_cost,
        second_best=second,
        ambiguous=(second <= radius) if len(pi) else np.empty(0, bool),
    )


def single_linkage_clusters(pts: np.ndarray, radius: float) -> np.ndarray:
    """Connected components of `pts` under a `radius` linking distance.

    Returns an (N,) int64 label array. Union-find over the thresholded pairwise
    graph -- N is a few dozen per frame, so the O(N^2) distance matrix is cheap
    and this avoids depending on scipy's cluster tie-breaking.
    """
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    n = len(pts)
    if n == 0:
        return np.empty(0, np.int64)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    ii, jj = np.where(np.triu(d <= radius, k=1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    roots: dict[int, int] = {}
    labels = np.empty(n, np.int64)
    for i in range(n):
        labels[i] = roots.setdefault(find(i), len(roots))
    return labels


def in_interior(pts: np.ndarray, width: int, height: int,
                margin: float = EDGE_MARGIN) -> np.ndarray:
    """Symmetric edge mask, applied identically to references and to both
    models' detections -- an asymmetric exclusion would charge a correct edge
    detection as a false positive."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.empty(0, bool)
    x, y = pts[:, 0], pts[:, 1]
    return ((x >= margin) & (x <= width - margin)
            & (y >= margin) & (y <= height - margin))


# --------------------------------------------------------------------------
# Logging / IO conventions used by every stage
# --------------------------------------------------------------------------

def setup_logging(log_dir: Path, name: str) -> Path:
    """Tee stdout to a timestamped log file. Every script in this harness logs;
    runs get reviewed after the fact, not watched live."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )
    logging.info("log file: %s", path)
    return path


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_json_default)


def save_table(df, path: Path) -> Path:
    """Parquet, falling back to CSV where pyarrow is unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as exc:
        csv = path.with_suffix(".csv")
        logging.warning("parquet write failed (%s) -- writing %s", exc, csv.name)
        df.to_csv(csv, index=False)
        return csv
