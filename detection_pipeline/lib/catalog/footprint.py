"""Scan a block's data/ dir (filename-only) for pipeline outputs + completeness.

Never opens .h5/.slp. When no external chunk-count source exists (the case for
newer colony blocks), the deepest stage reached (usually aruco) supplies the
honest expected-chunk denominator, and completeness_state is reported as
"internal" so the number is never mistaken for a duration-based ground truth.
"""
import os

from . import const
from .model import Footprint

_STAGE_ORDER = ["none", "chunk", "sleap", "aruco", "tracks", "stitched", "interactions"]


def _data_files(path):
    """Yield (name, DirEntry) for regular files in data/."""
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_file():
                        yield e.name, e
                except OSError:
                    continue
    except OSError:
        return


def _small(entry):
    try:
        return entry.stat().st_size < const.TRUNCATED_H5_BYTES
    except OSError:
        return False


def _compute_stage(fp, subdirs_lower):
    reached = "none"

    def up(stage):
        nonlocal reached
        if _STAGE_ORDER.index(stage) > _STAGE_ORDER.index(reached):
            reached = stage

    if fp.has_data_dir:
        up("chunk")
    if fp.n_slp:
        up("sleap")
    if fp.n_aruco_det:
        up("aruco")
    if "tracks" in subdirs_lower:
        up("tracks")
    if "stitched" in subdirs_lower:
        up("stitched")
    if "interactions" in subdirs_lower:
        up("interactions")
    return reached


def scan_footprint(unit, check_sizes=False):
    """Return a Footprint for the unit (its block dir + data/ dir)."""
    fp = Footprint()
    subdirs_lower = [s.lower() for s in unit.subdir_names]
    fp.has_hpc_logs = any(h in subdirs_lower for h in const.HPC_LOG_DIRNAMES)
    fp.downstream = sorted({s.lower() for s in unit.subdir_names
                            if s.lower() in const.DOWNSTREAM_DIRS})

    if not unit.has_data_dir or not unit.data_dir:
        fp.has_data_dir = False
        fp.completeness_state = "n/a"
        fp.stage_reached = _compute_stage(fp, subdirs_lower)
        return fp

    fp.has_data_dir = True
    slp, det, trk, sdat = {}, {}, {}, {}
    legacy = False
    truncated = 0
    file_count = 0

    for nm, e in _data_files(unit.data_dir):
        file_count += 1
        if const.LEGACY_FOOTPRINT_RE.search(nm):
            legacy = True
        m = const.SLP_RE.search(nm)
        if m:
            slp.setdefault(nm[:m.start()], set()).add(m.group(1))
            continue
        m = const.ARUCO_DET_RE.search(nm)
        if m:
            det.setdefault(nm[:m.start()], set()).add(m.group(1))
            if check_sizes and _small(e):
                truncated += 1
            continue
        m = const.ARUCO_TRK_RE.search(nm)
        if m:
            trk.setdefault(nm[:m.start()], set()).add(m.group(1))
            continue
        m = const.SLEAP_DATA_RE.search(nm)
        if m:
            sdat.setdefault(nm[:m.start()], set()).add(m.group(1))
            continue

    fp.n_slp = sum(len(v) for v in slp.values())
    fp.n_aruco_det = sum(len(v) for v in det.values())
    fp.n_aruco_tracks = sum(len(v) for v in trk.values())
    fp.n_sleap_data = sum(len(v) for v in sdat.values())
    fp.truncated_files = truncated
    fp.has_legacy_markers = legacy
    fp.file_count = file_count
    # Retain per-video chunk index sets (int) for the recovery layer.
    fp.chunk_sets = {
        "slp": {v: set(int(x) for x in s) for v, s in slp.items()},
        "det": {v: set(int(x) for x in s) for v, s in det.items()},
        "trk": {v: set(int(x) for x in s) for v, s in trk.items()},
        "sdat": {v: set(int(x) for x in s) for v, s in sdat.items()},
    }

    # A modern 4-tuple block is defined by presence of *_aruco_detections.h5.
    if fp.n_aruco_det > 0:
        fp.pipeline_format = "h5_4tuple"
    elif legacy or fp.n_slp or fp.n_sleap_data:
        fp.pipeline_format = "legacy"
    else:
        fp.pipeline_format = "none"

    fp.stage_reached = _compute_stage(fp, subdirs_lower)

    if fp.pipeline_format != "h5_4tuple":
        fp.completeness_state = "n/a" if fp.pipeline_format == "none" else "unverifiable"
        return fp

    # Deepest-stage expected chunks per video (contiguous 0..max assumed).
    vnames = set(slp) | set(det) | set(trk) | set(sdat)
    expected = {}
    for v in vnames:
        idxs = set()
        for d in (slp, det, trk, sdat):
            idxs |= {int(x) for x in d.get(v, ())}
        expected[v] = (max(idxs) + 1) if idxs else 0
    fp.expected_per_video = expected
    fp.expected_total = sum(expected.values())

    present = fp.n_slp + fp.n_aruco_det + fp.n_aruco_tracks + fp.n_sleap_data
    denom = 4 * fp.expected_total
    if denom:
        fp.completeness_pct = round(present / denom, 4)
        fp.completeness_state = "internal"
    else:
        fp.completeness_state = "unverifiable"
    return fp
