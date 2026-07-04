"""Walk basler/ and enumerate catalog units (blocks / flat sessions / aux rows).

One os.scandir per directory. Descends at most: top -> session -> block, or
top -> nested-container -> session -> block. Footprint/probe/sess parsing are
done later (given the paths this module records on each Unit).
"""
import os

from . import const, naming
from .classify import classify_top_entry, parse_session_date
from .model import Unit


def _scandir(path):
    try:
        with os.scandir(path) as it:
            return list(it)
    except OSError:
        return []


def _safe_isdir(entry) -> bool:
    try:
        return entry.is_dir()
    except OSError:
        return False


def _is_regular_file(entry) -> bool:
    try:
        return entry.is_file(follow_symlinks=True)
    except OSError:
        return False


def _is_sess_file(name: str) -> bool:
    low = name.lower()
    if low.startswith("sess_") and low.endswith(".txt"):
        return True
    if low == "stim_timing.txt":
        return True
    return low.endswith(".txt") and "stim" in low


def _dir_has_grid_video(path: str) -> bool:
    for e in _scandir(path):
        if _is_regular_file(e) and naming.is_video(e.name):
            if naming.parse_video_name(e.name).naming_style in ("new", "legacy"):
                return True
    return False


def _apply_date(unit: Unit, top_name: str) -> None:
    d = parse_session_date(top_name)
    unit.date_start = d["date_start"]
    unit.date_end = d["date_end"]
    unit.date_kind = d["date_kind"]
    for lab in d["labels"]:
        if lab not in unit.labels:
            unit.labels.append(lab)


def _discover_block(path, session_id, block, kind, is_test, layout, top_name) -> Unit:
    u = Unit(session_id=session_id, block=block, path=path,
             session_kind=kind, layout=layout, is_test=is_test)
    _apply_date(u, top_name)

    colony, n_global, dead, sess, subdirs = [], 0, 0, [], []
    has_data, data_dir, raw_chunked, legacy_named = False, "", False, False

    for e in _scandir(path):
        nm = e.name
        if _safe_isdir(e):
            subdirs.append(nm)
            if nm.lower() == "data":
                has_data, data_dir = True, e.path
            continue
        if naming.is_video(nm):
            vn = naming.parse_video_name(nm)
            vn.path = e.path
            if vn.is_global:
                if _is_regular_file(e):
                    n_global += 1
                else:
                    dead += 1
                continue
            if vn.naming_style == "unknown":
                continue
            if not _is_regular_file(e):
                dead += 1
                continue
            raw_chunked = raw_chunked or vn.chunk_idx is not None
            legacy_named = legacy_named or vn.naming_style == "legacy"
            colony.append(vn)
        elif _is_sess_file(nm):
            sess.append(e.path)

    u.video_names = colony
    u.n_global = n_global
    u.dead_symlinks = dead
    u.sess_paths = sorted(sess)
    u.subdir_names = subdirs
    u.has_data_dir = has_data
    u.data_dir = data_dir

    footprint_dirs = any(s.lower() in const.PURE_ANALYSIS_DIRS for s in subdirs)
    if u.session_kind == "unknown" and (colony or sess or has_data or footprint_dirs):
        u.session_kind = "session"
    if not colony and (has_data or footprint_dirs) and u.session_kind == "session":
        u.session_kind = "pure_analysis"

    if raw_chunked:
        u.extra_hazards.append(const.HZ_RAW_CHUNKED)
    if dead:
        u.extra_hazards.append(const.HZ_DEAD_SYMLINK)
    if legacy_named:
        u.extra_hazards.append(const.HZ_CAM_NAMING_LEGACY)
    if colony and not sess and u.session_kind == "session":
        u.extra_hazards.append(const.HZ_NO_SESS_FILE)
    return u


def _nonblock_video_targets(path, name):
    """(block_label, video_dir) for a non-block sibling dir.

    Returns the dir itself if it holds grid videos, else its immediate subdirs
    that do (calibration datasets nest one level deeper, e.g.
    20260414_calibration_dataset/set0_patterns_elevated_by_2mm/).
    """
    if _dir_has_grid_video(path):
        return [(name, path)]
    targets = []
    for e in _scandir(path):
        if (_safe_isdir(e) and e.name not in const.WALK_BLACKLIST
                and not e.name.startswith((".", "_")) and _dir_has_grid_video(e.path)):
            targets.append((f"{name}/{e.name}", e.path))
    return targets


def _aux_unit(name, path) -> Unit:
    u = Unit(session_id=name, block="", path=path, session_kind="aux", layout="flat")
    _apply_date(u, name)
    return u


def _scan_session(path, session_id, kind, is_test, top_name) -> list:
    entries = _scandir(path)
    subdirs = [e for e in entries if _safe_isdir(e)]
    block_dirs = [e for e in subdirs if const.BLOCK_DIR_RE.match(e.name)]

    if not block_dirs:
        flat = _discover_block(path, session_id, "", kind, is_test, "flat", top_name)
        # A flat session with videos, or a pure-analysis dir, is one row as-is.
        if flat.video_names or flat.session_kind == "pure_analysis":
            return [flat]
        # No top-level videos: catalog subdirs that nest video sets
        # (calibration datasets, e.g. calibration_dataset/set0_.../*.avi).
        nested = []
        for e in subdirs:
            if e.name in const.WALK_BLACKLIST or e.name.startswith((".", "_")):
                continue
            if _dir_has_grid_video(e.path):
                is_calib = "calib" in (session_id + e.name).lower()
                u = _discover_block(e.path, session_id, e.name,
                                    "aux" if is_calib else kind, is_test, "block", top_name)
                u.extra_hazards.append(const.HZ_NONBLOCK_VIDEO_DIR)
                if is_calib and "calibration" not in u.labels:
                    u.labels.append("calibration")
                nested.append(u)
        return nested if nested else [flat]

    units = [_discover_block(b.path, session_id, b.name, kind, is_test, "block", top_name)
             for b in sorted(block_dirs, key=lambda e: e.name)]

    # Non-block sibling dirs that themselves hold grid videos
    # (e.g. 20260414_calibration_dataset alongside block01..04).
    for sd in subdirs:
        if const.BLOCK_DIR_RE.match(sd.name):
            continue
        if sd.name in const.WALK_BLACKLIST or sd.name.startswith((".", "_")):
            continue
        is_calib = "calib" in sd.name.lower()
        for label, vdir in _nonblock_video_targets(sd.path, sd.name):
            u = _discover_block(vdir, session_id, label,
                                "aux" if is_calib else kind, is_test, "block", top_name)
            u.extra_hazards.append(const.HZ_NONBLOCK_VIDEO_DIR)
            if is_calib and "calibration" not in u.labels:
                u.labels.append("calibration")
            units.append(u)
    return units


def enumerate_units(root, only=None):
    """Return (units, ignored). `only` is an optional set of top-level names."""
    root = os.path.abspath(root)
    units, ignored = [], []
    for entry in sorted(_scandir(root), key=lambda e: e.name):
        name = entry.name
        if only and name not in only:
            continue
        cls = classify_top_entry(name, _safe_isdir(entry))
        if cls.action == "ignore":
            ignored.append({"path": entry.path, "reason": cls.reason})
            continue
        if cls.action == "thin":
            units.append(_aux_unit(name, entry.path))
            continue
        if cls.action == "recurse":
            for child in sorted(_scandir(entry.path), key=lambda e: e.name):
                sid = f"{name}/{child.name}"
                if not _safe_isdir(child):
                    ignored.append({"path": child.path, "reason": "loose file in container"})
                    continue
                ccls = classify_top_entry(child.name, True)
                if ccls.action == "ignore":
                    ignored.append({"path": child.path, "reason": ccls.reason})
                    continue
                units += _scan_session(child.path, sid, ccls.session_kind,
                                       ccls.is_test, child.name)
            continue
        units += _scan_session(entry.path, name, cls.session_kind, cls.is_test, name)
    return units, ignored
