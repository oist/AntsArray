"""Calibration registry: which camera-array homography applies to which block.

A calibration is a filming of the ArUco board under ``cameraArray_calib/``;
its directory name carries the board recording date
(``20260623_calib_elevated_by_2mm_from_arenafloor``) and it yields one or more
homography stacks (``frame0/aruco_stitch/aruco_H_mats.npz``, ``initial_H_mats``,
``refined_H_mats``). The npz itself records no date, and a block tracked with a
stale stack is indistinguishable on disk from one tracked with the right one --
so the registry, not the file, has to say what applies when.

Rule: a block's *expected* calibration is the enabled registry entry with the
latest ``valid_from`` on or before the block's recording date. ``valid_from``
defaults to the calibration date and can be overridden per calibration in
``<outdir>/calibration_overrides.csv`` (also ``enabled=false`` to retire one,
``note`` for provenance). Compared against the calibration the tracking record
says was used (lib/tracking_state.py), a difference is ``HMAT_MISMATCH``: the
block should be re-tracked.

Applied AFTER the scan cache, like the label overlay: dropping a new calibration
under ``cameraArray_calib`` (or editing the overrides) re-derives every block's
expected calibration on the next ``catalog.py all`` without a rescan. That is
what makes "update on every calibration" a one-command operation.

Kept python 3.6-compatible: this runs on the deigo login node.
"""
import ast
import csv
import os
import re
import zipfile
from datetime import datetime

from . import classify, const
from .tracking import tracking_state   # shared file fingerprint (sha256/size/mtime)

CALIB_DIRNAME = "cameraArray_calib"
OVERRIDES_FILENAME = "calibration_overrides.csv"
OUTPUT_FILENAME = "calibrations.csv"
MAX_DEPTH = 5   # <calib_id>/<set>/frame0/aruco_stitch/<file> is depth 4

# The homography stacks the tracking pipeline can consume (--hmats), by name.
_H_FILE_RE = re.compile(
    r"^(aruco_H_mats|aruco_board_H_mats|refined_H_mats|initial_H_mats)\.npz$", re.I)
_VARIANT = {"aruco_h_mats": "aruco_stitch", "aruco_board_h_mats": "board",
            "refined_h_mats": "refined", "initial_h_mats": "initial"}
# Older calibration folders are dashed: 2023-12-26-22-42_AruCo_...
_DASHED_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_TRUE = ("1", "true", "yes", "y", "on")


def calib_date_from_name(name):
    """'20260623_calib_x' -> '2026-06-23'; '2023-12-26-22-42_x' -> '2023-12-26'; else ''."""
    d = classify.parse_session_date(name)
    if d.get("date_kind") in ("single", "range", "timestamp") and d.get("date_start"):
        return d["date_start"]
    m = _DASHED_DATE_RE.match(name)
    if m and "2000" <= m.group(1) <= "2099" and "01" <= m.group(2) <= "12" \
            and "01" <= m.group(3) <= "31":
        return "%s-%s-%s" % m.groups()
    return ""


def _valid_date(s):
    """True for a real calendar date written YYYY-MM-DD."""
    if not s or not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return False
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def npz_h_shape(path):
    """Shape of array ``H`` inside an .npz, read from the .npy header only.

    No numpy: the catalog runs where numpy may be absent, and the header is a
    python literal (``{'descr': '<f8', 'fortran_order': False, 'shape': (25, 3, 3)}``).
    Returns a tuple, or None when the file is not an npz with an ``H`` member.
    """
    try:
        with zipfile.ZipFile(path) as z:
            name = "H.npy" if "H.npy" in z.namelist() else None
            if name is None:
                return None
            with z.open(name) as f:
                magic = f.read(6)
                if magic != b"\x93NUMPY":
                    return None
                major = f.read(1)[0]
                f.read(1)  # minor
                if major == 1:
                    n = int.from_bytes(f.read(2), "little")
                else:
                    n = int.from_bytes(f.read(4), "little")
                header = f.read(n).decode("latin1")
        meta = ast.literal_eval(header.strip())
        shape = meta.get("shape")
        return tuple(int(x) for x in shape) if isinstance(shape, tuple) else None
    except (OSError, zipfile.BadZipFile, ValueError, SyntaxError, IndexError, KeyError):
        return None


def _row(root, calib_dir, calib_id, path):
    rel = os.path.relpath(path, root).replace("\\", "/")
    stem = os.path.basename(path).rsplit(".", 1)[0].lower()
    fp = tracking_state.file_fingerprint(path)      # {} when unreadable
    sha, size, computed_at = fp.get("sha256", ""), fp.get("size", ""), fp.get("mtime", "")
    shape = npz_h_shape(path)
    n_cams = shape[0] if shape and len(shape) == 3 and shape[1:] == (3, 3) else ""
    date = calib_date_from_name(calib_id)
    return {
        "calib_id": calib_id, "calib_date": date, "valid_from": date, "enabled": "true",
        "variant": _VARIANT.get(stem, stem), "n_cams": n_cams,
        "hmats_path": path.replace("\\", "/"), "hmats_rel": rel, "sha256": sha,
        "size": size, "computed_at": computed_at,
        "blocks_expected": 0, "blocks_tracked": 0, "note": "",
    }


def discover(root, log=None):
    """One registry row per homography stack under <root>/cameraArray_calib.

    Walks each calibration directory to MAX_DEPTH. Files that are not one of the
    known stacks (previews, montages, the 2023 ``cam_homographies.pkl``) are
    ignored: only what ``--hmats`` can consume is a calibration here.
    """
    base = os.path.join(root, CALIB_DIRNAME)
    rows = []
    if not os.path.isdir(base):
        if log:
            log("[calib] no %s under %s" % (CALIB_DIRNAME, root))
        return rows
    for entry in sorted(os.listdir(base)):
        calib_dir = os.path.join(base, entry)
        if entry.startswith((".", "_")) or not os.path.isdir(calib_dir):
            continue
        for dirpath, dirnames, filenames in os.walk(calib_dir):
            depth = dirpath[len(calib_dir):].count(os.sep)
            if depth >= MAX_DEPTH:
                dirnames[:] = []
            dirnames[:] = sorted(d for d in dirnames if not d.startswith((".", "_")))
            for fn in sorted(filenames):
                if _H_FILE_RE.match(fn):
                    rows.append(_row(root, calib_dir, entry, os.path.join(dirpath, fn)))
    rows.sort(key=lambda r: (r["calib_date"], r["calib_id"], r["variant"], r["hmats_rel"]))
    return rows


def load_overrides(path, log=None):
    """{calib_id: {valid_from, enabled, note}} from calibration_overrides.csv, or {}.

    A calib_id that appears on several lines is merged field by field (later
    lines win only for the fields they set) and reported, so an appended
    correction cannot silently drop an earlier line's valid_from.
    """
    out = {}
    if not path or not os.path.isfile(path):
        return out
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(_strip_comments(f)):
                cid = (row.get("calib_id") or "").strip()
                if not cid:
                    continue
                if cid in out and log:
                    log("[WARN] calibration_overrides: %s appears more than once; "
                        "later fields override earlier ones" % cid)
                o = out.setdefault(cid, {})
                vf = (row.get("valid_from") or "").strip()
                if vf:
                    if not _valid_date(vf):
                        if log:
                            log("[WARN] calibration_overrides: %s valid_from %r is not "
                                "a YYYY-MM-DD calendar date; ignored" % (cid, vf))
                    else:
                        o["valid_from"] = vf
                en = (row.get("enabled") or "").strip()
                if en:
                    o["enabled"] = "true" if en.lower() in _TRUE else "false"
                note = (row.get("note") or "").strip()
                if note:
                    o["note"] = note
    except (OSError, csv.Error) as e:
        if log:
            log("[WARN] calibration_overrides: could not read %s: %s" % (path, e))
    return out


def _strip_comments(lines):
    for line in lines:
        if line.lstrip().startswith("#") or not line.strip():
            continue
        yield line


def apply_overrides(rows, overrides, log=None):
    """Return new rows with per-calibration overrides merged in."""
    known = set(r["calib_id"] for r in rows)
    for cid in sorted(overrides):
        if cid not in known and log:
            log("[WARN] calibration_overrides: %s is not a calibration under %s; ignored"
                % (cid, CALIB_DIRNAME))
    out = []
    for r in rows:
        o = overrides.get(r["calib_id"])
        if not o:
            out.append(dict(r))
            continue
        nr = dict(r)
        nr.update(o)
        out.append(nr)
    return out


_is_full_date = _valid_date


def expected_for(rows, date_start):
    """(calib_id, valid_from) in effect on date_start, or ('', '').

    Latest valid_from on or before the date, enabled entries only. A calibration
    with several stacks (initial + aruco_stitch) counts once: the unit is the
    filming, not the file. Fuzzy dates ('2025-09') get no answer rather than a
    guess.
    """
    if not _is_full_date(date_start):
        return "", ""
    best = ("", "")
    for r in rows:
        if r.get("enabled", "true") != "true":
            continue
        vf = r.get("valid_from", "")
        if not _is_full_date(vf) or vf > date_start:
            continue
        if (vf, r["calib_id"]) > (best[1], best[0]):
            best = (r["calib_id"], vf)
    return best


def apply_to_rows(catalog_rows, calib_rows):
    """Set calib_expected / calib_expected_from on every catalog row, flag
    HMAT_MISMATCH where the tracking record names a different calibration, and
    fill the registry's per-calibration block counts. Returns (n_expected, n_mismatch).

    Rows for the calibration filmings themselves (session_kind 'aux' or a
    'calibration' label) get no expectation: they are the calibrations.
    """
    by_id = dict((r["calib_id"], r) for r in calib_rows)
    for r in calib_rows:
        r["blocks_expected"] = 0
        r["blocks_tracked"] = 0
    n_exp = n_mis = 0
    for row in catalog_rows:
        row.setdefault("calib_expected", "")
        row.setdefault("calib_expected_from", "")
        labels = (row.get("labels") or "").split(const.TOKEN_JOIN)
        if row.get("session_kind") == "aux" or "calibration" in labels:
            continue
        cid, vf = expected_for(calib_rows, row.get("date_start", ""))
        row["calib_expected"], row["calib_expected_from"] = cid, vf
        if cid:
            n_exp += 1
            if cid in by_id:
                by_id[cid]["blocks_expected"] += 1
        used = row.get("tracking_hmats", "")
        if used and used in by_id:
            by_id[used]["blocks_tracked"] += 1
        if used and cid and used != cid:
            flags = [f for f in (row.get("hazard_flags") or "").split(const.TOKEN_JOIN) if f]
            if const.HZ_HMAT_MISMATCH not in flags:
                flags.append(const.HZ_HMAT_MISMATCH)
            row["hazard_flags"] = const.TOKEN_JOIN.join(flags)
            n_mis += 1
    # blocks_expected counts distinct blocks per calibration; a calibration with
    # two stacks shares the count across its rows (same calib_id, same filming).
    for r in calib_rows:
        src = by_id[r["calib_id"]]
        r["blocks_expected"], r["blocks_tracked"] = src["blocks_expected"], src["blocks_tracked"]
    return n_exp, n_mis
